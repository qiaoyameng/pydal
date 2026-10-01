import datetime

from pydal import DAL, Field, RowValue
from pydal.ast_translate import set_to_select, to_ast
from pydal.compilers.mssql import MSSQLCompiler
from pydal.compilers.sql import ParamSQL

from ._compat import unittest


class RowValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DAL("sqlite:memory")
        cls.db.define_table(
            "rv",
            Field("a", "integer"),
            Field("b", "integer"),
            Field("c", "integer"),
            Field("created_at", "datetime"),
        )
        cls.db.define_table(
            "other",
            Field("x", "integer"),
            Field("y", "integer"),
            Field("z", "integer"),
        )
        cls.rows = [(None, None), (None, 0), (None, 1), (0, None), (0, 0),
                    (0, 1), (1, None), (1, 0), (1, 1)]
        for values in cls.rows:
            cls.db.rv.insert(a=values[0], b=values[1])
        cls.compiler = cls.db._adapter.compiler

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def compile_predicate(self, query, emulated):
        original = self.compiler.supports_row_values
        self.compiler.supports_row_values = not emulated
        try:
            node = set_to_select(self.db(query), (self.db.rv.id), {})
            return self.compiler.compile_select(node)
        finally:
            self.compiler.supports_row_values = original

    def truth_table(self, query, emulated):
        sql = self.compile_predicate(query, emulated)
        command = "SELECT id FROM rv WHERE %s ORDER BY id;" % str(sql).split(" WHERE ", 1)[1][:-1]
        self.db._adapter.execute(sql.__class__(command, getattr(sql, "params", ())))
        rows = self.db._adapter.cursor.fetchall()
        ids = {row[0] for row in rows}
        return {index: index in ids for index in range(1, len(self.rows) + 1)}

    def test_two_column_comparisons_match_native_and_fallback(self):
        operators = (
            ("eq", lambda row, value: row == value),
            ("ne", lambda row, value: row != value),
            ("lt", lambda row, value: row < value),
            ("lte", lambda row, value: row <= value),
            ("gt", lambda row, value: row > value),
            ("gte", lambda row, value: row >= value),
        )
        for value in self.rows:
            for name, _ in operators:
                row_value = RowValue(self.db.rv.a, self.db.rv.b)
                if name == "eq":
                    query = row_value == value
                elif name == "ne":
                    query = row_value != value
                elif name == "lt":
                    query = row_value < value
                elif name == "lte":
                    query = row_value <= value
                elif name == "gt":
                    query = row_value > value
                else:
                    query = row_value >= value
                native = self.truth_table(query, False)
                emulated = self.truth_table(query, True)
                self.assertEqual(native, emulated, (name, value))

    def test_two_column_in_lists_match_native_and_fallback(self):
        row_lists = (
            (),
            ((0, 0),),
            ((0, 1), (1, 0)),
            ((None, None), (0, None), (None, 0), (1, 1)),
        )
        for values in row_lists:
            for negated in (False, True):
                row_value = RowValue(self.db.rv.a, self.db.rv.b)
                query = (
                    row_value.notbelongs(values)
                    if negated else row_value.belongs(values)
                )
                native = self.truth_table(query, False)
                emulated = self.truth_table(query, True)
                self.assertEqual(native, emulated, (values, negated))

    def test_three_column_sample_matches_native_and_fallback(self):
        triples = [
            (a, b, c)
            for a in (None, 0, 1)
            for b in (None, 0, 1)
            for c in (None, 0, 1)
        ]
        self.db.rv.truncate()
        for values in triples:
            self.db.rv.insert(a=values[0], b=values[1], c=values[2])
        samples = (
            (0, 0, 0), (0, None, 1), (1, None, 0), (None, 1, None)
        )
        for value in samples:
            row_value = RowValue(self.db.rv.a, self.db.rv.b, self.db.rv.c)
            for query in (
                row_value == value,
                row_value != value,
                row_value < value,
                row_value <= value,
                row_value > value,
                row_value >= value,
                row_value.belongs(((0, 0, 0), (1, None, 0))),
                row_value.notbelongs(((0, 0, 0), (1, None, 0))),
            ):
                native = self.truth_table(query, False)
                emulated = self.truth_table(query, True)
                self.assertEqual(native, emulated, (value, str(query)))

    def test_empty_and_duplicate_constant_rows_are_deterministic(self):
        row_value = RowValue(self.db.rv.a, self.db.rv.b)
        empty = self.compile_predicate(row_value.belongs([]), True)
        self.assertIn("(1=0)", empty)
        self.assertEqual(empty.params, ())
        empty_not = self.compile_predicate(row_value.notbelongs([]), True)
        self.assertIn("(1=1)", empty_not)
        self.assertEqual(empty_not.params, ())

        values = [(1, 2), (0, None), (1, 2), (0, None)]
        sql = self.compile_predicate(row_value.belongs(values), False)
        self.assertEqual(sql.params, (0, 1, 2))
        self.assertEqual(sql.count("?"), 3)
        self.assertEqual(str(sql).count("NULL"), 1)
        sql_from_set = self.compile_predicate(row_value.belongs(set(values)), False)
        self.assertEqual(sql, sql_from_set)
        self.assertEqual(sql_from_set.params, sql.params)
        self.assertNotIn("'2026'", sql)

    def test_user_values_use_bound_parameters(self):
        malicious = "2026-10-02 00:00:00'); DROP TABLE rv;--"
        row_value = RowValue(self.db.rv.a, self.db.rv.created_at)
        sql = self.compile_predicate(row_value < (1, malicious), False)
        self.assertIsInstance(sql, ParamSQL)
        self.assertNotIn(malicious, sql)
        self.assertEqual(sql.params[-2:], (1, malicious))

    def test_aliased_left_columns(self):
        a = self.db.rv.a.with_alias("x")
        b = self.db.rv.b.with_alias("y")
        query = RowValue(a, b) == (1, 2)
        sql = self.compile_predicate(query, True)
        self.assertNotIn(" AS x", sql)
        self.assertEqual(sql.params, (1, 2))
        self.assertEqual(self.truth_table(query, True), self.truth_table(query, False))

    def test_select_field_aliases_remain_in_select_list(self):
        query = RowValue(self.db.rv.a, self.db.rv.b) == (1, 2)
        sql = self.compiler.compile_select(
            set_to_select(
                self.db(query),
                (
                    self.db.rv.a.with_alias("aa"),
                    self.db.rv.b.with_alias("bb"),
                ),
                {},
            )
        )
        self.assertIn('"rv"."a" AS aa', sql)
        self.assertIn('"rv"."b" AS bb', sql)
        self.assertEqual(sql.params, (1, 2))

    def test_ast_and_legacy_subqueries(self):
        for subquery in (
            self.db(self.db.other.x == self.db.rv.a).subselect(
                self.db.other.x, self.db.other.y
            ),
            self.db(self.db.other.x == self.db.rv.a).nested_select(
                self.db.other.x, self.db.other.y
            ),
        ):
            self.db.other.truncate()
            self.db.other.insert(x=1, y=1)
            self.db.other.insert(x=0, y=None)
            row_value = RowValue(self.db.rv.a, self.db.rv.b)
            for negated in (False, True):
                query = (
                    row_value.notbelongs(subquery)
                    if negated else row_value.belongs(subquery)
                )
                native = self.truth_table(query, False)
                emulated = self.truth_table(query, True)
                self.assertEqual(native, emulated, (type(subquery), negated))

    def test_independent_nested_subquery(self):
        subquery = self.db(self.db.other.y > 0).subselect(
            self.db.other.x, self.db.other.y
        )
        query = RowValue(self.db.rv.a, self.db.rv.b).belongs(subquery)
        self.assertEqual(self.truth_table(query, False), self.truth_table(query, True))

    def test_nested_subquery_with_known_dimensions(self):
        self.db.define_table(
            "middle",
            Field("x", "integer"),
            Field("y", "integer"),
        )
        self.db.define_table(
            "inner",
            Field("a", "integer"),
            Field("b", "integer"),
        )
        self.db.middle.insert(x=1, y=2)
        self.db.inner.insert(a=1, b=2)
        inner = self.db(self.db.inner.a > 0).subselect(
            self.db.inner.a, self.db.inner.b, correlated=False
        )
        middle = self.db(
            RowValue(self.db.middle.x, self.db.middle.y).belongs(inner)
        ).subselect(self.db.middle.x, self.db.middle.y, correlated=False)
        query = RowValue(self.db.rv.a, self.db.rv.b).belongs(middle)
        native = self.truth_table(query, False)
        emulated = self.truth_table(query, True)
        self.assertEqual(native, emulated)

    def test_dimension_and_type_validation(self):
        with self.assertRaises(ValueError):
            RowValue(self.db.rv.a)
        with self.assertRaises(ValueError):
            RowValue(self.db.rv.a, self.db.rv.b) == (1,)
        with self.assertRaises(ValueError):
            RowValue(self.db.rv.a, self.db.rv.b) == (1, 2, 3)
        with self.assertRaises(ValueError):
            RowValue(self.db.rv.a, self.db.rv.b).belongs(
                self.db(self.db.other.x > 0).subselect(self.db.other.x)
            )
        with self.assertRaises(ValueError):
            RowValue(self.db.rv.a, self.db.rv.b) < ("bad", 2)
        with self.assertRaises(TypeError):
            RowValue(self.db.rv.a, self.db.rv.b).belongs(1, 2)

    def test_single_column_comparison_unchanged(self):
        sql = self.compiler.compile_select(
            set_to_select(
                self.db(self.db.rv.a == 1), (self.db.rv.id,), {}
            )
        )
        self.assertIn('("rv"."a" = ?)', sql)
        self.assertEqual(sql.params, (1,))

    def test_selected_unsupported_dialect_uses_fallback(self):
        compiler = MSSQLCompiler(
            adapter=self.db._adapter,
            represent=self.db._adapter.represent,
            parameterize=True,
        )
        self.assertFalse(compiler.supports_row_values)
        query = RowValue(self.db.rv.a, self.db.rv.b) < (1, 2)
        sql = compiler.compile_expression(to_ast(query))
        self.assertNotIn("(?,?)", sql)
        self.assertIn(" OR ", sql)
        self.assertIn("%s", sql)
        self.assertNotIn("?", sql)
        self.assertEqual(sql.params, (1, 1, 2))


if __name__ == "__main__":
    unittest.main()
