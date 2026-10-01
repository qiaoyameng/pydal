# -*- coding: utf-8 -*-
"""Tests for typed SQL row-value expressions."""

import itertools

from pydal import DAL, Field, RowValue
from pydal import ast
from pydal.ast_translate import to_ast
from pydal.compilers.sql import ParamSQL
from pydal.compilers.sqlite import SQLiteCompiler

from ._adapt import IS_NOSQL
from ._compat import unittest


@unittest.skipIf(IS_NOSQL, "SQL-only")
class TestRowValues(unittest.TestCase):
    def setUp(self):
        self.db = DAL("sqlite:memory")
        self.db.define_table(
            "thing",
            Field("created_at", "datetime"),
            Field("seq", "integer"),
            Field("name"),
            Field("extra", "integer"),
        )
        self.db.define_table(
            "cursor_page",
            Field("created_at", "datetime"),
            Field("seq", "integer"),
        )

    def tearDown(self):
        self.db.close()

    def test_constant_comparison_is_native_and_parameterized(self):
        query = RowValue(self.db.thing.created_at, self.db.thing.seq) > (
            "2024-01-01 00:00:00",
            7,
        )
        sql = self.db._adapter._select_wcols(
            query, [self.db.thing.id], {}
        )[1]
        self.assertIsInstance(sql, ParamSQL)
        self.assertEqual(sql.params, ("2024-01-01 00:00:00", 7))
        self.assertIn('("thing"."created_at","thing"."seq") > (?,?)', sql)
        self.assertNotIn("2024-01-01 00:00:00", str(sql).replace("?", ""))
        self.assertNotIn("(?,7)", str(sql))

    def test_row_in_deduplicates_retains_order_and_binds_row_major(self):
        rows = [
            ("2024-01-02 00:00:00", 2),
            ("2024-01-01 00:00:00", 1),
            ("2024-01-02 00:00:00", 2),
        ]
        query = RowValue(self.db.thing.created_at, self.db.thing.seq).belongs(rows)
        sql = self.db._adapter._select_wcols(
            query, [self.db.thing.id], {}
        )[1]
        self.assertEqual(
            sql.params,
            ("2024-01-02 00:00:00", 2, "2024-01-01 00:00:00", 1),
        )
        self.assertEqual(sql.count("(?,?)"), 2)

    def test_empty_row_in_and_not_in_are_deterministic(self):
        row = RowValue(self.db.thing.created_at, self.db.thing.seq)
        in_sql = self.db._adapter._select_wcols(row.belongs([]), [self.db.thing.id], {})[1]
        not_in_sql = self.db._adapter._select_wcols(
            row.not_belongs([]), [self.db.thing.id], {}
        )[1]
        self.assertIn("(0=1)", in_sql)
        self.assertIn("(1=1)", not_in_sql)
        self.assertEqual(in_sql.params, ())
        self.assertEqual(not_in_sql.params, ())

    def test_native_and_fallback_two_column_null_truth_tables_match(self):
        self._assert_null_truth_table(2)

    def test_native_and_fallback_three_column_null_truth_tables_match(self):
        self._assert_null_truth_table(3)

    def _assert_null_truth_table(self, arity):
        values = (None, 0, 1)
        fields = tuple(
            getattr(self.db.thing, "created_at" if index == 0 else (
                "seq" if index == 1 else "extra"
            ))
            for index in range(arity)
        )
        types = tuple(field.type for field in fields)
        compiler = SQLiteCompiler(adapter=self.db._adapter)
        compiler.supports_row_values = False
        symbols = {
            "eq": "=",
            "ne": "<>",
            "lt": "<",
            "lte": "<=",
            "gt": ">",
            "gte": ">=",
        }
        for left_values in itertools.product(values, repeat=arity):
            for right_values in itertools.product(values, repeat=arity):
                for op, symbol in symbols.items():
                    native = self._native_comparison_result(
                        arity, left_values, right_values, symbol
                    )
                    fallback = self._fallback_comparison_result(
                        compiler,
                        fields,
                        types,
                        op,
                        left_values,
                        right_values,
                    )
                    self.assertEqual(
                        fallback,
                        native,
                        "%s %s %s" % (left_values, symbol, right_values),
                    )

    def _comparison_node(self, fields, types, op, right_values):
        left = ast.RowValue(
            tuple(
                ast.FieldRef("thing", field.name, sqlsafe=field._rname, type=field.type)
                for field in fields
            ),
            types,
        )
        right = ast.RowValue(
            tuple(ast.Literal(value, field_type) for value, field_type in zip(right_values, types)),
            types,
        )
        return ast.RowCompare(op, left, right)

    def _native_comparison_result(self, arity, left, right, symbol):
        def literal(value):
            return "CAST(%s AS INTEGER)" % (
                "NULL" if value is None else str(value)
            )

        row = lambda values: "(%s)" % ",".join(literal(v) for v in values)
        predicate = "%s %s %s" % (row(left), symbol, row(right))
        return self._three_value_result(predicate, arity)

    def _fallback_comparison_result(
        self, compiler, fields, types, op, left_values, right_values
    ):
        sql = compiler.compile_expression(
            self._comparison_node(fields, types, op, right_values)
        )
        columns = ", ".join(
            "CAST(NULL AS INTEGER) AS %s" % field.name
            if value is None
            else "CAST(%d AS INTEGER) AS %s" % (value, field.name)
            for value, field in zip(left_values, fields)
        )
        return self._three_value_result(sql, len(fields), columns, sql.params)

    def _three_value_result(self, predicate, arity, columns=None, params=()):
        if columns is None:
            columns = ", ".join(
                "CAST(NULL AS INTEGER) AS c%d" % index
                for index in range(arity)
            )
        cursor = self.db._adapter.cursor
        sql = (
            "SELECT CASE WHEN %s THEN 'true' "
            "WHEN NOT (%s) THEN 'false' ELSE 'unknown' END FROM (SELECT %s)"
            % (predicate, predicate, columns)
        )
        if params:
            repeat = sql.count("?") // len(params)
            params = tuple(params) * repeat
        cursor.execute(sql, params)
        return cursor.fetchone()[0]

    def test_native_and_fallback_row_in_null_truth_table_matches(self):
        values = (None, 0, 1)
        fields = (self.db.thing.seq, self.db.thing.extra)
        types = tuple(field.type for field in fields)
        compiler = SQLiteCompiler(adapter=self.db._adapter)
        compiler.supports_row_values = False
        row_sets = (
            [],
            [(0, 0)],
            [(0, None)],
            [(0, 0), (0, None)],
            [(0, 0), (1, 1)],
        )
        for left_values in itertools.product(values, repeat=2):
            for rows in row_sets:
                native = self._native_in_result(left_values, rows, False)
                fallback = self._fallback_in_result(
                    compiler, fields, types, left_values, rows, False
                )
                self.assertEqual(fallback, native, (left_values, rows, "IN"))
                native_not = self._native_in_result(left_values, rows, True)
                fallback_not = self._fallback_in_result(
                    compiler, fields, types, left_values, rows, True
                )
                self.assertEqual(
                    fallback_not, native_not, (left_values, rows, "NOT IN")
                )

    def _in_predicate(self, left, rows, negate):
        def literal(value):
            return "CAST(%s AS INTEGER)" % (
                "NULL" if value is None else str(value)
            )

        row = lambda values: "(%s)" % ",".join(literal(v) for v in values)
        right = "(" + ",".join(row(row_values) for row_values in rows) + ")"
        return "%s%s IN %s" % (
            row(left),
            " NOT" if negate else "",
            right,
        )

    def _native_in_result(self, left, rows, negate):
        return self._three_value_result(
            self._in_predicate(left, rows, negate),
            2,
            ", ".join(
                "CAST(NULL AS INTEGER) AS %s" % field.name
                if value is None
                else "CAST(%d AS INTEGER) AS %s" % (value, field.name)
                for value, field in zip(left, (self.db.thing.seq, self.db.thing.extra))
            ),
        )

    def _fallback_in_result(self, compiler, fields, types, left, rows, negate):
        left_node = ast.RowValue(
            tuple(
                ast.FieldRef("thing", field.name, sqlsafe=field._rname, type=field.type)
                for field in fields
            ),
            types,
        )
        values = tuple(
            ast.RowValue(
                tuple(ast.Literal(value, field_type) for value, field_type in zip(row, types)),
                types,
            )
            for row in rows
        )
        sql = compiler.compile_expression(
            ast.RowInList(left_node, values, negate=negate)
        )
        columns = ", ".join(
            "CAST(NULL AS INTEGER) AS %s" % field.name
            if value is None
            else "CAST(%d AS INTEGER) AS %s" % (value, field.name)
            for value, field in zip(left, fields)
        )
        return self._three_value_result(sql, 2, columns, sql.params)

    def test_aliased_expression_can_be_a_row_member(self):
        aliased = (self.db.thing.seq + 1).with_alias("next_seq")
        query = RowValue(self.db.thing.name, aliased) == ("alice", 2)
        node = to_ast(query)
        self.assertIsInstance(node, ast.RowCompare)
        sql = self.db._adapter.compiler.compile_expression(node)
        self.assertIn(" AS next_seq", sql)
        self.assertEqual(sql.params, (1, "alice", 2))

    def test_row_in_accepts_subquery_with_known_arity(self):
        sub = self.db(self.db.thing.name == "alice").subselect(
            self.db.cursor_page.created_at, self.db.cursor_page.seq
        )
        query = RowValue(self.db.thing.created_at, self.db.thing.seq).belongs(sub)
        sql = self.db._adapter._select_wcols(query, [self.db.thing.id], {})[1]
        self.assertIn(" IN (SELECT", sql)
        self.assertIn("?", sql)
        self.assertEqual(sql.params, ("alice",))
        self.assertEqual(len(self.db(query).select()), 0)

    def test_nested_legacy_subquery_works(self):
        sub = self.db(self.db.thing.name == "alice").nested_select(
            self.db.cursor_page.created_at, self.db.cursor_page.seq
        )
        query = RowValue(self.db.thing.created_at, self.db.thing.seq).belongs(sub)
        rows = self.db(query).select()
        self.assertEqual(len(rows), 0)

    def test_fallback_row_in_and_comparison_subqueries_execute(self):
        self.db.thing.insert(created_at="2024-01-01 00:00:00", seq=1, name="a")
        self.db.cursor_page.insert(created_at="2024-01-01 00:00:00", seq=1)
        self.db._adapter.compiler.supports_row_values = False
        try:
            sub = self.db(self.db.thing.name == "a").subselect(
                self.db.cursor_page.created_at, self.db.cursor_page.seq
            )
            row = RowValue(self.db.thing.created_at, self.db.thing.seq)
            self.assertEqual(len(self.db(row.belongs(sub)).select()), 1)
            self.assertEqual(len(self.db(row.not_belongs(sub)).select()), 0)
            self.assertEqual(len(self.db(row == sub).select()), 1)
            self.assertEqual(len(self.db(row < sub).select()), 0)
            self.assertEqual(len(self.db(row <= sub).select()), 1)
            empty = self.db(self.db.thing.name == "missing").subselect(
                self.db.cursor_page.created_at, self.db.cursor_page.seq
            )
            self.assertEqual(len(self.db(row == empty).select()), 0)
        finally:
            self.db._adapter.compiler.supports_row_values = True

    def test_invalid_constant_dimension_rejected(self):
        with self.assertRaises(ValueError):
            RowValue(self.db.thing.created_at, self.db.thing.seq) > (
                "2024-01-01 00:00:00",
            )

    def test_invalid_subquery_dimension_rejected(self):
        sub = self.db().subselect(self.db.thing.created_at)
        with self.assertRaises(ValueError):
            RowValue(self.db.thing.created_at, self.db.thing.seq).belongs(sub)

    def test_invalid_subquery_column_type_rejected(self):
        sub = self.db().subselect(self.db.thing.name, self.db.thing.seq)
        with self.assertRaises(TypeError):
            RowValue(self.db.thing.created_at, self.db.thing.seq).belongs(sub)

    def test_invalid_value_type_rejected(self):
        with self.assertRaises(TypeError):
            RowValue(self.db.thing.seq, self.db.thing.extra) > ("bad", 1)

    def test_row_column_type_mismatch_rejected(self):
        with self.assertRaises(TypeError):
            RowValue(self.db.thing.seq, self.db.thing.name) > (
                RowValue(self.db.thing.name, self.db.thing.seq)
            )

    def test_single_column_none_comparison_remains_is_null(self):
        query = self.db.thing.name == None  # noqa: E711
        sql = self.db._adapter._select_wcols(query, [self.db.thing.id], {})[1]
        self.assertIn('"name" IS NULL', sql)
        self.assertEqual(sql.params, ())

    def test_scalar_single_column_comparison_unchanged(self):
        query = self.db.thing.seq > 7
        sql = self.db._adapter._select_wcols(query, [self.db.thing.id], {})[1]
        self.assertEqual(sql.params, (7,))
        self.assertIn('"seq" > ?', sql)
        self.assertNotIn("(?,?)", sql)

    def test_row_value_requires_two_columns(self):
        with self.assertRaises(ValueError):
            RowValue(self.db.thing.seq)
