import unittest

from pydal import (
    DAL,
    Field,
    QueryBuilder,
    QueryBuilderConfigError,
    QueryParseError,
)


class TestQueryBuilder(unittest.TestCase):
    def test_query_builder(self):
        db = DAL("sqlite:memory")
        db.define_table("thing", Field("name"), Field("solid", "boolean"))
        builder = QueryBuilder(db.thing)
        query = builder.parse("name is null")
        self.assertEqual(str(query), '("thing"."name" IS NULL)')
        query = builder.parse("name is not null")
        self.assertEqual(str(query), '("thing"."name" IS NOT NULL)')
        query = builder.parse("solid is true")
        self.assertEqual(str(query), '("thing"."solid" = \'T\')')
        query = builder.parse("solid is false")
        self.assertEqual(str(query), '("thing"."solid" = \'F\')')
        query = builder.parse("name is Chair")
        self.assertEqual(str(query), '("thing"."name" = \'Chair\')')
        query = builder.parse("name is equal to Chair")
        self.assertEqual(str(query), '("thing"."name" = \'Chair\')')
        query = builder.parse("name == Chair")
        self.assertEqual(str(query), '("thing"."name" = \'Chair\')')
        query = builder.parse('name == "Chair"')
        self.assertEqual(str(query), '("thing"."name" = \'Chair\')')
        query = builder.parse('name == "Cha\\"ir"')
        self.assertEqual(str(query), '("thing"."name" = \'Cha\\"ir\')')
        query = builder.parse('name != "Chair"')
        self.assertEqual(str(query), '("thing"."name" <> \'Chair\')')
        query = builder.parse('name < "Chair"')
        self.assertEqual(str(query), '("thing"."name" < \'Chair\')')
        query = builder.parse('name > "Chair"')
        self.assertEqual(str(query), '("thing"."name" > \'Chair\')')
        query = builder.parse('name <= "Chair"')
        self.assertEqual(str(query), '("thing"."name" <= \'Chair\')')
        query = builder.parse('name >= "Chair"')
        self.assertEqual(str(query), '("thing"."name" >= \'Chair\')')
        query = builder.parse("name in Chair, Table")
        self.assertEqual(str(query), "(\"thing\".\"name\" IN ('Chair','Table'))")
        query = builder.parse("name belongs Chair, Table")
        self.assertEqual(str(query), "(\"thing\".\"name\" IN ('Chair','Table'))")
        query = builder.parse('name belongs "Chair", "Table"')
        self.assertEqual(str(query), "(\"thing\".\"name\" IN ('Chair','Table'))")
        query = builder.parse('name belongs "Chair, Table"')
        self.assertEqual(str(query), '("thing"."name" IN (\'Chair, Table\'))')
        query = builder.parse('name contains "Chair"')
        self.assertEqual(
            str(query), "(LOWER(\"thing\".\"name\") LIKE '%chair%' ESCAPE '\\')"
        )
        query = builder.parse('name startswith "Chair"')
        self.assertEqual(str(query), "(\"thing\".\"name\" LIKE 'Chair%' ESCAPE '\\')")
        query = builder.parse('name starts with "Chair"')
        self.assertEqual(str(query), "(\"thing\".\"name\" LIKE 'Chair%' ESCAPE '\\')")
        query = builder.parse("name lower == chair")
        self.assertEqual(str(query), '(LOWER("thing"."name") = \'chair\')')
        query = builder.parse("name lower is equal to chair")
        self.assertEqual(str(query), '(LOWER("thing"."name") = \'chair\')')
        query = builder.parse("name upper == CHAIR")
        self.assertEqual(str(query), '(UPPER("thing"."name") = \'CHAIR\')')
        query = builder.parse("not name == Chair")
        self.assertEqual(str(query), '(NOT ("thing"."name" = \'Chair\'))')

        query = builder.parse("not (name == Chair)")
        self.assertEqual(str(query), '(NOT ("thing"."name" = \'Chair\'))')

        query = builder.parse("name == Chair or name is Table")
        self.assertEqual(
            str(query), '(("thing"."name" = \'Chair\') OR ("thing"."name" = \'Table\'))'
        )

        query = builder.parse("name == Chair and not name is Table")
        self.assertEqual(
            str(query),
            '(("thing"."name" = \'Chair\') AND (NOT ("thing"."name" = \'Table\')))',
        )

        query = builder.parse("not ((name == Chair) and not (name == Table))")
        self.assertEqual(
            str(query),
            '(NOT (("thing"."name" = \'Chair\') AND (NOT ("thing"."name" = \'Table\'))))',
        )
        query = builder.parse('id > "5" and id < "10"')
        self.assertEqual(str(query), '(("thing"."id" > 5) AND ("thing"."id" < 10))')

    def test_translations(self):
        db = DAL("sqlite:memory")
        db.define_table("thing", Field("name"))
        field_aliases = {"id": "id", "nome": "name"}
        token_aliases = {"non è nullo": "is not null", "è uguale a": "=="}
        builder = QueryBuilder(
            db.thing, field_aliases=field_aliases, token_aliases=token_aliases
        )

        query = builder.parse("nome non è nullo")
        self.assertEqual(str(query), '("thing"."name" IS NOT NULL)')
        query = builder.parse("nome è uguale a Chair")
        self.assertEqual(str(query), '("thing"."name" = \'Chair\')')


class TestQueryBuilderStandardMode(unittest.TestCase):
    def setUp(self):
        self.db = DAL("sqlite:memory")
        self.db.define_table(
            "thing", Field("name"), Field("solid", "boolean"), Field("qty", "integer")
        )
        self.builder = QueryBuilder(self.db.thing, mode="standard")

    def names(self, query):
        rows = self.db(query).select(self.db.thing.name)
        return {row.name for row in rows}

    def test_precedence_without_brackets(self):
        # Same table, same rows, same words: only the precedence differs.
        db = self.db
        db.thing.insert(name="a", qty=1)
        db.thing.insert(name="b", qty=2)
        db.thing.insert(name="c", qty=3)
        # ``and`` binds tighter than ``or`` in standard mode...
        query = self.builder.parse("name == a or name == b and name == c")
        self.assertEqual(
            str(query),
            '(("thing"."name" = \'a\') OR '
            '(("thing"."name" = \'b\') AND ("thing"."name" = \'c\')))',
        )
        self.assertEqual(self.names(query), {"a"})
        # ...while legacy mode keeps its historical left-to-right reading.
        legacy = QueryBuilder(db.thing)
        self.assertEqual(self.names(legacy.parse("name == a or name == b and name == c")), set())
        # AND-first expressions agree across modes.
        text = "name == a and qty > 0 or name == c"
        self.assertEqual(self.names(self.builder.parse(text)), {"a", "c"})
        self.assertEqual(self.names(legacy.parse(text)), {"a", "c"})
        # Parentheses override precedence.
        query = self.builder.parse("(name == a or name == b) and qty > 1")
        self.assertEqual(self.names(query), {"b"})

    def test_consecutive_not(self):
        query = self.builder.parse("not not name == Chair")
        self.assertEqual(str(query), '("thing"."name" = \'Chair\')')
        query = self.builder.parse("not not not name == Chair")
        self.assertEqual(str(query), '(NOT ("thing"."name" = \'Chair\'))')

    def test_values_are_content(self):
        # and/or/commas/parens inside quoted strings are data.
        query = self.builder.parse('name == "and (or), not"')
        self.assertEqual(str(query), '("thing"."name" = \'and (or), not\')')
        query = self.builder.parse('name == "Cha\\"ir"')
        self.assertEqual(str(query), '("thing"."name" = \'Cha\\"ir\')')
        query = self.builder.parse('name belongs "a,x", b, "c"')
        self.assertEqual(str(query), "(\"thing\".\"name\" IN ('a,x','b','c'))")
        # Reserved words must be quoted to be used as bare values.
        with self.assertRaises(QueryParseError):
            self.builder.parse("name == and")

    def test_unicode(self):
        db = DAL("sqlite:memory")
        db.define_table("thing", Field("name"))
        builder = QueryBuilder(
            db.thing,
            field_aliases={"名称": "name"},
            token_aliases={"等于": "==", "且": "and"},
            mode="standard",
        )
        query = builder.parse("名称 等于 椅子 且 名称 != 桌子")
        self.assertEqual(
            str(query), '(("thing"."name" = \'椅子\') AND ("thing"."name" <> \'桌子\'))'
        )

    def test_aliases(self):
        db = DAL("sqlite:memory")
        db.define_table("thing", Field("name"))
        builder = QueryBuilder(
            db.thing,
            field_aliases={"nome": "name"},
            token_aliases={"non è nullo": "is not null", "è uguale a": "=="},
            mode="standard",
        )
        query = builder.parse("nome non è nullo")
        self.assertEqual(str(query), '("thing"."name" IS NOT NULL)')
        query = builder.parse("nome è uguale a Chair")
        self.assertEqual(str(query), '("thing"."name" = \'Chair\')')

    def test_error_positions(self):
        # Truncated list: the error points at the end of the input.
        text = "name belongs Chair,"
        with self.assertRaises(QueryParseError) as ctx:
            self.builder.parse(text)
        self.assertEqual((ctx.exception.start, ctx.exception.end), (len(text), len(text)))
        # Missing operand after an operator.
        text = "name =="
        with self.assertRaises(QueryParseError) as ctx:
            self.builder.parse(text)
        self.assertEqual((ctx.exception.start, ctx.exception.end), (len(text), len(text)))
        # Missing operand after ``and``.
        text = "name == Chair and"
        with self.assertRaises(QueryParseError) as ctx:
            self.builder.parse(text)
        self.assertEqual((ctx.exception.start, ctx.exception.end), (len(text), len(text)))
        # Errors inside nested brackets carry offsets into the
        # original input, not into the innermost substring.
        text = "((name == ))"
        with self.assertRaises(QueryParseError) as ctx:
            self.builder.parse(text)
        self.assertEqual((ctx.exception.start, ctx.exception.end), (10, 11))
        self.assertIn("10-11", str(ctx.exception))
        # Unterminated string: span covers the opening quote onward.
        text = 'name == "abc'
        with self.assertRaises(QueryParseError) as ctx:
            self.builder.parse(text)
        self.assertEqual((ctx.exception.start, ctx.exception.end), (8, 12))

    def test_max_depth(self):
        builder = QueryBuilder(self.db.thing, mode="standard", max_depth=1)
        query = builder.parse("(name == a)")
        self.assertEqual(str(query), '("thing"."name" = \'a\')')
        with self.assertRaises(QueryParseError) as ctx:
            builder.parse("((name == a))")
        self.assertEqual((ctx.exception.start, ctx.exception.end), (1, 2))
        self.assertIn("maximum depth of 1", str(ctx.exception))

    def test_config_validation(self):
        # Field aliases colliding with reserved logical words are
        # rejected at construction time, not at parse time.
        with self.assertRaises(QueryBuilderConfigError):
            QueryBuilder(
                self.db.thing, field_aliases={"and": "name"}, mode="standard"
            )
        # So are token aliases that hijack a reserved word...
        with self.assertRaises(QueryBuilderConfigError):
            QueryBuilder(self.db.thing, token_aliases={"and": "=="}, mode="standard")
        # ...aliases targeting unknown tokens...
        with self.assertRaises(QueryBuilderConfigError):
            QueryBuilder(
                self.db.thing, token_aliases={"circa": "almost"}, mode="standard"
            )
        # ...and aliases redefining a canonical operator.
        with self.assertRaises(QueryBuilderConfigError):
            QueryBuilder(
                self.db.thing, token_aliases={"contains": "=="}, mode="standard"
            )
        # Legacy mode is untouched: the same configurations still build.
        QueryBuilder(self.db.thing, field_aliases={"and": "name"})
        QueryBuilder(self.db.thing, token_aliases={"and": "=="})

    def test_empty_input(self):
        self.assertEqual(str(self.builder.parse("")), '("thing"."id" > 0)')
