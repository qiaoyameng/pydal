import unittest

from pydal import (
    DAL,
    Field,
    QueryBuilder,
    QueryBuilderConfigurationError,
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

    def test_standard_precedence_against_memory_table(self):
        db = DAL("sqlite:memory")
        db.define_table(
            "flag",
            Field("a", "integer"),
            Field("b", "integer"),
            Field("c", "integer"),
        )
        db.flag.insert(a=1, b=0, c=0)
        db.flag.insert(a=0, b=1, c=1)
        db.flag.insert(a=0, b=1, c=0)

        expression = "a == 1 or b == 1 and c == 1"
        legacy = QueryBuilder(db.flag)
        standard = QueryBuilder(db.flag, mode="standard")

        self.assertEqual([row.id for row in db(legacy.parse(expression)).select()], [2])
        self.assertEqual(
            [row.id for row in db(standard.parse(expression)).select()], [1, 2]
        )
        grouped = "(a == 1 or b == 1) and c == 1"
        self.assertEqual([row.id for row in db(standard.parse(grouped)).select()], [2])
        self.assertEqual(
            str(standard.parse("not a == 1 and b == 1")),
            '((NOT ("flag"."a" = 1)) AND ("flag"."b" = 1))',
        )

    def test_standard_tokens_aliases_lists_and_quoted_values(self):
        db = DAL("sqlite:memory")
        db.define_table("thing", Field("name"), Field("level", "integer"))
        field_aliases = {"名前": "name"}
        token_aliases = {
            "等於": "==",
            "屬於": "belongs",
            "與": "and",
            "或": "or",
            "非": "not",
        }
        builder = QueryBuilder(
            db.thing,
            field_aliases=field_aliases,
            token_aliases=token_aliases,
            mode="standard",
        )

        query = builder.parse(
            "(名前 等於 A 或 名前 等於 B) 與 非 名前 等於 C"
        )
        self.assertEqual(
            str(query),
            '((("thing"."name" = \'A\') OR ("thing"."name" = \'B\')) '
            'AND (NOT ("thing"."name" = \'C\')))',
        )
        query = builder.parse("名前 屬於 A, B, café")
        self.assertEqual(
            str(query), '("thing"."name" IN (\'A\',\'B\',\'café\'))'
        )
        query = builder.parse('名前 等於 "A 與 B, (café)"')
        self.assertEqual(str(query), '("thing"."name" = \'A 與 B, (café)\')')

        plain = QueryBuilder(db.thing, mode="standard")
        self.assertEqual(str(plain.parse("not not name == A")), '("thing"."name" = \'A\')')
        self.assertEqual(
            str(plain.parse('name == "Cha\\"ir"')),
            '("thing"."name" = \'Cha"ir\')',
        )
        self.assertEqual(
            str(plain.parse('name == "__import__(\'os\')"')),
            "(\"thing\".\"name\" = '__import__(''os'')')",
        )
        bad_integer_text = 'level == "__import__(\'os\')"'
        bad_integer = plain.parse
        with self.assertRaises(QueryParseError) as caught:
            bad_integer(bad_integer_text)
        self.assertEqual(
            caught.exception.span,
            (bad_integer_text.index('"'), len(bad_integer_text)),
        )

    def test_standard_syntax_error_spans_and_depth(self):
        db = DAL("sqlite:memory")
        db.define_table("thing", Field("name"))
        builder = QueryBuilder(db.thing, mode="standard", max_depth=2)

        cases = [
            ("(name == A) and", (12, 15), "and"),
            ("name belongs A,", (14, 15), ","),
            ("(name ==)", (8, 9), ")"),
            ("name == A and (name ==)", (22, 23), ")"),
            ("(name == A", (0, 1), "("),
            ("(((name == A)))", (2, 3), "("),
            ('name == "unterminated', (8, 21), '"unterminated'),
        ]
        for text, span, fragment in cases:
            with self.subTest(text=text):
                with self.assertRaises(QueryParseError) as caught:
                    builder.parse(text)
                self.assertEqual(caught.exception.text, text)
                self.assertEqual(caught.exception.span, span)
                self.assertEqual(text[slice(*span)], fragment)

    def test_standard_configuration_conflicts(self):
        db = DAL("sqlite:memory")
        db.define_table("thing", Field("name"))

        with self.assertRaises(QueryBuilderConfigurationError) as caught:
            QueryBuilder(db.thing, mode="standard", token_aliases={"and": "or"})
        self.assertIn("'and'", str(caught.exception))

        with self.assertRaises(QueryBuilderConfigurationError):
            QueryBuilder(
                db.thing,
                mode="standard",
                field_aliases={"and": "name"},
            )

        with self.assertRaises(QueryBuilderConfigurationError):
            QueryBuilder(
                db.thing,
                mode="standard",
                token_aliases={"unknown phrase": "not an operator"},
            )

        legacy = QueryBuilder(db.thing, token_aliases={"and": "or"})
        self.assertEqual(
            str(legacy.parse("name == A and name == B")),
            '(("thing"."name" = \'A\') OR ("thing"."name" = \'B\'))',
        )

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
