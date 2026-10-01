# -*- coding: utf-8 -*-
"""
Tests for `Rows.assemble`: explicit graph assembly over a flat join product.
"""

from pydal import Field
from pydal.exceptions import AssemblyConflict, AssemblyError
from pydal.objects import Row, Rows

from ._compat import unittest
from ._helpers import DALtest


class TestAssemble(DALtest):
    def define_shop(self, db):
        db.define_table("orders", Field("code"))
        db.define_table(
            "order_items", Field("order_id", "reference orders"), Field("sku")
        )
        db.define_table(
            "order_tags", Field("order_id", "reference orders"), Field("tag")
        )
        return db.orders, db.order_items, db.order_tags

    def test_fold_product_rows(self):
        db = self.connect()
        orders, items, tags = self.define_shop(db)
        o1 = orders.insert(code="A1")
        o2 = orders.insert(code="B2")
        items.insert(order_id=o1, sku="s1")
        items.insert(order_id=o1, sku="s2")
        tags.insert(order_id=o1, tag="t1")
        tags.insert(order_id=o1, tag="t2")
        items.insert(order_id=o2, sku="s3")
        db.commit()

        rows = db(db.orders.id == db.order_items.order_id).select(
            orders.ALL,
            items.ALL,
            tags.ALL,
            left=tags.on(tags.order_id == orders.id),
            orderby=orders.id,
        )
        # the raw product is untouched: o1 -> 2 items x 2 tags, o2 -> 1 item
        self.assertEqual(len(rows), 5)

        graph = rows.assemble(
            "orders",
            identity="id",
            relations=[
                {"name": "items", "alias": "order_items", "many": True},
                {"name": "tags", "alias": "order_tags", "many": True},
            ],
        )
        self.assertEqual(len(graph), 2)
        first, second = graph
        self.assertEqual(first.code, "A1")
        self.assertEqual(second.code, "B2")
        self.assertEqual([item.sku for item in first.items], ["s1", "s2"])
        self.assertEqual([tag.tag for tag in first.tags], ["t1", "t2"])
        self.assertEqual([item.sku for item in second.items], ["s3"])
        self.assertEqual(second.tags, [])
        # nodes are plain Row objects with the alias' scalar columns
        self.assertIsInstance(first, Row)
        self.assertEqual(first.id, o1)

    def test_missing_children_and_order(self):
        db = self.connect()
        orders, items, tags = self.define_shop(db)
        o1 = orders.insert(code="A1")
        o2 = orders.insert(code="B2")  # no items, no tags
        o3 = orders.insert(code="C3")
        items.insert(order_id=o3, sku="x2")
        items.insert(order_id=o3, sku="x1")
        db.commit()

        rows = db(db.orders).select(
            orders.ALL,
            items.ALL,
            orderby=(~orders.id, items.sku),
            left=items.on(items.order_id == orders.id),
        )
        graph = rows.assemble(
            "orders",
            identity="id",
            relations=[{"name": "items", "alias": "order_items", "many": True}],
        )
        # root order follows the select order; childless roots get empty lists
        self.assertEqual([node.code for node in graph], ["C3", "B2", "A1"])
        self.assertEqual([item.sku for item in graph[0].items], ["x1", "x2"])
        self.assertEqual(graph[1].items, [])
        self.assertEqual(graph[2].items, [])

    def test_same_numeric_ids_across_aliases(self):
        db = self.connect()
        orders, items, tags = self.define_shop(db)
        o1 = orders.insert(code="A1")  # id 1
        # item id 1 and tag id 1 share the root's numeric identity
        i1 = items.insert(order_id=o1, sku="s1")
        t1 = tags.insert(order_id=o1, tag="t1")
        self.assertEqual((o1, i1, t1), (1, 1, 1))
        db.commit()

        rows = db(db.orders).select(
            orders.ALL,
            items.ALL,
            tags.ALL,
            left=[
                items.on(items.order_id == orders.id),
                tags.on(tags.order_id == orders.id),
            ],
        )
        graph = rows.assemble(
            "orders",
            identity="id",
            relations=[
                {"name": "items", "alias": "order_items", "many": True},
                {"name": "tags", "alias": "order_tags", "many": True},
            ],
        )
        self.assertEqual(len(graph), 1)
        self.assertEqual(graph[0].id, 1)
        self.assertEqual([item.id for item in graph[0].items], [1])
        self.assertEqual([tag.id for tag in graph[0].tags], [1])
        self.assertEqual(graph[0].items[0].sku, "s1")
        self.assertEqual(graph[0].tags[0].tag, "t1")

    def test_self_join_aliases(self):
        db = self.connect()
        db.define_table(
            "node", Field("name"), Field("parent_id", "reference node")
        )
        root = db.node.insert(name="root")
        child = db.node.insert(name="leaf", parent_id=root)
        db.commit()
        parent = db.node.with_alias("parent")

        rows = db(db.node.parent_id == parent.id).select(db.node.ALL, parent.ALL)
        graph = rows.assemble(
            "parent",
            identity="id",
            relations=[{"name": "children", "alias": "node", "many": True}],
        )
        self.assertEqual(len(graph), 1)
        self.assertEqual(graph[0].name, "root")
        self.assertEqual([node.name for node in graph[0].children], ["leaf"])
        self.assertEqual(graph[0].children[0].id, child)

    def test_composite_identity(self):
        db = self.connect()
        db.define_table(
            "pkg",
            Field("region"),
            Field("code"),
            Field("label"),
            primarykey=["region", "code"],
        )
        db.define_table("pkg_item", Field("region"), Field("code"), Field("sku"))
        db.pkg.insert(region="eu", code="01", label="EU 01")
        db.pkg.insert(region="us", code="01", label="US 01")
        db.pkg_item.insert(region="eu", code="01", sku="e1")
        db.pkg_item.insert(region="us", code="01", sku="u1")
        db.pkg_item.insert(region="us", code="01", sku="u2")
        db.commit()

        rows = db(
            (db.pkg.region == db.pkg_item.region)
            & (db.pkg.code == db.pkg_item.code)
        ).select(db.pkg.ALL, db.pkg_item.ALL, orderby=db.pkg.region)
        graph = rows.assemble(
            "pkg",
            identity=("region", "code"),
            relations=[{"name": "items", "alias": "pkg_item", "many": True}],
        )
        self.assertEqual(len(graph), 2)
        self.assertEqual(graph[0].label, "EU 01")
        self.assertEqual([item.sku for item in graph[0].items], ["e1"])
        self.assertEqual(graph[1].label, "US 01")
        self.assertEqual([item.sku for item in graph[1].items], ["u1", "u2"])

    def test_one_to_one(self):
        db = self.connect()
        orders, items, tags = self.define_shop(db)
        o1 = orders.insert(code="A1")
        o2 = orders.insert(code="B2")  # no tag
        tags.insert(order_id=o1, tag="t1")
        db.commit()
        rows = db(db.orders).select(
            orders.ALL,
            tags.ALL,
            orderby=orders.id,
            left=tags.on(tags.order_id == orders.id),
        )
        graph = rows.assemble(
            "orders",
            identity="id",
            relations=[{"name": "tag", "alias": "order_tags"}],
        )
        self.assertEqual(graph[0].tag.tag, "t1")
        self.assertIsNone(graph[1].tag)

        # a second identity under the same parent is an error
        tags.insert(order_id=o1, tag="t2")
        db.commit()
        rows = db(db.orders).select(
            orders.ALL,
            tags.ALL,
            orderby=(orders.id, tags.id),
            left=tags.on(tags.order_id == orders.id),
        )
        with self.assertRaises(AssemblyError):
            rows.assemble(
                "orders",
                identity="id",
                relations=[{"name": "tag", "alias": "order_tags"}],
            )

    def test_scalar_conflict(self):
        db = self.connect()
        orders, items, tags = self.define_shop(db)
        records = [
            Row(orders={"id": 1, "code": "A1"}),
            Row(orders={"id": 1, "code": "CHANGED"}),
        ]
        rows = Rows(
            db, records, colnames=["orders.id", "orders.code"], compact=False
        )
        with self.assertRaises(AssemblyConflict):
            rows.assemble("orders", identity="id")

    def test_unknown_alias(self):
        db = self.connect()
        orders, items, tags = self.define_shop(db)
        orders.insert(code="A1")
        db.commit()
        rows = db(db.orders).select(orders.ALL)
        with self.assertRaises(AssemblyError):
            rows.assemble("missing_table", identity="id")

    def test_no_cross_assembly_caching(self):
        db = self.connect()
        orders, items, tags = self.define_shop(db)
        o1 = orders.insert(code="A1")
        items.insert(order_id=o1, sku="s1")
        db.commit()
        rows = db(db.orders).select(
            orders.ALL,
            items.ALL,
            left=items.on(items.order_id == orders.id),
        )
        spec = dict(
            identity="id",
            relations=[{"name": "items", "alias": "order_items", "many": True}],
        )
        first = rows.assemble("orders", **spec)
        second = rows.assemble("orders", **spec)
        self.assertEqual(len(first), len(second))
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[0].items[0], second[0].items[0])
        # plain iteration still exposes the raw product rows
        self.assertEqual(len(list(rows)), 1)
        self.assertEqual(rows.first().orders.code, "A1")


if __name__ == "__main__":
    unittest.main()
