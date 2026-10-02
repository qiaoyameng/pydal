# -*- coding: utf-8 -*-

"""Tests for ``Set.recursive_traverse`` — the explicit recursive
traversal constructor built on pydal's CTE support.

Covered on SQLite:

* plain trees (paths, depths, extra node fields)
* diamond DAGs — a shared node on two branches is NOT a cycle
* self-loops and cross-level cycles, in both ``on_cycle`` modes
* max_depth: truncated branches vs. natural leaves at the depth limit
* multi-root anchor queries
* composite node keys
* keys containing the path delimiter, quotes, backslashes, Unicode
* bound parameters flowing through the root query and recursive part
* clear NotImplementedError on dialects without the required support
"""

from pydal import DAL, Field

from ._adapt import IS_NOSQL
from ._compat import unittest


@unittest.skipIf(IS_NOSQL, "SQL-only")
class TestRecursiveTraverseTree(unittest.TestCase):
    """Traversals over a self-referencing table (org chart)."""

    def setUp(self):
        self.db = DAL("sqlite:memory")
        self.db.define_table("org", Field("name"), Field("boss", "reference org"))

    def tearDown(self):
        self.db.close()

    def _make_tree(self):
        # alice -> bob -> carol, bob -> dan
        db = self.db
        alice = db.org.insert(name="alice")
        bob = db.org.insert(name="bob", boss=alice)
        carol = db.org.insert(name="carol", boss=bob)
        dan = db.org.insert(name="dan", boss=bob)
        db.commit()
        return alice, bob, carol, dan

    def _traverse(self, roots, **kwargs):
        kwargs.setdefault("key", self.db.org.id)
        kwargs.setdefault("link", lambda parent, child: child.boss == parent.id)
        kwargs.setdefault("fields", (self.db.org.name,))
        return roots.recursive_traverse(**kwargs)

    def test_tree_paths_and_depth(self):
        alice, bob, carol, dan = self._make_tree()
        walk = self._traverse(self.db(self.db.org.boss == None))  # noqa: E711
        rows = self.db().select(walk.ALL, orderby=walk.path)
        got = [(r.name, r.depth, r.cycle, r.truncated, r.path) for r in rows]
        self.assertEqual(
            got,
            [
                ("alice", 0, 0, 0, "|%d|" % alice),
                ("bob", 1, 0, 0, "|%d|%d|" % (alice, bob)),
                ("carol", 2, 0, 0, "|%d|%d|%d|" % (alice, bob, carol)),
                ("dan", 2, 0, 0, "|%d|%d|%d|" % (alice, bob, dan)),
            ],
        )

    def test_max_depth_truncation_flag(self):
        alice, bob, carol, dan = self._make_tree()
        walk = self._traverse(self.db(self.db.org.boss == None), max_depth=1)  # noqa: E711
        rows = self.db().select(walk.ALL, orderby=walk.path)
        got = [(r.name, r.depth, r.truncated) for r in rows]
        # bob sits at the depth limit and still has children -> truncated;
        # alice is above the limit -> not truncated.
        self.assertEqual(got, [("alice", 0, 0), ("bob", 1, 1)])

    def test_max_depth_natural_leaf_not_truncated(self):
        alice, bob, carol, dan = self._make_tree()
        # chain alice -> bob -> carol is exactly 2 deep; carol is a leaf.
        walk = self._traverse(self.db(self.db.org.boss == None), max_depth=2)  # noqa: E711
        rows = self.db().select(walk.ALL, orderby=walk.path)
        got = [(r.name, r.depth, r.truncated) for r in rows]
        self.assertEqual(
            got,
            [
                ("alice", 0, 0),
                ("bob", 1, 0),
                ("carol", 2, 0),  # natural leaf at the limit
                ("dan", 2, 0),  # natural leaf at the limit
            ],
        )
        # one level less: bob becomes a truncated branch
        walk = self._traverse(self.db(self.db.org.boss == None), max_depth=1)  # noqa: E711
        rows = self.db().select(walk.ALL, orderby=walk.path)
        self.assertEqual(
            [(r.name, r.truncated) for r in rows], [("alice", 0), ("bob", 1)]
        )

    def test_max_depth_zero(self):
        alice, bob, carol, dan = self._make_tree()
        walk = self._traverse(self.db(self.db.org.boss == None), max_depth=0)  # noqa: E711
        rows = self.db().select(walk.ALL)
        self.assertEqual(
            [(r.name, r.depth, r.truncated) for r in rows], [("alice", 0, 1)]
        )

    def test_self_loop_report(self):
        db = self.db
        x = db.org.insert(name="x")
        db(db.org.id == x).update(boss=x)
        db.commit()
        walk = self._traverse(db(db.org.id == x))
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.name, r.depth, r.cycle, r.path) for r in rows]
        # the loop edge is reported once with cycle=1, then expansion stops
        self.assertEqual(
            got,
            [("x", 0, 0, "|%d|" % x), ("x", 1, 1, "|%d|%d|" % (x, x))],
        )

    def test_self_loop_ignore(self):
        db = self.db
        x = db.org.insert(name="x")
        db(db.org.id == x).update(boss=x)
        db.commit()
        walk = self._traverse(db(db.org.id == x), on_cycle="ignore")
        rows = db().select(walk.ALL)
        self.assertEqual(
            [(r.name, r.depth, r.cycle) for r in rows], [("x", 0, 0)]
        )

    def test_cross_level_cycle_report(self):
        # a -> b -> c -> a (cycle back to the root)
        db = self.db
        a = db.org.insert(name="a")
        b = db.org.insert(name="b", boss=a)
        c = db.org.insert(name="c", boss=b)
        db(db.org.id == a).update(boss=c)
        db.commit()
        walk = self._traverse(db(db.org.id == a))
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.name, r.depth, r.cycle, r.path) for r in rows]
        self.assertEqual(
            got,
            [
                ("a", 0, 0, "|%d|" % a),
                ("b", 1, 0, "|%d|%d|" % (a, b)),
                ("c", 2, 0, "|%d|%d|%d|" % (a, b, c)),
                ("a", 3, 1, "|%d|%d|%d|%d|" % (a, b, c, a)),
            ],
        )

    def test_cross_level_cycle_ignore(self):
        db = self.db
        a = db.org.insert(name="a")
        b = db.org.insert(name="b", boss=a)
        c = db.org.insert(name="c", boss=b)
        db(db.org.id == a).update(boss=c)
        db.commit()
        walk = self._traverse(db(db.org.id == a), on_cycle="ignore")
        rows = db().select(walk.ALL, orderby=walk.path)
        self.assertEqual(
            [(r.name, r.depth, r.cycle) for r in rows],
            [("a", 0, 0), ("b", 1, 0), ("c", 2, 0)],
        )

    def test_two_node_cycle_terminates(self):
        # a <-> b mutual cycle, no depth limit: must still terminate
        db = self.db
        a = db.org.insert(name="a")
        b = db.org.insert(name="b", boss=a)
        db(db.org.id == a).update(boss=b)
        db.commit()
        # report: a(0), b(1), a(2, cycle) — the cycle row is not expanded;
        # ignore: a(0), b(1) — the b->a edge is dropped
        for mode, expected in (("report", 3), ("ignore", 2)):
            walk = self._traverse(
                db(db.org.id == a), on_cycle=mode, name="walk_%s" % mode
            )
            rows = db().select(walk.ALL)
            self.assertEqual(len(rows), expected)

    def test_values_are_bound_parameters(self):
        alice, bob, carol, dan = self._make_tree()
        db = self.db
        seen = []
        orig = db._adapter.execute

        def tap(*args, **kwargs):
            seen.append((str(args[0]), getattr(args[0], "params", None)))
            return orig(*args, **kwargs)

        db._adapter.execute = tap
        try:
            walk = self._traverse(db(db.org.name == "ali'ce|"), max_depth=3)
            db().select(walk.ALL)
        finally:
            db._adapter.execute = orig
        sql, params = seen[-1]
        # the root-query value and max_depth ride as bound parameters...
        self.assertIn("ali'ce|", params)
        self.assertIn(3, params)
        # ...and are never interpolated into the SQL text
        self.assertNotIn("ali'ce|", sql)
        self.assertNotIn("'3'", sql)

    def test_invalid_arguments(self):
        db = self.db
        with self.assertRaises(ValueError):
            self._traverse(db(db.org), on_cycle="explode")
        with self.assertRaises(ValueError):
            self._traverse(db(db.org), max_depth=-1)
        with self.assertRaises(TypeError):
            db(db.org).recursive_traverse(
                key="id", link=lambda p, c: c.boss == p.id
            )

    def test_unsupported_dialect_raises(self):
        db = self.db
        dialect = db._adapter.dialect
        saved = dialect.instr
        dialect.instr = None  # simulate a dialect without string-position support
        try:
            with self.assertRaises(NotImplementedError):
                self._traverse(db(db.org))
        finally:
            dialect.instr = saved


@unittest.skipIf(IS_NOSQL, "SQL-only")
class TestRecursiveTraverseGraph(unittest.TestCase):
    """Traversals over an explicit edge table (generic graphs/DAGs)."""

    def setUp(self):
        self.db = DAL("sqlite:memory")
        self.db.define_table("node", Field("name"))
        self.db.define_table(
            "edge", Field("src", "reference node"), Field("dst", "reference node")
        )

    def tearDown(self):
        self.db.close()

    def _link(self, parent, child):
        db = self.db
        return (db.edge.src == parent.id) & (child.id == db.edge.dst)

    def test_diamond_shared_node_is_not_a_cycle(self):
        # a -> b, a -> c, b -> d, c -> d : d is shared, not a cycle
        db = self.db
        a = db.node.insert(name="a")
        b = db.node.insert(name="b")
        c = db.node.insert(name="c")
        d = db.node.insert(name="d")
        for src, dst in ((a, b), (a, c), (b, d), (c, d)):
            db.edge.insert(src=src, dst=dst)
        db.commit()
        walk = db(db.node.id == a).recursive_traverse(
            key=db.node.id, link=self._link, fields=(db.node.name,)
        )
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.name, r.depth, r.cycle, r.path) for r in rows]
        self.assertEqual(
            got,
            [
                ("a", 0, 0, "|%d|" % a),
                ("b", 1, 0, "|%d|%d|" % (a, b)),
                ("d", 2, 0, "|%d|%d|%d|" % (a, b, d)),
                ("c", 1, 0, "|%d|%d|" % (a, c)),
                ("d", 2, 0, "|%d|%d|%d|" % (a, c, d)),
            ],
        )

    def test_diamond_with_back_edge_reports_only_real_cycles(self):
        # diamond plus d -> a: only paths that revisit a close a cycle
        db = self.db
        a = db.node.insert(name="a")
        b = db.node.insert(name="b")
        c = db.node.insert(name="c")
        d = db.node.insert(name="d")
        for src, dst in ((a, b), (a, c), (b, d), (c, d), (d, a)):
            db.edge.insert(src=src, dst=dst)
        db.commit()
        walk = db(db.node.id == a).recursive_traverse(
            key=db.node.id, link=self._link, fields=(db.node.name,)
        )
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.name, r.depth, r.cycle) for r in rows]
        self.assertEqual(
            got,
            [
                ("a", 0, 0),
                ("b", 1, 0),
                ("d", 2, 0),
                ("a", 3, 1),
                ("c", 1, 0),
                ("d", 2, 0),
                ("a", 3, 1),
            ],
        )

    def test_multi_root(self):
        # two independent roots: a -> b and x -> y
        db = self.db
        a = db.node.insert(name="a")
        b = db.node.insert(name="b")
        x = db.node.insert(name="x")
        y = db.node.insert(name="y")
        db.edge.insert(src=a, dst=b)
        db.edge.insert(src=x, dst=y)
        db.commit()
        roots = db((db.node.id == a) | (db.node.id == x))
        walk = roots.recursive_traverse(
            key=db.node.id, link=self._link, fields=(db.node.name,)
        )
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.name, r.depth, r.path) for r in rows]
        self.assertEqual(
            got,
            [
                ("a", 0, "|%d|" % a),
                ("b", 1, "|%d|%d|" % (a, b)),
                ("x", 0, "|%d|" % x),
                ("y", 1, "|%d|%d|" % (x, y)),
            ],
        )


@unittest.skipIf(IS_NOSQL, "SQL-only")
class TestRecursiveTraverseKeys(unittest.TestCase):
    """Composite keys and hostile key values (delimiters, quotes, Unicode)."""

    def setUp(self):
        self.db = DAL("sqlite:memory")
        self.db.define_table("n", Field("region"), Field("code"))
        self.db.define_table("e", Field("fr"), Field("fc"), Field("tr"), Field("tc"))

    def tearDown(self):
        self.db.close()

    def _link(self, parent, child):
        db = self.db
        return (
            (db.e.fr == parent.region)
            & (db.e.fc == parent.code)
            & (child.region == db.e.tr)
            & (child.code == db.e.tc)
        )

    def test_composite_key(self):
        db = self.db
        db.n.insert(region="r1", code="a")
        db.n.insert(region="r1", code="b")
        db.n.insert(region="r2", code="a")
        db.e.insert(fr="r1", fc="a", tr="r1", tc="b")
        db.e.insert(fr="r1", fc="b", tr="r2", tc="a")
        db.commit()
        roots = db((db.n.region == "r1") & (db.n.code == "a"))
        walk = roots.recursive_traverse(key=[db.n.region, db.n.code], link=self._link)
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.region, r.code, r.depth, r.path) for r in rows]
        self.assertEqual(
            got,
            [
                ("r1", "a", 0, "|r1|a|"),
                ("r1", "b", 1, "|r1|a|r1|b|"),
                ("r2", "a", 2, "|r1|a|r1|b|r2|a|"),
            ],
        )

    def test_composite_key_components_not_confused(self):
        # ("a", "b|c") and ("a|b", "c") encode to different path elements:
        # walking from ("a", "b|c") to ("a|b", "c") must NOT look like a cycle
        db = self.db
        db.n.insert(region="a", code="b|c")
        db.n.insert(region="a|b", code="c")
        db.e.insert(fr="a", fc="b|c", tr="a|b", tc="c")
        db.commit()
        roots = db((db.n.region == "a") & (db.n.code == "b|c"))
        walk = roots.recursive_traverse(key=[db.n.region, db.n.code], link=self._link)
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.region, r.code, r.depth, r.cycle) for r in rows]
        self.assertEqual(got, [("a", "b|c", 0, 0), ("a|b", "c", 1, 0)])

    def test_delimiter_in_key_is_not_a_false_cycle(self):
        # "a|b" -> "b": naive path "|a|b|" would false-match needle "|b|"
        db = self.db
        db.n.insert(region="a|b", code="")
        db.n.insert(region="b", code="")
        db.e.insert(fr="a|b", fc="", tr="b", tc="")
        db.commit()
        walk = db(db.n.region == "a|b").recursive_traverse(
            key=db.n.region, link=self._link_single()
        )
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.region, r.depth, r.cycle, r.path) for r in rows]
        self.assertEqual(
            got,
            [("a|b", 0, 0, "|a\\pb|"), ("b", 1, 0, "|a\\pb|b|")],
        )

    def _link_single(self):
        db = self.db
        return lambda parent, child: (db.e.fr == parent.region) & (
            child.region == db.e.tr
        )

    def test_hostile_keys_cycle_still_detected(self):
        # "a|b" -> "b" -> "\" -> "a|b" : a real cycle through hostile keys
        db = self.db
        db.n.insert(region="a|b", code="")
        db.n.insert(region="b", code="")
        db.n.insert(region="\\", code="")
        db.e.insert(fr="a|b", fc="", tr="b", tc="")
        db.e.insert(fr="b", fc="", tr="\\", tc="")
        db.e.insert(fr="\\", fc="", tr="a|b", tc="")
        db.commit()
        walk = db(db.n.region == "a|b").recursive_traverse(
            key=db.n.region, link=self._link_single()
        )
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.region, r.depth, r.cycle) for r in rows]
        self.assertEqual(
            got,
            [("a|b", 0, 0), ("b", 1, 0), ("\\", 2, 0), ("a|b", 3, 1)],
        )

    def test_quotes_and_unicode_keys(self):
        db = self.db
        db.n.insert(region="o'clock", code="")
        db.n.insert(region="中文字符", code="")
        db.n.insert(region='quote"double', code="")
        db.e.insert(fr="o'clock", fc="", tr="中文字符", tc="")
        db.e.insert(fr="中文字符", fc="", tr='quote"double', tc="")
        db.commit()
        walk = db(db.n.region == "o'clock").recursive_traverse(
            key=db.n.region, link=self._link_single()
        )
        rows = db().select(walk.ALL, orderby=walk.path)
        got = [(r.region, r.depth, r.cycle, r.path) for r in rows]
        self.assertEqual(
            got,
            [
                ("o'clock", 0, 0, "|o'clock|"),
                ("中文字符", 1, 0, "|o'clock|中文字符|"),
                ('quote"double', 2, 0, "|o'clock|中文字符|quote\"double|"),
            ],
        )
