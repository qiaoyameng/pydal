# -*- coding: utf-8 -*-

"""Tests for Set.recursive_walk — the explicit recursive-traversal
constructor built on pydal's CTE support.

Covers, on SQLite:

* plain trees (depths, paths, termination)
* diamond DAGs (shared node reached via two paths is NOT a cycle)
* self-loops and cross-level cycles, under both cycle policies
  ("report" emits a cycle-flagged closing row, "ignore" drops the edge)
* max_depth: natural leaves vs. actively truncated nodes
* multi-root walks and composite node keys
* path exactness with separators, quotes and Unicode in keys
* parameter binding (root-query values and max_depth ride as params)
* clear NotImplementedError on unsupported dialects
"""

from pydal import DAL, Field

from ._adapt import IS_SQLITE
from ._compat import unittest


def decode_path(path):
    """'/hex(k1):hex(k2)/.../' -> [[k1, k2], ...] as strings."""
    return [
        [bytes.fromhex(part).decode("utf-8") for part in elem.split(":")]
        for elem in path.strip("/").split("/")
    ]


@unittest.skipUnless(IS_SQLITE, "recursive_walk is SQLite-only for now")
class TestRecursiveWalk(unittest.TestCase):

    def setUp(self):
        self.db = DAL("sqlite:memory")
        self.db.define_table(
            "org", Field("name"), Field("boss", "reference org")
        )

    def tearDown(self):
        self.db.close()

    def _org_tree(self):
        org = self.db.org
        a = org.insert(name="A")
        b = org.insert(name="B", boss=a)
        c = org.insert(name="C", boss=a)
        d = org.insert(name="D", boss=b)
        e = org.insert(name="E", boss=d)
        return dict(A=a, B=b, C=c, D=d, E=e)

    def _walk(self, query, **kwargs):
        return self.db(query).recursive_walk(
            join=lambda w, child: child.boss == w.id, **kwargs
        )

    def _rows(self, walk):
        return self.db().select(walk.ALL, orderby=walk.path)

    # ---------- trees ----------

    def test_tree_depths_and_paths(self):
        ids = self._org_tree()
        walk = self._walk(self.db.org.boss == None)  # noqa: E711
        rows = self._rows(walk)
        got = {r.name: (r.depth, [[i] for i in decode_path(r.path)][-1]) for r in rows}
        self.assertEqual(
            {r.name: r.depth for r in rows},
            {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3},
        )
        # path of each row ends with the node's own key and starts at A
        for r in rows:
            elems = decode_path(r.path)
            self.assertEqual(elems[0], [str(ids["A"])])
            self.assertEqual(elems[-1], [str(ids[r.name])])
            self.assertEqual(len(elems), r.depth + 1)
            self.assertEqual(r.cycle, 0)
            self.assertEqual(r.truncated, 0)
        # full path of E is A,B,D,E
        e_row = [r for r in rows if r.name == "E"][0]
        self.assertEqual(
            [int(e[0]) for e in decode_path(e_row.path)],
            [ids["A"], ids["B"], ids["D"], ids["E"]],
        )

    def test_multi_root(self):
        org = self.db.org
        a = org.insert(name="A")
        org.insert(name="B", boss=a)
        org.insert(name="R2")  # second root, no children
        walk = self._walk(self.db.org.boss == None)  # noqa: E711
        rows = self._rows(walk)
        self.assertEqual(
            sorted((r.name, r.depth) for r in rows),
            [("A", 0), ("B", 1), ("R2", 0)],
        )

    def test_default_key_is_primary_key(self):
        self._org_tree()
        # no explicit key: the table's primary key is used
        walk = self.db(self.db.org.boss == None).recursive_walk(  # noqa: E711
            join=lambda w, child: child.boss == w.id
        )
        rows = self._rows(walk)
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            {r.name: r.depth for r in rows},
            {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3},
        )

    # ---------- diamond DAG (via edge table) ----------

    def _diamond(self):
        db = self.db
        db.define_table("node", Field("name"))
        db.define_table(
            "edge",
            Field("src", "reference node"),
            Field("dst", "reference node"),
        )
        n = {k: db.node.insert(name=k) for k in "ABCD"}
        for s, d in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]:
            db.edge.insert(src=n[s], dst=n[d])
        return n

    def test_diamond_shared_node_is_not_a_cycle(self):
        n = self._diamond()
        db = self.db
        walk = db(db.node.name == "A").recursive_walk(
            key=db.node.id,
            join=lambda w, child: (child.id == db.edge.dst)
            & (db.edge.src == w.id),
        )
        rows = self._rows(walk)
        # D reached via B and via C: two rows, distinct paths, no cycle
        d_rows = [r for r in rows if r.id == n["D"]]
        self.assertEqual(len(d_rows), 2)
        self.assertNotEqual(d_rows[0].path, d_rows[1].path)
        self.assertEqual({r.cycle for r in rows}, {0})
        self.assertEqual({r.depth for r in d_rows}, {2})
        # termination: exactly 5 rows (A, B, C, D, D)
        self.assertEqual(len(rows), 5)

    # ---------- cycles ----------

    def _cyclic_org(self):
        org = self.db.org
        x = org.insert(name="X")
        y = org.insert(name="Y", boss=x)
        z = org.insert(name="Z", boss=y)
        # cross-level cycle: X's boss is Z (X -> Y -> Z -> X)
        self.db(org.id == x).update(boss=z)
        # self-loop
        s = org.insert(name="S")
        self.db(org.id == s).update(boss=s)
        return dict(X=x, Y=y, Z=z, S=s)

    def test_cross_level_cycle_report(self):
        ids = self._cyclic_org()
        walk = self._walk(self.db.org.name == "X", on_cycle="report")
        rows = self._rows(walk)
        self.assertEqual(
            [(r.name, r.depth, r.cycle) for r in rows],
            [("X", 0, 0), ("Y", 1, 0), ("Z", 2, 0), ("X", 3, 1)],
        )
        closing = rows[-1]
        self.assertEqual(
            [int(e[0]) for e in decode_path(closing.path)],
            [ids["X"], ids["Y"], ids["Z"], ids["X"]],
        )

    def test_cross_level_cycle_ignore(self):
        self._cyclic_org()
        walk = self._walk(self.db.org.name == "X", on_cycle="ignore")
        rows = self._rows(walk)
        # the Z->X edge is silently dropped; terminates with 3 rows
        self.assertEqual(
            [(r.name, r.depth, r.cycle) for r in rows],
            [("X", 0, 0), ("Y", 1, 0), ("Z", 2, 0)],
        )

    def test_self_loop_report(self):
        ids = self._cyclic_org()
        walk = self._walk(self.db.org.name == "S", on_cycle="report")
        rows = self._rows(walk)
        self.assertEqual(
            [(r.name, r.depth, r.cycle) for r in rows],
            [("S", 0, 0), ("S", 1, 1)],
        )
        self.assertEqual(
            [int(e[0]) for e in decode_path(rows[-1].path)],
            [ids["S"], ids["S"]],
        )

    def test_self_loop_ignore(self):
        self._cyclic_org()
        walk = self._walk(self.db.org.name == "S", on_cycle="ignore")
        rows = self._rows(walk)
        self.assertEqual([(r.name, r.depth) for r in rows], [("S", 0)])

    # ---------- max_depth / truncation ----------

    def test_max_depth_distinguishes_leaf_from_truncated(self):
        self._org_tree()  # A -> B -> D -> E, A -> C
        walk = self._walk(self.db.org.boss == None, max_depth=1)  # noqa: E711
        rows = self._rows(walk)
        got = {r.name: (r.depth, r.truncated) for r in rows}
        # D and E are beyond the cap; B is cut (has child D), C is a
        # natural leaf at the cap, A is below the cap
        self.assertEqual(
            got, {"A": (0, 0), "B": (1, 1), "C": (1, 0)}
        )
        # paths of the emitted rows are still exact
        for r in rows:
            self.assertEqual(len(decode_path(r.path)), r.depth + 1)

    def test_max_depth_zero(self):
        self._org_tree()
        walk = self._walk(self.db.org.boss == None, max_depth=0)  # noqa: E711
        rows = self._rows(walk)
        self.assertEqual(
            [(r.name, r.depth, r.truncated) for r in rows], [("A", 0, 1)]
        )

    def test_max_depth_with_ignore_policy(self):
        # Under "ignore" a self-loop edge is dropped, so a self-looping
        # node at the depth cap is NOT truncated; a node with a real
        # child beyond the cap IS.
        self._cyclic_org()
        walk = self._walk(
            self.db.org.name.belongs(["X", "S"]),
            max_depth=1,
            on_cycle="ignore",
        )
        rows = self._rows(walk)
        got = {(r.name, r.depth): r.truncated for r in rows}
        self.assertEqual(
            got,
            {
                ("X", 0): 0,   # child Y was emitted
                ("Y", 1): 1,   # child Z cut by the cap
                ("S", 0): 0,   # only child is itself (ignored edge)
            },
        )

    def test_max_depth_binds_as_parameter(self):
        self._org_tree()
        seen = []
        orig = self.db._adapter.execute

        def tap(*a, **kw):
            seen.append((a[0], getattr(a[0], "params", None)))
            return orig(*a, **kw)

        self.db._adapter.execute = tap
        try:
            walk = self._walk(self.db.org.boss == None, max_depth=2)  # noqa: E711
            rows = self._rows(walk)
        finally:
            self.db._adapter.execute = orig
        self.assertEqual(len(rows), 4)  # A, B, C, D — E is cut
        params = [p for _, ps in seen if ps for p in ps]
        self.assertIn(2, params)

    # ---------- composite keys / exact paths ----------

    def test_composite_keys_with_nasty_values(self):
        db = self.db
        db.define_table(
            "kk",
            Field("k1"),
            Field("k2"),
            Field("parent_k1"),
            Field("parent_k2"),
        )
        kk = db.kk
        kk.insert(k1="a/b", k2='q"1')
        kk.insert(k1="ün|ä", k2="x:y", parent_k1="a/b", parent_k2='q"1')
        kk.insert(k1="leaf", k2="1", parent_k1="ün|ä", parent_k2="x:y")
        # decoys: prefix-similar keys must NOT be conflated by path match
        kk.insert(k1="a", k2="b")
        kk.insert(k1="ab", k2="c", parent_k1="a", parent_k2="b")
        walk = db((kk.parent_k1 == None) & (kk.parent_k2 == None)).recursive_walk(  # noqa: E711
            key=(kk.k1, kk.k2),
            join=lambda w, child: (child.parent_k1 == w.k1)
            & (child.parent_k2 == w.k2),
        )
        rows = self._rows(walk)
        self.assertEqual(len(rows), 5)
        by_root = {}
        for r in rows:
            elems = decode_path(r.path)
            by_root.setdefault(tuple(elems[0]), []).append((elems, r.depth))
        chain = by_root[("a/b", 'q"1')]
        self.assertEqual(
            sorted((tuple(map(tuple, e)), d) for e, d in chain),
            [
                ((("a/b", 'q"1'),), 0),
                ((("a/b", 'q"1'), ("ün|ä", "x:y")), 1),
                ((("a/b", 'q"1'), ("ün|ä", "x:y"), ("leaf", "1")), 2),
            ],
        )
        decoy = by_root[("a", "b")]
        self.assertEqual(
            sorted((tuple(map(tuple, e)), d) for e, d in decoy),
            [((("a", "b"),), 0), ((("a", "b"), ("ab", "c")), 1)],
        )

    # ---------- parameter binding ----------

    def test_root_query_values_are_bound_not_inlined(self):
        org = self.db.org
        a = org.insert(name="alice")
        org.insert(name="bob", boss=a)
        seen = []
        orig = self.db._adapter.execute

        def tap(*args, **kw):
            seen.append((args[0], getattr(args[0], "params", None)))
            return orig(*args, **kw)

        self.db._adapter.execute = tap
        try:
            walk = self._walk(self.db.org.name == "alice")
            rows = self._rows(walk)
        finally:
            self.db._adapter.execute = orig
        self.assertEqual([r.name for r in rows], ["alice", "bob"])
        params = [p for _, ps in seen if ps for p in ps]
        self.assertIn("alice", params)
        for sql, _ in seen:
            self.assertNotIn("alice", str(sql))

    # ---------- unsupported dialects / validation ----------

    def test_unsupported_dialect_raises_clear_error(self):
        self._org_tree()
        self.db._adapter.dbengine = "oracle"
        try:
            with self.assertRaises(NotImplementedError) as ctx:
                self._walk(self.db.org.boss == None)  # noqa: E711
            self.assertIn("oracle", str(ctx.exception))
        finally:
            del self.db._adapter.__dict__["dbengine"]

    def test_invalid_arguments(self):
        self._org_tree()
        with self.assertRaises(ValueError):
            self._walk(self.db.org.boss == None, on_cycle="explode")  # noqa: E711
        with self.assertRaises(ValueError):
            self._walk(self.db.org.boss == None, max_depth=-1)  # noqa: E711
        with self.assertRaises(ValueError):
            self.db(self.db.org.boss == None).recursive_walk(join=None)  # noqa: E711

    def test_reserved_column_names_rejected(self):
        db = self.db
        db.define_table("bad", Field("depth", "integer"))
        with self.assertRaises(ValueError):
            db(db.bad).recursive_walk(
                join=lambda w, child: child.id == w.id
            )


if __name__ == "__main__":
    unittest.main()
