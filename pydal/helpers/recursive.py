"""
Explicit recursive traversal constructor built on pydal's CTE support.

``Set.recursive_walk(...)`` turns a root query plus a parent/child join
condition into a recursive CTE ("walk") that yields, for every visited
node occurrence, the node's own columns plus:

* ``depth``     — 0 for the roots, +1 per traversed edge;
* ``path``      — the encoded chain of node keys from the root to this
  node (exact, collision-free, see below);
* ``cycle``     — 1 when this row closes a cycle (``on_cycle="report"``);
* ``truncated`` — 1 when the node sits at ``max_depth`` and still has
  children that were cut off by the depth cap (0 for natural leaves).

Cycle detection is path-based: two branches reaching the same shared
node (a diamond DAG) is NOT a cycle — only a node already present on
the current path is. ``on_cycle="report"`` emits the closing row once
(flagged with ``cycle=1``) and stops descending; ``on_cycle="ignore"``
silently drops the edge that would close the cycle.

Path encoding (SQLite): every key component is rendered with
``hex(<col>)`` — the hex alphabet cannot collide with the ``:``
component separator or the ``/`` element separator — so keys containing
separators, quotes or Unicode are matched exactly. Membership is tested
as ``instr(path, '/' || <element> || '/')`` on the ``/``-wrapped path,
which only ever matches whole elements.

All user values (root-query filters, ``max_depth``) flow through
pydal's normal expression machinery and arrive as bound parameters;
nothing is interpolated into the SQL text.

Only dialects with recursive CTEs and the required string-encoding
functions are supported; anything else raises ``NotImplementedError``.
"""

from ..objects import Expression, Field

#: column names added by the walk; node tables may not reuse them
RESERVED_COLUMNS = ("depth", "path", "cycle", "truncated")


def _sqlite_element(tbl, keys):
    """SQL fragment encoding one path element from the key columns."""
    return " || ':' || ".join("hex(%s)" % tbl[k.name].sqlsafe for k in keys)


def _sqlite_on_path(path_sql, tbl, keys):
    """SQL fragment testing whether tbl's key is already on the path."""
    return "instr(%s, '/' || %s || '/')" % (path_sql, _sqlite_element(tbl, keys))


_SQLITE_OPS = (_sqlite_element, _sqlite_on_path)

#: dbengine -> (element_sql, on_path_sql); dialects without recursive
#: CTEs or the required exact-encoding functions are simply absent.
_PATH_OPS = {
    "sqlite": _SQLITE_OPS,
    "spatialite": _SQLITE_OPS,
}


def _path_ops(adapter):
    engine = getattr(adapter, "dbengine", None)
    ops = _PATH_OPS.get(engine)
    if ops is None:
        raise NotImplementedError(
            "recursive_walk is not supported on the %r dialect: it "
            "requires recursive CTEs and exact path-encoding functions "
            "(currently implemented for SQLite only)" % (engine,)
        )
    return ops


def recursive_walk(db_set, key=None, join=None, max_depth=None,
                   on_cycle="report", name="walk"):
    """
    Build a recursive-traversal CTE over ``db_set`` (the root query).

    * ``key`` — a Field or tuple of Fields identifying a node
      (composite keys supported); defaults to the table's primary key.
    * ``join`` — callable ``(walk, child) -> Query`` linking a parent
      row of the walk to a child row of the node table, e.g.
      ``lambda walk, child: child.boss == walk.id``.
    * ``max_depth`` — maximum emitted depth (roots are 0); ``None``
      means unlimited (cycle detection still guarantees termination).
    * ``on_cycle`` — ``"report"`` (emit the cycle-closing row once with
      ``cycle=1`` and stop) or ``"ignore"`` (silently drop the edge).
    * ``name`` — name of the resulting CTE.

    Returns an aliased ``Select`` (the CTE) exposing the node table's
    own columns plus ``depth``, ``path``, ``cycle`` and ``truncated``;
    use it like any other pydal CTE: ``db().select(walk.ALL)``.
    """
    db = db_set.db
    adapter = db._adapter
    element_sql, on_path_sql = _path_ops(adapter)
    if not callable(join):
        raise ValueError(
            "recursive_walk: 'join' must be a callable "
            "(walk, child) -> Query"
        )
    if on_cycle not in ("report", "ignore"):
        raise ValueError(
            "recursive_walk: on_cycle must be 'report' or 'ignore', "
            "got %r" % (on_cycle,)
        )
    if max_depth is not None and (
        not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError(
            "recursive_walk: max_depth must be a non-negative integer "
            "or None"
        )
    if key is None:
        table = adapter.get_table(db_set.query)
        keys = [getattr(table, "_id", None)]
        if keys[0] is None:
            raise ValueError(
                "recursive_walk: 'key' is required when the table has "
                "no primary key"
            )
    else:
        keys = list(key) if isinstance(key, (list, tuple)) else [key]
        table = None
    if not keys or not all(isinstance(k, Field) for k in keys):
        raise ValueError("recursive_walk: 'key' must be Field(s)")
    for k in keys:
        if table is None:
            table = k.table
        elif k.table is not table:
            raise ValueError(
                "recursive_walk: all key fields must belong to the "
                "same table"
            )
    clash = set(table.fields) & set(RESERVED_COLUMNS)
    if clash:
        raise ValueError(
            "recursive_walk: node table %r uses reserved column "
            "name(s): %s" % (table._tablename, ", ".join(sorted(clash)))
        )

    inner_name = "%s__inner" % name
    node_fields = [table[fname] for fname in table.fields]

    def literal_zero():
        return Expression(db, "0", type="integer")

    def case_one_zero(query):
        return Expression(
            db, db._adapter.dialect.case, query, (1, 0), type="integer"
        )

    # ---- anchor: the root query at depth 0 ----
    anchor_fields = node_fields + [
        literal_zero().with_alias("depth"),
        Expression(
            db, "'/' || %s || '/'" % element_sql(table, keys), type="string"
        ).with_alias("path"),
        literal_zero().with_alias("cycle"),
    ]
    inner = db_set.cte(inner_name, *anchor_fields)

    # ---- recursive term: children of the current frontier ----
    def recurse(walk):
        cond = join(walk, table) & (walk.cycle == 0)
        if max_depth is not None:
            cond = cond & (walk.depth < max_depth)
        on_path = Expression(
            db, on_path_sql(walk.path.sqlsafe, table, keys), type="integer"
        )
        if on_cycle == "ignore":
            cond = cond & (on_path == 0)
            cycle_col = literal_zero().with_alias("cycle")
        else:
            cycle_col = case_one_zero(on_path > 0).with_alias("cycle")
        fields = node_fields + [
            (walk.depth + 1).with_alias("depth"),
            Expression(
                db,
                "%s || %s || '/'" % (walk.path.sqlsafe, element_sql(table, keys)),
                type="string",
            ).with_alias("path"),
            cycle_col,
        ]
        return db(cond).nested_select(*fields)

    inner = inner.union_all(recurse)

    # ---- outer CTE: attach the truncated flag ----
    out_fields = [inner[fname] for fname in table.fields] + [
        inner.depth, inner.path, inner.cycle,
    ]
    if max_depth is None:
        # no depth cap: nothing can ever be truncated
        out_fields.append(literal_zero().with_alias("truncated"))
        return db().cte(name, *out_fields)
    child = table.with_alias("%s__child" % name)
    has_more = join(inner, child)
    if on_cycle == "ignore":
        # only children that would actually be emitted count
        has_more = has_more & (
            Expression(
                db, on_path_sql(inner.path.sqlsafe, child, keys), type="integer"
            ) == 0
        )
    truncated = case_one_zero(
        (inner.depth == max_depth) & (child[keys[0].name] != None)  # noqa: E711
    )
    out_fields.append(truncated.max().with_alias("truncated"))
    return db().cte(
        name,
        *out_fields,
        left=child.on(has_more),
        groupby=[inner[fname] for fname in table.fields]
        + [inner.depth, inner.path, inner.cycle],
    )
