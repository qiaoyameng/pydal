# -*- coding: utf-8 -*-
r"""
Explicit recursive traversal constructor, built on pydal's CTE support.

``recursive_traverse`` turns a root query plus a parent/child join
condition into a ``WITH RECURSIVE`` common table expression that walks
a graph (org charts, category trees, generic DAGs, possibly cyclic)
and yields, for every visited path, the node key, its depth and the
encoded path from the root to it.

Cycle handling is *path-based*: a node that is reachable through two
different branches (a diamond DAG) is visited once per path and is NOT
a cycle; a cycle is reported only when the next node already appears
on the *current* path. Two policies are available via ``on_cycle``:

* ``"report"`` — emit the row that closes the cycle flagged with
  ``cycle = 1`` and do not expand it any further;
* ``"ignore"`` — silently drop the edge that would close the cycle.

Every path is finite (paths never repeat a node), so traversal always
terminates even without ``max_depth``.

Path encoding
-------------
``path`` is a text column of the form ``|<enc(k1)>|<enc(k2)>|...``
(one ``enc(...)`` block per key component per visited node, always
delimiter-terminated). Keys are cast to text and encoded so the
encoded form can never contain the delimiter: ``\`` becomes ``\\``
and ``|`` becomes ``\p``. Because no encoded key contains ``|``, a
plain substring search for ``|<enc(key)>|`` is an exact
path-membership test — keys containing separators, quotes or Unicode
are matched correctly. Composite keys simply concatenate their
encoded components, which stays unambiguous for the same reason.

Values (root-query filters, join constants, ``max_depth``, path
delimiters, ...) are always bound parameters, never interpolated into
the SQL text.

Dialect support
---------------
Requires recursive CTEs plus a string-position function. Currently
implemented for SQLite; other dialects raise ``NotImplementedError``.
"""

from .objects import DialectOp, Expression, Field, Query, Table

__all__ = ["recursive_traverse"]

#: column names appended after the node key/extra fields
DEPTH_COLUMN = "depth"
PATH_COLUMN = "path"
CYCLE_COLUMN = "cycle"
TRUNCATED_COLUMN = "truncated"

#: element delimiter of the encoded path
PATH_DELIMITER = "|"
#: escape character used inside encoded keys
PATH_ESCAPE = "\\"
#: ``PATH_DELIMITER`` inside a key is encoded as ``PATH_ESCAPE + "p"``;
#: the replacement contains no delimiter, so encoded keys are always
#: delimiter-free and substring matching stays exact.
_PATH_DELIMITER_ENCODED = PATH_ESCAPE + "p"


def recursive_traverse(
    root_set,
    key,
    link,
    max_depth=None,
    on_cycle="report",
    name="walk",
    fields=(),
):
    """
    Build a recursive-traversal CTE over ``root_set``.

    :param root_set: a ``Set`` (``db(query)``) selecting the root
        node(s); multiple roots are supported.
    :param key: a ``Field`` or a non-empty sequence of ``Field``\\ s
        (composite key) identifying a node; all fields must belong to
        the same (node) table.
    :param link: callable ``(parent, child) -> Query`` describing one
        edge. ``parent`` is the traversal row (the node table in the
        anchor part, the walk CTE in the recursive part), ``child`` is
        a fresh alias of the node table. The returned query may
        reference additional tables (e.g. an edge table) captured from
        the enclosing scope.
    :param max_depth: optional non-negative int; rows deeper than this
        are not emitted. Roots have depth 0.
    :param on_cycle: ``"report"`` (emit the cycle-closing row with
        ``cycle = 1`` and stop expanding it) or ``"ignore"`` (drop the
        cycle-closing edge).
    :param name: name of the generated CTE.
    :param fields: optional extra fields of the node table to carry
        through the traversal (e.g. a label column).

    :returns: a pydal ``Select`` (aliased CTE) with the key fields,
        the extra ``fields``, and the columns ``depth`` (int),
        ``path`` (encoded path string), ``cycle`` (1 when the row
        closes a cycle, ``on_cycle="report"`` only) and ``truncated``
        (1 when the row sits at ``max_depth`` and still had unfollowed
        outgoing edges — distinguishing an actively truncated branch
        from a natural leaf). Use it like any CTE::

            walk = db(db.org.boss == None).recursive_traverse(
                key=db.org.id,
                link=lambda parent, child: child.boss == parent.id,
            )
            rows = db().select(walk.ALL, orderby=walk.path)
    """
    db = root_set.db
    adapter = db._adapter
    dialect = adapter.dialect

    if on_cycle not in ("report", "ignore"):
        raise ValueError("on_cycle must be 'report' or 'ignore', got %r" % (on_cycle,))
    if max_depth is not None and (
        not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0
    ):
        raise ValueError("max_depth must be a non-negative integer or None")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    keys = list(key) if isinstance(key, (list, tuple)) else [key]
    if not keys or not all(isinstance(k, Field) for k in keys):
        raise TypeError("key must be a Field or a non-empty sequence of Fields")
    node_table = keys[0]._table
    if not isinstance(node_table, Table):
        raise TypeError("key fields must belong to a table")
    for k in keys[1:]:
        if k._table is not node_table:
            raise ValueError("all key fields must belong to the same table")
    extra_fields = list(fields)
    for f in extra_fields:
        if not isinstance(f, Field) or f._table is not node_table:
            raise ValueError("fields must be Fields of the key's table")

    if not callable(getattr(dialect, "instr", None)):
        raise NotImplementedError(
            "recursive_traverse is not supported on dialect %r: it requires "
            "recursive CTEs and a string-position function (instr), currently "
            "available on SQLite only" % type(dialect).__name__
        )

    from .ast_translate import set_to_select

    def _literal(value, type_):
        # constant value, emitted as a bound parameter (never inlined)
        return Expression(db, DialectOp(db, "literal"), value, type=type_)

    def _concat(*items):
        result = None
        for item in items:
            if result is None:
                result = item
            else:
                result = Expression(db, dialect.concat, result, item, type="string")
        return result

    def _encode(key_expression):
        # delimiter-free text encoding of one key component
        return (
            key_expression.cast("text")
            .replace(PATH_ESCAPE, PATH_ESCAPE * 2)
            .replace(PATH_DELIMITER, _PATH_DELIMITER_ENCODED)
        )

    def _encoded_key_parts(key_expressions):
        # enc(k1), "|", enc(k2), "|", ... — trailing delimiter included
        parts = []
        for kx in key_expressions:
            parts.append(_encode(kx))
            parts.append(_literal(PATH_DELIMITER, "string"))
        return parts

    def _has_child(parent_ref):
        # EXISTS (<node/edge tables> WHERE link(parent_ref, probe))
        probe = node_table.with_alias("%s__sub" % name)
        link_query = link(parent_ref, probe)
        if not isinstance(link_query, Query):
            raise TypeError("link must return a pydal Query")
        sub = set_to_select(
            db(link_query), (probe[keys[0].name],), {}, _in_cte_body=True
        )
        return Expression(db, DialectOp(db, "exists"), sub, type="integer")

    # ---------------- anchor (roots) ----------------
    anchor_fields = (
        list(keys)
        + extra_fields
        + [
            _literal(0, "integer").with_alias(DEPTH_COLUMN),
            _concat(
                _literal(PATH_DELIMITER, "string"), *_encoded_key_parts(keys)
            ).with_alias(PATH_COLUMN),
            _literal(0, "integer").with_alias(CYCLE_COLUMN),
        ]
    )
    if max_depth == 0:
        anchor_truncated = _has_child(node_table)
    else:
        # no max_depth, or roots are strictly below it: never truncated
        anchor_truncated = _literal(0, "integer")
    anchor_fields.append(anchor_truncated.with_alias(TRUNCATED_COLUMN))

    walk = root_set.cte(name, *anchor_fields)

    # ---------------- recursive step ----------------
    def _recursive_step(w):
        child = node_table.with_alias("%s__step" % name)
        child_keys = [child[k.name] for k in keys]
        link_query = link(w, child)
        if not isinstance(link_query, Query):
            raise TypeError("link must return a pydal Query")

        needle = _concat(
            _literal(PATH_DELIMITER, "string"), *_encoded_key_parts(child_keys)
        )
        seen = Expression(db, dialect.instr, w[PATH_COLUMN], needle, type="integer")

        query = link_query
        if max_depth is not None:
            query = query & (w[DEPTH_COLUMN] < max_depth)
        if on_cycle == "report":
            # never expand a row that already closed a cycle
            query = query & (w[CYCLE_COLUMN] == 0)
            cycle_expr = Expression(db, DialectOp(db, "gt"), seen, 0, type="integer")
        else:
            # drop edges whose target is already on the current path
            query = query & (seen == 0)
            cycle_expr = _literal(0, "integer")

        if max_depth is None:
            truncated_expr = _literal(0, "integer")
        else:
            # the emitted row is the child: truncated means "sits at the
            # depth limit and still has unfollowed outgoing edges"
            truncated_expr = Expression(
                db,
                DialectOp(db, "_and"),
                (w[DEPTH_COLUMN] + 1) == max_depth,
                _has_child(child),
                type="integer",
            )

        step_fields = (
            child_keys
            + [child[f.name] for f in extra_fields]
            + [
                (w[DEPTH_COLUMN] + 1).with_alias(DEPTH_COLUMN),
                _concat(w[PATH_COLUMN], *_encoded_key_parts(child_keys)).with_alias(
                    PATH_COLUMN
                ),
                cycle_expr.with_alias(CYCLE_COLUMN),
                truncated_expr.with_alias(TRUNCATED_COLUMN),
            ]
        )
        return db(query).nested_select(*step_fields)

    walk.union_all(_recursive_step)
    return walk
