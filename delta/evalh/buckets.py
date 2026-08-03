"""Classify gold SQL into Spider difficulty buckets.

Prefers the official labels computed from Spider's pre-parsed ``sql`` field —
the same structure and counting rules as ``taoyds/spider/evaluation.py`` — so
stratified splits and per-difficulty reporting match published Spider numbers.
Falls back to a ``sqlglot`` approximation when only a raw SQL string is available.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import sqlglot
from sqlglot import exp

DIFFICULTIES = ("easy", "medium", "hard", "extra")

AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
_AGG_NONE = AGG_OPS.index("none")
_WHERE_LIKE = WHERE_OPS.index("like")


def difficulty_histogram(labels: list[str]) -> dict[str, int]:
    counts = Counter(labels)
    return {d: counts.get(d, 0) for d in DIFFICULTIES}


def _has_agg(unit: Any) -> bool:
    return isinstance(unit, (list, tuple)) and len(unit) > 0 and unit[0] != _AGG_NONE


def _count_agg(units: Any) -> int:
    if not units:
        return 0
    return sum(1 for unit in units if _has_agg(unit))


def _get_nested_sql(sql: dict) -> list[dict]:
    nested: list[dict] = []
    conds = (
        (sql.get("from") or {}).get("conds") or []
    )[::2] + (sql.get("where") or [])[::2] + (sql.get("having") or [])[::2]
    for cond_unit in conds:
        if not isinstance(cond_unit, (list, tuple)) or len(cond_unit) < 5:
            continue
        if isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
        if isinstance(cond_unit[4], dict):
            nested.append(cond_unit[4])
    for key in ("intersect", "except", "union"):
        if sql.get(key) is not None:
            nested.append(sql[key])
    return nested


def _count_component1(sql: dict) -> int:
    count = 0
    if len(sql.get("where") or []) > 0:
        count += 1
    if len(sql.get("groupBy") or []) > 0:
        count += 1
    if len(sql.get("orderBy") or []) > 0:
        count += 1
    if sql.get("limit") is not None:
        count += 1
    table_units = (sql.get("from") or {}).get("table_units") or []
    if len(table_units) > 0:
        count += len(table_units) - 1

    ao = (
        ((sql.get("from") or {}).get("conds") or [])[1::2]
        + (sql.get("where") or [])[1::2]
        + (sql.get("having") or [])[1::2]
    )
    count += sum(1 for token in ao if token == "or")

    cond_units = (
        ((sql.get("from") or {}).get("conds") or [])[::2]
        + (sql.get("where") or [])[::2]
        + (sql.get("having") or [])[::2]
    )
    count += sum(
        1
        for cond_unit in cond_units
        if isinstance(cond_unit, (list, tuple))
        and len(cond_unit) > 1
        and cond_unit[1] == _WHERE_LIKE
    )
    return count


def _count_component2(sql: dict) -> int:
    return len(_get_nested_sql(sql))


def _count_others(sql: dict) -> int:
    """Literal port of Spider's ``count_others``."""
    count = 0
    select_cols = (sql.get("select") or [False, []])[1] or []
    agg_count = _count_agg(select_cols)
    agg_count += _count_agg((sql.get("where") or [])[::2])
    agg_count += _count_agg(sql.get("groupBy") or [])
    order_by = sql.get("orderBy") or []
    if len(order_by) > 0:
        order_units = order_by[1] if len(order_by) > 1 else []
        agg_count += _count_agg(
            [unit[1] for unit in order_units if isinstance(unit, (list, tuple)) and unit[1]]
            + [unit[2] for unit in order_units if isinstance(unit, (list, tuple)) and len(unit) > 2 and unit[2]]
        )
    # Official code passes the whole having list, including connector tokens.
    agg_count += _count_agg(sql.get("having") or [])
    if agg_count > 1:
        count += 1
    if len(select_cols) > 1:
        count += 1
    if len(sql.get("where") or []) > 1:
        count += 1
    if len(sql.get("groupBy") or []) > 1:
        count += 1
    return count


def hardness_from_parsed(sql: dict) -> str:
    """Official Spider hardness from a pre-parsed ``sql`` dict."""
    comp1 = _count_component1(sql)
    comp2 = _count_component2(sql)
    others = _count_others(sql)

    if comp1 <= 1 and others == 0 and comp2 == 0:
        return "easy"
    if (others <= 2 and comp1 <= 1 and comp2 == 0) or (
        comp1 <= 2 and others < 2 and comp2 == 0
    ):
        return "medium"
    if (
        (others > 2 and comp1 <= 2 and comp2 == 0)
        or (2 < comp1 <= 3 and others <= 2 and comp2 == 0)
        or (comp1 <= 1 and others == 0 and comp2 <= 1)
    ):
        return "hard"
    return "extra"


def _parse(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def _sqlglot_components(sql: str) -> tuple[int, int, int]:
    root = _parse(sql)
    if root is None:
        return (0, 0, 0)

    set_ops = (exp.Union, exp.Intersect, exp.Except)
    selects: list[exp.Select] = []
    if isinstance(root, exp.Select):
        selects = [root]
    elif isinstance(root, set_ops):
        stack: list[exp.Expression] = [root]
        while stack:
            node = stack.pop()
            if isinstance(node, exp.Select):
                selects.append(node)
            elif isinstance(node, set_ops):
                if node.this is not None:
                    stack.append(node.this)
                if node.expression is not None:
                    stack.append(node.expression)
    else:
        first = root.find(exp.Select)
        selects = [first] if first is not None else []

    if not selects:
        return (0, 0, 0)

    comp1 = 0
    # Each UNION / INTERSECT / EXCEPT counts toward component2 (Spider IUEN).
    comp2 = sum(1 for _ in root.find_all(set_ops))
    others = 0
    for select in selects:
        if select.args.get("where") is not None:
            comp1 += 1
        if select.args.get("group") is not None:
            comp1 += 1
        if select.args.get("order") is not None:
            comp1 += 1
        if select.args.get("limit") is not None:
            comp1 += 1
        comp1 += len(list(select.find_all(exp.Join)))
        comp1 += sum(1 for _ in select.find_all(exp.Or))
        comp1 += sum(1 for _ in select.find_all((exp.Like, exp.ILike)))
        comp2 += sum(1 for child in select.find_all(exp.Select) if child is not select)

        aggs = sum(
            1
            for fn in select.find_all(exp.Func)
            if isinstance(fn, exp.AggFunc)
            or fn.sql_name().upper() in {"COUNT", "SUM", "AVG", "MIN", "MAX"}
        )
        if aggs > 1:
            others += 1
        if len(select.expressions or []) > 1:
            others += 1
        where = select.args.get("where")
        if where is not None:
            pred = where.this if isinstance(where, exp.Where) else where
            n_conn = len(list(pred.find_all((exp.And, exp.Or)))) if pred else 0
            if pred is not None and (n_conn + 1) > 1:
                others += 1
        group = select.args.get("group")
        if group is not None and len(getattr(group, "expressions", []) or []) > 1:
            others += 1

    return (comp1, comp2, others)


def hardness_from_sql(sql: str) -> str:
    """Approximate Spider hardness from a raw SQL string via sqlglot."""
    if not (sql or "").strip():
        return "easy"
    comp1, comp2, others = _sqlglot_components(sql)
    if comp1 <= 1 and others == 0 and comp2 == 0:
        return "easy"
    if (others <= 2 and comp1 <= 1 and comp2 == 0) or (
        comp1 <= 2 and others < 2 and comp2 == 0
    ):
        return "medium"
    if (
        (others > 2 and comp1 <= 2 and comp2 == 0)
        or (2 < comp1 <= 3 and others <= 2 and comp2 == 0)
        or (comp1 <= 1 and others == 0 and comp2 <= 1)
    ):
        return "hard"
    return "extra"


classify_hardness = hardness_from_sql
DIFFICULTY_LEVELS = DIFFICULTIES


def with_difficulty(examples: list) -> list:
    return [ex.with_difficulty(hardness_from_sql(ex.gold)) for ex in examples]
