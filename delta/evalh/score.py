"""Score predicted SQL by executing it and comparing result sets to gold.

This is *execution accuracy*: a prediction is correct when running it returns the
same data as running the gold query. That is the metric Spider adopted as
official in November 2020, and it is far more meaningful than string similarity,
which punishes correct queries written differently.

Three comparison rules, following the official Spider evaluator:

- Rows are compared as a **multiset** unless the gold query's *outermost*
  ``ORDER BY`` makes order significant. An ``ORDER BY`` inside a subquery does
  not count. Duplicate rows are meaningful either way.
- **Column order is not significant.** ``SELECT name, age`` and ``SELECT age, name``
  both answer "names and ages", so a permutation search is used when the direct
  comparison fails.
- Values are **normalized** before comparison so that ``3`` and ``3.0`` match, and
  so float noise from different aggregation paths does not cause false negatives.

Deliberate divergences from the official evaluator, documented here because they
affect the reported numbers:

1. The official *test-suite* metric executes against several synthetically
   perturbed copies of each database to catch predictions that are only
   accidentally right on the shipped data. Delta scores against the single
   shipped database, which is marginally more generous.
2. Numeric strings are coerced to floats before comparison (``'3'`` matches
   ``3``). The official evaluator is stricter about types. Models sometimes
   quote numbers that SQLite still treats numerically; this avoids punishing
   that harmless style difference.

Both are applied identically to every condition being compared, so they do not
bias the deltas between them, which is what this project actually measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import sqlglot
from sqlglot import exp

from delta.evalh.execute import DEFAULT_MAX_ROWS, DEFAULT_TIMEOUT_S, ExecResult, execute_sql

# Above this width, trying every column permutation gets expensive and the
# ambiguity it resolves is vanishingly rare in Spider.
MAX_PERMUTATION_COLUMNS = 6

FLOAT_PRECISION = 6


class ScoreReason:
    CORRECT = "correct"
    WRONG_ROWS = "wrong_rows"
    PRED_FAILED = "pred_failed"
    GOLD_FAILED = "gold_failed"


@dataclass
class ScoreResult:
    correct: bool
    reason: str
    pred: ExecResult | None = None
    gold: ExecResult | None = None
    detail: str | None = None


def gold_is_ordered(gold_sql: str) -> bool:
    """Whether the gold query's *outermost* row order is part of the answer.

    An ``ORDER BY`` inside a subquery (for example to support ``LIMIT`` before
    an aggregate) does not make the outer result ordered. Matching the keyword
    anywhere would score such gold order-sensitively and punish correct
    unordered predictions.
    """
    if not (gold_sql or "").strip():
        return False
    try:
        root = sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return "order by" in gold_sql.lower()
    # Walk set-op peers; any top-level SELECT with ORDER BY counts.
    set_ops = (exp.Union, exp.Intersect, exp.Except)
    stack: list = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, exp.Select):
            if node.args.get("order") is not None:
                return True
        elif isinstance(node, set_ops):
            if node.this is not None:
                stack.append(node.this)
            if node.expression is not None:
                stack.append(node.expression)
        elif isinstance(node, exp.Subquery) and node.this is not None:
            stack.append(node.this)
    return False


def _normalize_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return round(float(value), FLOAT_PRECISION)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        stripped = value.strip()
        # Numeric strings are compared numerically: some models quote numbers,
        # and SQLite happily stores them either way.
        try:
            return round(float(stripped), FLOAT_PRECISION)
        except ValueError:
            return stripped
    return value


def _normalize_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_normalize_value(v) for v in row) for row in rows]


def _rows_equal(pred_rows: list[tuple], gold_rows: list[tuple], ordered: bool) -> bool:
    if len(pred_rows) != len(gold_rows):
        return False
    if ordered:
        return pred_rows == gold_rows
    return sorted(pred_rows, key=repr) == sorted(gold_rows, key=repr)


def _equal_under_column_permutation(
    pred_rows: list[tuple], gold_rows: list[tuple], ordered: bool
) -> bool:
    """Try to match gold by reordering the predicted columns.

    ``SELECT age, name`` should score the same as ``SELECT name, age``.
    """
    if not pred_rows or not gold_rows:
        return False
    width = len(gold_rows[0])
    if len(pred_rows[0]) != width or width > MAX_PERMUTATION_COLUMNS:
        return False

    for order in permutations(range(width)):
        if order == tuple(range(width)):
            continue  # already tried as the direct comparison
        permuted = [tuple(row[i] for i in order) for row in pred_rows]
        if _rows_equal(permuted, gold_rows, ordered):
            return True
    return False


def score_prediction(
    db_path: str | Path,
    predicted_sql: str,
    gold_sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> ScoreResult:
    """Execute both queries and decide whether the prediction is correct."""
    gold = execute_sql(db_path, gold_sql, timeout_s=timeout_s, max_rows=max_rows)
    if not gold.ok:
        # A broken gold query means the benchmark row is unusable. Surface it
        # loudly rather than silently marking the model wrong.
        return ScoreResult(
            correct=False,
            reason=ScoreReason.GOLD_FAILED,
            gold=gold,
            detail=f"gold query failed ({gold.status}): {gold.error}",
        )

    pred = execute_sql(db_path, predicted_sql, timeout_s=timeout_s, max_rows=max_rows)
    if not pred.ok:
        return ScoreResult(
            correct=False,
            reason=ScoreReason.PRED_FAILED,
            pred=pred,
            gold=gold,
            detail=f"{pred.status}: {pred.error}",
        )

    ordered = gold_is_ordered(gold_sql)
    pred_rows = _normalize_rows(pred.rows)
    gold_rows = _normalize_rows(gold.rows)

    if _rows_equal(pred_rows, gold_rows, ordered) or _equal_under_column_permutation(
        pred_rows, gold_rows, ordered
    ):
        return ScoreResult(correct=True, reason=ScoreReason.CORRECT, pred=pred, gold=gold)

    return ScoreResult(
        correct=False,
        reason=ScoreReason.WRONG_ROWS,
        pred=pred,
        gold=gold,
        detail=f"returned {len(pred_rows)} rows, expected {len(gold_rows)}",
    )
