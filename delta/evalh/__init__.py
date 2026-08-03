"""Evaluation harness: execute model-generated SQL and score it against gold."""

from delta.evalh.buckets import DIFFICULTIES, hardness_from_parsed, hardness_from_sql
from delta.evalh.dataset import Example, load_sample, load_spider, spider_available
from delta.evalh.evaluate import EvalReport, ExampleResult, evaluate_prompt, group_by_database
from delta.evalh.execute import ExecResult, execute_sql
from delta.evalh.sample import stratified_sample
from delta.evalh.score import ScoreResult, gold_is_ordered, score_prediction
from delta.evalh.splits import SplitIds, build_splits, ensure_splits, resolve_split
from delta.evalh.stats import (
    compare_all,
    compare_pair,
    holm_correction,
    mcnemar_exact,
    paired_bootstrap_ci,
    strict_gate_decision,
)

__all__ = [
    "DIFFICULTIES",
    "EvalReport",
    "ExampleResult",
    "ExecResult",
    "Example",
    "ScoreResult",
    "SplitIds",
    "build_splits",
    "compare_all",
    "compare_pair",
    "ensure_splits",
    "evaluate_prompt",
    "execute_sql",
    "gold_is_ordered",
    "group_by_database",
    "hardness_from_parsed",
    "hardness_from_sql",
    "holm_correction",
    "load_sample",
    "load_spider",
    "mcnemar_exact",
    "paired_bootstrap_ci",
    "resolve_split",
    "score_prediction",
    "spider_available",
    "stratified_sample",
    "strict_gate_decision",
]
