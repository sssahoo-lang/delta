"""Evaluation harness: execute model-generated SQL and score it against gold."""

from delta.evalh.dataset import Example, load_sample, load_spider, spider_available
from delta.evalh.evaluate import EvalReport, ExampleResult, evaluate_prompt
from delta.evalh.execute import ExecResult, execute_sql
from delta.evalh.score import ScoreResult, score_prediction

__all__ = [
    "EvalReport",
    "ExampleResult",
    "ExecResult",
    "Example",
    "ScoreResult",
    "evaluate_prompt",
    "execute_sql",
    "load_sample",
    "load_spider",
    "score_prediction",
    "spider_available",
]
