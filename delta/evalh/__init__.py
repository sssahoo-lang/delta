"""Evaluation harness: execute model-generated SQL and score it against gold."""

from delta.evalh.execute import ExecResult, execute_sql
from delta.evalh.score import ScoreResult, score_prediction

__all__ = ["ExecResult", "execute_sql", "ScoreResult", "score_prediction"]
