"""Validate the offline fixture itself.

This is the sanity check that guards every number the project will report: if the
scorer cannot score the gold queries as correct against themselves, then the
scorer is broken and no downstream accuracy figure means anything.
"""

from __future__ import annotations

import pytest

from delta.evalh.execute import execute_sql
from delta.evalh.score import score_prediction

VALID_DIFFICULTIES = {"easy", "medium", "hard", "extra"}


def test_every_gold_query_executes(sample_db, sample_questions):
    for q in sample_questions:
        result = execute_sql(sample_db, q["gold"])
        assert result.ok, f"{q['id']} gold query failed: {result.error}"


def test_gold_queries_self_score_at_100_percent(sample_db, sample_questions):
    """The scorer must be perfectly accurate on a set where the answer is known."""
    for q in sample_questions:
        assert score_prediction(sample_db, q["gold"], q["gold"]).correct, q["id"]


def test_no_gold_query_returns_empty(sample_db, sample_questions):
    """An empty result set makes a question trivially passable by a broken query."""
    for q in sample_questions:
        result = execute_sql(sample_db, q["gold"])
        assert result.rows, f"{q['id']} returns no rows, which makes it a weak test case"


def test_question_metadata_is_well_formed(sample_questions):
    ids = [q["id"] for q in sample_questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"

    for q in sample_questions:
        assert q["question"].strip()
        assert q["gold"].strip()
        assert q["difficulty"] in VALID_DIFFICULTIES


def test_difficulty_spread_is_broad(sample_questions):
    """A single-difficulty fixture would hide per-bucket regressions."""
    present = {q["difficulty"] for q in sample_questions}
    assert present == VALID_DIFFICULTIES, f"missing difficulties: {VALID_DIFFICULTIES - present}"


@pytest.mark.parametrize("bad_sql", ["SELECT name FROM students", "SELECT count(*) FROM courses"])
def test_wrong_answers_do_not_pass(sample_db, sample_questions, bad_sql):
    """Guard against a scorer so lenient that any query passes."""
    passed = sum(1 for q in sample_questions if score_prediction(sample_db, bad_sql, q["gold"]).correct)
    assert passed <= 1, f"{bad_sql!r} passed {passed} questions; scorer is too lenient"
