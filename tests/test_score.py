"""Tests for execution-accuracy scoring.

The scorer decides every number this project reports, so its edge cases are
tested directly: order sensitivity, column permutation, numeric normalization,
and the distinction between a wrong answer and a broken query.
"""

from __future__ import annotations

from delta.evalh.score import ScoreReason, gold_is_ordered, score_prediction


def test_identical_query_is_correct(sample_db):
    sql = "SELECT name FROM students WHERE gpa > 3.7"
    assert score_prediction(sample_db, sql, sql).correct


def test_semantically_equivalent_phrasing_is_correct(sample_db):
    """Different SQL, same rows: execution accuracy must accept it."""
    gold = "SELECT name FROM students WHERE gpa > 3.7"
    pred = "SELECT s.name FROM students AS s WHERE s.gpa > 3.7000"
    assert score_prediction(sample_db, pred, gold).correct


def test_wrong_rows_is_incorrect(sample_db):
    gold = "SELECT name FROM students WHERE gpa > 3.7"
    pred = "SELECT name FROM students WHERE gpa > 3.0"
    result = score_prediction(sample_db, pred, gold)
    assert not result.correct
    assert result.reason == ScoreReason.WRONG_ROWS


def test_row_order_ignored_when_gold_is_unordered(sample_db):
    gold = "SELECT name FROM students WHERE dept_id = 1"
    pred = "SELECT name FROM students WHERE dept_id = 1 ORDER BY name DESC"
    assert score_prediction(sample_db, pred, gold).correct


def test_row_order_enforced_when_gold_has_order_by(sample_db):
    gold = "SELECT name FROM students ORDER BY gpa DESC LIMIT 3"
    wrong_direction = "SELECT name FROM students ORDER BY gpa ASC LIMIT 3"
    assert not score_prediction(sample_db, wrong_direction, gold).correct
    assert score_prediction(sample_db, gold, gold).correct


def test_column_order_ignored(sample_db):
    gold = "SELECT name, gpa FROM students WHERE dept_id = 1"
    swapped = "SELECT gpa, name FROM students WHERE dept_id = 1"
    assert score_prediction(sample_db, swapped, gold).correct


def test_integer_and_float_forms_match(sample_db):
    gold = "SELECT count(*) FROM students"
    pred = "SELECT CAST(count(*) AS REAL) FROM students"
    assert score_prediction(sample_db, pred, gold).correct


def test_duplicate_rows_are_significant(sample_db):
    gold = "SELECT dept_id FROM students"
    deduped = "SELECT DISTINCT dept_id FROM students"
    assert not score_prediction(sample_db, deduped, gold).correct


def test_failed_prediction_is_labeled_as_such(sample_db):
    result = score_prediction(sample_db, "SELEKT nonsense", "SELECT count(*) FROM students")
    assert not result.correct
    assert result.reason == ScoreReason.PRED_FAILED


def test_broken_gold_is_surfaced_distinctly(sample_db):
    """A bad benchmark row must not be silently charged to the model."""
    result = score_prediction(sample_db, "SELECT 1", "SELECT * FROM table_that_is_not_there")
    assert not result.correct
    assert result.reason == ScoreReason.GOLD_FAILED


def test_extra_column_is_incorrect(sample_db):
    gold = "SELECT name FROM students WHERE dept_id = 1"
    pred = "SELECT name, gpa FROM students WHERE dept_id = 1"
    assert not score_prediction(sample_db, pred, gold).correct


def test_gold_is_ordered_detection():
    assert gold_is_ordered("SELECT a FROM t ORDER BY a")
    assert gold_is_ordered("select a from t order   by a desc")
    assert not gold_is_ordered("SELECT a FROM t GROUP BY a")
    assert not gold_is_ordered("SELECT a FROM t")
    # ORDER BY inside a subquery must not make the outer result ordered.
    assert not gold_is_ordered(
        "SELECT count(*) FROM (SELECT a FROM t ORDER BY a LIMIT 3) AS s"
    )
