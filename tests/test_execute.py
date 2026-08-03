"""Tests for the SQL execution sandbox.

The safety properties here are load-bearing: the SQL under test is written by a
language model, and a successful write would corrupt the benchmark for every
subsequent run.
"""

from __future__ import annotations

from delta.evalh.execute import ExecStatus, execute_sql


def test_simple_select(sample_db):
    result = execute_sql(sample_db, "SELECT count(*) FROM students")
    assert result.ok
    assert result.rows == [(12,)]
    assert result.column_count == 1


def test_trailing_semicolon_is_tolerated(sample_db):
    result = execute_sql(sample_db, "SELECT count(*) FROM students;  ")
    assert result.ok
    assert result.rows == [(12,)]


def test_syntax_error_is_a_result_not_an_exception(sample_db):
    result = execute_sql(sample_db, "SELEKT * FROM students")
    assert not result.ok
    assert result.status == ExecStatus.SQL_ERROR
    assert result.error


def test_missing_table_is_a_result(sample_db):
    result = execute_sql(sample_db, "SELECT * FROM unicorns")
    assert result.status == ExecStatus.SQL_ERROR
    assert "unicorns" in result.error


def test_empty_sql(sample_db):
    assert execute_sql(sample_db, "   ").status == ExecStatus.EMPTY_SQL


def test_missing_database_file(tmp_path):
    result = execute_sql(tmp_path / "nope.sqlite", "SELECT 1")
    assert result.status == ExecStatus.NO_DATABASE


def test_writes_are_rejected(sample_db):
    """The database is opened read-only, so mutation must fail."""
    for sql in (
        "DROP TABLE students",
        "DELETE FROM students",
        "UPDATE students SET gpa = 0.0",
        "INSERT INTO students VALUES (99, 'Ghost', 1, 2024, 4.0)",
        "CREATE TABLE evil (x INTEGER)",
    ):
        result = execute_sql(sample_db, sql)
        assert result.status == ExecStatus.SQL_ERROR, f"{sql!r} was not rejected"

    # And the data is still intact afterwards.
    assert execute_sql(sample_db, "SELECT count(*) FROM students").rows == [(12,)]


def test_row_cap_truncates(sample_db):
    # A deliberate cross join produces far more rows than the cap allows.
    result = execute_sql(
        sample_db,
        "SELECT T1.name, T2.name FROM students AS T1, students AS T2, courses AS T3",
        max_rows=50,
    )
    assert result.ok
    assert result.truncated
    assert len(result.rows) == 50


def test_timeout_is_reported(sample_db):
    """A runaway recursive query must be cut off, not hang the harness."""
    runaway = """
    WITH RECURSIVE forever(x) AS (
        SELECT 1 UNION ALL SELECT x + 1 FROM forever
    )
    SELECT count(*) FROM forever
    """
    result = execute_sql(sample_db, runaway, timeout_s=1.0)
    assert result.status == ExecStatus.TIMEOUT
    assert result.elapsed_ms < 5_000
