"""Execute model-generated SQL against a SQLite database.

Two properties matter more than speed here, because the SQL is written by a
language model and is untrusted:

1. The database is opened read-only. A generated ``DROP TABLE`` must not be able
   to corrupt the benchmark, which would silently poison every later run.
2. Execution is bounded in time and in rows. Generated SQL can produce accidental
   cross joins that never finish or that return millions of rows.

A failed query is a *scored outcome*, not an exception. Syntax errors, missing
columns, and timeouts are all just "incorrect" as far as the optimizer cares, so
they are captured rather than raised.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_ROWS = 10_000


class ExecStatus:
    OK = "ok"
    TIMEOUT = "timeout"
    SQL_ERROR = "sql_error"
    NO_DATABASE = "no_database"
    EMPTY_SQL = "empty_sql"


@dataclass
class ExecResult:
    status: str
    rows: list[tuple] = field(default_factory=list)
    column_count: int = 0
    error: str | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == ExecStatus.OK


def execute_sql(
    db_path: str | Path,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> ExecResult:
    """Run ``sql`` read-only against the SQLite file at ``db_path``.

    Never raises for bad SQL. Returns an :class:`ExecResult` whose ``status``
    describes what happened.
    """
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return ExecResult(status=ExecStatus.EMPTY_SQL, error="empty SQL string")

    db_path = Path(db_path)
    if not db_path.exists():
        return ExecResult(status=ExecStatus.NO_DATABASE, error=f"no such database: {db_path}")

    # mode=ro makes any write attempt fail at the SQLite level rather than
    # relying on us to detect write statements by parsing, which is defeatable.
    uri = f"file:{db_path}?mode=ro"
    started = time.perf_counter()
    conn: sqlite3.Connection | None = None
    timer: threading.Timer | None = None
    timed_out = threading.Event()

    try:
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_s, check_same_thread=False)
        # Spider databases contain values that are not valid UTF-8; decoding
        # them strictly would crash on rows that the official evaluator handles.
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")

        def _interrupt() -> None:
            timed_out.set()
            if conn is not None:
                conn.interrupt()

        timer = threading.Timer(timeout_s, _interrupt)
        timer.daemon = True
        timer.start()

        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        if truncated:
            rows = rows[:max_rows]

        column_count = len(cursor.description) if cursor.description else 0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ExecResult(
            status=ExecStatus.OK,
            rows=[tuple(r) for r in rows],
            column_count=column_count,
            truncated=truncated,
            elapsed_ms=elapsed_ms,
        )

    except sqlite3.Error as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if timed_out.is_set():
            return ExecResult(
                status=ExecStatus.TIMEOUT,
                error=f"exceeded {timeout_s}s",
                elapsed_ms=elapsed_ms,
            )
        return ExecResult(status=ExecStatus.SQL_ERROR, error=str(exc), elapsed_ms=elapsed_ms)

    finally:
        if timer is not None:
            timer.cancel()
        if conn is not None:
            conn.close()
