"""Turn a SQLite database into the schema text the model sees.

Rendering is held **fixed** across every experimental condition. Only the system
prompt evolves. If schema formatting changed alongside the prompt, an accuracy
delta could not be attributed to either one, and the entire comparison would be
confounded.

The format is ``CREATE TABLE`` DDL, which is the standard representation in the
text-to-SQL literature and the one models handle best, since it is what they saw
during pretraining.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

# Sample rows help models pick the right literal casing and date format, but
# they cost context. Kept modest and configurable.
DEFAULT_SAMPLE_ROWS = 0
MAX_SAMPLE_ROWS = 5
MAX_CELL_CHARS = 40


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _render_table(conn: sqlite3.Connection, table: str, sample_rows: int) -> str:
    cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    fks = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()

    lines = [f"CREATE TABLE {table} ("]
    body: list[str] = []
    for _cid, name, ctype, _notnull, _default, _pk in cols:
        body.append(f"  {name} {ctype or 'TEXT'}")

    pks = [c[1] for c in cols if c[5]]
    if pks:
        body.append(f"  PRIMARY KEY ({', '.join(pks)})")
    for fk in fks:
        # (id, seq, referenced_table, from_col, to_col, on_update, on_delete, match)
        body.append(f"  FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})")

    lines.append(",\n".join(body))
    lines.append(");")

    if sample_rows > 0:
        lines.append(_render_samples(conn, table, sample_rows))

    return "\n".join(line for line in lines if line)


def _render_samples(conn: sqlite3.Connection, table: str, n: int) -> str:
    n = min(n, MAX_SAMPLE_ROWS)
    try:
        cursor = conn.execute(f'SELECT * FROM "{table}" LIMIT {n}')
        rows = cursor.fetchall()
    except sqlite3.Error:
        return ""
    if not rows:
        return ""

    headers = [d[0] for d in cursor.description]

    def cell(v: object) -> str:
        s = "NULL" if v is None else str(v)
        return s[:MAX_CELL_CHARS] + "..." if len(s) > MAX_CELL_CHARS else s

    out = [f"/* {n} example row(s) from {table}:", "   " + " | ".join(headers)]
    out.extend("   " + " | ".join(cell(v) for v in row) for row in rows)
    out.append("*/")
    return "\n".join(out)


@lru_cache(maxsize=256)
def render_schema(db_path: str, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> str:
    """Render ``db_path`` as CREATE TABLE DDL. Cached, since it is re-rendered
    for every question against the same database."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"no such database: {path}")

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        return "\n\n".join(_render_table(conn, t, sample_rows) for t in _table_names(conn))
    finally:
        conn.close()


def render_user_message(db_path: str, question: str, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> str:
    """The user turn: schema, then question. Identical across all conditions."""
    schema = render_schema(str(db_path), sample_rows)
    return f"Database schema:\n\n{schema}\n\nQuestion: {question}"
