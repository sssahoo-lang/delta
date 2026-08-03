#!/usr/bin/env python3
"""Measure rendered prompt sizes across Spider dev databases.

Offline: no API key. Uses chars/4 as a tokenizer stand-in so budget planning has
a measured number rather than a guess. Writes ``results/token_distribution.json``.

    python scripts/measure_tokens.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from delta.config import RESULTS_DIR  # noqa: E402
from delta.evalh.dataset import DatasetNotAvailableError, load_spider  # noqa: E402
from delta.target_agent.prompt import V0_WEAK, V_HANDTUNED  # noqa: E402
from delta.target_agent.schema_render import render_schema, render_user_message  # noqa: E402


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def main() -> int:
    try:
        dev = load_spider("dev")
    except DatasetNotAvailableError as exc:
        print(exc)
        return 1

    by_db: dict[str, list] = defaultdict(list)
    for ex in dev:
        by_db[ex.db_id].append(ex)

    rows = []
    for db_id, examples in sorted(by_db.items()):
        schema = render_schema(str(examples[0].db_path), 0)
        msg = render_user_message(str(examples[0].db_path), examples[0].question, 0)
        schema_tok = est_tokens(schema)
        msg_tok = est_tokens(msg)
        rows.append(
            {
                "db_id": db_id,
                "n": len(examples),
                "tables": schema.count("CREATE TABLE"),
                "schema_tok_est": schema_tok,
                "msg_tok_est": msg_tok,
                "full_v0_tok_est": est_tokens(V0_WEAK.text + msg),
                "full_ht_tok_est": est_tokens(V_HANDTUNED.text + msg),
            }
        )

    n = sum(r["n"] for r in rows)
    w_schema = sum(r["schema_tok_est"] * r["n"] for r in rows) / n
    w_msg = sum(r["msg_tok_est"] * r["n"] for r in rows) / n
    w_v0 = sum(r["full_v0_tok_est"] * r["n"] for r in rows) / n
    total_no_cache = sum(r["full_v0_tok_est"] * r["n"] for r in rows)
    # Schema paid once per db; subsequent calls pay question-only approx.
    total_prefix = sum(
        r["full_v0_tok_est"]
        + max(0, r["msg_tok_est"] - r["schema_tok_est"]) * (r["n"] - 1)
        for r in rows
    )

    payload = {
        "method": "chars//4 estimate (offline; not an API tokenizer)",
        "per_db": rows,
        "summary": {
            "n_dev": n,
            "n_dbs": len(rows),
            "weighted_mean_schema_tok": round(w_schema, 1),
            "weighted_mean_msg_tok": round(w_msg, 1),
            "weighted_mean_full_v0_tok": round(w_v0, 1),
            "one_pass_no_prefix_cache_tok": total_no_cache,
            "one_pass_with_prefix_cache_tok_est": total_prefix,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "token_distribution.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'db_id':<28} {'n':>4} {'schema':>7} {'msg':>6} {'+v0':>5}")
    for r in rows:
        print(
            f"{r['db_id']:<28} {r['n']:4d} {r['schema_tok_est']:7d} "
            f"{r['msg_tok_est']:6d} {r['full_v0_tok_est']:5d}"
        )
    print()
    print(f"weighted mean full v0 prompt ≈ {w_v0:.0f} tokens")
    print(f"one pass without prefix cache ≈ {total_no_cache:,} tokens")
    print(f"one pass with prefix cache    ≈ {total_prefix:,} tokens")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
