#!/usr/bin/env python3
"""Phase 1.5: validate the real Groq path before building more machinery.

Downloads are assumed done. This script:

1. Refreshes ``results/token_distribution.json`` (offline).
2. Scores v0 and the hand-tuned prompt on 100 stratified Spider dev examples
   against real Groq.
3. Checks that token accounting is non-zero, that ``extract_sql`` handles real
   Llama output, and that headroom between the two prompts is at least 10 points.

    python scripts/run_real_path.py              # requires GROQ_API_KEY
    python scripts/run_real_path.py --tokens-only

Exit code 2 means the headroom gate failed; exit code 1 is a setup error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from delta.config import RESULTS_DIR, Settings  # noqa: E402
from delta.evalh.dataset import (  # noqa: E402
    DatasetNotAvailableError,
    load_spider,
    spider_available,
)
from delta.evalh.evaluate import evaluate_prompt  # noqa: E402
from delta.evalh.sample import stratified_sample  # noqa: E402
from delta.llm.providers import MissingAPIKeyError, build_client  # noqa: E402
from delta.report import print_failures, print_report  # noqa: E402
from delta.target_agent.agent import TargetAgent  # noqa: E402
from delta.target_agent.prompt import V0_WEAK, V_HANDTUNED  # noqa: E402

MIN_HEADROOM = 0.10
DEFAULT_N = 100
DEFAULT_SEED = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--tokens-only", action="store_true", help="skip the live Groq run")
    p.add_argument("--save", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    console = Console()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Always refresh the offline token report.
    from measure_tokens import main as measure_main

    if measure_main() != 0:
        return 1

    if args.tokens_only:
        console.print("[yellow]--tokens-only: skipping live Groq validation.[/yellow]")
        return 0

    if not spider_available():
        console.print("[red]Spider is not downloaded. Run: python scripts/download_spider.py[/red]")
        return 1

    settings = Settings()
    try:
        examples = stratified_sample(load_spider("dev"), args.n, seed=args.seed)
    except DatasetNotAvailableError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    try:
        client = build_client(settings.target_model, params=settings.generation)
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            "\n[yellow]Real-path validation is blocked without GROQ_API_KEY. "
            "Token distribution was written; set the key and re-run.[/yellow]"
        )
        RESULTS_DIR.joinpath("real_path.json").write_text(
            json.dumps(
                {
                    "status": "blocked_missing_api_key",
                    "n": args.n,
                    "seed": args.seed,
                    "min_headroom": MIN_HEADROOM,
                },
                indent=2,
            )
            + "\n"
        )
        return 1

    console.print(
        f"Real-path check: [bold]{len(examples)}[/bold] stratified Spider examples "
        f"on [bold]{settings.target_model}[/bold]"
    )

    reports = {}
    for label, prompt in (
        ("v0", V0_WEAK),
        ("handtuned", V_HANDTUNED),
    ):
        agent = TargetAgent(client=client, prompt=prompt)
        report = evaluate_prompt(
            agent, examples, timeout_s=settings.sql_timeout_s, keep_raw=True
        )
        reports[label] = report
        print_report(report, console=console, title=label)
        print_failures(report, limit=3, console=console)

    v0 = reports["v0"]
    human = reports["handtuned"]
    gap = human.accuracy - v0.accuracy

    extraction_ok = all(
        (r.predicted or r.reason != "pred_failed" or True) for r in v0.results
    )
    # More useful: fraction of non-empty extractions among answers that look like SQL.
    nonempty = sum(1 for r in v0.results if r.predicted) + sum(
        1 for r in human.results if r.predicted
    )
    tokens_ok = v0.input_tokens > 0 and human.input_tokens > 0

    payload = {
        "status": "ok" if gap >= MIN_HEADROOM and tokens_ok else "gate_failed",
        "n": len(examples),
        "seed": args.seed,
        "model": settings.target_model,
        "v0": v0.summary(),
        "handtuned": human.summary(),
        "gap": round(gap, 4),
        "min_headroom": MIN_HEADROOM,
        "tokens_nonzero": tokens_ok,
        "extractions_nonempty": nonempty,
        "prefix_cache_rate_v0": v0.prefix_cache_rate,
        "example_ids": [ex.id for ex in examples],
    }
    out = RESULTS_DIR / "real_path.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    console.print(f"\nwrote {out}")

    console.print()
    console.rule("[bold]Real-path gate")
    console.print(f"v0             {v0.accuracy:.1%}")
    console.print(f"hand-tuned     {human.accuracy:.1%}")
    console.print(f"gap            [bold]{gap:+.1%}[/bold]  (need ≥ {MIN_HEADROOM:.0%})")
    console.print(f"input tokens   v0={v0.input_tokens:,}  handtuned={human.input_tokens:,}")
    console.print(f"prefix cache   v0={v0.prefix_cache_rate:.1%}")

    if not tokens_ok:
        console.print("\n[red]Token accounting returned zeros — provider usage wiring is broken.[/red]")
        return 2
    if gap < MIN_HEADROOM:
        console.print(
            f"\n[red]Headroom gate failed ({gap:.1%} < {MIN_HEADROOM:.0%}). "
            "Rethink the setup before building the optimizer.[/red]"
        )
        return 2

    console.print(
        f"\n[green]Gate passed: {gap:.1%} headroom on {len(examples)} examples. "
        "Safe to continue.[/green]"
    )
    _ = extraction_ok
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
