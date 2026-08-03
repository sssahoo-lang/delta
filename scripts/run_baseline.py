#!/usr/bin/env python3
"""Measure the starting accuracy of the v0 prompt.

This produces the number every later phase is compared against. It also acts as
the project's smoke test: if this runs clean, then the provider layer, prompt
rendering, SQL extraction, execution sandbox, and scorer are all wired correctly.

    python scripts/run_baseline.py --mock          # offline, no API key
    python scripts/run_baseline.py                 # real model
    python scripts/run_baseline.py --compare       # v0 against the human baseline

The ``--compare`` mode is worth running early. If the deliberately weak v0 prompt
does not score meaningfully below the hand-written one, then the setup has no
headroom and there is nothing for an optimizer to find, which would need fixing
before building anything on top.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from delta.config import MOCK_MODEL, RUNS_DIR, Settings  # noqa: E402
from delta.evalh.dataset import (  # noqa: E402
    DatasetNotAvailableError,
    load_sample,
    load_spider,
    spider_available,
)
from delta.evalh.evaluate import evaluate_prompt  # noqa: E402
from delta.llm.providers import MissingAPIKeyError, build_client  # noqa: E402
from delta.report import print_failures, print_report  # noqa: E402
from delta.target_agent.agent import TargetAgent  # noqa: E402
from delta.target_agent.prompt import V0_WEAK, V_HANDTUNED  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mock", action="store_true", help="use the offline deterministic model")
    p.add_argument(
        "--dataset",
        choices=["sample", "spider"],
        default="sample",
        help="offline fixture (default) or Spider dev",
    )
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    p.add_argument(
        "--compare",
        action="store_true",
        help="also score the hand-tuned prompt, to confirm headroom exists",
    )
    p.add_argument("--show-failures", type=int, default=5, help="how many failures to print")
    p.add_argument("--save", action="store_true", help="write the report to runs/")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    console = Console()
    settings = Settings()

    try:
        examples = load_spider("dev") if args.dataset == "spider" else load_sample()
    except DatasetNotAvailableError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    if args.dataset == "spider" and not spider_available():
        console.print("[red]Spider is not downloaded. Run: python scripts/download_spider.py[/red]")
        return 1

    if args.limit:
        examples = examples[: args.limit]

    model_id = MOCK_MODEL if args.mock else settings.target_model
    try:
        client = build_client(model_id, params=settings.generation)
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(
        f"Scoring [bold]{len(examples)}[/bold] examples from "
        f"[bold]{args.dataset}[/bold] with [bold]{model_id}[/bold]"
    )

    conditions = [("v0 (weak baseline)", V0_WEAK)]
    if args.compare:
        conditions.append(("hand-tuned (human baseline)", V_HANDTUNED))

    reports = {}
    for label, prompt in conditions:
        agent = TargetAgent(client=client, prompt=prompt)
        report = evaluate_prompt(agent, examples, timeout_s=settings.sql_timeout_s)
        reports[label] = report
        print_report(report, console=console, title=label)
        if args.show_failures:
            print_failures(report, limit=args.show_failures, console=console)

    if args.compare:
        baseline = reports["v0 (weak baseline)"]
        human = reports["hand-tuned (human baseline)"]
        gap = human.accuracy - baseline.accuracy
        console.print()
        console.rule("[bold]Headroom check")
        console.print(f"v0             {baseline.accuracy:.1%}")
        console.print(f"hand-tuned     {human.accuracy:.1%}")
        console.print(f"gap            [bold]{gap:+.1%}[/bold]")
        if gap <= 0.05:
            console.print(
                "\n[yellow]Warning: the gap is too small to optimize into. "
                "A better prompt is barely beating a weak one, so an optimizer "
                "would have almost nothing to discover.[/yellow]"
            )
        else:
            console.print(
                f"\n[green]Headroom confirmed: {gap:.1%} is available to a prompt "
                f"optimizer on this setup.[/green]"
            )

    if args.save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / f"baseline_{args.dataset}_{model_id.replace('/', '-')}.json"
        out.write_text(
            json.dumps(
                {label: r.summary() for label, r in reports.items()},
                indent=2,
            )
            + "\n"
        )
        console.print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
