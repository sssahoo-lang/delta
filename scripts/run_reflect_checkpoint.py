#!/usr/bin/env python3
"""Phase 3 quality checkpoint: diagnose failures, propose a new prompt, re-score.

In simple words: take the weak prompt, see where it fails, ask the analyzer what
is wrong, ask the proposer for a fix, then measure whether accuracy went up.

    python scripts/run_reflect_checkpoint.py --mock          # offline
    python scripts/run_reflect_checkpoint.py                 # needs Gemini key
    python scripts/run_reflect_checkpoint.py --mock --rounds 3

This is a *manual* checkpoint, not the full optimization loop. Look at the
printed diagnosis and proposed prompt with your own eyes before Phase 4 wires
them into an automatic search.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from delta.config import MOCK_MODEL, RUNS_DIR, Settings  # noqa: E402
from delta.evalh.dataset import (  # noqa: E402
    DatasetNotAvailableError,
    load_sample,
    load_spider,
    spider_available,
)
from delta.evalh.evaluate import evaluate_prompt  # noqa: E402
from delta.evalh.sample import stratified_sample  # noqa: E402
from delta.llm.providers import MissingAPIKeyError, build_client  # noqa: E402
from delta.optimizer.analyzer import Analyzer  # noqa: E402
from delta.optimizer.proposer import Proposer  # noqa: E402
from delta.report import print_report  # noqa: E402
from delta.target_agent.agent import TargetAgent  # noqa: E402
from delta.target_agent.prompt import V0_WEAK, PromptVersion  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mock", action="store_true", help="offline deterministic reflection")
    p.add_argument("--dataset", choices=["sample", "spider"], default="sample")
    p.add_argument("--limit", type=int, default=None, help="stratified sample size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=1, help="analyze/propose/score cycles")
    p.add_argument("--save", action="store_true")
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
        console.print("[red]Spider not downloaded.[/red]")
        return 1
    if args.limit:
        examples = stratified_sample(examples, args.limit, seed=args.seed)

    # Target stays on mock or Groq; reflection uses mock or Gemini.
    target_model = MOCK_MODEL if args.mock else settings.target_model
    reflection_model = MOCK_MODEL if args.mock else settings.reflection_model
    try:
        target_client = build_client(target_model, params=settings.generation)
        reflect_client = build_client(reflection_model, params=settings.generation)
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            "\n[yellow]Tip: use --mock to practice the checkpoint with no keys, "
            "or set GROQ_API_KEY / GEMINI_API_KEY in .env[/yellow]"
        )
        return 1

    analyzer = Analyzer(reflect_client)
    proposer = Proposer(reflect_client)
    prompt: PromptVersion = V0_WEAK
    history = []

    console.print(
        f"Reflect checkpoint on [bold]{len(examples)}[/bold] {args.dataset} examples "
        f"(target={target_model}, reflection={reflection_model}, rounds={args.rounds})"
    )

    for round_i in range(1, args.rounds + 1):
        console.rule(f"[bold]Round {round_i}")
        agent = TargetAgent(client=target_client, prompt=prompt)
        report = evaluate_prompt(agent, examples, timeout_s=settings.sql_timeout_s)
        print_report(report, console=console, title=f"prompt {prompt.version_id}")

        diagnosis = analyzer.diagnose(prompt, report)
        console.print()
        console.print(
            Panel(
                f"[bold]summary[/bold]\n{diagnosis.summary}\n\n"
                f"[bold]failure modes[/bold]\n"
                + "\n".join(f"• {m}" for m in diagnosis.failure_modes)
                + "\n\n[bold]recommendations[/bold]\n"
                + "\n".join(f"• {r}" for r in diagnosis.recommendations),
                title="Diagnosis — read this carefully",
            )
        )

        candidate = proposer.propose(prompt, diagnosis)
        console.print()
        console.print(
            Panel(
                candidate.text,
                title=f"Proposed prompt {candidate.version_id} ({candidate.char_count} chars)",
            )
        )

        improved = TargetAgent(client=target_client, prompt=candidate)
        after = evaluate_prompt(improved, examples, timeout_s=settings.sql_timeout_s)
        print_report(after, console=console, title=f"re-score {candidate.version_id}")
        delta = after.accuracy - report.accuracy
        console.print(f"\nRound delta: [bold]{delta:+.1%}[/bold]")

        history.append(
            {
                "round": round_i,
                "parent_id": prompt.version_id,
                "candidate_id": candidate.version_id,
                "before": report.summary(),
                "after": after.summary(),
                "delta": round(delta, 4),
                "diagnosis": diagnosis.to_dict(),
                "candidate_prompt": candidate.to_dict(),
            }
        )
        prompt = candidate

    if args.save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out = RUNS_DIR / f"reflect_{args.dataset}_{'mock' if args.mock else 'live'}.json"
        out.write_text(json.dumps(history, indent=2) + "\n")
        console.print(f"\nwrote {out}")

    console.print(
        "\n[green]Checkpoint done.[/green] If the diagnosis and proposal look sensible "
        "to you, Phase 4 can wrap them in the automatic loop. If they look like keyword "
        "stuffing (especially on --mock), that is expected — judge quality on real traces."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
