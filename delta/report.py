"""Shared console rendering for evaluation reports.

Kept in one place so the baseline script, the optimizer, and the final comparison
all present numbers the same way. A reader comparing two runs should never have to
work out whether a difference is real or just formatting.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from delta.evalh.evaluate import EvalReport

DIFFICULTY_ORDER = ["easy", "medium", "hard", "extra", "unknown"]


def difficulty_sort_key(name: str) -> tuple[int, str]:
    try:
        return (DIFFICULTY_ORDER.index(name), name)
    except ValueError:
        return (len(DIFFICULTY_ORDER), name)


def print_report(report: EvalReport, console: Console | None = None, title: str = "") -> None:
    console = console or Console()

    heading = title or f"prompt {report.prompt_version_id} ({report.prompt_origin})"
    console.print()
    console.rule(f"[bold]{heading}")

    console.print(f"model           {report.model_id}")
    console.print(
        f"accuracy        [bold]{report.accuracy:.1%}[/bold] "
        f"({report.n_correct}/{report.n})"
    )
    console.print(f"wall clock      {report.wall_clock_s:.1f}s")
    console.print(
        f"tokens          {report.input_tokens:,} in / {report.output_tokens:,} out"
    )
    if report.cache_hits:
        console.print(f"cache hits      {report.cache_hits}/{report.n}")

    buckets = report.by_difficulty
    if len(buckets) > 1:
        table = Table(title="By difficulty", title_justify="left", show_edge=False)
        table.add_column("difficulty")
        table.add_column("correct", justify="right")
        table.add_column("n", justify="right")
        table.add_column("accuracy", justify="right")
        for name in sorted(buckets, key=difficulty_sort_key):
            stats = buckets[name]
            table.add_row(name, str(stats.n_correct), str(stats.n), f"{stats.accuracy:.1%}")
        console.print()
        console.print(table)

    reasons = report.failure_reasons
    if reasons:
        console.print()
        console.print("[bold]Failure modes[/bold]")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            console.print(f"  {reason:<16} {count}")
        if report.extraction_failures:
            console.print(f"  (no SQL extracted: {report.extraction_failures})")


def print_failures(report: EvalReport, limit: int = 5, console: Console | None = None) -> None:
    """Show a few concrete failures, which is how prompt weaknesses become visible."""
    console = console or Console()
    failures = report.failures()
    if not failures:
        return

    console.print()
    console.print(f"[bold]Sample failures[/bold] (showing {min(limit, len(failures))} of {len(failures)})")
    for r in failures[:limit]:
        console.print()
        console.print(f"  [dim]{r.example_id} · {r.difficulty} · {r.reason}[/dim]")
        console.print(f"  Q     {r.question}")
        console.print(f"  gold  {r.gold}")
        console.print(f"  pred  {r.predicted or '(nothing extracted)'}")
