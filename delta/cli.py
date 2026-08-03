"""Command-line entry point.

Thin by design: each subcommand wires existing library pieces together and prints
a report. Anything with real logic lives in the library so it can be tested
without going through argument parsing.
"""

from __future__ import annotations

import typer
from rich.console import Console

from delta.config import MOCK_MODEL, Settings
from delta.evalh.dataset import (
    DatasetNotAvailableError,
    load_sample,
    load_spider,
    spider_available,
)
from delta.evalh.evaluate import evaluate_prompt
from delta.llm.providers import MissingAPIKeyError, build_client
from delta.report import print_failures, print_report
from delta.target_agent.agent import TargetAgent
from delta.target_agent.prompt import V0_WEAK, V_HANDTUNED

app = typer.Typer(
    add_completion=False,
    help="Delta: a self-improving text-to-SQL agent, measured honestly.",
)
console = Console()

PROMPTS = {"v0": V0_WEAK, "handtuned": V_HANDTUNED}


def _load(dataset: str, limit: int | None):
    examples = load_spider("dev") if dataset == "spider" else load_sample()
    return examples[:limit] if limit else examples


@app.command()
def diagnose(
    dataset: str = typer.Option("sample", help="'sample' or 'spider'"),
    prompt: str = typer.Option("v0", help=f"one of {', '.join(PROMPTS)}"),
    mock: bool = typer.Option(False, "--mock", help="use the offline model, no API key needed"),
    limit: int | None = typer.Option(None, help="evaluate only the first N examples"),
    show_failures: int = typer.Option(5, help="how many failures to print"),
) -> None:
    """Score a prompt and show where it fails."""
    settings = Settings()

    if prompt not in PROMPTS:
        console.print(f"[red]unknown prompt {prompt!r}. Choose from: {', '.join(PROMPTS)}[/red]")
        raise typer.Exit(1)

    if dataset == "spider" and not spider_available():
        console.print("[red]Spider is not downloaded. Run: python scripts/download_spider.py[/red]")
        raise typer.Exit(1)

    try:
        examples = _load(dataset, limit)
    except DatasetNotAvailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    try:
        client = build_client(
            MOCK_MODEL if mock else settings.target_model, params=settings.generation
        )
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    agent = TargetAgent(client=client, prompt=PROMPTS[prompt])
    report = evaluate_prompt(agent, examples, timeout_s=settings.sql_timeout_s)
    print_report(report, console=console, title=f"{prompt} on {dataset}")
    if show_failures:
        print_failures(report, limit=show_failures, console=console)


@app.command()
def info() -> None:
    """Show configuration and what is runnable right now."""
    from delta.config import SAMPLE_DB, api_key_for

    settings = Settings()
    console.print()
    console.rule("[bold]Delta")
    console.print(f"target model      {settings.target_model}")
    console.print(f"reflection model  {settings.reflection_model}")
    console.print(f"temperature       {settings.generation.temperature}")

    console.print()
    console.print("[bold]Availability[/bold]")
    for label, ok, hint in [
        ("sample dataset", SAMPLE_DB.exists(), "python scripts/make_sample_db.py"),
        ("spider dataset", spider_available(), "python scripts/download_spider.py"),
        ("groq key", bool(api_key_for(settings.target_model)), "set GROQ_API_KEY in .env"),
        ("gemini key", bool(api_key_for(settings.reflection_model)), "set GEMINI_API_KEY in .env"),
    ]:
        mark = "[green]yes[/green]" if ok else "[yellow]no[/yellow] "
        suffix = "" if ok else f"  ({hint})"
        console.print(f"  {label:<17} {mark}{suffix}")

    console.print()
    console.print("Everything runs offline with --mock; no key is required.")


if __name__ == "__main__":
    app()
