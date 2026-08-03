#!/usr/bin/env python3
"""Draw the seeded three-way split and write it to data/splits.json.

Run once. The output is committed so that every later result refers to the same
examples, and so that no phase can quietly redraw the split after seeing a
number it did not like.

    python scripts/make_splits.py
    python scripts/make_splits.py --show      # inspect without writing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from delta.config import SPLITS_PATH  # noqa: E402
from delta.evalh.buckets import DIFFICULTIES  # noqa: E402
from delta.evalh.dataset import DatasetNotAvailableError  # noqa: E402
from delta.evalh.splits import (  # noqa: E402
    DEFAULT_N_SCREEN,
    DEFAULT_N_TEST,
    DEFAULT_N_TRAIN,
    DEFAULT_N_VAL,
    DEFAULT_SEED,
    build_splits,
)


def print_splits(splits, console: Console) -> None:
    table = Table(title=f"Three-way split (seed {splits.seed})")
    table.add_column("split")
    table.add_column("n", justify="right")
    for difficulty in DIFFICULTIES:
        table.add_column(difficulty, justify="right")
    table.add_column("source")

    sources = {
        "train": "Spider train",
        "val": f"{len(splits.val_databases)} dev databases",
        "val_screen": "subset of val",
        "test": f"{len(splits.test_databases)} dev databases",
    }
    for name in ("train", "val", "val_screen", "test"):
        ids = getattr(splits, name)
        counts = splits.difficulty.get(name, {})
        row = [name, str(len(ids))]
        row += [
            f"{counts.get(d, 0)} ({counts.get(d, 0) / len(ids):.0%})" for d in DIFFICULTIES
        ]
        row.append(sources[name])
        table.add_row(*row)
    console.print(table)

    console.print(f"\n[bold]validation databases[/bold] ({len(splits.val_databases)})")
    console.print("  " + ", ".join(splits.val_databases))
    console.print(f"[bold]test databases[/bold] ({len(splits.test_databases)})")
    console.print("  " + ", ".join(splits.test_databases))

    overlap = set(splits.val_databases) & set(splits.test_databases)
    if overlap:
        console.print(f"\n[red]databases appear in both splits: {sorted(overlap)}[/red]")
    else:
        console.print(
            "\n[green]No database appears in both validation and test, so a prompt "
            "cannot be tuned to a schema it is later scored on.[/green]"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--val", type=int, default=DEFAULT_N_VAL)
    parser.add_argument("--test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--screen", type=int, default=DEFAULT_N_SCREEN)
    parser.add_argument("--show", action="store_true", help="print without writing")
    args = parser.parse_args()

    console = Console()
    try:
        splits = build_splits(
            seed=args.seed,
            n_train=args.train,
            n_val=args.val,
            n_test=args.test,
            n_screen=args.screen,
        )
    except DatasetNotAvailableError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    print_splits(splits, console)

    if args.show:
        console.print("\n[yellow]--show given, nothing written.[/yellow]")
        return 0

    if SPLITS_PATH.exists():
        console.print(
            f"\n[yellow]{SPLITS_PATH} already exists. Refusing to overwrite: "
            "redrawing a split after seeing results invalidates them. Delete it "
            "deliberately if you really mean to.[/yellow]"
        )
        return 1

    splits.save(SPLITS_PATH)
    console.print(f"\nwrote {SPLITS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
