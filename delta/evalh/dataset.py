"""Loading benchmark examples, from either the offline fixture or Spider.

Both sources normalize to the same :class:`Example` shape so every downstream
component (evaluator, optimizer, acceptance gate, baselines) is agnostic to which
one it is looking at. That is what lets the whole pipeline be developed and tested
against the 15-question fixture and then pointed at Spider's 1,034 dev examples
without touching anything but the loader call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from delta.config import SAMPLE_DB, SAMPLE_QUESTIONS, SPIDER_DIR
from delta.evalh.buckets import hardness_from_parsed, hardness_from_sql

UNKNOWN_DIFFICULTY = "unknown"


@dataclass(frozen=True)
class Example:
    """One benchmark question with its gold answer and the database to run against."""

    id: str
    question: str
    gold: str
    db_id: str
    db_path: Path
    difficulty: str = UNKNOWN_DIFFICULTY

    def with_difficulty(self, difficulty: str) -> Example:
        return Example(
            id=self.id,
            question=self.question,
            gold=self.gold,
            db_id=self.db_id,
            db_path=self.db_path,
            difficulty=difficulty,
        )


class DatasetNotAvailableError(FileNotFoundError):
    """Raised with actionable guidance when benchmark files are absent."""


def load_sample() -> list[Example]:
    """The offline fixture. Requires no download and no API key."""
    if not SAMPLE_QUESTIONS.exists() or not SAMPLE_DB.exists():
        raise DatasetNotAvailableError(
            "Sample dataset not built. Run: python scripts/make_sample_db.py"
        )

    payload = json.loads(SAMPLE_QUESTIONS.read_text())
    return [
        Example(
            id=row["id"],
            question=row["question"],
            gold=row["gold"],
            db_id=row.get("db_id", "sample"),
            db_path=SAMPLE_DB,
            difficulty=row.get("difficulty", UNKNOWN_DIFFICULTY),
        )
        for row in payload
    ]


def spider_available() -> bool:
    return (SPIDER_DIR / "dev.json").exists()


def load_spider(split: str = "dev") -> list[Example]:
    """Spider examples for ``split`` ("dev" or "train").

    Difficulty comes from the ``sql`` field Spider ships alongside each question,
    which is the same pre-parsed structure the official evaluator consumes, so
    the labels here are the official ones rather than an approximation.
    """
    filename = {"dev": "dev.json", "train": "train_spider.json"}.get(split)
    if filename is None:
        raise ValueError(f"unknown Spider split: {split!r}")

    path = SPIDER_DIR / filename
    if not path.exists():
        raise DatasetNotAvailableError(
            f"Spider not downloaded ({path} missing). "
            "Run: python scripts/download_spider.py"
        )

    rows = json.loads(path.read_text())
    examples: list[Example] = []
    for i, row in enumerate(rows):
        db_id = row["db_id"]
        db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            # A handful of Spider databases are known to be absent from some
            # mirrors. Skipping is correct; silently scoring them wrong is not.
            continue
        parsed = row.get("sql")
        difficulty = (
            hardness_from_parsed(parsed)
            if isinstance(parsed, dict)
            else hardness_from_sql(row["query"])
        )
        examples.append(
            Example(
                id=f"{split}-{i:05d}",
                question=row["question"],
                gold=row["query"],
                db_id=db_id,
                db_path=db_path,
                difficulty=difficulty,
            )
        )
    return examples


def group_by_difficulty(examples: list[Example]) -> dict[str, list[Example]]:
    out: dict[str, list[Example]] = {}
    for ex in examples:
        out.setdefault(ex.difficulty, []).append(ex)
    return out
