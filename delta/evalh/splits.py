"""Deterministic three-way split for the optimization experiment.

Train comes from Spider train (databases are disjoint from dev by construction).
Validation and test split Spider **dev by database**, not by example, so a
candidate prompt cannot be tuned to a specific held-out schema.

A small ``val_screen`` subset of validation is carved out for the cheap screening
stage before a candidate is promoted to the full validation set.

The assignment is seeded and re-drawn until difficulty distributions on val and
test are comparable. The resulting id lists are written to ``data/splits.json``
and committed so every later run uses the same cut.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from delta.config import SPLITS_PATH
from delta.evalh.buckets import DIFFICULTIES, difficulty_histogram
from delta.evalh.dataset import Example, load_spider
from delta.evalh.sample import stratified_sample

DEFAULT_SEED = 42
DEFAULT_N_TRAIN = 100
DEFAULT_N_VAL = 350
DEFAULT_N_TEST = 400
DEFAULT_N_SCREEN = 60
# Max total-variation distance between val and test difficulty distributions.
MAX_DIFFICULTY_TV = 0.08
MAX_REDRAW_ATTEMPTS = 2_000


@dataclass(frozen=True)
class SplitIds:
    train: list[str]
    val: list[str]
    val_screen: list[str]
    test: list[str]
    seed: int
    val_databases: list[str]
    test_databases: list[str]
    difficulty: dict[str, dict[str, int]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "train": self.train,
            "val": self.val,
            "val_screen": self.val_screen,
            "test": self.test,
            "val_databases": self.val_databases,
            "test_databases": self.test_databases,
            "difficulty": self.difficulty,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(payload: dict) -> SplitIds:
        return SplitIds(
            train=list(payload["train"]),
            val=list(payload["val"]),
            val_screen=list(payload.get("val_screen", [])),
            test=list(payload["test"]),
            seed=int(payload.get("seed", DEFAULT_SEED)),
            val_databases=list(payload.get("val_databases", [])),
            test_databases=list(payload.get("test_databases", [])),
            difficulty=dict(payload.get("difficulty", {})),
            meta=dict(payload.get("meta", {})),
        )

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else SPLITS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @staticmethod
    def load(path: str | Path | None = None) -> SplitIds:
        path = Path(path) if path else SPLITS_PATH
        return SplitIds.from_dict(json.loads(path.read_text()))


def _difficulty_dist(examples: list[Example]) -> dict[str, float]:
    counts = Counter(ex.difficulty for ex in examples)
    n = len(examples) or 1
    return {k: counts.get(k, 0) / n for k in DIFFICULTIES}


def _total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _db_groups(examples: list[Example]) -> dict[str, list[Example]]:
    groups: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        groups[ex.db_id].append(ex)
    return dict(groups)


def _pick_db_partition(
    groups: dict[str, list[Example]],
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[list[str], list[str], float]:
    """Assign whole databases to val/test; leftover databases stay unused.

    Not every example has to be used. Leaving surplus databases out is how we
    hit the ~350 / ~400 targets despite coarse database sizes.
    """
    rng = random.Random(seed)
    db_ids = sorted(groups)
    best: tuple[list[str], list[str], float, float] | None = None

    for _ in range(MAX_REDRAW_ATTEMPTS):
        order = db_ids[:]
        rng.shuffle(order)
        val_dbs: list[str] = []
        test_dbs: list[str] = []
        val_n = 0
        test_n = 0
        for db_id in order:
            n = len(groups[db_id])
            # Stop filling a side once it would overshoot its target by more than
            # this database's size; leftover databases remain unused.
            val_room = n_val - val_n
            test_room = n_test - test_n
            if val_room >= test_room and val_room >= n // 2:
                val_dbs.append(db_id)
                val_n += n
            elif test_room >= n // 2:
                test_dbs.append(db_id)
                test_n += n
            # else: leave unused

        if not val_dbs or not test_dbs:
            continue

        val_ex = [ex for d in val_dbs for ex in groups[d]]
        test_ex = [ex for d in test_dbs for ex in groups[d]]
        tv = _total_variation(_difficulty_dist(val_ex), _difficulty_dist(test_ex))
        size_penalty = abs(len(val_ex) - n_val) + abs(len(test_ex) - n_test)
        score = tv + size_penalty / 500.0

        if best is None or score < best[3]:
            best = (sorted(val_dbs), sorted(test_dbs), tv, score)
            if tv <= MAX_DIFFICULTY_TV and size_penalty <= 100:
                return best[0], best[1], best[2]

    if best is None:
        raise RuntimeError("could not find a val/test database partition")
    return best[0], best[1], best[2]


def build_splits(
    seed: int = DEFAULT_SEED,
    n_train: int = DEFAULT_N_TRAIN,
    n_val: int = DEFAULT_N_VAL,
    n_test: int = DEFAULT_N_TEST,
    n_screen: int = DEFAULT_N_SCREEN,
) -> SplitIds:
    """Build the train / val / val_screen / test id lists."""
    train_raw = load_spider("train")
    dev_raw = load_spider("dev")

    train = stratified_sample(train_raw, n_train, seed=seed, by="difficulty")

    groups = _db_groups(dev_raw)
    val_dbs, test_dbs, tv = _pick_db_partition(groups, n_val, n_test, seed)
    val_ex = [ex for d in val_dbs for ex in groups[d]]
    test_ex = [ex for d in test_dbs for ex in groups[d]]

    screen = stratified_sample(val_ex, min(n_screen, len(val_ex)), seed=seed + 1)

    difficulty = {
        "train": difficulty_histogram([ex.difficulty for ex in train]),
        "val": difficulty_histogram([ex.difficulty for ex in val_ex]),
        "val_screen": difficulty_histogram([ex.difficulty for ex in screen]),
        "test": difficulty_histogram([ex.difficulty for ex in test_ex]),
    }
    meta = {
        "train_n": len(train),
        "val_n": len(val_ex),
        "val_screen_n": len(screen),
        "test_n": len(test_ex),
        "difficulty_tv": round(tv, 4),
        "max_difficulty_tv": MAX_DIFFICULTY_TV,
    }
    return SplitIds(
        train=[ex.id for ex in train],
        val=[ex.id for ex in val_ex],
        val_screen=[ex.id for ex in screen],
        test=[ex.id for ex in test_ex],
        seed=seed,
        val_databases=val_dbs,
        test_databases=test_dbs,
        difficulty=difficulty,
        meta=meta,
    )


def resolve_split(
    splits: SplitIds,
    which: str,
    examples: list[Example] | None = None,
) -> list[Example]:
    """Materialize a named split (``train`` / ``val`` / ``val_screen`` / ``test``)."""
    ids = getattr(splits, which)
    if examples is None:
        source = "train" if which == "train" else "dev"
        examples = load_spider(source)
    by_id = {ex.id: ex for ex in examples}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} ids from splits.json not found in dataset")
    return [by_id[i] for i in ids]


def ensure_splits(path: str | Path | None = None, force: bool = False) -> SplitIds:
    """Load committed splits, or build and save them if absent."""
    path = Path(path) if path else SPLITS_PATH
    if path.exists() and not force:
        return SplitIds.load(path)
    split = build_splits()
    split.save(path)
    return split
