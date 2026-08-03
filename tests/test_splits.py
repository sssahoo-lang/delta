"""Tests for the seeded three-way split."""

from __future__ import annotations

import pytest

from delta.evalh.dataset import load_spider, spider_available
from delta.evalh.sample import stratified_sample
from delta.evalh.splits import build_splits

requires_spider = pytest.mark.skipif(not spider_available(), reason="Spider not downloaded")


@requires_spider
class TestBuildSplits:
    def test_no_database_overlap_between_val_and_test(self):
        splits = build_splits(seed=42)
        assert set(splits.val_databases).isdisjoint(splits.test_databases)

    def test_sizes_near_targets(self):
        splits = build_splits(seed=42)
        assert 80 <= len(splits.train) <= 100
        assert 280 <= len(splits.val) <= 450
        assert 300 <= len(splits.test) <= 500
        assert len(splits.val_screen) == 60
        assert set(splits.val_screen).issubset(set(splits.val))

    def test_deterministic(self):
        a = build_splits(seed=7)
        b = build_splits(seed=7)
        assert a.to_dict() == b.to_dict()

    def test_train_ids_come_from_train_split(self):
        splits = build_splits(seed=42)
        train_ids = {ex.id for ex in load_spider("train")}
        assert set(splits.train).issubset(train_ids)

    def test_difficulty_tv_under_threshold(self):
        splits = build_splits(seed=42)
        assert splits.meta["difficulty_tv"] <= 0.08


class TestStratifiedSample:
    def test_balances_buckets(self):
        from pathlib import Path

        from delta.evalh.dataset import Example

        examples = [
            Example(
                id=f"{d}-{i}",
                question="q",
                gold="SELECT 1",
                db_id="db",
                db_path=Path("x"),
                difficulty=d,
            )
            for d in ("easy", "medium", "hard", "extra")
            for i in range(10)
        ]
        sample = stratified_sample(examples, 8, seed=0)
        from collections import Counter

        counts = Counter(ex.difficulty for ex in sample)
        assert counts == {"easy": 2, "medium": 2, "hard": 2, "extra": 2}
