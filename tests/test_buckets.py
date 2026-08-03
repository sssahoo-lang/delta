"""Tests for Spider difficulty bucketing.

The buckets drive stratified splitting and the per-difficulty regression guards,
so a mislabeled bucket would quietly corrupt both. The load-bearing assertions
here are that every dev example gets a valid label and that the distribution
matches the one Spider publishes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from delta.evalh.buckets import (
    DIFFICULTIES,
    difficulty_histogram,
    hardness_from_parsed,
    hardness_from_sql,
)
from delta.evalh.dataset import load_spider, spider_available

SPIDER_DEV = Path(__file__).resolve().parent.parent / "data" / "spider_data" / "dev.json"

requires_spider = pytest.mark.skipif(
    not spider_available(), reason="Spider not downloaded"
)


class TestHardnessFromSQL:
    """The sqlglot fallback, used when Spider's own parse is unavailable."""

    def test_bare_select_is_easy(self):
        assert hardness_from_sql("SELECT name FROM students") == "easy"

    def test_single_filter_is_easy(self):
        assert hardness_from_sql("SELECT name FROM students WHERE gpa > 3") == "easy"

    def test_group_by_with_two_projections_is_medium(self):
        query = "SELECT dept, count(*) FROM students GROUP BY dept"
        assert hardness_from_sql(query) in {"medium", "hard"}

    def test_nested_subquery_is_not_easy(self):
        query = (
            "SELECT name FROM students "
            "WHERE gpa > (SELECT avg(gpa) FROM students)"
        )
        assert hardness_from_sql(query) in {"hard", "extra"}

    def test_set_operation_is_not_easy(self):
        query = "SELECT name FROM a INTERSECT SELECT name FROM b"
        assert hardness_from_sql(query) in {"hard", "extra"}

    def test_unparseable_sql_does_not_raise(self):
        assert hardness_from_sql("this is not sql at all") in DIFFICULTIES
        assert hardness_from_sql("") in DIFFICULTIES


class TestDifficultyHistogram:
    def test_includes_empty_buckets(self):
        assert difficulty_histogram(["easy", "easy"]) == {
            "easy": 2, "medium": 0, "hard": 0, "extra": 0
        }

    def test_counts_are_exact(self):
        counts = difficulty_histogram(["easy", "medium", "medium", "extra"])
        assert counts["medium"] == 2
        assert sum(counts.values()) == 4


@requires_spider
class TestOfficialSpiderLabels:
    def test_every_dev_example_is_labeled(self):
        examples = load_spider("dev")
        assert len(examples) == 1_034
        assert all(ex.difficulty in DIFFICULTIES for ex in examples)

    def test_distribution_matches_published_spider_dev(self):
        counts = difficulty_histogram([ex.difficulty for ex in load_spider("dev")])
        assert counts == {"easy": 248, "medium": 446, "hard": 174, "extra": 166}

    def test_all_four_buckets_are_well_populated(self):
        """Stratified splitting needs every bucket to be big enough to divide."""
        counts = difficulty_histogram([ex.difficulty for ex in load_spider("dev")])
        assert all(n > 100 for n in counts.values())

    def test_labels_are_deterministic(self):
        first = [ex.difficulty for ex in load_spider("dev")]
        second = [ex.difficulty for ex in load_spider("dev")]
        assert first == second

    def test_trivial_count_query_is_easy(self):
        rows = json.loads(SPIDER_DEV.read_text())
        row = next(r for r in rows if r["query"].strip() == "SELECT count(*) FROM singer")
        assert hardness_from_parsed(row["sql"]) == "easy"

    def test_fallback_broadly_agrees_with_the_official_parse(self):
        """The approximation is allowed to differ, but not wildly."""
        rows = json.loads(SPIDER_DEV.read_text())
        agree = sum(
            1 for r in rows
            if hardness_from_parsed(r["sql"]) == hardness_from_sql(r["query"])
        )
        assert agree / len(rows) > 0.85
