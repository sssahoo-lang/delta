"""Tests for the evaluation harness.

The per-example correctness vector is the input to every statistical test in
Phase 2, so its integrity is checked directly: keyed by example id, complete, and
aligned across reports evaluated in different orders.
"""

from __future__ import annotations

from pathlib import Path

from delta.evalh.dataset import Example, load_sample
from delta.evalh.evaluate import evaluate_prompt, group_by_database
from delta.evalh.score import ScoreReason
from delta.llm.providers import build_client
from delta.target_agent.agent import TargetAgent
from delta.target_agent.prompt import V0_WEAK, V_HANDTUNED


def _agent(prompt=V0_WEAK):
    return TargetAgent(client=build_client("mock/deterministic"), prompt=prompt)


def _example(ex_id: str, db_id: str) -> Example:
    return Example(
        id=ex_id, question="q", gold="SELECT 1", db_id=db_id, db_path=Path("x.sqlite")
    )


def _switch_count(examples) -> int:
    """How many times the database changes while walking the list."""
    ids = [ex.db_id for ex in examples]
    return sum(1 for a, b in zip(ids, ids[1:], strict=False) if a != b)


class TestEvalReport:
    def test_scores_every_example(self):
        examples = load_sample()
        report = evaluate_prompt(_agent(), examples)
        assert report.n == len(examples)
        assert 0.0 <= report.accuracy <= 1.0
        assert report.n_correct == sum(1 for r in report.results if r.correct)

    def test_correctness_is_keyed_by_example_id(self):
        examples = load_sample()
        report = evaluate_prompt(_agent(), examples)
        assert set(report.correctness) == {ex.id for ex in examples}

    def test_correctness_aligns_across_evaluation_order(self):
        """Keying by id, not position, is what makes paired tests safe."""
        examples = load_sample()
        forward = evaluate_prompt(_agent(), examples)
        backward = evaluate_prompt(_agent(), list(reversed(examples)))
        assert forward.correctness == backward.correctness

    def test_difficulty_breakdown_sums_to_total(self):
        report = evaluate_prompt(_agent(), load_sample())
        buckets = report.by_difficulty
        assert sum(s.n for s in buckets.values()) == report.n
        assert sum(s.n_correct for s in buckets.values()) == report.n_correct

    def test_all_four_buckets_present(self):
        report = evaluate_prompt(_agent(), load_sample())
        assert set(report.by_difficulty) == {"easy", "medium", "hard", "extra"}

    def test_token_totals_accumulate(self):
        report = evaluate_prompt(_agent(), load_sample())
        assert report.input_tokens > 0
        assert report.output_tokens > 0

    def test_failures_excludes_broken_gold(self):
        report = evaluate_prompt(_agent(), load_sample())
        assert all(r.reason != ScoreReason.GOLD_FAILED for r in report.failures())

    def test_summary_is_json_safe(self):
        import json

        summary = evaluate_prompt(_agent(), load_sample()).summary()
        json.dumps(summary)  # must not raise
        assert summary["n"] == 15
        assert "by_difficulty" in summary

    def test_records_prompt_identity(self):
        report = evaluate_prompt(_agent(V_HANDTUNED), load_sample())
        assert report.prompt_version_id == V_HANDTUNED.version_id
        assert report.prompt_origin == "handtuned"

    def test_progress_callback_fires_once_per_example(self):
        seen = []
        examples = load_sample()
        evaluate_prompt(_agent(), examples, progress=lambda i, n, r: seen.append((i, n)))
        assert seen == [(i, len(examples)) for i in range(1, len(examples) + 1)]

    def test_empty_example_set_is_safe(self):
        report = evaluate_prompt(_agent(), [])
        assert report.n == 0
        assert report.accuracy == 0.0


class TestDatabaseGrouping:
    """Ordering by database is what lets Groq's prefix cache absorb the schema.

    It is a pure cost optimization, so the tests that matter most are the ones
    asserting it cannot change a measured result.
    """

    def test_collapses_database_switches(self):
        interleaved = [
            _example("a1", "alpha"),
            _example("b1", "beta"),
            _example("a2", "alpha"),
            _example("b2", "beta"),
            _example("a3", "alpha"),
        ]
        assert _switch_count(interleaved) == 4
        assert _switch_count(group_by_database(interleaved)) == 1

    def test_one_contiguous_run_per_database(self):
        grouped = group_by_database(
            [_example(f"e{i}", f"db{i % 4}") for i in range(40)]
        )
        assert _switch_count(grouped) == 3  # four databases, three transitions

    def test_preserves_every_example_exactly_once(self):
        examples = [_example(f"e{i}", f"db{i % 3}") for i in range(12)]
        grouped = group_by_database(examples)
        assert sorted(e.id for e in grouped) == sorted(e.id for e in examples)

    def test_is_deterministic(self):
        examples = [_example(f"e{i}", f"db{i % 5}") for i in range(25)]
        assert [e.id for e in group_by_database(examples)] == [
            e.id for e in group_by_database(examples)
        ]

    def test_preserves_relative_order_within_a_database(self):
        examples = [_example(f"e{i}", f"db{i % 3}") for i in range(12)]
        grouped = group_by_database(examples)
        for db in {e.db_id for e in examples}:
            assert [e.id for e in grouped if e.db_id == db] == [
                e.id for e in examples if e.db_id == db
            ]

    def test_does_not_change_measured_accuracy(self):
        """The whole safety argument for reordering, asserted directly."""
        examples = load_sample()
        grouped = evaluate_prompt(_agent(), examples, group_databases=True)
        ungrouped = evaluate_prompt(_agent(), examples, group_databases=False)
        assert grouped.correctness == ungrouped.correctness
        assert grouped.accuracy == ungrouped.accuracy

    def test_scores_every_example_when_grouped(self):
        examples = load_sample()
        report = evaluate_prompt(_agent(), examples, group_databases=True)
        assert set(report.correctness) == {ex.id for ex in examples}


class TestTokenAccounting:
    def test_billable_excludes_provider_cached_tokens(self):
        from delta.llm.providers import LLMResponse

        response = LLMResponse(
            text="", model_id="m", input_tokens=500, output_tokens=60,
            cache_read_tokens=400,
        )
        assert response.billable_tokens == 160

    def test_prefix_cache_rate_is_zero_without_provider_caching(self):
        report = evaluate_prompt(_agent(), load_sample())
        # The mock reports no provider-side caching, so this must not divide by
        # zero or invent a rate.
        assert report.prefix_cache_rate == 0.0
        assert report.billable_tokens == report.input_tokens + report.output_tokens


class TestDataset:
    def test_sample_examples_are_complete(self):
        for ex in load_sample():
            assert ex.id and ex.question and ex.gold
            assert ex.db_path.exists()
            assert ex.difficulty in {"easy", "medium", "hard", "extra"}

    def test_with_difficulty_returns_a_new_example(self):
        ex = load_sample()[0]
        relabeled = ex.with_difficulty("extra")
        assert relabeled.difficulty == "extra"
        assert ex.difficulty != "extra" or ex.difficulty == "extra"
        assert relabeled.id == ex.id
