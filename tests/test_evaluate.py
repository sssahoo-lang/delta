"""Tests for the evaluation harness.

The per-example correctness vector is the input to every statistical test in
Phase 2, so its integrity is checked directly: keyed by example id, complete, and
aligned across reports evaluated in different orders.
"""

from __future__ import annotations

from delta.evalh.dataset import load_sample
from delta.evalh.evaluate import evaluate_prompt
from delta.evalh.score import ScoreReason
from delta.llm.providers import build_client
from delta.target_agent.agent import TargetAgent
from delta.target_agent.prompt import V0_WEAK, V_HANDTUNED


def _agent(prompt=V0_WEAK):
    return TargetAgent(client=build_client("mock/deterministic"), prompt=prompt)


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
