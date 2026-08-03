"""Tests for the decoupled acceptance gate."""

from __future__ import annotations

from delta.evalh.evaluate import EvalReport, ExampleResult
from delta.optimizer.gate import (
    ScreeningConfig,
    permissive_accept,
    screen_candidates,
)


def _report(correctness: dict[str, bool], difficulties: dict[str, str] | None = None) -> EvalReport:
    difficulties = difficulties or {eid: "easy" for eid in correctness}
    report = EvalReport(prompt_version_id="x", prompt_origin="test", model_id="mock")
    for eid, ok in correctness.items():
        report.results.append(
            ExampleResult(
                example_id=eid,
                question="q",
                difficulty=difficulties[eid],
                gold="SELECT 1",
                predicted="SELECT 1",
                correct=ok,
                reason="correct" if ok else "wrong_rows",
            )
        )
    return report


class TestPermissiveAccept:
    def test_admits_point_improvement(self):
        base = _report({f"e{i}": i < 5 for i in range(10)})
        cand = _report({f"e{i}": True for i in range(10)})
        decision = permissive_accept(base, cand, record_strict=False)
        assert decision.admit
        assert decision.reason == "admit"
        assert decision.delta > 0

    def test_rejects_non_positive_delta(self):
        base = _report({f"e{i}": True for i in range(10)})
        cand = _report({f"e{i}": i < 8 for i in range(10)})
        decision = permissive_accept(base, cand, record_strict=False)
        assert not decision.admit
        assert decision.reason == "non_positive_delta"

    def test_rejects_bucket_regression(self):
        # Overall can rise while a bucket falls: fix many easy, break some hard.
        difficulties = {f"e{i}": ("easy" if i < 5 else "hard") for i in range(10)}
        base_c = {f"e{i}": (i >= 5) for i in range(10)}  # easy 0/5, hard 5/5 → 50%
        cand_c = {f"e{i}": (i < 8) for i in range(10)}  # easy 5/5, hard 3/5 → 80%
        decision = permissive_accept(
            _report(base_c, difficulties),
            _report(cand_c, difficulties),
            bucket_tolerance=0.05,
            record_strict=False,
        )
        assert decision.delta > 0
        assert not decision.admit
        assert decision.reason.startswith("bucket_regression")

    def test_records_strict_counterfactual(self):
        base = _report({f"e{i}": False for i in range(30)})
        cand = _report({f"e{i}": True for i in range(30)})
        decision = permissive_accept(base, cand, record_strict=True)
        assert decision.admit
        assert decision.strict_counterfactual.get("admit") is True


class TestScreening:
    def test_promotes_top_third(self):
        scores = [(f"c{i}", float(i)) for i in range(9)]
        promoted = screen_candidates(scores, ScreeningConfig(promote_fraction=1 / 3))
        assert promoted == ["c8", "c7", "c6"]

    def test_empty(self):
        assert screen_candidates([]) == []
