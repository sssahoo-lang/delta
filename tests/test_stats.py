"""Tests for paired bootstrap, McNemar, and Holm correction."""

from __future__ import annotations

import numpy as np

from delta.evalh.stats import (
    compare_all,
    holm_correction,
    mcnemar_exact,
    paired_bootstrap_ci,
    strict_gate_decision,
)


def _perfect(n: int = 20) -> dict[str, bool]:
    return {f"e{i}": True for i in range(n)}


def _flip(base: dict[str, bool], wrong_to_right: list[str], right_to_wrong: list[str] | None = None):
    out = dict(base)
    for i in wrong_to_right:
        out[i] = True
    for i in right_to_wrong or []:
        out[i] = False
    return out


class TestPairedBootstrap:
    def test_zero_delta_ci_includes_zero(self):
        base = _perfect(50)
        boot = paired_bootstrap_ci(base, dict(base), n_bootstrap=2_000, rng=np.random.default_rng(0))
        assert boot.delta == 0.0
        assert not boot.excludes_zero

    def test_large_improvement_excludes_zero(self):
        base = {f"e{i}": i % 2 == 0 for i in range(100)}
        # Flip every false to true.
        cand = {k: True for k in base}
        boot = paired_bootstrap_ci(base, cand, n_bootstrap=2_000, rng=np.random.default_rng(0))
        assert boot.delta == 0.5
        assert boot.excludes_zero
        assert boot.ci_low > 0


class TestMcNemar:
    def test_no_discordant_pairs(self):
        base = _perfect(10)
        result = mcnemar_exact(base, dict(base))
        assert result.b == 0 and result.c == 0
        assert result.p_value == 1.0
        assert not result.significant

    def test_clear_improvement_is_significant(self):
        base = {f"e{i}": False for i in range(30)}
        cand = {f"e{i}": True for i in range(30)}
        result = mcnemar_exact(base, cand)
        assert result.b == 30 and result.c == 0
        assert result.significant


class TestHolm:
    def test_empty(self):
        assert holm_correction([]) == []

    def test_step_down(self):
        # Smallest p clears, next fails and stops the rest.
        flags = holm_correction([0.01, 0.04, 0.03], alpha=0.05)
        assert flags[0] is True
        assert flags[1] is False  # 0.04 > 0.05/2


class TestStrictGate:
    def test_admits_clear_win(self):
        base = {f"e{i}": False for i in range(40)}
        cand = {f"e{i}": True for i in range(40)}
        decision = strict_gate_decision(base, cand, n_bootstrap=1_000, rng=np.random.default_rng(0))
        assert decision["admit"] is True

    def test_rejects_noise(self):
        base = _perfect(40)
        cand = dict(base)
        cand["e0"] = False
        decision = strict_gate_decision(base, cand, n_bootstrap=1_000, rng=np.random.default_rng(0))
        assert decision["admit"] is False


class TestCompareAll:
    def test_holm_attached(self):
        base = {f"e{i}": i < 20 for i in range(40)}
        better = {k: True for k in base}
        same = dict(base)
        pairs = compare_all(
            {"v0": base, "delta": better, "random": same},
            baseline_id="v0",
            n_bootstrap=500,
            seed=0,
        )
        assert len(pairs) == 2
        assert all(p.holm_significant is not None for p in pairs)
