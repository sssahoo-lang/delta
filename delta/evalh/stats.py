"""Paired statistical tests for comparing two evaluation reports.

The confirmatory test lives here. Search uses a permissive rule elsewhere; this
module is for the single held-out comparison (and for the counterfactual strict
gate ablation).

All resampling shares one RNG stream seeded by the caller so that six conditions
compared pairwise are paired across the same bootstrap draws.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class BootstrapDelta:
    """Paired bootstrap CI on accuracy(cand) - accuracy(base)."""

    delta: float
    ci_low: float
    ci_high: float
    n: int
    n_bootstrap: int
    excludes_zero: bool

    def to_dict(self) -> dict:
        return {
            "delta": round(self.delta, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "n": self.n,
            "n_bootstrap": self.n_bootstrap,
            "excludes_zero": self.excludes_zero,
        }


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar test on discordant pairs (b = base-wrong/cand-right, etc.)."""

    b: int  # base incorrect, candidate correct
    c: int  # base correct, candidate incorrect
    p_value: float
    significant: bool

    def to_dict(self) -> dict:
        return {
            "b": self.b,
            "c": self.c,
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
        }


@dataclass(frozen=True)
class PairComparison:
    baseline_id: str
    candidate_id: str
    bootstrap: BootstrapDelta
    mcnemar: McNemarResult
    holm_significant: bool | None = None

    def to_dict(self) -> dict:
        return {
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "bootstrap": self.bootstrap.to_dict(),
            "mcnemar": self.mcnemar.to_dict(),
            "holm_significant": self.holm_significant,
        }


def _aligned_pairs(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Return parallel boolean arrays for the intersection of example ids."""
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        raise ValueError("no overlapping example ids between the two reports")
    base = np.array([bool(baseline[i]) for i in ids], dtype=bool)
    cand = np.array([bool(candidate[i]) for i in ids], dtype=bool)
    return base, cand


def paired_bootstrap_ci(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    rng: np.random.Generator | None = None,
) -> BootstrapDelta:
    """95% (by default) CI on the paired accuracy delta via bootstrap."""
    base, cand = _aligned_pairs(baseline, candidate)
    n = len(base)
    deltas = cand.astype(np.float64) - base.astype(np.float64)
    point = float(deltas.mean())

    if n == 0:
        return BootstrapDelta(0.0, 0.0, 0.0, 0, n_bootstrap, False)

    rng = rng or np.random.default_rng(0)
    # Resample example indices with replacement.
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot = deltas[idx].mean(axis=1)
    low = float(np.quantile(boot, alpha / 2))
    high = float(np.quantile(boot, 1 - alpha / 2))
    return BootstrapDelta(
        delta=point,
        ci_low=low,
        ci_high=high,
        n=n,
        n_bootstrap=n_bootstrap,
        excludes_zero=(low > 0.0) or (high < 0.0),
    )


def mcnemar_exact(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    alpha: float = DEFAULT_ALPHA,
) -> McNemarResult:
    """Two-sided exact McNemar (binomial) test on discordant pairs."""
    base, cand = _aligned_pairs(baseline, candidate)
    b = int((~base & cand).sum())  # base wrong, cand right
    c = int((base & ~cand).sum())  # base right, cand wrong
    n = b + c
    if n == 0:
        return McNemarResult(b=0, c=0, p_value=1.0, significant=False)

    # Exact two-sided p-value: sum of probabilities as extreme as observed.
    k = min(b, c)
    # P(X <= k) + P(X >= n-k) under Binomial(n, 0.5); for symmetric case = 2*cdf.
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    p = min(1.0, 2.0 * cdf)
    return McNemarResult(b=b, c=c, p_value=p, significant=p < alpha)


def holm_correction(
    p_values: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
) -> list[bool]:
    """Holm–Bonferroni step-down correction. Returns significance flags in input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    significant = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            significant[idx] = True
        else:
            # Once one fails, all remaining (larger p) fail.
            break
    return significant


def compare_pair(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    baseline_id: str = "baseline",
    candidate_id: str = "candidate",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    rng: np.random.Generator | None = None,
) -> PairComparison:
    boot = paired_bootstrap_ci(
        baseline, candidate, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng
    )
    mac = mcnemar_exact(baseline, candidate, alpha=alpha)
    return PairComparison(
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        bootstrap=boot,
        mcnemar=mac,
    )


def compare_all(
    conditions: Mapping[str, Mapping[str, bool]],
    baseline_id: str,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> list[PairComparison]:
    """Compare every non-baseline condition against ``baseline_id``, Holm-corrected."""
    if baseline_id not in conditions:
        raise KeyError(f"baseline {baseline_id!r} not in conditions")
    rng = np.random.default_rng(seed)
    pairs: list[PairComparison] = []
    for cid, correctness in conditions.items():
        if cid == baseline_id:
            continue
        # Fresh generator state per pair but same parent seed stream.
        pair_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        pairs.append(
            compare_pair(
                conditions[baseline_id],
                correctness,
                baseline_id=baseline_id,
                candidate_id=cid,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                rng=pair_rng,
            )
        )
    flags = holm_correction([p.mcnemar.p_value for p in pairs], alpha=alpha)
    return [
        PairComparison(
            baseline_id=p.baseline_id,
            candidate_id=p.candidate_id,
            bootstrap=p.bootstrap,
            mcnemar=p.mcnemar,
            holm_significant=flag,
        )
        for p, flag in zip(pairs, flags, strict=True)
    ]


def strict_gate_decision(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    rng: np.random.Generator | None = None,
) -> dict:
    """Counterfactual: would the original 95% in-loop gate have admitted this?"""
    boot = paired_bootstrap_ci(
        baseline, candidate, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng
    )
    mac = mcnemar_exact(baseline, candidate, alpha=alpha)
    admit = boot.excludes_zero and boot.delta > 0 and mac.significant and mac.b > mac.c
    reason = "admit"
    if not (boot.delta > 0):
        reason = "non_positive_delta"
    elif not boot.excludes_zero:
        reason = "ci_includes_zero"
    elif not mac.significant:
        reason = "mcnemar_not_significant"
    elif not (mac.b > mac.c):
        reason = "mcnemar_wrong_direction"
    return {
        "admit": admit,
        "reason": reason,
        "bootstrap": boot.to_dict(),
        "mcnemar": mac.to_dict(),
    }
