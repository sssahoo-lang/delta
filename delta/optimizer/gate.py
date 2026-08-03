"""Decoupled acceptance: permissive search rule + counterfactual strict gate.

During search, a candidate is accepted when the point estimate on validation
improves and no difficulty bucket regresses beyond tolerance. The original 95%
paired-bootstrap / McNemar gate is recorded as a counterfactual ablation, not
used to stop the search.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from delta.evalh.evaluate import EvalReport
from delta.evalh.stats import strict_gate_decision

DEFAULT_BUCKET_TOLERANCE = 0.05  # absolute accuracy points
DEFAULT_SCREEN_TRAIN = 40
DEFAULT_SCREEN_VAL = 60
DEFAULT_PROMOTE_FRACTION = 1 / 3


@dataclass(frozen=True)
class ScreeningConfig:
    train_minibatch: int = DEFAULT_SCREEN_TRAIN
    val_screen: int = DEFAULT_SCREEN_VAL
    promote_fraction: float = DEFAULT_PROMOTE_FRACTION


@dataclass
class GateDecision:
    admit: bool
    reason: str
    delta: float
    bucket_deltas: dict[str, float] = field(default_factory=dict)
    strict_counterfactual: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "admit": self.admit,
            "reason": self.reason,
            "delta": round(self.delta, 6),
            "bucket_deltas": {k: round(v, 6) for k, v in self.bucket_deltas.items()},
            "strict_counterfactual": self.strict_counterfactual,
        }


def _bucket_accuracy(report: EvalReport) -> dict[str, float]:
    return {name: stats.accuracy for name, stats in report.by_difficulty.items()}


def _bucket_deltas(
    baseline: EvalReport,
    candidate: EvalReport,
) -> dict[str, float]:
    base = _bucket_accuracy(baseline)
    cand = _bucket_accuracy(candidate)
    keys = set(base) | set(cand)
    return {k: cand.get(k, 0.0) - base.get(k, 0.0) for k in sorted(keys)}


def permissive_accept(
    baseline: EvalReport,
    candidate: EvalReport,
    bucket_tolerance: float = DEFAULT_BUCKET_TOLERANCE,
    record_strict: bool = True,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> GateDecision:
    """Accept on positive point delta and no per-bucket regression beyond tolerance."""
    delta = candidate.accuracy - baseline.accuracy
    buckets = _bucket_deltas(baseline, candidate)
    regressions = {
        k: v for k, v in buckets.items() if v < -bucket_tolerance
    }

    strict: dict = {}
    if record_strict:
        strict = strict_gate_decision(
            baseline.correctness,
            candidate.correctness,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            rng=rng,
        )

    if delta <= 0:
        return GateDecision(
            admit=False,
            reason="non_positive_delta",
            delta=delta,
            bucket_deltas=buckets,
            strict_counterfactual=strict,
        )
    if regressions:
        worst = min(regressions, key=regressions.get)  # type: ignore[arg-type]
        return GateDecision(
            admit=False,
            reason=f"bucket_regression:{worst}",
            delta=delta,
            bucket_deltas=buckets,
            strict_counterfactual=strict,
        )
    return GateDecision(
        admit=True,
        reason="admit",
        delta=delta,
        bucket_deltas=buckets,
        strict_counterfactual=strict,
    )


def strict_gate_counterfactual(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    **kwargs,
) -> dict:
    """Thin wrapper so callers can log the strict gate without importing stats."""
    return strict_gate_decision(baseline, candidate, **kwargs)


def screen_candidates(
    scores: Sequence[tuple[str, float]],
    config: ScreeningConfig | None = None,
) -> list[str]:
    """Promote the top third of candidates (by screen score) to full validation.

    ``scores`` is a sequence of ``(candidate_id, screen_accuracy)`` already
    measured on the train minibatch + val screen. Returns ids to evaluate on
    the full validation set, sorted best-first.
    """
    config = config or ScreeningConfig()
    if not scores:
        return []
    ordered = sorted(scores, key=lambda kv: kv[1], reverse=True)
    k = max(1, int(round(len(ordered) * config.promote_fraction)))
    return [cid for cid, _ in ordered[:k]]
