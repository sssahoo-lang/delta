"""Score a prompt over a set of examples.

This is the measurement primitive the whole project is built on. The optimizer
calls it, the acceptance gate calls it twice and compares, and the final
comparison table calls it once per condition.

The important design choice is that :class:`EvalReport` keeps the **per-example**
outcome, not just an average. Aggregate accuracy cannot support a paired
statistical test, and a paired test is what distinguishes a real improvement from
noise on a few hundred examples. Everything else here exists to make that vector
trustworthy and cheap to obtain.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from delta.evalh.dataset import Example
from delta.evalh.score import ScoreReason, score_prediction
from delta.target_agent.agent import TargetAgent

ProgressFn = Callable[[int, int, "ExampleResult"], None]


@dataclass
class ExampleResult:
    example_id: str
    question: str
    difficulty: str
    gold: str
    predicted: str
    correct: bool
    reason: str
    raw_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    detail: str | None = None

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "difficulty": self.difficulty,
            "gold": self.gold,
            "predicted": self.predicted,
            "correct": self.correct,
            "reason": self.reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "cached": self.cached,
            "detail": self.detail,
        }


@dataclass
class DifficultyStats:
    n: int = 0
    n_correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {"n": self.n, "n_correct": self.n_correct, "accuracy": round(self.accuracy, 4)}


@dataclass
class EvalReport:
    prompt_version_id: str
    prompt_origin: str
    model_id: str
    results: list[ExampleResult] = field(default_factory=list)
    wall_clock_s: float = 0.0

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def correctness(self) -> dict[str, bool]:
        """Per-example outcomes keyed by id.

        Keyed rather than positional so that paired tests cannot silently
        misalign two reports that were evaluated in a different order.
        """
        return {r.example_id: r.correct for r in self.results}

    @property
    def by_difficulty(self) -> dict[str, DifficultyStats]:
        out: dict[str, DifficultyStats] = {}
        for r in self.results:
            stats = out.setdefault(r.difficulty, DifficultyStats())
            stats.n += 1
            stats.n_correct += int(r.correct)
        return out

    @property
    def failure_reasons(self) -> dict[str, int]:
        return dict(Counter(r.reason for r in self.results if not r.correct))

    @property
    def extraction_failures(self) -> int:
        """Answers where no SQL could be found at all.

        Worth watching separately: a spike here usually means the prompt stopped
        asking for bare SQL, which is a prompt regression rather than a reasoning
        one.
        """
        return sum(1 for r in self.results if not r.predicted)

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)

    @property
    def cache_hits(self) -> int:
        return sum(1 for r in self.results if r.cached)

    def failures(self) -> list[ExampleResult]:
        """Incorrect results, excluding rows where the benchmark itself is broken."""
        return [
            r for r in self.results if not r.correct and r.reason != ScoreReason.GOLD_FAILED
        ]

    def summary(self) -> dict:
        return {
            "prompt_version_id": self.prompt_version_id,
            "prompt_origin": self.prompt_origin,
            "model_id": self.model_id,
            "n": self.n,
            "n_correct": self.n_correct,
            "accuracy": round(self.accuracy, 4),
            "by_difficulty": {k: v.to_dict() for k, v in sorted(self.by_difficulty.items())},
            "failure_reasons": self.failure_reasons,
            "extraction_failures": self.extraction_failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hits": self.cache_hits,
            "wall_clock_s": round(self.wall_clock_s, 2),
        }


def evaluate_prompt(
    agent: TargetAgent,
    examples: Sequence[Example],
    timeout_s: float = 10.0,
    progress: ProgressFn | None = None,
    keep_raw: bool = False,
) -> EvalReport:
    """Generate and score SQL for every example.

    Sequential by design. The target model's free tier allows 30 requests per
    minute, so concurrency would mostly produce 429s, and serial execution keeps
    the trace order deterministic.
    """
    report = EvalReport(
        prompt_version_id=agent.prompt.version_id,
        prompt_origin=agent.prompt.origin,
        model_id=agent.model_id,
    )

    started = time.perf_counter()
    total = len(examples)
    for i, ex in enumerate(examples, start=1):
        generation = agent.generate_sql(str(ex.db_path), ex.question)
        scored = score_prediction(
            ex.db_path, generation.sql, ex.gold, timeout_s=timeout_s
        )

        result = ExampleResult(
            example_id=ex.id,
            question=ex.question,
            difficulty=ex.difficulty,
            gold=ex.gold,
            predicted=generation.sql,
            correct=scored.correct,
            reason=scored.reason,
            raw_text=generation.raw_text if keep_raw else "",
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            latency_ms=generation.latency_ms,
            cached=generation.cached,
            detail=scored.detail,
        )
        report.results.append(result)

        if progress is not None:
            progress(i, total, result)

    report.wall_clock_s = time.perf_counter() - started
    return report
