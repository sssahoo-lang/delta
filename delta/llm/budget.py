"""Runtime token/request accounting against free-tier caps.

Tracked in **tokens as well as requests**, because on Spider the token caps bind
first. Disk-cache hits cost nothing; provider prefix-cache reads are subtracted
from billable tokens so Groq's published behavior is reflected accurately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from delta.config import Budget
from delta.llm.providers import LLMResponse
from delta.llm.tokens import GROQ_FREE_TPD, RateLimits


class BudgetExceededError(RuntimeError):
    """Raised when a call would breach a configured hard stop."""


@dataclass
class BudgetTracker:
    """Accumulates usage and enforces :class:`Budget` ceilings."""

    budget: Budget = field(default_factory=Budget)
    limits: RateLimits = field(default_factory=RateLimits)
    calls: int = 0
    reflection_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    disk_cache_hits: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def billable_tokens(self) -> int:
        return max(0, self.input_tokens - self.cache_read_tokens) + self.output_tokens

    @property
    def prefix_cache_rate(self) -> float:
        return self.cache_read_tokens / self.input_tokens if self.input_tokens else 0.0

    def record(self, response: LLMResponse, reflection: bool = False) -> None:
        """Record one model response, raising if a hard stop would be breached.

        Disk-cache hits (``response.cached``) consume no quota: the call never
        left the machine. Provider prefix-cache reads still count as a request
        but their tokens are excluded from the billable total.
        """
        if response.cached:
            self.disk_cache_hits += 1
            return

        billable = response.billable_tokens
        next_calls = self.calls + (0 if reflection else 1)
        next_reflection = self.reflection_calls + (1 if reflection else 0)
        next_billable = self.billable_tokens + billable

        if not reflection and next_calls > self.budget.max_target_calls:
            raise BudgetExceededError(
                f"call budget exceeded: {next_calls} > {self.budget.max_target_calls}"
            )
        if reflection and next_reflection > self.budget.max_reflection_calls:
            raise BudgetExceededError(
                f"reflection call budget exceeded: "
                f"{next_reflection} > {self.budget.max_reflection_calls}"
            )
        if next_billable > self.budget.max_billable_tokens:
            raise BudgetExceededError(
                f"token budget exceeded: {next_billable} > {self.budget.max_billable_tokens}"
            )
        if time.time() - self.started_at > self.budget.max_wall_clock_s:
            raise BudgetExceededError(
                f"wall-clock budget exceeded: {self.budget.max_wall_clock_s}s"
            )

        if reflection:
            self.reflection_calls += 1
        else:
            self.calls += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.cache_read_tokens += response.cache_read_tokens

    def quota_days(self) -> float:
        """Fraction of a free-tier day consumed by billable tokens so far."""
        return self.limits.days_for(self.calls, self.billable_tokens)

    def remaining_calls_today(self) -> int:
        """How many more average-cost calls fit under the daily token cap."""
        if self.calls == 0:
            # Assume a cold Spider call until we have an observation.
            cost = 494
        else:
            cost = max(1, self.billable_tokens // self.calls)
        remaining_tokens = max(0, GROQ_FREE_TPD - self.billable_tokens)
        by_tokens = remaining_tokens // cost
        by_requests = max(0, self.limits.rpd - self.calls)
        return int(min(by_tokens, by_requests))

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "reflection_calls": self.reflection_calls,
            "disk_cache_hits": self.disk_cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "billable_tokens": self.billable_tokens,
            "prefix_cache_rate": round(self.prefix_cache_rate, 4),
            "quota_days": round(self.quota_days(), 4),
            "remaining_calls_today": self.remaining_calls_today(),
            "elapsed_s": round(time.time() - self.started_at, 1),
        }
