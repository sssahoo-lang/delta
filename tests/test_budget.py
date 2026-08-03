"""Tests for token-aware budget accounting.

The point of these is that the token ceiling binds before the request ceiling on
this workload. A tracker that only counted requests would let an overnight run
believe it had roughly fourteen times the quota it really has.
"""

from __future__ import annotations

import pytest

from delta.config import Budget
from delta.llm.budget import BudgetExceededError, BudgetTracker
from delta.llm.providers import LLMResponse
from delta.llm.tokens import GROQ_FREE_TPD, RateLimits


def _response(input_tokens=400, output_tokens=60, cache_read=0, cached=False):
    return LLMResponse(
        text="SELECT 1",
        model_id="groq/llama-3.1-8b-instant",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cached=cached,
    )


class TestBudgetTracker:
    def test_accumulates_calls_and_tokens(self):
        tracker = BudgetTracker()
        for _ in range(3):
            tracker.record(_response())
        assert tracker.calls == 3
        assert tracker.input_tokens == 1_200
        assert tracker.billable_tokens == 1_380

    def test_disk_cache_hits_cost_no_quota(self):
        tracker = BudgetTracker()
        tracker.record(_response(cached=True))
        assert tracker.calls == 0
        assert tracker.billable_tokens == 0

    def test_provider_cached_tokens_are_not_billable(self):
        tracker = BudgetTracker()
        tracker.record(_response(input_tokens=500, output_tokens=60, cache_read=450))
        assert tracker.billable_tokens == 110
        assert tracker.prefix_cache_rate == pytest.approx(0.9)

    def test_token_ceiling_trips_before_the_call_ceiling(self):
        """The whole reason this module exists."""
        budget = Budget(max_target_calls=10_000, max_billable_tokens=5_000)
        tracker = BudgetTracker(budget=budget)
        with pytest.raises(BudgetExceededError, match="token budget"):
            for _ in range(100):
                tracker.record(_response())
        assert tracker.calls < 10_000

    def test_call_ceiling_is_enforced(self):
        budget = Budget(max_target_calls=3, max_billable_tokens=10**9)
        tracker = BudgetTracker(budget=budget)
        with pytest.raises(BudgetExceededError, match="call budget"):
            for _ in range(10):
                tracker.record(_response())

    def test_quota_days_reflects_the_binding_cap(self):
        tracker = BudgetTracker(budget=Budget(max_billable_tokens=10**9))
        # Half a day's tokens, but a trivial number of requests.
        tracker.record(_response(input_tokens=GROQ_FREE_TPD // 2, output_tokens=0))
        assert tracker.quota_days() == pytest.approx(0.5)

    def test_remaining_calls_uses_observed_cost_per_call(self):
        tracker = BudgetTracker(budget=Budget(max_billable_tokens=10**9))
        tracker.record(_response(input_tokens=434, output_tokens=60))
        # ~494 billable tokens/call against a 500k daily cap is ~1,011 calls,
        # far below the 14,400 the request cap alone would suggest.
        assert 950 <= tracker.remaining_calls_today() <= 1_050

    def test_prefix_caching_multiplies_the_daily_call_allowance(self):
        cold = BudgetTracker(budget=Budget(max_billable_tokens=10**9))
        cold.record(_response(input_tokens=434, output_tokens=60))

        warm = BudgetTracker(budget=Budget(max_billable_tokens=10**9))
        warm.record(_response(input_tokens=434, output_tokens=60, cache_read=409))

        assert warm.remaining_calls_today() > 4 * cold.remaining_calls_today()

    def test_summary_is_json_safe(self):
        import json

        tracker = BudgetTracker()
        tracker.record(_response())
        json.dumps(tracker.summary())


class TestRateLimits:
    def test_token_cap_binds_before_the_request_cap_per_minute(self):
        limits = RateLimits()
        # 12 calls at 494 tokens is under 30 requests but over 6,000 tokens.
        minutes = limits.minutes_for(calls=12, billable_tokens=12 * 494)
        assert minutes > 12 / limits.rpm

    def test_empty_run_costs_nothing(self):
        assert RateLimits().days_for(0, 0) == 0.0
