"""Unit tests for Groq TPM retry-after parsing and wait strategy."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from delta.llm.providers import RateLimitedError, _parse_retry_after_seconds, _wait_rate_limit


class ParseRetryAfterTests(unittest.TestCase):
    def test_groq_try_again_seconds(self) -> None:
        msg = (
            'Rate limit reached ... on tokens per minute (TPM): Limit 6000, '
            'Used 5583, Requested 537. Please try again in 1.2s.'
        )
        self.assertAlmostEqual(_parse_retry_after_seconds(msg), 1.2)

    def test_retry_after_header_style(self) -> None:
        self.assertAlmostEqual(_parse_retry_after_seconds("Retry-After: 5"), 5.0)

    def test_milliseconds(self) -> None:
        self.assertAlmostEqual(_parse_retry_after_seconds("try again in 800ms"), 0.8)

    def test_missing(self) -> None:
        self.assertIsNone(_parse_retry_after_seconds("something else went wrong"))


class WaitRateLimitTests(unittest.TestCase):
    def test_uses_stated_delay_when_larger(self) -> None:
        wait = _wait_rate_limit()
        state = Mock()
        state.attempt_number = 1
        state.outcome = Mock()
        state.outcome.exception.return_value = RateLimitedError(
            "Please try again in 45s. Need more tokens?"
        )
        # First exponential step is 2s; stated 45s + 0.5 buffer should win.
        self.assertGreaterEqual(wait(state), 45.0)

    def test_falls_back_to_exponential(self) -> None:
        wait = _wait_rate_limit()
        state = Mock()
        state.attempt_number = 1
        state.outcome = Mock()
        state.outcome.exception.return_value = RateLimitedError("rate limit 429")
        self.assertGreaterEqual(wait(state), 2.0)


if __name__ == "__main__":
    unittest.main()
