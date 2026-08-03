"""Token counting and the provider rate limits that actually bind.

The project's original budget was written in requests, which turns out to be the
wrong unit. Groq's free tier caps ``llama-3.1-8b-instant`` at 30 requests/minute
and 14,400 requests/day, but *also* at 6,000 tokens/minute and 500,000
tokens/day. With Spider schemas in every prompt the token caps bind first, and by
a wide margin: at roughly 700 tokens per call the daily token cap allows about
700 calls, not 14,400.

The saving grace is that Groq caches identical prompt *prefixes* automatically,
and cached tokens do not count against either token limit
(https://console.groq.com/docs/prompt-caching). Since the rendered message is
``[system prompt][schema][question]`` and Spider dev has only 20 databases, every
question after the first against a given database can reuse the schema prefix,
provided the evaluation visits databases in contiguous runs. That is what
:func:`delta.evalh.evaluate.evaluate_prompt` now does, and it is the difference
between a 40-day run and a 7-day one.

Counting note: Llama 3.1's tokenizer is not available offline without pulling
``transformers`` and a gated HF repo, so ``cl100k_base`` is used as a stand-in.
Both are ~100k-vocab byte-level BPEs and agree closely on SQL and DDL text. This
is a budget-planning estimate, not billing, and callers should treat it as
accurate to roughly +/-10%.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# Groq free tier, llama-3.1-8b-instant, verified August 2026.
GROQ_FREE_RPM = 30
GROQ_FREE_RPD = 14_400
GROQ_FREE_TPM = 6_000
GROQ_FREE_TPD = 500_000

# Groq's prefix cache is dropped after this long without a hit, which is why
# examples sharing a database must be evaluated contiguously rather than
# interleaved.
PREFIX_CACHE_TTL_S = 2 * 60 * 60


# Fallback ratio for when tiktoken cannot load its BPE table, which needs a
# one-time network fetch. Measured against cl100k_base over all 20 rendered
# Spider dev schemas, which are what dominate a prompt's token count.
CHARS_PER_TOKEN_FALLBACK = 4.03


@lru_cache(maxsize=1)
def _encoder():
    """The tokenizer, or ``None`` if its table cannot be loaded offline."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Counting is for budget planning, so a degraded estimate beats a crash
        # in an offline CI run.
        return None


def count_tokens(text: str) -> int:
    """Approximate the token count of ``text`` for budget planning."""
    if not text:
        return 0
    encoder = _encoder()
    if encoder is None:
        return int(len(text) / CHARS_PER_TOKEN_FALLBACK)
    return len(encoder.encode(text))


def tokenizer_is_exact() -> bool:
    """Whether counts come from a real tokenizer rather than the fallback."""
    return _encoder() is not None


@dataclass(frozen=True)
class RateLimits:
    """A provider's caps, in both request and token units."""

    rpm: int = GROQ_FREE_RPM
    rpd: int = GROQ_FREE_RPD
    tpm: int = GROQ_FREE_TPM
    tpd: int = GROQ_FREE_TPD

    def minutes_for(self, calls: int, billable_tokens: int) -> float:
        """Wall-clock floor, whichever of the two per-minute caps binds."""
        by_requests = calls / self.rpm if self.rpm else 0.0
        by_tokens = billable_tokens / self.tpm if self.tpm else 0.0
        return max(by_requests, by_tokens)

    def days_for(self, calls: int, billable_tokens: int) -> float:
        """Quota-days consumed, whichever of the two daily caps binds."""
        by_requests = calls / self.rpd if self.rpd else 0.0
        by_tokens = billable_tokens / self.tpd if self.tpd else 0.0
        return max(by_requests, by_tokens)


@dataclass(frozen=True)
class PassCost:
    """The cost of evaluating one prompt over one set of examples.

    Built from exact per-database counts rather than averages, because Spider
    schema sizes vary by more than an order of magnitude and an average would
    misstate the cold-prefix term badly.
    """

    schema_tokens_by_db: dict[str, int]
    calls_by_db: dict[str, int]
    question_tokens: int
    output_tokens: int
    system_tokens_per_call: int

    @property
    def calls(self) -> int:
        return sum(self.calls_by_db.values())

    @property
    def n_databases(self) -> int:
        return len(self.calls_by_db)

    @property
    def uncached_tokens(self) -> int:
        """Every call pays for the full system prompt and schema."""
        schema = sum(
            self.schema_tokens_by_db.get(db, 0) * n for db, n in self.calls_by_db.items()
        )
        return (
            schema
            + self.question_tokens
            + self.output_tokens
            + self.system_tokens_per_call * self.calls
        )

    @property
    def cached_tokens(self) -> int:
        """Tokens billed when databases are visited in contiguous runs.

        The ``[system][schema]`` prefix costs one cold miss per database instead
        of one per question. Questions and generated output are always billed,
        since they differ on every call.
        """
        cold_prefix = sum(
            self.schema_tokens_by_db.get(db, 0) + self.system_tokens_per_call
            for db in self.calls_by_db
        )
        return cold_prefix + self.question_tokens + self.output_tokens

    @property
    def savings_ratio(self) -> float:
        if not self.uncached_tokens:
            return 0.0
        return 1.0 - self.cached_tokens / self.uncached_tokens
