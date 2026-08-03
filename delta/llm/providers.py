"""Uniform completion interface over Strands, plus caching and retries.

Every model in the project (target, analyzer, proposer) is reached through
:class:`ModelClient`. That uniformity is what makes the two-tier routing in
Phase 6 a configuration change rather than a rewrite.

All providers go through Strands' LiteLLM model so there is exactly one code
path, using LiteLLM's ``provider/model`` naming. The mock provider implements the
same interface without any network, so the entire pipeline (including the
optimization loop) runs offline in CI at zero cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from delta.config import (
    MOCK_MODEL,
    GenerationParams,
    api_key_for,
    missing_key_message,
)
from delta.llm.cache import ResponseCache, cache_key


class MissingAPIKeyError(RuntimeError):
    """Raised with actionable guidance when a provider key is absent."""


class RateLimitedError(RuntimeError):
    """Retryable provider throttling."""


@dataclass
class LLMResponse:
    text: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    stop_reason: str | None = None
    # Input tokens the provider served from its own prefix cache. Groq excludes
    # these from its rate limits, so this is the number that determines whether
    # a run fits in the free tier, and it is the direct evidence that grouping
    # examples by database is working.
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def billable_tokens(self) -> int:
        """Tokens that count against the provider's rate limits."""
        return max(0, self.input_tokens - self.cache_read_tokens) + self.output_tokens


class ModelClient(Protocol):
    model_id: str

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...


def _looks_like_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(s in text for s in ("rate limit", "429", "too many requests", "quota"))


class StrandsClient:
    """A real model, reached through Strands' LiteLLM provider.

    A fresh Strands ``Agent`` is constructed per call. That is deliberate: the
    agent object accumulates conversation history, and carrying history between
    independent benchmark questions would leak one question's answer into the
    next, quietly inflating accuracy.
    """

    def __init__(
        self,
        model_id: str,
        params: GenerationParams | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        api_key = api_key_for(model_id)
        if not api_key:
            raise MissingAPIKeyError(missing_key_message(model_id))

        self.model_id = model_id
        self.params = params or GenerationParams()
        self.cache = cache if cache is not None else ResponseCache()
        self._api_key = api_key
        self._model = None  # built lazily so import stays cheap

    def _build_model(self):
        if self._model is None:
            from strands.models.litellm import LiteLLMModel

            self._model = LiteLLMModel(
                client_args={"api_key": self._api_key},
                model_id=self.model_id,
                params=self.params.as_litellm_params(),
            )
        return self._model

    @retry(
        retry=retry_if_exception_type(RateLimitedError),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _invoke(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        from strands import Agent

        agent = Agent(
            model=self._build_model(),
            system_prompt=system_prompt,
            callback_handler=None,  # suppress Strands' streaming console output
        )

        started = time.perf_counter()
        try:
            result = agent(user_prompt)
        except Exception as exc:
            if _looks_like_rate_limit(exc):
                raise RateLimitedError(str(exc)) from exc
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0

        return LLMResponse(
            text=_extract_text(result),
            model_id=self.model_id,
            input_tokens=_usage(result, "inputTokens"),
            output_tokens=_usage(result, "outputTokens"),
            latency_ms=latency_ms,
            cached=False,
            stop_reason=getattr(result, "stop_reason", None),
            cache_read_tokens=_usage(result, "cacheReadInputTokens"),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        key = cache_key(
            self.model_id, system_prompt, user_prompt, self.params.as_litellm_params()
        )
        hit = self.cache.get(key)
        if hit is not None:
            # Tolerate entries written by an older field set, so adding a field
            # does not invalidate an expensively accumulated cache.
            known = {f.name for f in fields(LLMResponse)} - {"cached"}
            return LLMResponse(**{k: v for k, v in hit.items() if k in known}, cached=True)

        response = self._invoke(system_prompt, user_prompt)
        self.cache.put(
            key,
            {
                "text": response.text,
                "model_id": response.model_id,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "stop_reason": response.stop_reason,
                "cache_read_tokens": response.cache_read_tokens,
            },
        )
        return response


def _extract_text(result) -> str:
    """Pull plain text out of a Strands AgentResult."""
    message = getattr(result, "message", None)
    if isinstance(message, dict):
        parts = [
            block["text"]
            for block in message.get("content", [])
            if isinstance(block, dict) and "text" in block
        ]
        if parts:
            return "\n".join(parts)
    return str(message) if message is not None else str(result)


def _usage(result, field: str) -> int:
    metrics = getattr(result, "metrics", None)
    usage = getattr(metrics, "accumulated_usage", None) if metrics else None
    if isinstance(usage, dict):
        return int(usage.get(field, 0) or 0)
    return 0


def build_client(
    model_id: str,
    params: GenerationParams | None = None,
    cache: ResponseCache | None = None,
) -> ModelClient:
    """Return a client for ``model_id``, or the offline mock."""
    if model_id == MOCK_MODEL or model_id.startswith("mock"):
        from delta.llm.mock import MockClient

        return MockClient(params=params)
    return StrandsClient(model_id=model_id, params=params, cache=cache)
