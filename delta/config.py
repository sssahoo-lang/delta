"""Central configuration: paths, model choices, and budget limits.

Model IDs use LiteLLM's ``provider/model`` notation so that every provider goes
through one code path. Defaults are the free tiers, verified as of the project
start: Groq allows 30 requests/minute and 14,400/day on ``llama-3.1-8b-instant``,
and Gemini allows 10/minute and 1,500/day on Flash. The target agent makes
thousands of calls and the reflection agents make dozens, so they are pointed at
those two tiers respectively.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / ".cache"

SAMPLE_DB = DATA_DIR / "sample.db"
SAMPLE_QUESTIONS = DATA_DIR / "sample_questions.json"
SPIDER_DIR = DATA_DIR / "spider_data"
SPLITS_PATH = DATA_DIR / "splits.json"

# The model that writes SQL. Cheap and fast, because it is called constantly.
# Free-tier binding limit is tokens (6k TPM / 500k TPD), not the 14.4k RPD.
TARGET_MODEL = "groq/llama-3.1-8b-instant"

# The models that read failures and propose prompt edits. Published ablations
# find that reflection quality collapses on small models, so this tier must stay
# strong even though it is called rarely.
REFLECTION_MODEL = "gemini/gemini-flash-latest"

# Optional stronger reflection, used only when explicitly requested.
STRONG_REFLECTION_MODEL = "anthropic/claude-sonnet-4-5"

MOCK_MODEL = "mock/deterministic"


@dataclass(frozen=True)
class GenerationParams:
    """Sampling settings.

    Temperature is zero everywhere by default. Non-determinism in the target
    agent would show up as phantom accuracy deltas and make the acceptance gate
    measure noise instead of improvement.
    """

    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0

    def as_litellm_params(self) -> dict:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


@dataclass(frozen=True)
class Budget:
    """Hard stops so a runaway loop cannot exhaust a daily quota.

    Tracked in **tokens as well as requests**, because on Spider the token caps
    bind first and by a wide margin. Measured over the dev set, an uncached call
    averages 494 tokens, so Groq's 500,000 tokens/day allows about 1,000 calls
    rather than the 14,400 its request cap suggests, and its 6,000 tokens/minute
    allows about 12 calls/minute rather than 30. A request-only budget would let
    a run believe it had 14x the quota it really has.

    ``max_target_calls`` covers the full six-condition experiment: roughly 9,750
    calls each for the Delta search, the random-search ablation, MIPROv2 and
    GEPA, plus 2,400 for final test scoring, with headroom for retries.
    """

    max_target_calls: int = 50_000
    max_reflection_calls: int = 400
    max_wall_clock_s: float = 3 * 60 * 60

    # Billable tokens, so provider-side prefix-cached tokens do not count. One
    # free-tier day is 500,000; this ceiling is a stop for a single run, not for
    # the whole project.
    max_billable_tokens: int = 450_000


@dataclass(frozen=True)
class Settings:
    target_model: str = TARGET_MODEL
    reflection_model: str = REFLECTION_MODEL
    generation: GenerationParams = field(default_factory=GenerationParams)
    budget: Budget = field(default_factory=Budget)
    sql_timeout_s: float = 10.0
    cache_enabled: bool = True


def api_key_for(model_id: str) -> str | None:
    """Return the env var value LiteLLM needs for ``model_id``'s provider."""
    provider = model_id.split("/", 1)[0]
    env_var = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider)
    return os.environ.get(env_var) if env_var else None


def missing_key_message(model_id: str) -> str:
    provider = model_id.split("/", 1)[0]
    env_var = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider, f"{provider.upper()}_API_KEY")
    return (
        f"No API key for {model_id}. Set {env_var} in .env "
        f"(copy .env.example), or pass --mock to run without any provider."
    )
