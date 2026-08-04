"""Proposer agent: diagnosis + current prompt --> candidate prompt.

Simple picture:
  diagnosis + old prompt  -->  proposer model  -->  new PromptVersion

Only the system prompt text changes. The proposer is capped at MAX_PROMPT_CHARS
so it cannot grow forever by appending caveats (a known failure mode of
reflective prompt optimizers).
"""

from __future__ import annotations

import re

from delta.llm.providers import ModelClient
from delta.optimizer.analyzer import Diagnosis
from delta.optimizer.reflect_prompts import PROPOSER_SYSTEM
from delta.target_agent.prompt import MAX_PROMPT_CHARS, PromptVersion

_FENCE_RE = re.compile(r"```(?:text|prompt|markdown)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def build_proposer_user_message(prompt: PromptVersion, diagnosis: Diagnosis) -> str:
    modes = "\n".join(f"- {m}" for m in diagnosis.failure_modes) or "- (none listed)"
    recs = "\n".join(f"- {r}" for r in diagnosis.recommendations) or "- (none listed)"
    return (
        f"Current system prompt ({prompt.char_count} chars, cap {MAX_PROMPT_CHARS}):\n"
        f"-----\n{prompt.text}\n-----\n\n"
        f"Diagnosis summary:\n{diagnosis.summary}\n\n"
        f"Failure modes:\n{modes}\n\n"
        f"Recommendations:\n{recs}\n\n"
        f"Write the full replacement system prompt now."
    )


def extract_prompt_text(raw: str) -> str:
    """Strip fences / lead-ins so we store instruction text, not chat wrapping."""
    text = (raw or "").strip()
    if not text:
        return ""
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    # Drop a common "Here is the prompt:" lead-in.
    lowered = text.lower()
    for prefix in (
        "here is the updated system prompt:",
        "here is the new system prompt:",
        "updated system prompt:",
        "new system prompt:",
        "system prompt:",
    ):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text.strip()


def enforce_cap(text: str, cap: int = MAX_PROMPT_CHARS) -> str:
    """Hard truncate at a paragraph boundary when possible."""
    if len(text) <= cap:
        return text
    clipped = text[:cap]
    # Prefer cutting at the last blank line so we do not mid-sentence truncate.
    cut = clipped.rfind("\n\n")
    if cut >= cap // 2:
        return clipped[:cut].rstrip()
    cut = clipped.rfind("\n")
    if cut >= cap // 2:
        return clipped[:cut].rstrip()
    return clipped.rstrip()


class Proposer:
    """Calls the reflection model to draft a revised system prompt."""

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def propose(self, prompt: PromptVersion, diagnosis: Diagnosis) -> PromptVersion:
        user_message = build_proposer_user_message(prompt, diagnosis)
        response = self.client.complete(PROPOSER_SYSTEM, user_message)
        text = enforce_cap(extract_prompt_text(response.text))
        if not text:
            # Empty proposals are useless; fall back to the parent so the loop
            # can reject on "no change" rather than crash.
            text = prompt.text
        return PromptVersion.make(
            text,
            parent_id=prompt.version_id,
            origin="proposer",
            notes=diagnosis.summary[:200],
            metadata={
                "model_id": response.model_id,
                "recommendations": diagnosis.recommendations,
                "failure_modes": diagnosis.failure_modes,
            },
        )
