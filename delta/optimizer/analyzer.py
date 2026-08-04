"""Analyzer agent: read failures, write a short diagnosis.

Simple picture:
  wrong answers + current prompt  -->  analyzer model  -->  diagnosis

The diagnosis is structured (summary, failure modes, recommendations) so the
proposer has something concrete to act on, and so we can inspect it by hand
before trusting the loop.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from delta.evalh.evaluate import EvalReport, ExampleResult
from delta.llm.providers import ModelClient
from delta.optimizer.reflect_prompts import ANALYZER_SYSTEM
from delta.target_agent.prompt import PromptVersion

# How many concrete failures to show the model. Too many wastes tokens and
# dilutes the signal; too few hides patterns.
DEFAULT_MAX_FAILURES = 12
DEFAULT_MAX_TOKENS = 1024


@dataclass
class Diagnosis:
    """What the analyzer believes is wrong with the current prompt."""

    summary: str
    failure_modes: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    example_ids: list[str] = field(default_factory=list)
    clusters: dict[str, int] = field(default_factory=dict)
    raw_text: str = ""
    model_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict) -> Diagnosis:
        return Diagnosis(
            summary=str(payload.get("summary", "")),
            failure_modes=list(payload.get("failure_modes") or []),
            recommendations=list(payload.get("recommendations") or []),
            example_ids=list(payload.get("example_ids") or []),
            clusters=dict(payload.get("clusters") or {}),
            raw_text=str(payload.get("raw_text", "")),
            model_id=str(payload.get("model_id", "")),
        )


def cluster_failures(failures: list[ExampleResult]) -> dict[str, list[ExampleResult]]:
    """Group failures by scorer reason and difficulty.

    Pure Python — no model call. Gives the analyzer a pre-digested view so it
    does not have to invent structure from a flat list.
    """
    buckets: dict[str, list[ExampleResult]] = defaultdict(list)
    for failure in failures:
        key = f"{failure.reason}|{failure.difficulty}"
        buckets[key].append(failure)
    return dict(sorted(buckets.items(), key=lambda kv: -len(kv[1])))


def _pick_examples(
    clusters: dict[str, list[ExampleResult]],
    max_failures: int,
) -> list[ExampleResult]:
    """Round-robin across clusters so one failure mode cannot dominate the prompt."""
    if max_failures <= 0:
        return []
    picked: list[ExampleResult] = []
    queues = {k: list(v) for k, v in clusters.items()}
    while len(picked) < max_failures and any(queues.values()):
        for key in list(queues):
            if queues[key]:
                picked.append(queues[key].pop(0))
                if len(picked) >= max_failures:
                    break
    return picked


def _format_failure(failure: ExampleResult) -> str:
    pred = failure.predicted or "(nothing extracted)"
    detail = f" ({failure.detail})" if failure.detail else ""
    return (
        f"- id={failure.example_id} difficulty={failure.difficulty} "
        f"reason={failure.reason}{detail}\n"
        f"  Q: {failure.question}\n"
        f"  gold: {failure.gold}\n"
        f"  pred: {pred}"
    )


def build_analyzer_user_message(
    prompt: PromptVersion,
    report: EvalReport,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> tuple[str, list[str], dict[str, int]]:
    """Render the analyzer's user turn. Returns (message, example_ids, cluster sizes)."""
    failures = report.failures()
    clusters = cluster_failures(failures)
    cluster_sizes = {k: len(v) for k, v in clusters.items()}
    samples = _pick_examples(clusters, max_failures)

    lines = [
        "Current system prompt:",
        "-----",
        prompt.text,
        "-----",
        "",
        f"Accuracy: {report.n_correct}/{report.n} ({report.accuracy:.1%})",
        f"Failure reason counts: {dict(Counter(f.reason for f in failures))}",
        f"Clusters (reason|difficulty -> n): {cluster_sizes}",
        "",
        "Sample failures:",
    ]
    if not samples:
        lines.append("(no failures — prompt already perfect on this set)")
    else:
        lines.extend(_format_failure(f) for f in samples)

    return "\n".join(lines), [f.example_id for f in samples], cluster_sizes


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_diagnosis(text: str) -> tuple[str, list[str], list[str]]:
    """Pull summary / modes / recommendations out of model text."""
    text = (text or "").strip()
    if not text:
        return "", [], []

    candidate = text
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        match = _JSON_BLOCK_RE.search(text)
        if match:
            candidate = match.group(0)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back: treat the whole reply as a summary so the loop can continue.
        return text[:500], [], []

    summary = str(payload.get("summary") or "").strip()
    modes = [str(m).strip() for m in (payload.get("failure_modes") or []) if str(m).strip()]
    recs = [
        str(r).strip() for r in (payload.get("recommendations") or []) if str(r).strip()
    ]
    return summary, modes, recs


class Analyzer:
    """Calls the reflection model to diagnose a prompt's failures."""

    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def diagnose(
        self,
        prompt: PromptVersion,
        report: EvalReport,
        max_failures: int = DEFAULT_MAX_FAILURES,
    ) -> Diagnosis:
        user_message, example_ids, cluster_sizes = build_analyzer_user_message(
            prompt, report, max_failures=max_failures
        )
        response = self.client.complete(ANALYZER_SYSTEM, user_message)
        summary, modes, recs = parse_diagnosis(response.text)
        return Diagnosis(
            summary=summary or "No summary returned.",
            failure_modes=modes,
            recommendations=recs,
            example_ids=example_ids,
            clusters=cluster_sizes,
            raw_text=response.text,
            model_id=response.model_id,
        )
