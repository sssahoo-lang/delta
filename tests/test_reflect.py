"""Tests for analyzer + proposer (Phase 3).

These run fully offline against the mock reflection client. They check wiring
and determinism, not that Gemini writes brilliant diagnoses — that needs a
manual checkpoint on real traces later.
"""

from __future__ import annotations

import json

from delta.evalh.dataset import load_sample
from delta.evalh.evaluate import evaluate_prompt
from delta.llm.providers import build_client
from delta.optimizer.analyzer import (
    Analyzer,
    cluster_failures,
    parse_diagnosis,
)
from delta.optimizer.proposer import Proposer, enforce_cap, extract_prompt_text
from delta.target_agent.agent import TargetAgent
from delta.target_agent.prompt import MAX_PROMPT_CHARS, V0_WEAK, V_HANDTUNED


def _weak_report():
    agent = TargetAgent(client=build_client("mock/deterministic"), prompt=V0_WEAK)
    return evaluate_prompt(agent, load_sample())


class TestClusterFailures:
    def test_groups_by_reason_and_difficulty(self):
        report = _weak_report()
        clusters = cluster_failures(report.failures())
        assert clusters
        assert all("|" in key for key in clusters)


class TestParseDiagnosis:
    def test_reads_json(self):
        raw = json.dumps(
            {
                "summary": "missing joins",
                "failure_modes": ["no JOIN guidance"],
                "recommendations": ["mention JOIN"],
            }
        )
        summary, modes, recs = parse_diagnosis(raw)
        assert summary == "missing joins"
        assert modes == ["no JOIN guidance"]
        assert recs == ["mention JOIN"]

    def test_reads_fenced_json(self):
        raw = '```json\n{"summary": "x", "failure_modes": [], "recommendations": []}\n```'
        summary, modes, recs = parse_diagnosis(raw)
        assert summary == "x"
        assert modes == [] and recs == []

    def test_fallback_on_garbage(self):
        summary, modes, recs = parse_diagnosis("not json at all, just prose")
        assert "not json" in summary
        assert modes == [] and recs == []


class TestAnalyzer:
    def test_returns_structured_diagnosis_on_mock(self):
        report = _weak_report()
        analyzer = Analyzer(build_client("mock/deterministic"))
        diagnosis = analyzer.diagnose(V0_WEAK, report)
        assert diagnosis.summary
        assert diagnosis.failure_modes
        assert diagnosis.recommendations
        assert diagnosis.example_ids
        assert diagnosis.model_id.startswith("mock")

    def test_handles_perfect_report(self):
        agent = TargetAgent(client=build_client("mock/deterministic"), prompt=V_HANDTUNED)
        # Hand-tuned still misses extra; may have failures. Build an empty-failure
        # report by filtering — use a one-example set the mock gets right.
        from delta.evalh.dataset import load_sample

        easy = [ex for ex in load_sample() if ex.difficulty == "easy"][:1]
        report = evaluate_prompt(agent, easy)
        analyzer = Analyzer(build_client("mock/deterministic"))
        diagnosis = analyzer.diagnose(V_HANDTUNED, report)
        assert isinstance(diagnosis.summary, str)


class TestProposer:
    def test_extract_strips_fences(self):
        assert extract_prompt_text("```\nhello\n```") == "hello"

    def test_enforce_cap_truncates(self):
        text = "para1\n\n" + ("x" * (MAX_PROMPT_CHARS + 100))
        clipped = enforce_cap(text)
        assert len(clipped) <= MAX_PROMPT_CHARS

    def test_propose_improves_weak_prompt_on_mock(self):
        report = _weak_report()
        client = build_client("mock/deterministic")
        diagnosis = Analyzer(client).diagnose(V0_WEAK, report)
        candidate = Proposer(client).propose(V0_WEAK, diagnosis)
        assert candidate.parent_id == V0_WEAK.version_id
        assert candidate.origin == "proposer"
        assert candidate.within_cap()
        # Mock should inject at least one skill the weak prompt lacks.
        assert candidate.text != V0_WEAK.text
        assert any(
            word in candidate.text.lower()
            for word in ("join", "group by", "order by", "distinct", "subquery")
        )

    def test_improved_prompt_raises_mock_accuracy(self):
        """End-to-end smoke: analyze -> propose -> re-score should not go down."""
        examples = load_sample()
        client = build_client("mock/deterministic")
        before = evaluate_prompt(TargetAgent(client=client, prompt=V0_WEAK), examples)
        diagnosis = Analyzer(client).diagnose(V0_WEAK, before)
        candidate = Proposer(client).propose(V0_WEAK, diagnosis)
        after = evaluate_prompt(TargetAgent(client=client, prompt=candidate), examples)
        assert after.accuracy >= before.accuracy
