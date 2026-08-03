"""Tests for the offline mock model.

The mock is what makes CI free and the optimization loop testable without an API
key, but it only serves that purpose if it is genuinely prompt-sensitive. These
tests assert the properties the rest of the project relies on: a better prompt
produces better SQL, every declared skill actually affects output, results are
deterministic, and the mock never sees the gold answer.
"""

from __future__ import annotations

import pytest

from delta.evalh.dataset import load_sample
from delta.evalh.evaluate import evaluate_prompt
from delta.llm.mock import (
    SKILL_TRIGGERS,
    MockClient,
    Skill,
    available_skills,
    parse_schema,
)
from delta.llm.providers import build_client
from delta.target_agent.agent import TargetAgent
from delta.target_agent.prompt import V0_WEAK, V_HANDTUNED
from delta.target_agent.schema_render import render_schema

# Skills that only take effect when another skill is also present. LIMIT is the
# real case: without ORDER BY it would pick an arbitrary row.
SKILL_PREREQUISITES: dict[str, tuple[str, ...]] = {
    Skill.LIMIT: (Skill.ORDER,),
}


class TestSchemaParsing:
    def test_recovers_tables_and_columns(self, sample_db):
        tables = parse_schema(render_schema(str(sample_db)))
        by_name = {t.name: t for t in tables}
        assert set(by_name) == {
            "courses",
            "departments",
            "enrollments",
            "instructors",
            "students",
        }
        assert "gpa" in by_name["students"].column_names()

    def test_recovers_foreign_keys(self, sample_db):
        tables = {t.name: t for t in parse_schema(render_schema(str(sample_db)))}
        assert tables["students"].fk_to("departments") is not None
        assert tables["departments"].fk_to("students") is None

    def test_constraint_lines_are_not_columns(self, sample_db):
        tables = {t.name: t for t in parse_schema(render_schema(str(sample_db)))}
        for col in tables["students"].column_names():
            assert not col.upper().startswith(("PRIMARY", "FOREIGN"))


class TestSkillGating:
    def test_weak_prompt_unlocks_nothing(self):
        assert available_skills(V0_WEAK.text) == set()

    def test_handtuned_prompt_unlocks_several(self):
        skills = available_skills(V_HANDTUNED.text)
        assert {Skill.GROUP, Skill.JOIN, Skill.ORDER, Skill.LIMIT} <= skills

    @pytest.mark.parametrize("skill", sorted(SKILL_TRIGGERS))
    def test_every_skill_is_reachable(self, skill):
        phrase = SKILL_TRIGGERS[skill][0]
        assert skill in available_skills(f"Please use {phrase} where appropriate.")

    @pytest.mark.parametrize("skill", sorted(SKILL_TRIGGERS))
    def test_every_skill_changes_some_output(self, skill, sample_db):
        """A declared skill that never alters SQL would be dead weight the
        optimizer could never exploit.

        Some skills are genuinely dependent: LIMIT without ORDER BY selects an
        arbitrary row, so it is only meaningful alongside ordering. Prerequisites
        are enabled in both arms so the comparison isolates the skill itself.
        """
        schema = render_schema(str(sample_db))
        client = MockClient()
        questions = [ex.question for ex in load_sample()]

        prereq_phrases = [SKILL_TRIGGERS[p][0] for p in SKILL_PREREQUISITES.get(skill, ())]
        base_prompt = "Write SQL. " + " ".join(f"Use {p}." for p in prereq_phrases)
        skill_prompt = base_prompt + f" Use {SKILL_TRIGGERS[skill][0]}."

        base = {q: client.complete(base_prompt, _user(schema, q)).text for q in questions}
        with_skill = {q: client.complete(skill_prompt, _user(schema, q)).text for q in questions}
        assert base != with_skill, f"skill {skill!r} never changed any output"


class TestDeterminism:
    def test_same_input_same_output(self, sample_db):
        schema = render_schema(str(sample_db))
        client = MockClient()
        user = _user(schema, "How many students are there?")
        first = client.complete(V0_WEAK.text, user).text
        for _ in range(5):
            assert client.complete(V0_WEAK.text, user).text == first


class TestNoGoldLeakage:
    def test_mock_never_receives_gold_sql(self, sample_db):
        """The mock must earn its accuracy from schema and question alone."""
        recorded: list[tuple[str, str]] = []
        client = MockClient()
        original = client.complete

        def spy(system_prompt: str, user_prompt: str):
            recorded.append((system_prompt, user_prompt))
            return original(system_prompt, user_prompt)

        client.complete = spy  # type: ignore[method-assign]
        examples = load_sample()
        agent = TargetAgent(client=client, prompt=V_HANDTUNED)
        evaluate_prompt(agent, examples)

        golds = {ex.gold for ex in examples}
        for system_prompt, user_prompt in recorded:
            blob = f"{system_prompt}\n{user_prompt}"
            for gold in golds:
                assert gold not in blob


class TestPromptSensitivity:
    def test_better_prompt_scores_higher(self):
        """The premise of the whole project, verified offline."""
        examples = load_sample()
        client = build_client("mock/deterministic")

        weak = evaluate_prompt(TargetAgent(client=client, prompt=V0_WEAK), examples)
        tuned = evaluate_prompt(TargetAgent(client=client, prompt=V_HANDTUNED), examples)

        assert tuned.accuracy > weak.accuracy, (
            f"no headroom: weak={weak.accuracy:.1%} tuned={tuned.accuracy:.1%}"
        )

    def test_headroom_is_large_enough_to_optimize_into(self):
        examples = load_sample()
        client = build_client("mock/deterministic")
        weak = evaluate_prompt(TargetAgent(client=client, prompt=V0_WEAK), examples)
        tuned = evaluate_prompt(TargetAgent(client=client, prompt=V_HANDTUNED), examples)
        assert tuned.accuracy - weak.accuracy >= 0.10

    def test_weak_prompt_is_not_accidentally_good(self):
        examples = load_sample()
        client = build_client("mock/deterministic")
        weak = evaluate_prompt(TargetAgent(client=client, prompt=V0_WEAK), examples)
        assert weak.accuracy < 0.5


def _user(schema: str, question: str) -> str:
    return f"Database schema:\n\n{schema}\n\nQuestion: {question}"
