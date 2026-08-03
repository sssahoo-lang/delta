"""Tests for SQL extraction and the target agent.

Extraction leniency is measured, not cosmetic: if a well-formed answer wrapped in
markdown were scored as a failure, the v0 baseline would be understated and every
later improvement would partly reflect the parser rather than the prompt.
"""

from __future__ import annotations

from delta.config import GenerationParams
from delta.llm.providers import build_client
from delta.target_agent.agent import TargetAgent, extract_sql
from delta.target_agent.prompt import V0_WEAK, V_HANDTUNED


class TestExtractSQL:
    def test_bare_sql(self):
        assert extract_sql("SELECT * FROM t") == "SELECT * FROM t"

    def test_sql_fence(self):
        assert extract_sql("```sql\nSELECT a FROM t\n```") == "SELECT a FROM t"

    def test_plain_fence(self):
        assert extract_sql("```\nSELECT a FROM t\n```") == "SELECT a FROM t"

    def test_sqlite_fence(self):
        assert extract_sql("```sqlite\nSELECT a FROM t\n```") == "SELECT a FROM t"

    def test_prose_prefix_is_dropped(self):
        assert extract_sql("Here is the query:\nSELECT a FROM t") == "SELECT a FROM t"

    def test_trailing_explanation_after_semicolon_is_dropped(self):
        text = "SELECT a FROM t; This query returns all values of a."
        assert extract_sql(text) == "SELECT a FROM t"

    def test_trailing_explanation_without_semicolon_is_dropped(self):
        text = "SELECT a FROM t\n\nThis query returns every row."
        assert extract_sql(text) == "SELECT a FROM t"

    def test_with_cte_is_recognized(self):
        text = "WITH x AS (SELECT 1) SELECT * FROM x"
        assert extract_sql(text) == text

    def test_multiline_sql_is_collapsed(self):
        text = "SELECT a,\n       b\nFROM t\nWHERE a > 1"
        assert extract_sql(text) == "SELECT a, b FROM t WHERE a > 1"

    def test_only_first_statement_is_taken(self):
        assert extract_sql("SELECT 1; SELECT 2") == "SELECT 1"

    def test_no_sql_returns_empty(self):
        assert extract_sql("I cannot answer that question.") == ""
        assert extract_sql("") == ""
        assert extract_sql("   ") == ""

    def test_label_prefix(self):
        assert extract_sql("SQL: SELECT a FROM t") == "SELECT a FROM t"


class TestExtractSQLRealModelPatterns:
    """Output shapes an instruction-tuned model actually produces.

    Each of these would otherwise be scored as a wrong answer for a query that
    is in fact correct, which would understate the baseline and inflate every
    later improvement.
    """

    def test_markdown_bold_explanation_header_is_dropped(self):
        text = "SELECT a FROM t\n\n**Explanation:**\nThis picks column a."
        assert extract_sql(text) == "SELECT a FROM t"

    def test_markdown_heading_note_is_dropped(self):
        text = "SELECT a FROM t\n\n### Note:\nAssumes t exists."
        assert extract_sql(text) == "SELECT a FROM t"

    def test_bulleted_explanation_is_dropped(self):
        text = "SELECT a FROM t\n\n- Selects column a\n- From table t"
        assert extract_sql(text) == "SELECT a FROM t"

    def test_numbered_explanation_is_dropped(self):
        text = "SELECT a FROM t\n\n1. First we select a\n2. Then we read t"
        assert extract_sql(text) == "SELECT a FROM t"

    def test_semicolon_inside_string_literal_is_not_a_statement_end(self):
        text = "SELECT a FROM t WHERE name = 'Smith; Jones'"
        assert extract_sql(text) == text

    def test_semicolon_after_a_literal_still_terminates(self):
        text = "SELECT a FROM t WHERE name = 'a;b'; This query filters rows."
        assert extract_sql(text) == "SELECT a FROM t WHERE name = 'a;b'"

    def test_escaped_quote_inside_literal_is_handled(self):
        text = "SELECT a FROM t WHERE name = 'O''Brien; Co'"
        assert extract_sql(text) == text

    def test_fenced_block_between_prose_on_both_sides(self):
        text = (
            "Sure! Here is the query you asked for:\n\n"
            "```sql\nSELECT a FROM t\n```\n\n"
            "Let me know if you need anything else."
        )
        assert extract_sql(text) == "SELECT a FROM t"

    def test_unterminated_fence_from_a_truncated_response(self):
        """max_tokens can cut the response before the closing fence."""
        text = "```sql\nSELECT a FROM t WHERE b = 1"
        assert extract_sql(text) == "SELECT a FROM t WHERE b = 1"

    def test_select_star_on_its_own_line_survives(self):
        """The bullet heuristic must not eat legitimate SQL."""
        text = "SELECT\n*\nFROM t"
        assert extract_sql(text) == "SELECT * FROM t"


class TestTargetAgent:
    def test_generates_sql_from_a_question(self, sample_db):
        agent = TargetAgent(client=build_client("mock/deterministic"), prompt=V0_WEAK)
        result = agent.generate_sql(str(sample_db), "How many students are there?")
        assert result.sql.upper().startswith("SELECT")
        assert not result.extraction_failed

    def test_with_prompt_swaps_prompt_and_shares_client(self, sample_db):
        agent = TargetAgent(client=build_client("mock/deterministic"), prompt=V0_WEAK)
        sibling = agent.with_prompt(V_HANDTUNED)
        assert sibling.prompt.version_id == V_HANDTUNED.version_id
        assert sibling.client is agent.client
        assert agent.prompt.version_id == V0_WEAK.version_id

    def test_prompt_changes_the_output(self, sample_db):
        """The prompt must be a real lever, or nothing downstream is measuring anything."""
        client = build_client("mock/deterministic")
        question = "Show the name of each student along with their department name."
        weak = TargetAgent(client=client, prompt=V0_WEAK).generate_sql(str(sample_db), question)
        tuned = TargetAgent(client=client, prompt=V_HANDTUNED).generate_sql(
            str(sample_db), question
        )
        assert weak.sql != tuned.sql

    def test_reports_token_usage(self, sample_db):
        agent = TargetAgent(client=build_client("mock/deterministic"), prompt=V0_WEAK)
        result = agent.generate_sql(str(sample_db), "How many students are there?")
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    def test_generation_params_are_deterministic_by_default(self):
        assert GenerationParams().temperature == 0.0
