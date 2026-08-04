"""A deterministic offline model, used for tests, CI, and `--mock` runs.

A mock that always returns the same string would let the plumbing be tested but
not the *optimizer*, because every candidate prompt would score identically and
the acceptance gate would never have anything to accept. This mock is therefore
built to be **prompt-sensitive**: it owns a set of SQL skills, and a skill is
only available if the system prompt actually instructs the model to use it.

The consequence is that a better prompt genuinely produces better SQL, so a full
optimization run can be exercised end to end with no API key and no network,
which is what makes CI both free and meaningful.

Two properties are enforced to keep this honest:

- It never sees gold SQL. It reads the same rendered schema and question text a
  real model gets, and nothing else.
- It is deterministic. No sampling, no clock, no randomness that is not seeded
  from the input itself, so results are reproducible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from delta.config import MOCK_MODEL, GenerationParams
from delta.llm.providers import LLMResponse


class Skill:
    GROUP = "group"
    JOIN = "join"
    ORDER = "order"
    LIMIT = "limit"
    SUBQUERY = "subquery"
    DISTINCT = "distinct"


# Phrases that unlock each skill. A prompt that never mentions grouping does not
# get grouping, which is exactly the weakness the optimizer has to discover.
SKILL_TRIGGERS: dict[str, tuple[str, ...]] = {
    Skill.GROUP: ("group by", "per group", "for each", "aggregate by"),
    Skill.JOIN: ("join", "foreign key", "related table", "multiple tables"),
    Skill.ORDER: ("order by", "sort", "ascending", "descending"),
    Skill.LIMIT: ("limit", "top n", "first n", "only the top"),
    Skill.SUBQUERY: ("subquery", "nested query", "nested select"),
    Skill.DISTINCT: ("distinct", "unique values", "deduplicate"),
}

# Column-name fragments too common to indicate that a question means a second
# table. Without this filter, "the names of students" appears to reference every
# table that happens to have a *_name column.
GENERIC_COLUMN_PARTS = {
    "name", "id", "title", "date", "year", "code", "type",
    "number", "count", "value", "text", "time", "info",
}

# Score a candidate partner table must clear before a join is emitted. A table
# name match scores 5, so this admits a named table or two distinctive columns.
MIN_JOIN_EVIDENCE = 4

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a million": 1_000_000, "one million": 1_000_000,
}

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL
)

_FK_RE = re.compile(
    r"FOREIGN\s+KEY\s*\(\s*(\w+)\s*\)\s*REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)", re.IGNORECASE
)


@dataclass
class Column:
    name: str
    type: str = "TEXT"

    @property
    def is_numeric(self) -> bool:
        return self.type.upper() in {"INTEGER", "REAL", "NUMERIC", "FLOAT", "DECIMAL"}


@dataclass
class ForeignKey:
    from_col: str
    ref_table: str
    ref_col: str


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def find(self, *candidates: str) -> Column | None:
        lowered = {c.name.lower(): c for c in self.columns}
        for cand in candidates:
            if cand.lower() in lowered:
                return lowered[cand.lower()]
        return None

    def fk_to(self, table_name: str) -> ForeignKey | None:
        return next(
            (fk for fk in self.foreign_keys if fk.ref_table.lower() == table_name.lower()), None
        )


def parse_schema(rendered: str) -> list[Table]:
    """Recover tables and columns from the rendered CREATE TABLE schema."""
    tables: list[Table] = []
    for match in _CREATE_TABLE_RE.finditer(rendered):
        name = match.group(1)
        body = match.group(2)
        columns: list[Column] = []
        foreign_keys = [
            ForeignKey(from_col=m.group(1), ref_table=m.group(2), ref_col=m.group(3))
            for m in _FK_RE.finditer(body)
        ]
        for line in body.splitlines():
            line = line.strip().rstrip(",").strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith(("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CHECK", "CONSTRAINT")):
                continue
            parts = line.split()
            if len(parts) >= 1 and re.fullmatch(r"\w+", parts[0]):
                col_type = parts[1] if len(parts) > 1 else "TEXT"
                columns.append(Column(name=parts[0], type=col_type.strip(",")))
        if columns:
            tables.append(Table(name=name, columns=columns, foreign_keys=foreign_keys))
    return tables


def available_skills(system_prompt: str) -> set[str]:
    text = (system_prompt or "").lower()
    return {skill for skill, phrases in SKILL_TRIGGERS.items() if any(p in text for p in phrases)}


def _extract_number(question: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", question)
    if m:
        return float(m.group(1))
    lowered = question.lower()
    # Longest phrase first, so "one million" is not shadowed by "one".
    for word in sorted(NUMBER_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return float(NUMBER_WORDS[word])
    return None


def _sentence_initial_words(question: str) -> set[str]:
    """Words capitalized by grammar rather than because they are names.

    Checked for every sentence, not just the first: questions like "How many
    students...? Show the department name." capitalize "Show" mid-question, and
    treating it as a proper noun invents a bogus WHERE clause.
    """
    starts = set()
    for sentence in re.split(r"(?<=[.?!])\s+", question.strip()):
        first = sentence.strip().split(" ")[:1]
        if first:
            starts.add(first[0].strip(",;:"))
    return starts


def _singular(word: str) -> str:
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _pick_table(question: str, tables: list[Table]) -> Table | None:
    if not tables:
        return None
    words = {w.lower() for w in re.findall(r"[a-zA-Z_]+", question)}
    words |= {_singular(w) for w in words}

    best, best_score = None, -1
    for table in tables:
        score = 0
        tname = table.name.lower()
        if tname in words or _singular(tname) in words:
            score += 10
        score += sum(1 for col in table.column_names() if col.lower() in words)
        # Weak tiebreak so selection is stable rather than dict-order dependent.
        if score > best_score or (score == best_score and best and table.name < best.name):
            best, best_score = table, score
    return best or tables[0]


def _label_column(table: Table) -> Column:
    return (
        table.find("name", "title", "dept_name")
        or next((c for c in table.columns if not c.is_numeric), table.columns[0])
    )


def _pick_join_partner(question: str, tables: list[Table], primary: Table) -> Table | None:
    """A second table the question refers to and that joins to ``primary``.

    Only direct foreign-key relationships in either direction are considered.
    Multi-hop join paths are intentionally out of reach, which keeps some of the
    hardest questions unreachable and preserves headroom above the mock's ceiling.
    """
    words = {w.lower() for w in re.findall(r"[a-zA-Z_]+", question)}
    words |= {_singular(w) for w in words}

    best, best_score = None, 0
    for candidate in tables:
        if candidate.name == primary.name:
            continue
        if not (primary.fk_to(candidate.name) or candidate.fk_to(primary.name)):
            continue

        score = 0
        cname = candidate.name.lower()
        if cname in words or _singular(cname) in words:
            score += 5
        # Only distinctive column names count. Matching on generic parts like
        # "name" or "id" would drag a second table into nearly every question,
        # since "the names of students" mentions neither departments nor a join.
        score += sum(
            2
            for col in candidate.column_names()
            if col.lower() in words
            or any(
                p in words
                for p in col.lower().split("_")
                if len(p) > 3 and p not in GENERIC_COLUMN_PARTS
            )
        )
        if score > best_score:
            best, best_score = candidate, score

    return best if best_score >= MIN_JOIN_EVIDENCE else None


def _join_clause(primary: Table, partner: Table) -> str | None:
    fk = primary.fk_to(partner.name)
    if fk:
        return (
            f"{primary.name} AS T1 JOIN {partner.name} AS T2 "
            f"ON T1.{fk.from_col} = T2.{fk.ref_col}"
        )
    fk = partner.fk_to(primary.name)
    if fk:
        return (
            f"{primary.name} AS T1 JOIN {partner.name} AS T2 "
            f"ON T1.{fk.ref_col} = T2.{fk.from_col}"
        )
    return None


def _mentioned_column(question: str, table: Table) -> Column | None:
    words = {w.lower() for w in re.findall(r"[a-zA-Z_]+", question)}
    for col in table.columns:
        if col.name.lower() in words:
            return col
        # gpa -> "gpa", enrollment_year -> "enrollment"/"year"
        if any(part in words for part in col.name.lower().split("_") if len(part) > 3):
            return col
    return None


def generate_sql(question: str, schema_text: str, skills: set[str]) -> str:
    """Rule-based text-to-SQL, limited by the skills the prompt unlocked."""
    tables = parse_schema(schema_text)
    if not tables:
        return "SELECT 1"

    q = question.lower().strip()
    table = _pick_table(question, tables)
    if table is None:
        return "SELECT 1"

    wants_count = any(p in q for p in ("how many", "number of", "count of", "count the"))
    wants_avg = "average" in q or "mean " in q
    wants_max = any(p in q for p in ("highest", "most", "largest", "maximum", "top "))
    wants_min = any(p in q for p in ("lowest", "fewest", "smallest", "minimum"))
    wants_each = any(p in q for p in ("each", "per ", "by department", "group"))
    wants_alpha = "alphabetical" in q or "alphabetically" in q

    label = _label_column(table)
    mentioned = _mentioned_column(question, table)

    wants_distinct = any(p in q for p in ("distinct", "different", "unique"))
    compares_to_aggregate = any(
        p in q for p in ("above the average", "than the average", "above average", "more than the average")
    )
    excludes = any(p in q for p in ("do not", "does not", "never", "without any", "no "))

    # Comparison against an aggregate needs a nested SELECT. Without the subquery
    # skill the mock produces a bare filter and gets it wrong, which is exactly
    # the failure the optimizer should learn to fix.
    if compares_to_aggregate and Skill.SUBQUERY in skills:
        num = (
            mentioned
            if mentioned and mentioned.is_numeric
            else next((c for c in table.columns if c.is_numeric), None)
        )
        if num is not None:
            return (
                f"SELECT {label.name} FROM {table.name} "
                f"WHERE {num.name} > (SELECT avg({num.name}) FROM {table.name})"
            )

    # Anti-join phrasing ("instructors who teach no course") needs NOT IN.
    partner_for_exclusion = _pick_join_partner(question, tables, table)
    if excludes and Skill.SUBQUERY in skills and partner_for_exclusion is not None:
        fk = partner_for_exclusion.fk_to(table.name)
        if fk:
            return (
                f"SELECT {label.name} FROM {table.name} WHERE {fk.ref_col} NOT IN "
                f"(SELECT {fk.from_col} FROM {partner_for_exclusion.name} "
                f"WHERE {fk.from_col} IS NOT NULL)"
            )

    # Two-table questions. The join is only reachable with the join skill, so a
    # prompt that never mentions joining silently answers from one table.
    partner = _pick_join_partner(question, tables, table) if Skill.JOIN in skills else None
    join_from = _join_clause(table, partner) if partner else None
    if join_from is not None:
        partner_label = _label_column(partner)
        if wants_each and Skill.GROUP in skills:
            agg = "count(*)"
            if wants_avg:
                num = next((c for c in table.columns if c.is_numeric), None)
                if num is not None:
                    agg = f"avg(T1.{num.name})"
            return (
                f"SELECT T2.{partner_label.name}, {agg} FROM {join_from} "
                f"GROUP BY T2.{partner_label.name}"
            )
        if wants_count:
            counted = (
                f"count(DISTINCT T1.{label.name})" if wants_distinct and Skill.DISTINCT in skills
                else "count(*)"
            )
            return f"SELECT {counted} FROM {join_from}"
        return f"SELECT T1.{label.name}, T2.{partner_label.name} FROM {join_from}"

    # WHERE clause, only for comparisons the question states explicitly.
    where = ""
    number = _extract_number(question)
    numeric_col = mentioned if (mentioned and mentioned.is_numeric) else None
    if numeric_col and number is not None:
        if any(p in q for p in ("above", "greater than", "more than", "over", "higher than")):
            where = f" WHERE {numeric_col.name} > {number:g}"
        elif any(p in q for p in ("below", "less than", "under", "fewer than", "lower than")):
            where = f" WHERE {numeric_col.name} < {number:g}"
    if not where:
        quoted = re.findall(r"'([^']+)'", question)
        proper = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b", question)
        sentence_starts = _sentence_initial_words(question)
        candidates = quoted + [p for p in proper if p.split(" ")[0] not in sentence_starts]
        text_col = table.find("name", "dept_name", "title")
        if candidates and text_col:
            where = f" WHERE {text_col.name} = '{candidates[0]}'"

    # GROUP BY requires both the skill and a joinable label, so a prompt that
    # never mentions grouping produces a plain aggregate and gets it wrong.
    if wants_each and Skill.GROUP in skills:
        agg = "count(*)"
        if wants_avg:
            num = next((c for c in table.columns if c.is_numeric and c is not mentioned), None)
            agg = f"avg({(mentioned or num or table.columns[-1]).name})"
        sql = f"SELECT {label.name}, {agg} FROM {table.name}{where} GROUP BY {label.name}"
        return sql

    if wants_count:
        if wants_distinct and Skill.DISTINCT in skills:
            target = mentioned or _label_column(table)
            return f"SELECT count(DISTINCT {target.name}) FROM {table.name}{where}"
        return f"SELECT count(*) FROM {table.name}{where}"

    if wants_avg:
        num = numeric_col or next((c for c in table.columns if c.is_numeric), table.columns[-1])
        return f"SELECT avg({num.name}) FROM {table.name}{where}"

    projection = (mentioned.name if mentioned and not mentioned.is_numeric else label.name)
    sql = f"SELECT {projection} FROM {table.name}{where}"

    if (wants_max or wants_min) and Skill.ORDER in skills:
        order_col = numeric_col or next(
            (c for c in table.columns if c.is_numeric), table.columns[-1]
        )
        direction = "DESC" if wants_max else "ASC"
        sql += f" ORDER BY {order_col.name} {direction}"
        if Skill.LIMIT in skills:
            n = int(number) if number is not None and number < 100 else 1
            sql += f" LIMIT {n}"
    elif wants_alpha and Skill.ORDER in skills:
        sql += f" ORDER BY {projection}"

    return sql


class MockClient:
    """Offline stand-in implementing the :class:`ModelClient` interface."""

    def __init__(self, params: GenerationParams | None = None) -> None:
        self.model_id = MOCK_MODEL
        self.params = params or GenerationParams()
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        system_l = (system_prompt or "").lower()

        # Reflection roles: the analyzer and proposer use fixed system prompts
        # that are not the evolvable text-to-SQL instruction. Route those first
        # so they do not fall through into SQL generation.
        if "delta's analyzer" in system_l or "delta analyzer" in system_l:
            text = _mock_analyze(user_prompt)
        elif "delta's proposer" in system_l or "delta proposer" in system_l:
            text = _mock_propose(user_prompt)
        else:
            skills = available_skills(system_prompt)
            question = _question_from_prompt(user_prompt)
            sql = generate_sql(question, user_prompt, skills)
            text = f"```sql\n{sql}\n```"

        return LLMResponse(
            text=text,
            model_id=self.model_id,
            input_tokens=len(system_prompt.split()) + len(user_prompt.split()),
            output_tokens=len(text.split()),
            latency_ms=0.0,
            cached=False,
            stop_reason="end_turn",
        )


def _mock_analyze(user_prompt: str) -> str:
    """Deterministic diagnosis from the failure dump the analyzer sends."""
    text = user_prompt.lower()
    modes: list[str] = []
    recs: list[str] = []

    # Infer missing skills from what the current prompt section contains.
    prompt_section = user_prompt
    if "-----" in user_prompt:
        parts = user_prompt.split("-----")
        if len(parts) >= 2:
            prompt_section = parts[1]
    prompt_skills = available_skills(prompt_section)

    if Skill.JOIN not in prompt_skills and (
        "join" in text or "dept_name" in text or "wrong_rows" in text
    ):
        modes.append("multi-table questions fail because the prompt never mentions JOINs")
        recs.append("Tell the model to JOIN tables on foreign keys when the question spans entities")
    if Skill.GROUP not in prompt_skills and ("each" in text or "group" in text or "per " in text):
        modes.append("aggregation-per-group questions fail without GROUP BY guidance")
        recs.append("When the question says each/per, require GROUP BY on that column")
    if Skill.ORDER not in prompt_skills and (
        "highest" in text or "lowest" in text or "order" in text or "top " in text
    ):
        modes.append("ranking questions fail without ORDER BY / LIMIT guidance")
        recs.append("For highest/lowest/top-N, require ORDER BY with LIMIT")
    if Skill.SUBQUERY not in prompt_skills and (
        "average" in text or "subquery" in text or "not in" in text
    ):
        modes.append("comparisons to aggregates fail without subquery guidance")
        recs.append("When comparing against an average or aggregate, use a subquery")
    if Skill.DISTINCT not in prompt_skills and (
        "distinct" in text or "different" in text or "unique" in text
    ):
        modes.append("distinct-count questions fail without DISTINCT guidance")
        recs.append("Use COUNT(DISTINCT x) when the question asks how many different things")
    if "nothing extracted" in text or "pred_failed" in text:
        modes.append("some answers are wrapped in prose instead of bare SQL")
        recs.append("Require output to be only the SQL query, with no explanation")

    if not modes:
        modes.append("residual wrong-row errors on harder questions")
        recs.append("Add SQLite dialect and column-selection guidance")

    payload = {
        "summary": "The system prompt is missing concrete SQL skills that show up in the failures.",
        "failure_modes": modes[:5],
        "recommendations": recs[:5],
    }
    return json.dumps(payload)


def _mock_propose(user_prompt: str) -> str:
    """Build a stronger prompt by unlocking skills the diagnosis asked for."""
    # Start from the current prompt block if present.
    current = ""
    if "-----" in user_prompt:
        parts = user_prompt.split("-----")
        if len(parts) >= 2:
            current = parts[1].strip()

    base = current or (
        "You are an expert SQLite analyst. Given a database schema and a question, "
        "write one SQL query that answers it."
    )
    lowered = user_prompt.lower()
    additions: list[str] = []
    if "join" in lowered and "join" not in base.lower():
        additions.append(
            "When the question spans entities in different tables, JOIN them on their foreign keys."
        )
    if "group" in lowered and "group by" not in base.lower():
        additions.append(
            "When the question says \"each\" or \"per\", use GROUP BY on that grouping column."
        )
    if ("order" in lowered or "limit" in lowered or "highest" in lowered) and (
        "order by" not in base.lower()
    ):
        additions.append(
            "When the question asks for the highest, lowest, top, or first, use ORDER BY with LIMIT."
        )
    if "subquery" in lowered and "subquery" not in base.lower():
        additions.append(
            "When comparing against an aggregate such as an average, use a subquery."
        )
    if "distinct" in lowered and "distinct" not in base.lower():
        additions.append(
            "Use COUNT(DISTINCT x) when the question asks how many different or distinct things there are."
        )
    if "only the sql" not in base.lower() and "no explanation" not in base.lower():
        additions.append(
            "Output only the SQL query, with no explanation and no markdown fences. Use SQLite syntax."
        )

    if not additions:
        return base
    return base.rstrip() + "\n\nRules:\n- " + "\n- ".join(additions)


def _question_from_prompt(user_prompt: str) -> str:
    """Recover the question from the rendered user message."""
    m = re.search(r"(?:Question|Q):\s*(.+?)(?:\n\s*\n|\Z)", user_prompt, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return user_prompt.strip().splitlines()[-1] if user_prompt.strip() else ""
