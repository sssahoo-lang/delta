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

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a million": 1_000_000, "one million": 1_000_000,
}

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL
)


@dataclass
class Column:
    name: str
    type: str = "TEXT"

    @property
    def is_numeric(self) -> bool:
        return self.type.upper() in {"INTEGER", "REAL", "NUMERIC", "FLOAT", "DECIMAL"}


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def find(self, *candidates: str) -> Column | None:
        lowered = {c.name.lower(): c for c in self.columns}
        for cand in candidates:
            if cand.lower() in lowered:
                return lowered[cand.lower()]
        return None


def parse_schema(rendered: str) -> list[Table]:
    """Recover tables and columns from the rendered CREATE TABLE schema."""
    tables: list[Table] = []
    for match in _CREATE_TABLE_RE.finditer(rendered):
        name = match.group(1)
        columns: list[Column] = []
        for line in match.group(2).splitlines():
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
            tables.append(Table(name=name, columns=columns))
    return tables


def available_skills(system_prompt: str) -> set[str]:
    text = (system_prompt or "").lower()
    return {skill for skill, phrases in SKILL_TRIGGERS.items() if any(p in text for p in phrases)}


def _extract_number(question: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", question)
    if m:
        return float(m.group(1))
    lowered = question.lower()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return float(value)
    return None


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
        # Skip the sentence-initial word, which is capitalized by grammar not by name.
        candidates = quoted + [p for p in proper if not question.strip().startswith(p)]
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
        skills = available_skills(system_prompt)

        question = _question_from_prompt(user_prompt)
        sql = generate_sql(question, user_prompt, skills)

        return LLMResponse(
            text=f"```sql\n{sql}\n```",
            model_id=self.model_id,
            # Deterministic stand-in for token counts so cost accounting has
            # something coherent to aggregate in mock runs.
            input_tokens=len(system_prompt.split()) + len(user_prompt.split()),
            output_tokens=len(sql.split()),
            latency_ms=0.0,
            cached=False,
            stop_reason="end_turn",
        )


def _question_from_prompt(user_prompt: str) -> str:
    """Recover the question from the rendered user message."""
    m = re.search(r"(?:Question|Q):\s*(.+?)(?:\n\s*\n|\Z)", user_prompt, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return user_prompt.strip().splitlines()[-1] if user_prompt.strip() else ""
