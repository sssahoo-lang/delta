"""The evolvable prompt, and its version lineage.

Only the **system prompt** evolves. The user message (schema plus question) is
rendered identically for every condition, so any measured accuracy difference is
attributable to the instruction text and nothing else.

The v0 prompt is deliberately weak. It says what the task is and nothing about
how to do it: no mention of grouping, joining, ordering, or any of the failure
modes that actually cost accuracy. That is not laziness, it is the experimental
setup. An optimizer needs real, diagnosable failures to find, and a strong
starting prompt would leave nothing to discover and no headroom to measure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The character cap is a regularizer, not cosmetics. Published ablations on
# reflective prompt optimization find that without a length limit the proposer
# appends caveats indefinitely, inflating cost and eventually hurting accuracy.
MAX_PROMPT_CHARS = 2400


@dataclass(frozen=True)
class PromptVersion:
    """One candidate instruction, with the lineage needed to audit the search."""

    text: str
    version_id: str
    parent_id: str | None = None
    origin: str = "seed"  # seed | proposer | random | handtuned
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make(
        text: str,
        parent_id: str | None = None,
        origin: str = "proposer",
        notes: str = "",
        metadata: dict | None = None,
    ) -> PromptVersion:
        text = text.strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return PromptVersion(
            text=text,
            version_id=digest,
            parent_id=parent_id,
            origin=origin,
            notes=notes,
            metadata=metadata or {},
        )

    @property
    def char_count(self) -> int:
        return len(self.text)

    def within_cap(self, cap: int = MAX_PROMPT_CHARS) -> bool:
        return self.char_count <= cap

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict) -> PromptVersion:
        return PromptVersion(
            text=payload["text"],
            version_id=payload["version_id"],
            parent_id=payload.get("parent_id"),
            origin=payload.get("origin", "unknown"),
            notes=payload.get("notes", ""),
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @staticmethod
    def load(path: str | Path) -> PromptVersion:
        return PromptVersion.from_dict(json.loads(Path(path).read_text()))


V0_WEAK_TEXT = """You are a helpful assistant. The user will give you a database schema and a question. Reply with a SQL query."""


V0_WEAK = PromptVersion.make(
    V0_WEAK_TEXT,
    origin="seed",
    notes=(
        "Deliberately minimal baseline. States the task and nothing else: no "
        "output-format contract, no dialect, no guidance on aggregation, joins, "
        "ordering, or column selection. Exists to create measurable headroom."
    ),
)


# Written by hand before seeing any optimizer output, so it is a fair human
# baseline rather than a post-hoc reconstruction of what the optimizer found.
HANDTUNED_TEXT = """You are an expert SQLite analyst. Given a database schema and a question, write one SQL query that answers it.

Rules:
- Output only the SQL query, with no explanation and no markdown fences.
- Use SQLite syntax.
- Select exactly the columns the question asks for, nothing extra.
- When the question says "each" or "per", use GROUP BY on that grouping column.
- When the question spans entities in different tables, JOIN them on their foreign keys.
- When the question asks for the highest, lowest, top, or first, use ORDER BY with LIMIT.
- When the question asks to compare against an aggregate such as an average, use a subquery.
- Use COUNT(DISTINCT x) when the question asks how many different or distinct things there are.
- Match text values exactly as they appear in the question."""

V_HANDTUNED = PromptVersion.make(
    HANDTUNED_TEXT,
    origin="handtuned",
    notes="Human baseline, authored before any optimizer run.",
)
