"""The agent under optimization: question plus schema in, SQL out.

Kept deliberately thin. The whole point of the project is that the *prompt* is the
variable under study, so this class holds no cleverness of its own: no retries on
bad SQL, no self-correction, no few-shot injection. Any of those would improve
accuracy in ways that confound the measurement, because an accuracy gain could
then come from the scaffold rather than the prompt being optimized.

The one non-trivial piece is :func:`extract_sql`. Models wrap SQL in markdown
fences, prefix it with "Here's the query:", and append explanations, none of which
is the model's fault when the prompt never forbade it. Parsing that leniently
matters: if extraction failed on well-formed answers, the v0 baseline would be
artificially low and every later improvement would be measuring the parser
getting lucky rather than the prompt getting better.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from delta.config import GenerationParams
from delta.llm.providers import LLMResponse, ModelClient
from delta.target_agent.prompt import PromptVersion
from delta.target_agent.schema_render import render_user_message

_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_START_RE = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)

# Lines that clearly begin prose rather than SQL. Used to trim trailing
# commentary when the model did not terminate the query with a semicolon.
_PROSE_PREFIXES = (
    "this query",
    "this sql",
    "the query",
    "note:",
    "note that",
    "explanation",
    "here,",
    "in this",
    "it returns",
    "this returns",
    "this will",
    "assumption",
)

# Markdown decoration the model wraps prose in. Stripped before matching the
# prefixes above, since Llama writes "**Explanation:**" far more often than a
# bare "Explanation:" and the bold markers would otherwise defeat the match.
_MARKDOWN_DECORATION = "*_`# \t"

# A trailing explanation is often a bulleted list rather than a sentence. Real
# SQL lines never start this way once leading whitespace is removed.
_BULLET_RE = re.compile(r"^(?:[-*+]\s+\w|\d+\.\s+\w)")


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of arbitrary model output.

    Returns an empty string when nothing SQL-shaped is present, which the scorer
    treats as an incorrect answer rather than an error.
    """
    if not text:
        return ""

    candidate = text.strip()

    # A fenced block is the strongest signal, so prefer it when present.
    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    # Drop any lead-in before the statement actually starts.
    start = _SQL_START_RE.search(candidate)
    if not start:
        return ""
    candidate = candidate[start.start() :]

    # A semicolon ends the statement; anything after it is commentary or a
    # second query we do not want.
    semicolon = _statement_end(candidate)
    if semicolon != -1:
        candidate = candidate[:semicolon]
    else:
        candidate = _trim_trailing_prose(candidate)

    return " ".join(candidate.split()).strip()


def _statement_end(candidate: str) -> int:
    """Index of the semicolon ending the statement, or -1.

    Semicolons inside string literals do not end anything. Truncating at one
    would silently corrupt an otherwise correct query into a syntax error, which
    the scorer would then charge to the prompt.
    """
    in_string = False
    i = 0
    while i < len(candidate):
        char = candidate[i]
        if char == "'":
            # SQL escapes a quote by doubling it, so '' stays inside the string.
            if in_string and i + 1 < len(candidate) and candidate[i + 1] == "'":
                i += 2
                continue
            in_string = not in_string
        elif char == ";" and not in_string:
            return i
        i += 1
    return -1


def _is_prose_line(line: str) -> bool:
    stripped = line.strip().lstrip(_MARKDOWN_DECORATION).strip().lower()
    if not stripped:
        return False
    return stripped.startswith(_PROSE_PREFIXES) or bool(_BULLET_RE.match(line.strip()))


def _trim_trailing_prose(candidate: str) -> str:
    kept: list[str] = []
    for line in candidate.splitlines():
        if _is_prose_line(line):
            break
        # A blank line followed by prose is the usual shape of an explanation.
        if not line.strip() and kept:
            continue
        kept.append(line)
    return "\n".join(kept)


@dataclass
class GenerationResult:
    """One SQL generation, with everything needed for tracing and cost accounting."""

    sql: str
    raw_text: str
    response: LLMResponse

    @property
    def extraction_failed(self) -> bool:
        return not self.sql

    @property
    def input_tokens(self) -> int:
        return self.response.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.response.output_tokens

    @property
    def latency_ms(self) -> float:
        return self.response.latency_ms

    @property
    def cached(self) -> bool:
        return self.response.cached

    @property
    def cache_read_tokens(self) -> int:
        return self.response.cache_read_tokens


class TargetAgent:
    """A text-to-SQL agent parameterized by the prompt under optimization."""

    def __init__(
        self,
        client: ModelClient,
        prompt: PromptVersion,
        sample_rows: int = 0,
        params: GenerationParams | None = None,
    ) -> None:
        self.client = client
        self.prompt = prompt
        self.sample_rows = sample_rows
        self.params = params or GenerationParams()

    @property
    def model_id(self) -> str:
        return self.client.model_id

    def with_prompt(self, prompt: PromptVersion) -> TargetAgent:
        """A sibling agent differing only in prompt.

        Candidate prompts are evaluated constantly, and sharing the client keeps
        the response cache and its accumulated hits intact across candidates.
        """
        return TargetAgent(
            client=self.client,
            prompt=prompt,
            sample_rows=self.sample_rows,
            params=self.params,
        )

    def generate_sql(self, db_path: str, question: str) -> GenerationResult:
        user_message = render_user_message(db_path, question, self.sample_rows)
        response = self.client.complete(self.prompt.text, user_message)
        return GenerationResult(
            sql=extract_sql(response.text),
            raw_text=response.text,
            response=response,
        )
