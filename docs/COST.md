# Cost model

Total recurring cost target: **$0**, on free tiers that do not require a credit
card.

## Provider limits (verified August 2026)

| Role | Model | Free tier | What binds |
|---|---|---|---|
| Target agent | `groq/llama-3.1-8b-instant` | 30 RPM, 14,400 RPD, **6,000 TPM, 500,000 TPD** | Tokens |
| Analyzer + proposer | Gemini Flash | 10 RPM, 1,500 RPD | Requests (few calls) |
| Offline mock | deterministic, local | unlimited | CI and development |

The original plan tracked requests. That was wrong. Measured over Spider dev, a
full v0 prompt averages ~296 tokens (chars/4 estimate; see
`results/token_distribution.json`). Against Groq's 500k TPD that is roughly
**1,700 uncached calls/day**, not 14,400. The 6k TPM cap similarly binds before
the 30 RPM cap.

## Prefix caching is what makes the project fit

Groq caches identical prompt prefixes automatically and
[cached tokens do not count toward rate limits](https://console.groq.com/docs/prompt-caching).
The rendered message is `[system prompt][schema][question]`. Spider dev has only
20 databases, so evaluating examples grouped by `db_id` (see
`delta.evalh.evaluate.group_by_database`) means each schema is paid for once per
candidate instead of once per question.

Measured one-pass costs on Spider dev:

| Mode | Estimated tokens |
|---|---|
| No prefix cache | ~306k |
| With prefix cache (schema once/db) | ~24k |

That is an ~92% reduction in billable tokens for a single pass, and it is why a
full six-condition experiment fits in roughly a week of free quota rather than
months.

The on-disk response cache (`delta.llm.cache`) is orthogonal: it makes **reruns**
free, but during search every candidate is a fresh system prompt, so disk cache
hits do not help the search itself.

## Right-sized experiment

| Split | Size | Source |
|---|---|---|
| Train | 100 | Spider train, stratified |
| Validation | ~350 | Dev databases (disjoint from test) |
| Val screen | 60 | Stratified subset of validation |
| Test | ~400 | Remaining assigned dev databases |

Leftover dev databases stay unused so the targets stay near 350/400 despite
coarse database sizes. See `data/splits.json`.

Screening (train minibatch + 60-example val screen, promote top third) keeps most
candidates off the full validation set.

## Fallback

If Gemini Flash proposal quality collapses, a few dollars of Claude Sonnet is
available behind `STRONG_REFLECTION_MODEL`. The provider layer already supports
it; it is never required for the core loop.
