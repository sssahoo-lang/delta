# Delta

> **Work in progress — not finished.**  
> Phases 0–3 are largely in place (measurement, splits, stats, analyzer/proposer).  
> The full optimization loop, DSPy/GEPA comparison table, and final results are **not done yet**.  
> Treat numbers below as interim findings, not a completed evaluation.

A self-improving setup that tries to improve a **text-to-SQL** agent's system prompt by
looking at failures, proposing a new prompt, and only trusting improvements that survive
honest measurement on held-out data.

Named for what it measures: the **delta**, with a confidence interval.

---

## Status (honest)

| Area | State |
|---|---|
| Execution + scoring harness | Done |
| Target agent (SQL writer) | Done |
| Seeded train/val/test splits | Done |
| Bootstrap / McNemar / acceptance primitives | Done |
| Analyzer + proposer agents | Done (offline mock OK; live Gemini checkpoint pending) |
| Full optimization loop (Phase 4) | **Not done** |
| Random search + DSPy MIPROv2 + GEPA comparison | **Not done** |
| Final `results/comparison.md` table | **Not done** |

**Interim real-model finding** (100 stratified Spider examples, Groq `llama-3.1-8b-instant`):

| Condition | Accuracy |
|---|---|
| Weak seed prompt (v0) | **75%** |
| Hand-tuned human prompt | **74%** |
| Gap | **−1%** |

The original “weak → human ≥10 points headroom” gate **failed**. Llama already follows a
minimal prompt well enough that a hand-written prompt does not beat it on this sample. The
project plan was adjusted: evaluate the optimizer against random search / DSPy from the weak
seed; treat hand-tuned as a reference line, not proof of headroom. See
[`results/real_path.json`](results/real_path.json).

---

## What this project is trying to do

1. A **target agent** turns a natural-language question + database schema into SQL.
2. We **execute** the SQL and check it against gold (execution accuracy — objective, no LLM judge).
3. An **analyzer** reads failures and writes a short diagnosis.
4. A **proposer** drafts a revised system prompt.
5. Search keeps candidates that improve validation without tanking a difficulty bucket.
6. At the end, a **paired statistical test** on a held-out test set is the confirmatory check.
7. The same splits/scoring are meant to compare: weak seed, human prompt, random search,
   DSPy MIPROv2, DSPy GEPA, and Delta’s loop.

The propose/evaluate/keep loop is **not claimed as novel** (APE → OPRO → DSPy/GEPA). The
deliverable is a careful comparison, including the chance that reflection does not beat
undirected search. See [docs/RELATED_WORK.md](docs/RELATED_WORK.md).

---

## Architecture

```mermaid
flowchart TD
    subgraph data [Data]
        train[Train pool]
        val[Validation]
        test[Held-out test]
    end

    subgraph target [Target agent]
        prompt[Evolvable system prompt]
        agent[Strands Agent via LiteLLM]
        sql[Generated SQL]
    end

    subgraph harness [Evaluation]
        exec[Read-only SQLite + timeout]
        score[Execution accuracy]
        stats[Bootstrap + McNemar]
    end

    subgraph smith [Optimizer - partial]
        analyzer[Analyzer]
        proposer[Proposer]
        gate[Acceptance gate]
        loop[Full loop - TODO]
    end

    train --> agent
    prompt --> agent
    agent --> sql --> exec --> score
    score --> analyzer --> proposer --> prompt
    score --> gate
    val --> gate
    gate -->|accept / reject| prompt
    test --> stats --> results[results/comparison.md TODO]
```

**Only the system prompt evolves.** Schema + question formatting stay fixed so accuracy
differences can be attributed to the instruction text.

### Main packages

| Path | Role |
|---|---|
| `delta/evalh/` | Dataset loaders, SQLite execution, scoring, splits, buckets, stats |
| `delta/target_agent/` | Prompt versions, schema render, thin SQL agent |
| `delta/llm/` | Providers (Strands/LiteLLM), disk cache, mock, budget/TPM pacing |
| `delta/optimizer/` | Gate, analyzer, proposer (loop not wired yet) |
| `scripts/` | Download Spider, baseline, real-path check, reflect checkpoint |
| `docs/` | Project log, cost model, related work |
| `data/splits.json` | Seeded train / val / test ids (val/test disjoint by database) |

---

## Plan (phases)

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Harness, scoring, offline fixture, Spider downloader | Done (`v0.1`) |
| 1 | Target agent, eval harness, mock baseline | Done |
| 2 | Splits, difficulty buckets, stats, gate primitives | Done (`v0.2`) |
| 3 | Analyzer + proposer | Mostly done |
| 4 | Pareto archive + optimization loop | **TODO** |
| 5 | Random / MIPROv2 / GEPA comparison table | **TODO** |
| 6–7 | Routing polish, docs/results | Partial |

Longer narrative: [docs/PROJECT.md](docs/PROJECT.md). Cost notes: [docs/COST.md](docs/COST.md).

---

## Stack

- Python 3.11
- Spider 1.0 (execution accuracy; downloaded, not vendored)
- Target model: Groq free tier `llama-3.1-8b-instant`
- Reflection (planned): Gemini Flash free tier; optional Anthropic
- Strands Agents + LiteLLM
- sqlglot, pytest, ruff

Cost target: **$0** on free tiers. Binding Groq limit is **tokens** (6k TPM / 500k TPD).

---

## Quickstart (offline)

```bash
make install
source .venv/bin/activate
make sample-db
make test
make baseline MOCK=1
make reflect MOCK=1
```

## Real API runs

1. Copy `.env.example` → `.env` (never commit `.env`).
2. Set `GROQ_API_KEY` from [console.groq.com](https://console.groq.com) (keys look like `gsk_...`).
3. Optional later: `GEMINI_API_KEY` for reflection.
4. Download Spider and run checks:

```bash
make spider
python scripts/run_real_path.py --n 100 --seed 0 --save
python scripts/run_reflect_checkpoint.py   # needs reflection key unless --mock
```

---

## Design choices that matter

1. **Decoupled gate** — permissive rule during search; strict bootstrap/McNemar once on test.
2. **Database-disjoint val/test** — a prompt cannot be tuned to a schema it is later scored on.
3. **Per-difficulty regression guards** — no trading easy questions for hard ones unnoticed.
4. **Random-search ablation** (planned) — does reflection beat undirected sampling?
5. **Mock is for CI only** — mock “improvements” are skill-keyword triggers, not real results.

---

## License

MIT for this code. Spider is CC BY-SA 4.0 and is downloaded, not vendored.
