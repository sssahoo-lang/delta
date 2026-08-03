# Delta

A self-improving agent that optimizes a text-to-SQL agent's prompt through reflective failure
analysis, and accepts a change **only when a statistical test on held-out data says it helped**.

Named for the thing it measures: the delta, with a confidence interval.

> **Status:** Phases 0–2 complete of 8. Measurement core, seeded splits, stats, and
> decoupled acceptance gate are in place. Optimizer agents (analyzer/proposer) are next.
> Tag `v0.2`.

## The idea in one paragraph

A target agent writes SQL. Most portfolio prompt-tuning projects stop at "I changed the prompt and
the number went up," which is usually noise. Delta instead treats each proposed prompt as a
hypothesis: an analyzer agent reads the failing traces and diagnoses *why* they fail, a proposer
agent drafts a candidate prompt, and an acceptance gate decides whether to keep it. Search uses a
permissive rule (point improvement, no per-difficulty regression); the paired bootstrap and McNemar
test run once on the held-out test set. The optimizer is then benchmarked head-to-head against
DSPy's MIPROv2 and GEPA on identical splits.

## Honest positioning

The propose/evaluate/keep loop is **not novel**. It is the contract of every DSPy optimizer, it
goes back to [APE (ICLR 2023)](https://arxiv.org/abs/2211.01910), and its current state of the art
is [GEPA (ICLR 2026 Oral)](https://arxiv.org/abs/2507.19457), which ships inside `dspy`. This
repository does not claim the idea. It claims a rigorous implementation and an honest measurement
against those baselines, including a random-search ablation that tests whether reflection beats
undirected sampling at all. See [docs/RELATED_WORK.md](docs/RELATED_WORK.md).

## Quickstart

Runs fully offline, no API key required, using a mock model and a built-in SQLite fixture:

```bash
make install
source .venv/bin/activate
make sample-db
make test
make baseline MOCK=1
```

For real runs you need two free API keys, neither requiring a credit card:
[console.groq.com](https://console.groq.com) and
[aistudio.google.com](https://aistudio.google.com). Copy `.env.example` to `.env` and fill them in.

Then:

```bash
make spider
python scripts/run_real_path.py    # 100 stratified examples; headroom gate
```

## Cost

$0 on free tiers. The binding Groq limit is **tokens** (6k TPM / 500k TPD), not requests.
Evaluation groups examples by database so Groq's automatic prefix cache absorbs the schema cost.
See [docs/COST.md](docs/COST.md) and `results/token_distribution.json`.

## Benchmark

[Spider 1.0](https://yale-lily.github.io/spider) dev, execution accuracy. Spider was chosen over
BIRD and Spider 2.0 deliberately: a [CIDR 2026 paper](https://www.cidrdb.org/cidr2026/papers/p5-jin.pdf)
found annotation error rates of 52.8% in BIRD Mini-Dev and 62.8% in Spider 2.0-Snow, which makes
small measured deltas on those sets untrustworthy.

Seeded splits live in [`data/splits.json`](data/splits.json): 100 train / ~350 val / ~400 test,
with validation and test disjoint by database.

## Project log

The full plan, architecture, and implementation notes are in [docs/PROJECT.md](docs/PROJECT.md).

## License

MIT for this code. Spider is CC BY-SA 4.0 and is downloaded, not vendored.
