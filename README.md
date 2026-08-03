# Delta

A self-improving agent that optimizes a text-to-SQL agent's prompt through reflective failure
analysis, and accepts a change **only when a statistical test on held-out data says it helped**.

Named for the thing it measures: the delta, with a confidence interval.

> **Status: under construction.** Phase 0 of 8. Results table lands in Phase 5.

## The idea in one paragraph

A target agent writes SQL. Most portfolio prompt-tuning projects stop at "I changed the prompt and
the number went up," which is usually noise. Delta instead treats each proposed prompt as a
hypothesis: an analyzer agent reads the failing traces and diagnoses *why* they fail, a proposer
agent drafts a candidate prompt, and an acceptance gate runs a paired bootstrap and a McNemar test
on a held-out set. Candidates that cannot clear significance, or that regress any difficulty
bucket, are rejected and logged with the reason. The optimizer is then benchmarked head-to-head
against DSPy's MIPROv2 and GEPA on identical splits.

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

## Cost

$0. The target agent runs on Groq's free tier (`llama-3.1-8b-instant`, 14,400 requests/day) and the
reflection agents on Gemini's free tier (1,500 requests/day). Every model response is cached to
disk by content hash, so reruns are free and deterministic. See [docs/COST.md](docs/COST.md).

## Benchmark

[Spider 1.0](https://yale-lily.github.io/spider) dev, execution accuracy. Spider was chosen over
BIRD and Spider 2.0 deliberately: a [CIDR 2026 paper](https://www.cidrdb.org/cidr2026/papers/p5-jin.pdf)
found annotation error rates of 52.8% in BIRD Mini-Dev and 62.8% in Spider 2.0-Snow, which makes
small measured deltas on those sets untrustworthy.

## License

MIT for this code. Spider is CC BY-SA 4.0 and is downloaded, not vendored.
