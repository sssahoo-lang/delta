# Related work

Delta does **not** claim the propose/evaluate/keep loop. That lineage is well
established; this document is the short map a reviewer will expect.

| Work | Year | Contribution |
|---|---|---|
| [APE](https://arxiv.org/abs/2211.01910) | ICLR 2023 | Treats the instruction as a program: LLM proposes candidates, select by score |
| [OPRO](https://arxiv.org/abs/2309.03409) | ICLR 2024 | LLM as optimizer over prior solutions and their scores |
| [PromptBreeder](https://arxiv.org/abs/2309.16797) | 2023 | Evolves task-prompts *and* the mutation-prompts |
| [TextGrad](https://www.nature.com/articles/s41586-025-08661-4) | Nature 2025 | Backpropagates textual feedback through a computation graph |
| DSPy MIPROv2 | ongoing | Bayesian optimization over joint instruction and demo space |
| [GEPA](https://arxiv.org/abs/2507.19457) | ICLR 2026 Oral | Reflective evolution over a Pareto frontier; ships in `dspy` 3.2.1 |
| [ADAS](https://arxiv.org/abs/2408.08435) | ICLR 2025 | Meta agent writes new agent *code*, not just prompts |
| [Darwin Godel Machine](https://arxiv.org/abs/2505.22954) | 2025 | Agent edits its own codebase |

## Honest framing

> Delta does not contribute the idea. It contributes a rigorous implementation of
> it in a specific domain, measured against the state of the art, including an
> ablation that tests whether reflection beats undirected random sampling at all.

**The deliverable is the comparison table, not the loop.** Six conditions on
identical splits with identical scoring: weak baseline, human hand-tuned prompt,
random search, DSPy MIPROv2, DSPy GEPA, and Delta's own optimizer.

If GEPA wins, that gets reported. Implementing and fairly evaluating against an
ICLR Oral method is the accomplishment; a suspicious win would not be.

## Why not "just use DSPy"?

Because the project's thesis is measurement discipline, not a new search
algorithm: a statistical confirmatory test on held-out data, per-difficulty
regression guards, a random-search ablation, and a database-disjoint split so a
prompt cannot be tuned to a schema it is later scored on. DSPy is a baseline in
that table, not a substitute for it.
