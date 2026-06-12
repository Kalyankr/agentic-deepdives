# 13 · Take-Home & Portfolio Guide

> Many frontier-lab loops include a **take-home** (or a live build) and a **project deep-dive**. These
> reward *engineering judgment* — scoping, testing, evaluation, and communication — far more than a
> clever trick. This file covers how to ace the take-home, what portfolio projects signal seniority,
> and how to present them.

---

## What take-homes actually look like

Common Applied/ML take-home shapes (4–8 hours, sometimes a weekend):
- **Build an eval harness** for some capability (summarization faithfulness, tool-use correctness) and
  report results with uncertainty.
- **Build a mini-agent or RAG pipeline** that solves a concrete task end-to-end.
- **Fine-tune / analyze** a small model and explain what changed and why.
- **Implement a core algorithm** (a sampler, an attention variant, a retrieval index) with tests.
- **Data/analysis task:** find signal in a messy dataset and defend your methodology.

> They're deliberately under-specified. **Stating assumptions and scoping** is part of the test.

## The rubric reviewers actually use

| Dimension | What "strong" looks like |
|-----------|--------------------------|
| **Correctness** | It runs from a clean clone; does what the prompt asked |
| **Code quality** | Readable, typed, modular; no dead code; sensible structure |
| **Tests** | Meaningful tests for the core logic; runnable in one command |
| **Evaluation** | You **measured** quality with a metric + uncertainty, not vibes |
| **Judgment** | Right scope; explicit trade-offs; you cut the right corners |
| **Communication** | A README that states the problem, decisions, results, and limits |
| **Reproducibility** | Pinned deps, a seed, one-command setup + run |

> The single biggest differentiator: a **README that explains your decisions and limitations**. Average
> submissions show code; strong submissions show *thinking*.

## A take-home playbook

1. **Scope first (15 min).** Restate the problem in your own words; list assumptions; pick a thin
   **end-to-end slice** to get working before adding depth. Write these in the README up front.
2. **Make it run, then make it good.** Get a working baseline early; iterate. A complete simple
   solution beats a half-finished sophisticated one.
3. **Measure.** Define a metric and a tiny eval set on day one; report a **confidence interval**
   ([12-math-stats](12-math-stats.md)). "Faithfulness 82% ± 4% on 60 cases" >> "works well."
4. **Test the core.** A few targeted tests on the conceptual heart signal seniority more than 100%
   coverage of glue.
5. **Handle failure.** Show you thought about errors, edge cases, latency, and cost — even if you only
   *note* them as "what I'd do next."
6. **Write the README last, read it first.** Problem → how to run → key decisions → results (with
   numbers) → limitations & next steps → time spent.
7. **Reproducibility.** Pin dependencies, set a seed, one command to set up and one to run.

## Anti-patterns that sink take-homes

- No README, or a README with no **decisions/limitations**.
- It doesn't run from a clean clone (missing deps, hard-coded paths, no seed).
- "It works" with **no evaluation** or a single number with no uncertainty.
- Over-engineering: a plugin framework for a one-off script; gold-plating instead of finishing.
- Ignoring the prompt's actual ask to show off something unrelated.
- Leaking test data into training / fitting preprocessing before the split ([11-debugging](11-debugging.md)).

---

## A portfolio that signals seniority

Three projects cover the surface area frontier labs care about. **This repo already gives you all
three** — finish the TODOs, write up the results, and they're portfolio-ready:

1. **From-scratch fundamentals** → [lab01 micrograd](../labs/lab01_micrograd/) + [lab02 nanoGPT](../labs/lab02_nanogpt/).
   *Signal:* you understand autograd and attention at the mechanism level.
2. **An applied LLM system** → [lab06 RAG](../labs/lab06_rag/) + [lab07 agent](../labs/lab07_agent/).
   *Signal:* you can build retrieval/agents with metrics and guardrails.
3. **A systems/perf or eval artifact** → [lab04 inference benchmark](../labs/lab04_inference_bench/)
   or a **CI eval gate** (see the [notebooks](../notebooks/README.md)).
   *Signal:* you reason in numbers — latency, throughput, cost, regression detection.

What turns a repo into a *portfolio piece*:
- A crisp README: what it is, what you learned, a **result with numbers**, and limitations.
- Tests that run in one command; a clean clone works.
- A short "**decisions & trade-offs**" section — the part interviewers actually read.
- Bonus: a one-paragraph **writeup** of a surprising finding (e.g. "GQA cut my KV cache 4× with <1% quality loss").

## The project deep-dive round (present like a senior)

Use a project-flavored **STAR** ([06-behavioral-mission](06-behavioral-mission.md)):
- **Context & constraints:** what, for whom, what mattered (latency? cost? accuracy?).
- **Your decisions:** the 2–3 real forks and **why you chose** each (name the trade-off).
- **Measurement:** how you knew it worked — the metric, the baseline, the delta + uncertainty.
- **Result & impact:** the number, the outcome, what you'd do with more time.
- **Reflection:** the hardest bug, what you'd change, what you learned.

Prepare for drill-down: *"Why that chunk size? What if traffic 10×? How did you measure quality? What
broke? What would you change?"* Have **numbers** ready and own the limitations — defensiveness reads
worse than a known gap.

> Reframe: the take-home and deep-dive aren't about a perfect artifact — they're evidence of how you'd
> **operate on the team**. Scope tightly, measure honestly, communicate clearly, and show a safety/eval
> reflex. The fastest prep is to finish a [lab](../labs/README.md), write its README like a submission,
> and run a mock deep-dive on it ([08-mock-interview](08-mock-interview.md)).
