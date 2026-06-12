# Module 09 · Evaluations

> **Goal:** Build rigorous evaluation systems for models, RAG, and agents. Evals are the single most important skill at a frontier lab — they're how you know if anything is actually working, catch regressions, and measure safety. "You can't improve what you can't measure."

**Duration:** ~4 weeks. **Prereqs:** [Module 03](03-llm-training-rlhf-dpo.md), [Module 07](07-agentic-systems.md).

---

## 9.1 Why evals are the job

- Models are stochastic; vibes don't scale. Evals turn "feels better" into a number.
- Evals gate releases, detect regressions, drive RLHF/DPO, and quantify safety.
- The flywheel: **logs → error analysis → eval cases → fix → re-eval**.
- Frontier labs treat eval design as a core research/engineering competency.

## 9.2 Foundations of measurement

- Metrics vs. benchmarks vs. evals; offline vs. online
- Statistical rigor: sample size, confidence intervals, variance across seeds, significance testing
- Pairwise comparison & **Elo / Bradley-Terry** ranking (Chatbot Arena style)
- Contamination/leakage, overfitting to benchmarks ("teaching to the test")
- Construct validity: are you measuring what you think you are?

## 9.3 Classic NLP & capability metrics

- Perplexity, accuracy, F1, exact match, BLEU/ROUGE/METEOR (and their limits)
- Pass@k for code; functional correctness via test execution
- Calibration (ECE), Brier score; does the model know what it knows?

## 9.4 LLM benchmarks (know what they measure & their flaws)

- Knowledge/reasoning: MMLU(-Pro), GPQA, BIG-Bench Hard, ARC, HellaSwag
- Math: GSM8K, MATH, AIME-style
- Code: HumanEval, MBPP, **SWE-bench (Verified)**, LiveCodeBench
- Long context: RULER, needle-in-a-haystack (and why NIAH is necessary-not-sufficient)
- Instruction following: IFEval
- Agentic: GAIA, **τ-bench**, WebArena, OSWorld
- Holistic: HELM; live/contamination-resistant leaderboards

## 9.5 LLM-as-judge

- The workhorse for open-ended quality at scale
- Pairwise vs. single-answer grading; reference-guided grading; rubrics
- **Biases:** position, verbosity, self-preference, formatting; mitigations (swap order, anonymize, calibrate)
- Judge agreement with humans; when to trust it; ensembling judges
- Pitfalls: judge ≠ ground truth; validate the judge itself

> **Build:** Build an LLM-as-judge harness, then *evaluate the judge*: measure its agreement with a small human-labeled set, test for position/verbosity bias, and report where it's unreliable.

## 9.6 RAG evaluation

- Retrieval: recall@k, MRR, nDCG, context precision/recall
- Generation: faithfulness/groundedness, answer relevance, citation accuracy
- Frameworks: RAGAS, TruLens, ARES (ties to [Module 06](06-rag-and-vector-databases.md))

## 9.7 Agent evaluation

- Outcome: task success rate, exact/functional correctness
- Process: trajectory quality, tool-call accuracy, steps-to-completion, cost, latency
- Partial credit & checkpoint-based scoring; environment-based eval (sandboxes)
- Reproducibility of stochastic trajectories; seeds, fixed environments

## 9.8 Safety & alignment evals

- Harmlessness: toxicity, bias/fairness, refusal correctness (over- vs. under-refusal)
- **Red-teaming** (manual + automated), adversarial suites, jailbreak robustness
- **Prompt-injection** resistance for tool-using agents
- Honesty/hallucination, sycophancy, deception probes
- Dangerous-capability evals (Anthropic's RSP / OpenAI Preparedness style), dual-use
- Bridges to [Module 10](10-ai-infrastructure-and-production.md)

## 9.9 Building an eval system (the engineering)

- Dataset curation: golden sets, slicing by category/difficulty, mining hard cases from prod logs
- Eval harness design: deterministic runners, caching, parallelism, cost control
- **Evals in CI/CD** — block regressions on every model/prompt change
- Online evaluation: A/B tests, interleaving, guardrail metrics, human feedback loops
- Tooling: `lm-evaluation-harness`, `inspect_ai` (UK AISI), OpenAI Evals, HELM, Braintrust/LangSmith/Phoenix
- Dashboards, trend tracking, regression alerts

> **Build:** A reusable eval harness that runs a suite (capability + safety) against any model/endpoint, caches results, reports CIs, and fails CI on regression.

---

## Module 09 capstone — **An eval platform**

1. A multi-suite harness (capability, RAG, agent, safety) runnable against any endpoint, with cached, parallel execution and statistical reporting (CIs, significance).
2. A **validated LLM-as-judge** with a bias audit and human-agreement numbers.
3. **CI integration** that blocks a PR when a prompt/model change regresses the suite.
4. A red-team / prompt-injection mini-suite for your Module 07 agent, with a findings report.
5. A write-up: what each metric does and doesn't tell you, and how you'd use this to drive model improvement.

## Exit criteria
- [ ] You can design an eval for a fuzzy capability and defend its validity.
- [ ] You can build and *validate* an LLM-as-judge and account for its biases.
- [ ] You can wire evals into CI and detect regressions automatically.
- [ ] You can design safety/red-team and prompt-injection evals.

## Core sources
- *Holistic Evaluation of Language Models (HELM)* — Liang et al., 2022
- *Judging LLM-as-a-Judge (MT-Bench / Chatbot Arena)* — Zheng et al., 2023
- *SWE-bench* — Jimenez et al., 2023; *τ-bench* — Yao et al., 2024; *GAIA* — Mialon et al., 2023
- *RAGAS* — Es et al., 2023
- Anthropic Responsible Scaling Policy; OpenAI Preparedness Framework
- `inspect_ai`, `lm-evaluation-harness` docs
- Hamel Husain — "Your AI Product Needs Evals"
