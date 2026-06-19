# Stage 4 — Evaluation (the underrated elite skill)

> **Objective:** Design rigorous evaluations, read benchmarks with healthy skepticism, and prove whether a model/system is actually good. The people who can *measure* well are rarer and more valuable than those who can only *train*.

[← Stage 3](../stage-3-adaptation-alignment/README.md) · [Index](../README.md) · Next: [Stage 5 — Inference](../stage-5-inference-optimization/README.md)

📝 **Interview prep:** [interview-questions.md](interview-questions.md) · ✅ [answer key](answers.md)

---

## Why this stage matters

You cannot improve what you cannot measure. Most LLM project failures are **evaluation failures** — teams ship on vibes, optimize the wrong metric, or trust a contaminated benchmark. Rigorous eval is the difference between "it seems good" and "it is good, and here's the evidence."

---

## Mental model

> Evaluation is **experiment design** applied to language models. Pick a question, choose a metric that actually reflects it, control for bias, and quantify uncertainty.

Three axes to always separate:
1. **Capability** (can it do the task?)
2. **Alignment/safety** (does it behave well?)
3. **Cost/latency** (is it deployable?) — covered more in Stage 5/6.

---

## Concept-by-concept deep dive

### 4.1 Intrinsic vs extrinsic metrics
- **Intrinsic:** measured on the modeling objective itself — e.g., **perplexity**. Good for tracking pretraining; weakly correlated with usefulness.
- **Extrinsic:** measured on downstream tasks (accuracy, pass@k, human ratings). What actually matters for users.

### 4.2 Automatic metrics & their failure modes
- **BLEU / ROUGE:** n-gram overlap (translation/summarization). Cheap but blind to meaning — a correct paraphrase scores low; a fluent wrong answer can score high.
- **Exact match / F1:** QA with short answers. Brittle to formatting.
- **pass@k:** code — fraction of problems solved within k samples (run the unit tests). Honest because it's *executable* — the gold standard pattern: **verify, don't guess**.
- **Lesson:** prefer metrics with a ground-truth *checker* (tests, calculators, constraints) over surface-overlap metrics.

### 4.3 Standard benchmarks (know what each tests)
- **MMLU:** broad multiple-choice knowledge (57 subjects).
- **HellaSwag / ARC / Winogrande:** commonsense reasoning.
- **GSM8K / MATH:** grade-school → competition math reasoning.
- **HumanEval / MBPP:** code generation (pass@k).
- **BIG-bench / BBH:** hard, diverse tasks.
- **HELM:** *holistic* — many scenarios × many metrics (accuracy, robustness, calibration, bias, efficiency). The mindset to emulate.
- **Chatbot Arena:** human pairwise preference at scale (Elo ranking).

### 4.4 Benchmark contamination (critical)
- If benchmark text leaked into training data, scores are inflated and meaningless.
- **Detect:** n-gram overlap between train and test, canary strings, "oracle" performance jumps, perplexity on test items.
- **Mitigate:** decontaminate training data; prefer **fresh/held-out** or private evals; report contamination checks.

### 4.5 LLM-as-a-judge (powerful but biased)
Use a strong model to score/compare outputs. Scales human judgment, but watch for:
- **Position bias:** favors the first (or second) answer → **swap order and average**.
- **Verbosity bias:** prefers longer answers regardless of quality.
- **Self-preference bias:** a model favors its own style/outputs.
- **Mitigations:** randomize order, control for length, use multiple judges, calibrate against human labels, use rubric-based scoring with explicit criteria.

### 4.6 Human evaluation
- **Design:** clear rubric, well-defined scales, training examples for annotators.
- **Reliability:** measure **inter-annotator agreement** (Cohen's/Fleiss' kappa). Low agreement = ill-defined task.
- **Pairwise > absolute:** humans compare more reliably than they score on an absolute scale.

### 4.7 Safety & robustness evaluation
- **Toxicity / bias:** standardized prompt sets; demographic parity checks.
- **Hallucination / factuality:** does the answer match a trusted source? Faithfulness vs the provided context.
- **Red-teaming:** adversarially probe for jailbreaks, harmful outputs, prompt injection (ties to Stage 8).
- **Calibration:** does stated/implied confidence match accuracy?

### 4.8 Evaluating RAG & agents (separate the stages!)
- **Retrieval:** recall@k, precision@k, **MRR**, nDCG — did we fetch the right context?
- **Generation:** **faithfulness/groundedness** (no claims beyond context), answer relevance, citation correctness.
- **Why separate:** a wrong answer might be a *retrieval* miss (right context never fetched) or a *generation* miss (context fetched but ignored). Measuring them together hides the root cause.

---

## Ordered learning path

1. Read the **HELM** paper — adopt the holistic mindset.
2. Study **MT-Bench / Chatbot Arena** for LLM-judge + Elo methodology.
3. Set up **`lm-evaluation-harness`** and run a real benchmark.
4. Read a recent **RAG evaluation** framework (e.g., RAGAS-style faithfulness metrics).
5. Do the labs.

---

## 🛠️ Hands-on labs

- [ ] **Lab A — Run a benchmark:** evaluate your Stage-3 models (base/SFT/DPO) on MMLU + GSM8K via lm-eval-harness; report numbers with confidence intervals.
- [ ] **Lab B — Custom eval set:** build a 50–100 item eval for a narrow task you care about, with a clear scoring rubric.
- [ ] **Lab C — LLM judge + bias audit:** implement pairwise judging; then **measure position bias** (swap order) and **verbosity bias** (control length). Quantify the effect.
- [ ] **Lab D — Judge vs human:** hand-label 30 items; compute agreement between your LLM judge and your labels.
- [ ] **Lab E — RAG eval:** for a small RAG pipeline, report retrieval recall@k *separately* from generation faithfulness.
- [ ] **Lab F — Contamination check:** n-gram overlap scan between a train set and a benchmark.

---

## ⚠️ Common pitfalls & gotchas

- Reporting a single number with no **confidence interval** or seed variation.
- Trusting a benchmark without checking **contamination**.
- Using BLEU/ROUGE where meaning matters.
- LLM-judge with fixed answer order → silent position bias.
- Evaluating RAG end-to-end only → can't localize failures.
- Optimizing the metric instead of the goal (**Goodhart's law**).
- Tiny eval sets → noise dominates; differences aren't significant.
- Prompt-format sensitivity: same model, different score just from formatting.

---

## 🔥 Mastery checks (answer without notes)

- [ ] Why can a model ace MMLU yet fail in production? Give concrete reasons.
- [ ] Design an evaluation for a task with **no ground-truth labels**.
- [ ] List three LLM-judge biases and a concrete mitigation for each.
- [ ] How do you detect benchmark contamination?
- [ ] In RAG, how do you prove a failure is retrieval vs generation?
- [ ] Why is pass@k a "trustworthy" metric while BLEU is not?
- [ ] Why prefer pairwise human comparison over absolute scoring?
- [ ] What is Goodhart's law and how does it bite LLM teams?

---

## ✅ Stage 4 checklist

- [ ] Read HELM + LLM-judge methodology
- [ ] Labs A–D complete (E/F for RAG/contamination depth)
- [ ] Built one custom eval with a rubric
- [ ] Measured judge bias quantitatively
- [ ] All mastery checks passable
- [ ] Notes in your own words

**When complete → proceed to [Stage 5](../stage-5-inference-optimization/README.md).**
