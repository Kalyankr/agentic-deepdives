# 13 — Track: Applied Scientist

> **Role in one line:** you *advance and rigorously measure* agent capability — designing methods (reasoning, training, eval), running hypothesis-driven experiments, and turning findings into reliable, evidenced improvements. Less "wire the API," more "prove it works and why."

> This lens emphasizes **methodology, evaluation rigor, training, and research judgment** on top of the core kit.

---

## What the interview actually tests

| They probe | Because the job is | Where in the kit |
|------------|--------------------|------------------|
| Evaluation methodology & statistics | Your claims must be defensible | [08](08-production-evaluation-security.md), below |
| Reasoning methods & ablations | You design/compare methods | [02](02-reasoning-and-planning.md) |
| Training: SFT, RLHF/RLAIF, DPO, RFT | You improve models, not just prompt them | below |
| Experiment design (hypothesis → metric → ablation) | Research is controlled experiments | below |
| Reading/critiquing papers | You build on and assess prior art | below |
| ML & math fundamentals | Coding/theory rounds | below |

**They expect depth on "why."** Buzzwords sink you here; mechanisms, trade-offs, and statistical care float you.

---

## Priority reading (in order)
1. [02 — Reasoning & Planning](02-reasoning-and-planning.md) — know each method's *mechanism* and when it wins.
2. [08 — Production, Eval & Security](08-production-evaluation-security.md) — eval is your core craft.
3. [09c — Follow-Ups](09c-followup-questions.md) §B, H, J — reasoning depth, eval rigor, RL/self-improvement.
4. [03 — Memory & Context](03-memory-and-context.md) — retrieval, embeddings, RAG eval.
5. [05](05-multi-agent-systems.md)/[07](07-system-design.md) — for applied/system framing.

---

## Applied-Scientist-specific Q&A (new)

### Evaluation methodology (the core skill)

**Q. Design an evaluation for a new agent capability from scratch.**
"Start from the *construct*: what does success mean operationally? Define tasks, a dataset (held-out, representative, with adversarial/edge slices), and metrics at two levels — outcome (task success) and trajectory (tool/step correctness). Choose graders: programmatic where possible, LLM-as-judge for nuance (validated against human labels), human for calibration. Control for contamination, report per-slice results with confidence intervals, and pre-register what 'better' means so I'm not p-hacking. Finally, tie offline gains to an online metric to confirm external validity."

**Q. How do you validate an LLM-as-judge before trusting it?**
"Treat it as a classifier to be measured. Collect human-labeled gold judgments, then compute the judge's agreement with humans (accuracy, Cohen's κ, or correlation for scores). Probe known biases — position, verbosity, self-preference — by swapping order, padding length, and cross-model checks. Prefer pairwise over absolute scoring, use a clear rubric, and only deploy the judge in regimes where its measured agreement is high; report that agreement alongside results."

**Q. Your new method beats baseline by 2 points. Is it real?**
"Depends on variance and n. I'd report a confidence interval (bootstrap over the eval set), run multiple seeds because generation is stochastic, and use a significance test or, better, effect size with CIs. I'd check it's not driven by one slice, not contaminated, and robust to prompt-format perturbations. Two points inside the noise band is not a result — I'd either gather more data or temper the claim."

**Q. How do you do error analysis on an agent?**
"Sample failed trajectories and build a *taxonomy*: retrieval miss, tool-selection error, bad arguments, reasoning error, hallucination, loop/non-termination, recovery failure, spec misread. Quantify each category's share, because that tells me where the marginal fix pays off. Then I form a hypothesis per top category and test a targeted intervention, re-running the same eval. Categorized failure rates beat a single aggregate number for driving research."

**Q. Design a benchmark for tool-using agents. What makes it good/bad?**
"Good: realistic, diverse tasks; verifiable success (programmatic checks, not vibes); resistance to gaming and memorization; separable difficulty; and trajectory-level signals, not just final answers. Bad: contaminated with training data, single-metric, easily shortcut, or unrepresentative of real use. I'd include a user-simulation component (like τ-bench) and report both success and efficiency (steps/tokens)."

### Training & optimization

**Q. SFT vs. RLHF vs. DPO vs. RFT — when each?**
"SFT (supervised fine-tuning) teaches behavior/format from demonstrations — the foundation. RLHF aligns to *preferences* via a reward model + PPO when you have comparison data and need nuanced optimization, at higher complexity. DPO achieves preference optimization *without* a separate reward model or RL loop — simpler and stable, often the default now. RFT / RL-from-verifiable-rewards shines when you have an automatic correctness signal (tests, math checkers) — great for agentic/tool tasks. Choose by what signal you have: demonstrations → SFT; preferences → DPO/RLHF; verifiable reward → RFT."

**Q. How would you improve an agent's tool use with training?**
"First exhaust prompting + tool redesign + evals, since most 'model' issues are scaffolding. If training is warranted: collect/curate successful and corrected trajectories, SFT on them for format and selection, then optimize against a *verifiable reward* (did the task succeed, were args valid) with RL or rejection-sampling/best-of-n distillation. Watch for reward hacking and distribution shift, and keep a held-out eval to confirm real gains, not overfitting to the reward proxy."

**Q. What is reward modeling and where does it break for agents?**
"A reward model scores outputs to stand in for human preference. It breaks via reward hacking (agent exploits the proxy), distribution shift (it's unreliable off the training distribution), and sparse/credit-assignment problems over long multi-step trajectories. Mitigations: process rewards (score steps, not just outcomes), verifiable rewards where available, ensembles, and conservative optimization (KL penalty to the base policy)."

**Q. LoRA / PEFT — why and trade-offs?**
"Parameter-efficient fine-tuning (e.g., LoRA) trains small low-rank adapters instead of all weights — far cheaper memory/compute, fast to swap per task, good when you adapt behavior on modest data. Trade-off: it may underperform full fine-tuning on large capability shifts, and serving many adapters adds routing complexity. For most applied behavior tuning, LoRA is the pragmatic default."

**Q. Generate synthetic data to improve an agent — how, and what's the risk?**
"Use a strong model (or self-play/bootstrapping) to produce trajectories, filter aggressively by a verifier or rubric, and distill into the student. Risks: distribution narrowing, error amplification, and model-collapse if you train on unfiltered self-output. I'd keep real held-out evals, mix in human data, and verify diversity/coverage rather than just volume."

### Reasoning & research judgment

**Q. Critique the ReAct paper as if reviewing it.**
"Strengths: a simple, general interleaving of reasoning and acting that grounds reasoning in observations, reducing hallucination and improving interpretability across QA and decision tasks. Limitations/threats: dependence on tool/environment quality, sensitivity to prompt format and exemplars, potential for loops, and evaluation on specific benchmarks that may not generalize. I'd ask for ablations isolating 'reason' vs. 'act' contributions and robustness to weaker base models."

**Q. Reasoning models vs. prompt-engineered CoT — research framing?**
"Reasoning models internalize long chains via RL on verifiable tasks, shifting reasoning from the scaffold into the weights and exposing a test-time-compute knob (more inference → higher accuracy). The scientific questions: where does extra test-time compute help vs. plateau, how to allocate it (search vs. longer chains), and how to evaluate hidden reasoning. For agents, they reduce prompt hand-holding but raise latency/cost and over-thinking on easy steps."

**Q. Propose a method to reduce compounding error in long-horizon agents.**
"Hypothesis: per-step verification + decomposition raises end-to-end success more than a stronger base model alone. Method: insert a lightweight verifier/critic between steps (process supervision), checkpoint sub-goals, and re-plan on verifier failure. Eval: measure end-to-end success and per-step error rate vs. baselines on a multi-step benchmark, ablating the verifier and decomposition independently, with CIs over seeds. Success = significant lift in success and a drop in cascaded failures at acceptable added cost."

---

## Coding / theory round (Applied Scientist)
- **ML fundamentals:** transformers/attention, softmax/temperature, embeddings, overfitting/regularization, bias-variance, evaluation metrics (precision/recall/F1, calibration).
- **Implement an algorithm:** e.g., self-consistency voting, beam search, a simple reranker, top-k/top-p sampling, or a metric (nDCG, BLEU-ish, pass@k).
- **Stats:** confidence intervals (bootstrap), significance tests, sample-size reasoning, multiple-comparison awareness.
- **Probability/math:** expectation, why p^n compounds, basic linear algebra for embeddings.

## Signals they grade
✅ Defines metrics from the construct · controls for contamination/variance · validates judges · categorizes errors · picks the right training signal · critiques prior art fairly · hypothesis → ablation → CI.
🚩 Single-number eval · trusts LLM-judge blindly · claims wins inside the noise · confuses SFT/RLHF/DPO · "just fine-tune it" reflex · no error analysis.

## 1-week plan
- **D1–2:** [02](02-reasoning-and-planning.md) + [09c §B/J](09c-followup-questions.md); be able to derive when each method wins.
- **D3:** [08](08-production-evaluation-security.md) eval section + this file's eval Q&A; design one eval end-to-end on paper.
- **D4:** Training Q&A (SFT/RLHF/DPO/RFT/LoRA); read summaries of ReAct, Reflexion, ToT, DPO.
- **D5:** Error-analysis + benchmark-design drills; do an ablation design out loud.
- **D6:** ML/stats fundamentals refresh; [11 mock](11-mock-interview.md) rounds 2 & 4.
- **D7:** Behavioral research stories (a method you shipped, a negative result you handled).
