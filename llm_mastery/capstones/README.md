# 🏆 Capstone Projects

> **Objective:** Prove mastery by shipping. Reading and labs build understanding; **capstones build evidence**. These are what you show in interviews, portfolios, and on GitHub. Pick **at least two** across different stages.

[← Index](../README.md)

---

## How to run a capstone (the standard)

For each project, deliver:
- [ ] **Working code** in a clean repo (README, reproducible setup, `requirements`/env).
- [ ] **A results writeup**: what you built, key decisions, metrics, plots, and **what you'd do differently**.
- [ ] **Evaluation** (Stage 4 mindset): numbers with baselines, not vibes.
- [ ] **A "how it works" explainer** — teach it. If you can't explain it simply, you don't own it yet.

> Treat each capstone as if a senior scientist will review it. Rigor and clarity > flashiness.

---

## Capstone 1 — End-to-End Model Pipeline
**Covers:** Stages 1–5

Build the full lifecycle on a small scale:
- [ ] Pretrain (or take a small base) → **SFT** → **DPO** → **evaluate** → **serve** (vLLM, quantized).
- [ ] Document the pipeline end to end with a diagram.
- [ ] Report: base vs SFT vs DPO quality, plus inference speed/memory at fp16 vs int4.

**Deliverable:** a repo that takes a base model to a served, aligned, optimized assistant with measured improvements at each step.

---

## Capstone 2 — Production RAG System
**Covers:** Stages 4, 6, 8

Grounded Q&A over a **real** corpus (docs you care about):
- [ ] Chunking → embeddings → vector store → retrieval → **reranking** → grounded generation with **citations**.
- [ ] Hybrid (dense + sparse) search.
- [ ] **Separate evals**: retrieval recall@k *and* generation faithfulness.
- [ ] Observability (tracing + online eval) + a cost/latency report.
- [ ] Security: indirect-injection red-team + guardrails (Stage 8).

**Deliverable:** a reliable, observable, attack-tested RAG app with a metrics dashboard.

---

## Capstone 3 — Inference Optimization Study
**Covers:** Stage 5 (+ 4)

Rigorous benchmark report on one model:
- [ ] Apply quantization (GPTQ/AWQ int4) + speculative decoding.
- [ ] Measure TTFT, TPOT, throughput, memory, and **quality delta** (reuse Stage-4 evals).
- [ ] Sweep batch size → plot the latency↔throughput tradeoff; recommend an SLA operating point.
- [ ] Compare ≥2 serving stacks (e.g., vLLM vs TGI) fairly.

**Deliverable:** a publishable benchmark writeup (quality vs speed vs cost) with clear methodology.

---

## Capstone 4 — Agent System
**Covers:** Stages 6, 8

Multi-tool agent that does something genuinely useful:
- [ ] Function calling + planning + multi-step orchestration (ReAct).
- [ ] Guardrails, step limits, output validation, **least-privilege** tools.
- [ ] Failure-mode analysis: where it breaks and how you contained it.
- [ ] Eval: task success rate + cost/latency per task.

**Deliverable:** a robust agent with a documented safety + failure analysis.

---

## Capstone 5 — Evaluation Framework
**Covers:** Stage 4 (+ 8)

The "measure well" capstone:
- [ ] Build a rigorous, **contamination-aware** eval suite for a domain.
- [ ] Include an **LLM-judge** with a **measured bias** report (position/verbosity) and human-agreement calibration.
- [ ] Provide confidence intervals and significance, not point estimates.

**Deliverable:** a reusable eval harness others could trust and adopt.

---

## Capstone 6 — Specialization Showcase (optional)
**Covers:** Stage 7

Implement and explain one advanced track artifact:
- [ ] e.g., a small **MoE** layer, a **long-context** extension + needle test, a **test-time-compute** scaling study, or an **interpretability** induction-head finding.

**Deliverable:** a focused deep-dive repo + explainer demonstrating frontier understanding.

---

## ✅ Capstone tracker

| Capstone | Picked? | Code done | Writeup done | Evaluated |
|----------|---------|-----------|--------------|-----------|
| 1. End-to-end model | [ ] | [ ] | [ ] | [ ] |
| 2. Production RAG | [ ] | [ ] | [ ] | [ ] |
| 3. Inference study | [ ] | [ ] | [ ] | [ ] |
| 4. Agent system | [ ] | [ ] | [ ] | [ ] |
| 5. Eval framework | [ ] | [ ] | [ ] | [ ] |
| 6. Specialization | [ ] | [ ] | [ ] | [ ] |

> **Bar for "cracked":** at least two capstones shipped, evaluated, and explainable end to end.
