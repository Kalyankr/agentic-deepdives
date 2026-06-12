# Module 12 · Capstones & Interview Prep

> **Goal:** Consolidate everything into portfolio-grade projects and prepare specifically for **Anthropic / OpenAI**–style interviews. By the end you have artifacts that prove senior-level ability and the muscle memory to perform live.

**Duration:** ongoing / final ~4 weeks.

---

## 12.1 Flagship capstone — build a "mini ChatGPT + agent" end-to-end

This single project exercises nearly every module. Ship it publicly (GitHub + write-up + demo).

**Scope:**
1. **Model:** start from an open base model; **SFT** it and **DPO**-align it on a chat dataset (Modules 02–03).
2. **Serving:** deploy with **vLLM** (PagedAttention, continuous batching); add **quantization** + **prefix caching**; tensor-parallel if needed (Modules 04–05).
3. **RAG:** ground answers in a document corpus with hybrid retrieval + reranking (Module 06).
4. **Agent:** add tool use (search, code exec), memory, and an **MCP** tool; multi-step task handling (Module 07).
5. **Orchestration:** structured outputs, routing, caching, fallbacks (Module 08).
6. **Evals:** a CI-gated eval suite (capability + RAG + agent + safety + LLM-judge) (Module 09).
7. **Production:** containerized, autoscaled, traced; latency/cost dashboards; safety layer (moderation, prompt-injection defense, sandboxed tools) (Module 10).
8. **Design doc:** architecture diagram + **capacity estimation** (QPS/GPU/storage/bandwidth) + cost analysis (Module 11).

**Deliverables:** repo, design doc, eval report, latency–throughput + cost analysis, demo video.

## 12.2 Smaller portfolio projects (pick 2–3 to go deep)

- **From-scratch GPT** with modern attention (RoPE/GQA/SwiGLU) + KV cache + benchmarks (Module 02).
- **DPO vs. PPO study** on a small model with win-rate evals (Module 03).
- **Inference optimization study:** quantization + speculative decoding + paged attention, with a full latency/throughput/cost characterization (Module 04).
- **Vector-search benchmark:** FAISS HNSW/IVF-PQ recall–latency–memory trade-off study (Module 06).
- **Agent eval harness** on SWE-bench/τ-bench-style tasks with trajectory analysis (Modules 07, 09).
- **A reproduction** of a notable paper (FlashAttention-lite, a small RLHF run, speculative decoding). Reproductions are strong signal.

## 12.3 Contributions & visibility

- Contribute to OSS you now understand: vLLM, transformers, TRL, FAISS, LangGraph, `inspect_ai`.
- Write technical posts explaining hard concepts (teaching = mastery signal).
- Reproduce/extend a recent paper and publish results.

---

## 12.4 Interviewing at Anthropic / OpenAI

> **Full question bank with answers:** [interview-prep/](../interview-prep/README.md) — a complete,
> answer-included drill set covering coding, ML/LLM depth, system design, applied LLM, safety, and
> behavioral rounds, plus 130+ rapid-fire flashcards. Use this section for *strategy*; use that folder
> for *practice*.

### What they look for
- **Strong coding** (often practical, ML-flavored, sometimes data-structures/algorithms).
- **ML/LLM depth** — transformers, training, inference, the trade-offs.
- **Systems design at scale** — serving, distributed training/inference, capacity & cost.
- **Research/eval literacy** — read a paper, reason about experiments, design evals.
- **Safety mindset** — these labs care deeply; be able to discuss alignment, misuse, prompt injection, responsible scaling.
- **Communication & collaboration** — clear thinking, design docs, mission alignment.

### Loop components to prepare
1. **Coding:** data structures & algorithms (LeetCode medium/hard) **and** practical ML coding (implement attention, a sampler, a training step, a dataloader, a metric). Practice in a plain editor.
2. **ML/LLM knowledge:** be able to whiteboard attention, derive params/FLOPs/KV-cache, explain RLHF/DPO, FlashAttention, GQA, quantization, parallelism.
3. **System design:** the exercises in [Module 11](11-system-design-and-capacity-planning.md) — practice out loud, manage assumptions, do the capacity math.
4. **Research/depth round:** discuss a paper or your own project deeply; defend choices; reason about ablations and failure modes.
5. **Behavioral / mission:** why this lab, how you think about AI safety and impact; bring concrete stories (STAR format).

### Drills (do these repeatedly)
- Whiteboard multi-head causal attention in <15 min.
- Given a model spec, estimate memory, FLOPs, KV-cache, and serving GPU count in <10 min.
- Implement top-k/top-p sampling and a KV cache from scratch.
- Design ChatGPT-scale serving end-to-end in 45 min with capacity + cost.
- Explain DPO vs. PPO and Constitutional AI clearly to a non-expert and to an expert.
- Threat-model an agent and propose layered defenses.

### Cadence (final 4–8 weeks)
- Weekly: 1 system-design mock, 2 coding sessions, 1 paper deep-dive, 1 behavioral rehearsal.
- Mock interviews with peers; record yourself; tighten communication.
- Keep your capstone demo and design docs polished and linkable.

---

## 12.5 Final exit criteria (portfolio + readiness)

- [ ] A flagship end-to-end project shipped publicly with docs, evals, and a design doc.
- [ ] 2–3 deep portfolio projects + ideally one paper reproduction.
- [ ] You can pass the drills in 12.4 cold.
- [ ] You can hold a substantive conversation on safety/alignment.
- [ ] You can clearly explain every trade-off in your projects with numbers.

## Resources
- See [resources.md](../resources.md) for the full library.
- Anthropic & OpenAI engineering/research blogs and their published interview guidance.
- "Machine Learning Systems Design" interview prep (Chip Huyen) + general system-design prep.
