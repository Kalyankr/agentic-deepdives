# Module 11 · Large-Scale System Design & Capacity Planning

> **Goal:** Tie everything together into the skill frontier labs interview hardest on: designing large-scale ML/LLM systems and doing **back-of-the-envelope capacity estimation** — QPS, GPU count, storage, and network bandwidth — plus cost optimization. This module is mostly drills and design docs.

**Duration:** ~4 weeks. **Prereqs:** Modules 04, 05, 09, 10.

---

## 11.1 The capacity-estimation toolkit

Senior engineers reason in numbers. Memorize these building blocks.

### Useful constants & approximations
- **Model FLOPs:** forward ≈ $2N$ FLOPs/token, training ≈ $6N$ FLOPs/token, where $N$ = #params.
- **Memory for weights:** bytes = params × bytes/param (FP16/BF16 = 2, FP8/INT8 = 1, INT4 = 0.5).
- **KV cache per token** = $2 \times L \times d_{kv} \times \text{bytes}$ (K and V). Per sequence, multiply by context length.
- **Decode is memory-bandwidth-bound:** tokens/sec/GPU ≈ (HBM bandwidth) / (bytes read per token). Bytes/token ≈ model size (weights) + KV read.
- **Prefill is compute-bound:** time ≈ (prompt_tokens × 2N) / (effective GPU FLOP/s).
- **Little's Law:** concurrency $L = \lambda W$ (arrival rate × latency). Core to QPS↔replica math.
- **Hardware (order of magnitude):** H100 ≈ ~1,000 TFLOP/s dense BF16, 80 GB HBM @ ~3.35 TB/s; A100 ≈ ~312 TFLOP/s BF16, 40/80 GB @ ~1.5–2 TB/s. Use real spec sheets in interviews.

### Method (say this out loud in interviews)
1. State assumptions (traffic, tokens in/out, model size, SLA, dtype).
2. Compute per-request cost (FLOPs, memory, tokens).
3. Scale to aggregate (QPS × per-request).
4. Convert to resources (GPUs, storage, bandwidth) with utilization headroom.
5. Add redundancy, peak/burst factor, and cost.

## 11.2 Worked example — serving an LLM API

**Assumptions:** 13B model, BF16; avg 500 input + 300 output tokens; SLA TTFT < 1s, TPOT < 50 ms; target **2,000 QPS** peak.

**Memory per GPU:**
- Weights: 13B × 2 B = **26 GB** (fits on one 80 GB GPU with room for KV cache; else use tensor parallelism).
- KV cache: with GQA, say ~0.5 MB/token → 800 tokens ≈ 0.4 GB/sequence; budget for batch.

**Decode throughput (memory-bound):**
- Per token a decode step reads ≈ weights (26 GB) + KV. At ~3 TB/s HBM, raw ≈ 3e12 / 26e9 ≈ ~115 tokens/s for batch=1 — but **batching** amortizes weight reads across many sequences, so aggregate throughput is far higher (e.g., thousands tokens/s/GPU at good batch sizes). This is *why* continuous batching matters.
- Suppose measured throughput = **4,000 output tokens/s/GPU** at the SLA batch size.

**Aggregate demand:**
- Output tokens/s = 2,000 QPS × 300 = **600,000 tokens/s**.
- GPUs for decode = 600,000 / 4,000 = **150 GPUs**. Add prefill cost, ~40% headroom, and redundancy → **~220–250 GPUs**.

**Network:** intra-node NVLink for tensor parallelism; inter-node for replica coordination is light (stateless requests). KV-cache-aware routing if using prefix caching.

> **Drill:** Redo this for a 70B model with tensor parallelism across 4 GPUs and TPOT < 30 ms. How does GPU count change? What if you quantize to FP8?

## 11.3 Worked example — capacity for the four pillars (from the brief)

For any LLM service, estimate:

### QPS
- From DAU and usage: QPS_avg = (DAU × requests/user/day) / 86,400. Peak = QPS_avg × peak factor (3–10×).
- Example: 10M DAU × 20 req = 200M/day → ~2,300 QPS avg → ~10–20k QPS peak.

### GPU count
- = (peak output tokens/s) / (tokens/s/GPU at SLA) ÷ utilization target, + prefill GPUs + redundancy. (See 11.2.)
- Cross-check against memory: must fit weights + peak KV cache; if not, add tensor parallelism (more GPUs per replica).

### Storage
- Model checkpoints: params × bytes (× #versions × #variants). A 70B FP16 ckpt ≈ 140 GB; training also stores optimizer state (~2× params in FP32).
- Datasets: pretraining corpora are TB–PB; logs/traces grow with QPS (e.g., 10k QPS × 5 KB/trace × 86,400 ≈ ~4 TB/day).
- Vector index: vectors × dim × bytes (+ ANN overhead). 1B × 768 × 4 B ≈ 3 TB raw; PQ compresses 10–30×.

### Network bandwidth
- Serving ingress/egress: QPS × avg payload. Token streaming is small per token but adds up.
- **Training is the network monster:** all-reduce of gradients each step ≈ 2 × params × bytes × (factor), needing TB/s-class interconnect (NVLink/InfiniBand). This dictates topology.
- Inter-node for distributed inference: KV transfer in disaggregated prefill/decode, expert routing (All-to-All) for MoE.

> **Drill:** Size storage + bandwidth for a RAG product: 100M documents, re-embedded quarterly, 5k QPS retrieval, traces retained 30 days.

## 11.4 Cost optimization (mastery area #14)

- **Cost per token / per request** as the north-star metric; cost per successful task for agents.
- Levers, roughly in order of impact:
  - Right-size the model (smallest that passes evals); **model cascades/routing**
  - **Quantization** (FP8/INT8/INT4) and **speculative decoding** (Module 04)
  - **Batching** & high GPU utilization; **prefix/prompt caching** (huge for agents/RAG)
  - **KV-cache** optimization (GQA, paging, quantized KV, offload)
  - Right hardware (newer GPUs often cheaper per token); spot/reserved capacity; autoscaling to demand
  - Prefill/decode disaggregation; distillation to a smaller model
  - Prompt/context compression to cut token counts
- Build vs. buy (API vs. self-host) break-even analysis
- FinOps: budgets, per-feature cost attribution, anomaly alerts (ties to [Module 10](10-ai-infrastructure-and-production.md))

> **Drill:** You're spending \$X/month on an API. Model the break-even point for self-hosting on H100s (capex/rental + utilization). State when each option wins.

## 11.5 End-to-end system design practice

Practice these as 45–60 min design exercises (assumptions → API → data model → architecture → scaling → bottlenecks → failure modes → cost):

1. Design **ChatGPT** (multi-turn chat at scale): serving fleet, KV/prefix caching, streaming, conversation storage, moderation, autoscaling, multi-region.
2. Design a **RAG assistant** over a company's documents (ingestion, indexing, retrieval, reranking, freshness, multi-tenant security).
3. Design an **AI coding agent** platform (sandboxes, tool execution, long-running tasks, durability, cost caps, eval/feedback loop).
4. Design the **inference platform** that serves many models/LoRAs to many tenants with SLAs.
5. Design a **training cluster** for a 100B+ model (parallelism, storage, checkpointing, fault tolerance, network).
6. Design an **eval/experimentation platform** (Module 09) that gates releases.

> **Build:** Write 2–3 full **design docs** (Google-doc style) for the above, with diagrams, capacity math, trade-offs, and failure analysis. These double as interview artifacts.

---

## Module 11 capstone — **The design portfolio**

1. A capacity-estimation cheat-sheet (your own) with the formulas and constants above.
2. ≥3 complete system-design docs (diagrams + capacity math + trade-offs + failure modes + cost).
3. A cost-optimization case study on one of your earlier capstones: baseline cost/token → optimizations → measured savings with no quality loss (evals from Module 09 prove it).

## Exit criteria
- [ ] You can estimate QPS, GPU count, storage, and bandwidth for a given product, out loud, with stated assumptions.
- [ ] You can derive serving GPU count from tokens/s and HBM bandwidth, and know why batching/caching change it.
- [ ] You can run a system-design interview end-to-end with trade-offs and failure analysis.
- [ ] You can drive cost/token down and prove quality held with evals.

## Core sources
- *Designing Data-Intensive Applications* — Kleppmann
- *Designing Machine Learning Systems* — Chip Huyen
- *System Design Interview* Vol. 1 & 2 — Alex Xu (general scaffolding)
- "How to Scale Your Model" (DeepMind) — LLM-specific capacity math
- Latency/throughput sections of the vLLM, TensorRT-LLM, and Mooncake/DistServe papers (Modules 04–05)
- Cloud GPU pricing pages + real GPU spec sheets (A100/H100/H200/B200)
