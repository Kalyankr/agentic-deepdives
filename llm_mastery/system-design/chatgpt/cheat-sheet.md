# 🃏 Design ChatGPT — One-Page Cheat-Sheet

> Last-minute recall card for the [full HLD](README.md). Drill the bold bits.

## The one idea
Unit of work = **stateful, multi-second, GPU-bound token stream** (not a cheap stateless request) → the whole system optimizes **GPU utilization (MFU) + tail latency (p99)**, measured as **TTFT** (responsiveness) and **TPOT** (flow speed, must beat reading ≈ 15–40 tok/s).

## Requirements → SLOs
TTFT p99 < 2 s · TPOT 30–80 ms · 99.9%+ avail · ~100M DAU · high safety recall · min **$/1M tokens** · tenant isolation + residency.

## Numbers (state assumptions)
- 100M DAU × 10 msgs = **1B msgs/day** → ~11.6K req/s avg → **~40K req/s peak**.
- × ~500 out-tok → **~20M output tok/s peak**.
- 70B fp8 on 8×H100 node (~26 TB/s, continuous batching) ≈ **~10K tok/s/node** → ~2K nodes ≈ **~16K H100s** for one model → fleet = **tens of thousands of GPUs** (MoE shrinks it).
- KV/token $= 2 \cdot L \cdot h_{kv} \cdot d_{head} \cdot \text{dtype}$; ~3 GB/seq @4K → **1M sessions impossible to keep hot** → GQA + KV-quant + paging + admission.
- Decode floor/token $\approx$ weight-bytes / HBM-bandwidth (**bandwidth-bound**).

## Architecture (data plane)
`Client → CDN/WAF → Gateway (auth · rate-limit · SSE) → Orchestrator → [input safety · context+RAG · router] → Scheduler → GPU inference (prefill→decode) → output safety → SSE back`. Control plane (registry, autoscaler, flywheel) is **async, off the hot path**. Keep data plane **stateless**; state in stores.

## Inference deep-dive (where you win)
- **Prefill** = parallel, **compute-bound**, sets TTFT. **Decode** = 1 tok/step, **bandwidth-bound**, sets TPOT. → **disaggregate** the two pools (or **chunked prefill**) to kill head-of-line blocking.
- **Continuous (in-flight) batching** = biggest throughput lever (token-granular; finished leave, new join).
- **Paged KV cache** (PagedAttention) = no fragmentation + **prefix sharing**; shrink KV with **GQA/MQA + KV-quant**.
- **Speculative decoding**: draft k, target verifies in 1 pass, accept w/ $\min(1, p_t/p_d)$ → 2–3× fewer steps (only if acceptance high).
- **Quantization** fp8/int4 + **MoE** (top-k experts) = fewer bytes/FLOPs per token.
- **Multi-LoRA**: one frozen base + per-request adapters → thousands of fine-tunes, ~base cost.
- **Scheduler**: admission control + priority queues (paid>free, interactive>batch) + length-aware + bin-pack; autoscale on **queue depth / TTFT / MFU**, warm pools (GPUs start slow).

## Streaming
**Concurrency, not bandwidth** → async gateways hold ~100K–1M SSE conns/node. Streams **sticky** to replica; **max-gen-time** + bounded buffers; on failure: clean fail + **idempotent retry** (or resume from checkpoint).

## Context & memory
Recent turns verbatim + **rolling summary** + RAG/memory chunks, budgeted by relevance/recency. **Prefix-cache** system prompts (skip re-prefill). Memory = per-user vector namespace, user-deletable.

## RAG / tools
Hybrid retrieve (dense+BM25) → **rerank** → ground + **citations** → abstain if unsupported; **eval retrieval & generation separately**. Tools = typed/schema-validated calls in a **sandbox**, ReAct loop with **step/cost budget**, least privilege, untrusted-content isolation.

## Safety (defense-in-depth / OWASP-LLM)
Input (moderation · injection · PII) → retrieval/tool boundary (delimit untrusted = **indirect injection** line) → aligned generation + scoped tools (**no excessive agency**) → output guard (never `eval/SQL/shell` raw) + stream-time moderation. Measure **over-refusal vs attack-success-rate**; red-team = release gate.

## Data flywheel
log → curate/PII-scrub → label/preferences → **SFT → RM → RLHF/DPO** → **offline eval + red-team gate → canary → A/B → ramp** (registry, 1-click rollback). **Never ship on offline metrics alone.**

## Storage
conversations → wide-column KV (by user) · vectors → sharded ANN (per-tenant) · docs → object+search · accounts/billing → Postgres · sessions/quotas → Redis · events → Kafka→lake · weights → registry. TTLs + right-to-be-forgotten.

## Reliability / cost
Regional **cells** + geo-DNS + residency; **graceful degradation** (smaller/quantized model, shorter ctx, queue honestly) > hard fail; model fallback chain. Cost order: **route to small model** → quant+MoE → MFU (batch/paging/spec) → caching → cap output/ctx → reserved+spot.

## Top tradeoffs / failure modes
GPU scarcity (warm pools, fallback) · long-ctx KV blowup & p99 (GQA/quant/length-isolation) · throughput↔latency (batch vs SLO) · quality↔cost (router+eval) · safety↔helpfulness (measure both) · hallucination (RAG+cite+abstain) · mid-stream failure (checkpoint/idempotent) · indirect injection (isolate+least-priv).

---
[← HLD](README.md) · [Q&A](questions.md) · [Answers](answers.md) · [Index](../../README.md)
