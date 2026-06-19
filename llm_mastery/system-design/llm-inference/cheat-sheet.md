# 🃏 LLM Inference Service — One-Page Cheat-Sheet

> Last-minute recall card for the [full HLD](README.md). Drill the bold bits.

## The one idea
Serve many models on a GPU fleet at **max tokens/s/GPU within TTFT/TPOT SLOs**. Everything follows from: **prefill = compute-bound (TTFT); decode = memory-bandwidth-bound (TPOT), intensity ≈ 1 → one stream can't saturate a GPU → BATCH.** Cost = GPU-seconds → **MFU is the budget**, headline = **$/1M tokens**.

## Requirements → SLOs
TTFT p99 < 1–2 s · TPOT 30–80 ms (15–40 tok/s) · maximize **tokens/s/GPU (MFU)** · multi-tenant fairness · 99.9%+ avail / zero-downtime deploys · min **$/1M tokens**.

## Numbers (state assumptions)
- 70B fp8 ≈ 70 GB weights; 8×H100 ≈ **26 TB/s** → decode floor $\frac{70}{26000}\approx$ **2.7 ms/tok (~370 tok/s single stream)**; batched ≈ **~10K tok/s/node**.
- Serve 2M out-tok/s → $\frac{2\text{M}}{10\text{K}}=$ **200 nodes = 1,600 H100s** (+ prefill/HA/headroom). MoE+quant shrink it.
- **KV/seq** $= 2 \cdot L \cdot h_{kv} \cdot d_{head} \cdot \text{seq} \cdot \text{dtype}$ → ~1.3 GB @4K (80L, GQA-8) → **KV, not weights, caps batch**.
- Intensity ≈ $B$ when batched → batch raises **MFU**.

## Architecture (data plane)
`Client → Gateway (auth · rate-limit · OpenAI API · SSE) → Router (model-aware · least-loaded · prefix-aware) → Scheduler (admission · priority · batch former) → Engine replicas (TP/PP/EP · prefill→decode · paged KV) → SSE back`. Control plane (registry · autoscaler · metrics) **off the hot path**.

## Engine deep-dive (where you win)
- **Prefill** parallel/compute-bound→TTFT; **decode** 1 tok/step/bandwidth-bound→TPOT → **disaggregate** or **chunked prefill** to stop head-of-line blocking.
- **Continuous (in-flight) batching** = biggest lever (token-granular; finished leave, new join) → ~2–4×+ vs static.
- **PagedAttention**: non-contiguous KV pages → no fragmentation + **prefix sharing** + higher batch.
- **Speculative decoding**: draft $k$ → target verifies in 1 pass, accept $\min(1, p_t/p_d)$, resample residual → 2–3× fewer steps **only if acceptance high & batch not saturated**.
- **Quantization** fp8/int4 + **MoE** = fewer bytes/FLOPs per token. **GQA/MQA + KV-quant** shrink KV.
- **Model parallelism**: TP in-node (NVLink), PP cross-node, EP for MoE; replicas = data-parallel throughput.

## Scheduler
**Admission control** (saturated → queue/429, never miss SLO) · **priority queues** (interactive>batch, paid>free, WFQ fairness) · **batch formation bounded by free KV pages** · **chunked prefill** + **length-aware** placement · **preemption** (swap KV out for high-pri).

## Multi-model / multi-LoRA
**Bin-pack** models by mem+load (big→dedicated TP replicas, small→share). **Multi-LoRA**: one frozen base + per-request adapters batched together → **thousands of fine-tunes ~base cost**. **Cold start** (100s GB load) → warm pools, predictive preload, **scale-to-zero** cold models only.

## Routing
Model-aware → **least-loaded/least-queue** (not round-robin) → **prefix-aware** (same prefix → same replica → prefix-cache hit, skip re-prefill) → **session affinity** (pin stream). Honor backpressure.

## Autoscaling
Scale on **queue depth / TTFT / MFU** — **never CPU**. GPUs start in minutes → **warm pools + predictive** scaling. Mix: reserved baseline + burst + **spot (batch tier)**. Scale-to-zero cold models.

## Reliability
Health checks + **graceful drain** (finish in-flight) for zero-downtime deploys · **canary + warm previous version** = instant rollback · mid-stream failure → **fail clean + idempotent retry** (or checkpoint+resume) · multi-AZ + capacity reservations (GPUs scarce).

## Observability
**TTFT/TPOT p50/p95/p99** · **tokens/s/GPU + MFU** · batch size · **KV occupancy** · prefix-cache hit rate · **speculative acceptance** · queue depth / admission rejects / preemptions · **$/1M tokens** per model/tenant.

## Cost order
**maximize batching/MFU** → quantize (fp8/int4) + MoE → speculative (high-accept) → route to smaller model → **multi-LoRA** → scale-to-zero + spot + bin-pack → cap output/ctx + cache.

## Top tradeoffs / failure modes
decode bandwidth-bound (→batch) · **KV memory caps batch / OOM** (page, GQA, quant, evict, admission) · throughput↔latency (batch vs SLO) · prefill↔decode interference (chunk/disaggregate) · cold start (warm pools) · per-tenant deploy cost (multi-LoRA) · hot/cold skew (bin-pack/scale-to-zero) · mid-stream failure (idempotent/checkpoint) · spec no-win (low accept / big batch) · GPU scarcity (reservations/predictive).

---
[← HLD](README.md) · [Q&A](questions.md) · [Answers](answers.md) · [Index](../../README.md)
