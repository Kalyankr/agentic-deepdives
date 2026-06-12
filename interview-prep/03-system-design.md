# 03 · System Design

> The system-design round tests whether you can take a vague prompt ("design ChatGPT") to a concrete
> architecture **with numbers**: capacity (QPS, GPUs, storage, bandwidth), latency budget, cost, and
> the failure/trade-off analysis. Drive the conversation; state assumptions; quantify everything.

Contents: [A reusable framework](#a-reusable-framework) · [Design 1: LLM serving at scale](#design-1--llm-inference-serving-at-scale-chatgpt-like) ·
[Design 2: RAG at scale](#design-2--rag-over-a-large-corpus) · [Design 3: Agent platform](#design-3--an-agent-platform) ·
[Design 4: Training cluster](#design-4--a-pretrainingfine-tuning-cluster) · [Capacity cheat-sheet](#capacity-estimation-cheat-sheet)

---

## A reusable framework

1. **Clarify scope & SLOs.** Users/QPS, prompt/output lengths, p50/p95 latency targets, quality bar,
   cost ceiling, safety requirements. *Write the assumptions on the board.*
2. **Back-of-envelope capacity.** Convert load → tokens/sec → GPUs; size KV-cache, storage, bandwidth.
3. **High-level architecture.** Boxes and arrows: gateway → router → model pool(s) → caches → stores.
4. **Drill into the hard parts.** Batching, KV cache, autoscaling, retrieval, evals, safety.
5. **Trade-offs & failure modes.** What breaks at 10×? Fallbacks, overload, degraded modes.
6. **Measure.** The dashboards/SLOs you'd watch (TTFT, TPOT, throughput, cost/req, error budget).

---

## Design 1 — LLM inference serving at scale (ChatGPT-like)

### Step 1 — Assumptions (say these out loud)
- 100 M DAU × ~10 messages/day ⇒ **1 B requests/day**.
- Average QPS = 1e9 / 86,400 ≈ **~11.6 k QPS**; peak ≈ 3× ⇒ **~35 k QPS**.
- Per request: ~500 input tokens (with history), ~300 output tokens.
- Model: a 70B-class model, **bf16** (140 GB of weights), GQA, 80 layers.
- SLO: **TTFT < 1 s**, **TPOT < 50 ms** (≈ 20 tok/s/user, faster than reading speed).

### Step 2 — Capacity math
- **Decode load:** 35 k req/s × 300 tok = **~10.5 M output tok/s** at peak.
- **Per-node throughput (assumption):** one 8×H100 node with TP + continuous batching ≈ **~5 k output
  tok/s** for a 70B model (order-of-magnitude; you'd measure it — see [lab04](../labs/lab04_inference_bench/)).
- **Nodes needed:** 10.5 M / 5 k ≈ **~2,100 nodes ≈ ~16.8 k GPUs** at peak. (State that real systems
  cut this hard with batching, quantization, caching, and smaller routed models.)
- **KV cache:** `2·layers·kv_heads·d_head·bytes` per token = `2·80·8·128·2 ≈ 320 KiB/token`. An
  800-token conversation ⇒ **~256 MiB/sequence**. After sharded weights (~17.5 GB/GPU), ~500 GB/node
  is free for KV ⇒ **~1,900 concurrent sequences/node** — this, not FLOPs, often caps concurrency.
- **Storage:** model checkpoint ~140 GB ×(versions, replicas across regions) + request logs (PII-aware)
  + eval datasets. Bandwidth: replicating a 140 GB checkpoint to 2,000 nodes is the real "deploy" cost
  → use a fast object store + peer-to-peer/torrent-style distribution.

### Step 3 — Architecture
```
Client → API gateway (authn, rate-limit, quotas)
       → Safety pre-filter (moderation, prompt-injection checks)
       → Router (model selection: small vs large; cache lookup)
       → Inference pool (vLLM/TGI; TP within node; continuous batching; paged KV)
            ├── Prefix cache (shared system prompts)
            └── Prompt/response cache (exact + semantic)
       → Safety post-filter (output moderation) → stream tokens back (SSE)
Cross-cutting: autoscaler, observability (TTFT/TPOT/cost), eval/canary pipeline
```

### Step 4 — The levers that matter
- **Continuous batching + PagedAttention** for throughput; **disaggregate prefill/decode** pools
  (compute-bound vs bandwidth-bound) so each scales independently.
- **Quantization** (INT8/FP8 weights, KV-cache quant) to cut bytes moved ⇒ more tok/s and bigger batch.
- **Caching:** prefix cache for the (large) shared system prompt; **semantic cache** for repeat
  queries; both slash TTFT and cost.
- **Routing / cascades:** send easy turns to a small/cheap model, escalate hard ones — most traffic
  stays cheap.
- **Autoscaling** on queue depth / TTFT, with regional pools for latency; **load-shed** gracefully
  (queue, then degrade to a smaller model) under overload.

### Step 5 — Trade-offs & failures
Latency vs throughput (bigger batch raises TPOT); cost vs quality (model size, quantization); the KV
cache is the scaling wall at long context. Failure modes: thundering-herd on a viral prompt (mitigate
with caching + admission control), a bad model rollout (canary + instant rollback), region outage
(multi-region replicas).

---

## Design 2 — RAG over a large corpus

### Assumptions
50 M documents, ~1 kB each; 1 k QPS of questions; answers must be **grounded with citations**;
freshness within minutes for new docs.

### Capacity
- **Chunks:** 50 M docs × ~3 chunks ≈ **150 M chunks**. Embeddings at 768-dim float32 = 3 KB each ⇒
  **~450 GB** of vectors (use float16 or **PQ** compression to cut 4–32×).
- **Index:** HNSW (high recall, more memory) or **IVF-PQ** (compressed, disk-friendly) for 150 M
  vectors; shard across nodes; replicate for QPS.
- **Ingestion:** chunk → embed → upsert; embedding is the throughput bottleneck (batch on GPU).

### Architecture
```
Query → (optional) query rewrite/expansion
      → Hybrid retrieval: BM25 (lexical) + ANN (semantic) → fuse (RRF)
      → Rerank top-100 → top-5/8 with a cross-encoder
      → Build grounded prompt (chunks + citations) → LLM → answer + citations
      → Eval/log: faithfulness, citation accuracy, retrieval recall@k
Offline: ingestion pipeline (chunk/embed/upsert), index build, eval set
```

### The hard parts (and senior answers)
- **Chunking:** semantic/structure-aware beats fixed-size; overlap to avoid cutting context. It's the
  highest-leverage knob on quality.
- **Hybrid + rerank:** dense retrieval misses exact terms (IDs, names); add BM25 and fuse (RRF), then
  a cross-encoder reranker — biggest quality win per unit effort.
- **Freshness:** stream upserts; handle deletes/updates; periodic index compaction.
- **Eval:** retrieval (recall@k, MRR/nDCG) **and** generation (groundedness/faithfulness, citation
  correctness). Without this you can't tell retrieval from generation failures.
- **Failure modes:** hallucination when retrieval is empty (detect low scores ⇒ "I don't know"),
  stale/duplicated chunks, context-window overflow (rerank + compress).

---

## Design 3 — An agent platform

### Scope
A service that runs tool-using agents (search, code-exec, internal APIs) for many tenants, reliably
and safely.

### Architecture
```
Task → Orchestrator (plan / ReAct loop, max-steps + budget caps)
     → Tool gateway (typed schemas, authz per tool, sandboxed execution)
     → Memory: short-term (scratchpad) + long-term (vector store)
     → Model pool (tool-calling) with prefix caching for system+tools
     → Trajectory logger (every step, tokens, latency, cost) → Eval harness
Guardrails: prompt-injection defense, allow-lists, human-approval for high-impact tools, spend limits
```

### Senior points
- **Reliability:** success/step `p` over `n` steps ≈ `pⁿ` — compounding errors. Cap steps, add
  verification/reflection, retries with backoff, and **checkpoint/resume** (durable execution) for
  long tasks.
- **Cost/latency:** agents fan out tokens fast; **prefix-cache** the (huge, repeated) system prompt +
  tool defs; route sub-steps to cheaper models; bound max steps and token budget per task.
- **Safety (critical):** treat tool output as **data, not instructions** (prompt injection); sandbox
  code-exec (no network, ephemeral FS, resource caps); allow-list actions; **human-in-the-loop** for
  irreversible/high-impact tools; least privilege + per-tenant spend caps.
- **Multi-agent only when it pays:** parallelizable subtasks or strong separation of concerns;
  otherwise the coordination + token overhead isn't worth it.
- **Observability:** full trajectories, an **eval suite** (≥dozens of tasks: success rate w/ CI, avg
  steps, cost, p95 latency) gating every change.

---

## Design 4 — A pretraining/fine-tuning cluster

### What to cover
- **Parallelism plan:** 3D = TP within node (NVLink) × PP across nodes × DP across replicas; **FSDP/
  ZeRO-3** when states exceed GPU memory (~16–20 B/param). Size it: 70B ⇒ ~1.2–1.4 TB of states ⇒
  shard across many GPUs.
- **Data pipeline:** dedup → quality filter → tokenize → **shard/pack** into fixed-length sequences;
  stream from object store; ensure **reproducible** shuffling and exactly-once-ish consumption.
- **Throughput:** target high **MFU** (model FLOPs utilization, ~40–55% is good); overlap comm with
  compute; gradient checkpointing for memory.
- **Reliability:** frequent **checkpointing** (failures are constant at 1000s of GPUs), automatic
  restart, loss-spike detection (skip/replay bad batches), elastic training.
- **Compute estimate:** `C ≈ 6ND`. e.g. 70B on 1.4T tokens ⇒ `6·7e10·1.4e12 ≈ 5.9e23` FLOPs; at
  ~400 TFLOP/s effective per H100 ⇒ ~`4.1e8` GPU-seconds ≈ **~410 k GPU-hours** ≈ ~1,000 H100s for
  ~2.5 weeks. (Show the math; they care about the method.)

---

## Capacity estimation cheat-sheet

| Quantity | Formula |
|----------|---------|
| Avg QPS | `daily_requests / 86,400` |
| Peak QPS | `avg_QPS × peak_factor` (2–5×) |
| FLOPs/token (inference) | `≈ 2 · N_params` |
| Training compute | `C ≈ 6 · N · D_tokens` |
| Weight memory | `N_params × bytes` (bf16 = 2) |
| Training state memory | `≈ 16–20 bytes × N_params` (mixed-precision Adam) |
| KV cache / token | `2 · n_layers · n_kv_heads · d_head · bytes` |
| Decode throughput (1 GPU) | `≈ HBM_bandwidth / bytes_read_per_token` |
| GPUs for serving | `peak_output_tok_per_s / tok_per_s_per_GPU` |
| Vector store size | `n_vectors × dim × bytes` (compress with PQ) |

> **Rules of thumb:** decode is **bandwidth-bound** (batch + quantize); prefill is **compute-bound**;
> the **KV cache** is usually the memory wall; **cache** the repeated prefix; **route** easy traffic
> to cheap models. Always close with "here's what I'd measure and how I'd scale to 10×."
