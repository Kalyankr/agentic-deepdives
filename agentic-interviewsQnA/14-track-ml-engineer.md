# 14 — Track: ML Engineer

> **Role in one line:** you make agentic systems *run* — serving, scaling, inference optimization, data/retrieval pipelines, deployment, monitoring, and cost-at-scale. Less "what should the agent do," more "make it fast, cheap, reliable, and observable in production."

> This lens emphasizes **infra, serving, MLOps, and systems** on top of the core kit.

---

## What the interview actually tests

| They probe | Because the job is | Where in the kit |
|------------|--------------------|------------------|
| LLM serving & inference optimization | You own latency/throughput/cost | below + [09c §H](09c-followup-questions.md) |
| Retrieval/vector infra at scale | RAG pipelines are data systems | [03](03-memory-and-context.md) + below |
| Deployment, autoscaling, reliability | Agents are distributed services | [07](07-system-design.md), [08](08-production-evaluation-security.md) |
| Observability & monitoring (drift, SLOs) | You keep it healthy | [08](08-production-evaluation-security.md) |
| Pipelines (ingestion, eval, CI/CD) | Repeatable, automated ops | [08](08-production-evaluation-security.md) |
| Cost/capacity engineering | Budgets at scale are real | below |

**They expect systems depth:** batching, KV cache, queues, sharding, SLOs, failure modes. Agent concepts matter, but infra mechanisms win this round.

---

## Priority reading (in order)
1. [09c — Follow-Ups](09c-followup-questions.md) §H — token economics, OTel spans, speculative decoding, idempotency.
2. [08 — Production, Eval & Security](08-production-evaluation-security.md) — observability, reliability, deployment.
3. [07 — System Design](07-system-design.md) — async/queue/worker, scaling, components.
4. [03 — Memory & Context](03-memory-and-context.md) — vector stores, embeddings, retrieval.
5. [04](04-tools-and-function-calling.md) — sandboxing/tool execution as infra.

---

## ML-Engineer-specific Q&A (new)

### Serving & inference optimization

**Q. How do you serve an LLM agent at scale? Walk the request path.**
"Client → gateway (auth, rate limit) → queue for async/long tasks → stateless agent workers that call a model-serving layer (self-hosted vLLM/TGI or a provider) and a tool layer → memory/vector/SQL stores → response, streamed back. Stateless workers autoscale behind the queue; long agent runs are jobs with IDs and checkpoints, not blocking calls. Cross-cutting: caching, tracing, budgets, and backpressure when downstream saturates."

**Q. Explain the big inference-optimization levers.**
"**Continuous (in-flight) batching** — merge requests dynamically to keep the GPU busy; the single biggest throughput win. **KV-cache** + **paged attention** (vLLM) — reuse attention state and page it like virtual memory to fit more concurrent sequences. **Quantization** (INT8/INT4/FP8) — shrink weights for memory/throughput at a small quality cost. **Speculative decoding** — a draft model proposes tokens the target verifies in parallel, cutting latency with identical output. **Prompt/prefix caching** — skip recompute of shared static prefixes. Plus tensor/pipeline parallelism for big models across GPUs."

**Q. Latency vs. throughput — how do you trade them?**
"Bigger batches raise throughput (tokens/sec/GPU) but add queuing latency per request; smaller batches cut latency but waste GPU. I set an SLO (e.g., p95 first-token < X, inter-token < Y), tune max batch size / batch timeout to meet it, and separate interactive traffic from batch/offline traffic onto different pools. Streaming hides total latency; speculative decoding helps the per-token tail."

**Q. How do you right-size capacity and control GPU cost?**
"Estimate load: QPS × tokens/request × steps (agents multiply by steps and agents). Benchmark tokens/sec/GPU at the target SLO to get GPUs needed, add headroom for peaks, and autoscale on queue depth / GPU utilization. Cost levers: quantization, continuous batching, routing cheap models for easy steps, caching, and spot/preemptible for offline jobs. Track $/request and tokens/request as SLO-level metrics."

**Q. Self-host vs. API — how do you decide?**
"API for speed-to-market, elastic scale, and frontier quality without ops burden. Self-host (vLLM/TGI/Triton) when volume makes it cheaper at scale, when you need data residency/privacy, custom/fine-tuned models, tight latency control, or no per-token vendor cost. Often hybrid: self-host the high-volume small model, call an API for the hard long-tail."

### Retrieval & data pipelines

**Q. Scale a vector store to a billion vectors.**
"Pick an ANN index for the recall/latency/memory budget — HNSW (fast, memory-heavy) or IVF-PQ (compressed, cheaper memory). Shard across nodes, replicate for QPS/HA, and use quantization (PQ/SQ) to fit memory. Metadata filters for tenant/security, a write path that batches upserts, and a reindex/compaction strategy. Monitor recall@k (it drifts as data grows), p99 query latency, and index build time."

**Q. Design the ingestion pipeline for RAG that stays fresh.**
"Source connectors → parse/clean → chunk → embed (batched) → upsert to the index with metadata, orchestrated by a workflow engine (Airflow/Dagster/etc.). Incremental updates via change-data-capture or event triggers, dedup and versioning, idempotent upserts, and dead-letter handling for failures. Backfills run as batch jobs; I monitor lag (doc → searchable), embedding throughput, and failure rates. Re-embed when the embedding model changes — that's a full reindex, planned."

**Q. The embedding model needs upgrading in production. How?**
"It's a reindex, since vectors aren't comparable across models. Build the new index in parallel (shadow), backfill embeddings as a batch job, validate retrieval quality (recall, downstream eval) on the new index, then cut over behind a flag with rollback. Dual-write during transition so nothing goes stale. Plan capacity for the temporary 2× storage."

### Deployment, reliability, observability

**Q. How do you deploy an agent/model change safely?**
"Version everything — model, prompts, tools, index. Roll out blue-green or canary: shadow first (log, don't act), then a small traffic %, comparing task success, latency, cost, and guardrail/error rates against control, with automatic rollback on regression. Gate on the offline eval set in CI before any traffic. For models, warm up/precompile and pre-load weights to avoid cold-start cliffs."

**Q. What do you monitor for an agent in prod, and what SLOs?**
"System SLOs: availability, p50/p95/p99 latency (first-token + total), error/timeout rates, queue depth, GPU utilization, $/request. Quality/health: task success rate, tool-error rate, loop/step-count distribution, guardrail-trigger and escalation rates, and **drift** in input distribution and output quality. I build on OpenTelemetry GenAI spans (LLM + tool calls with token/cost attributes) plus an LLM-observability tool, and alert on SLO burn and cost spikes."

**Q. How do you detect and handle drift?**
"Monitor input distribution (embeddings/topics), output quality proxies (guardrail triggers, user feedback, judge scores on a sample), and retrieval quality (recall@k). Alert on shifts, keep a rolling eval set sampled from live traffic, and have a retrain/re-index or prompt-fix playbook. For RAG specifically, stale indexes masquerade as model drift — check freshness first."

**Q. Downstream provider rate-limits / a tool is down. What happens?**
"Backpressure and graceful degradation: token-bucket rate limiting, request queue with bounded depth, retries with exponential backoff + jitter, and circuit breakers that fail fast when a dependency is unhealthy. Fall back to a secondary model/provider or cached/partial results, shed load on non-critical paths, and surface a degraded-but-working experience rather than cascading failures. Idempotency keys so retries don't double-execute side effects."

**Q. How do you sandbox code-execution / computer-use tools safely at scale?**
"Run them in isolated, ephemeral containers/microVMs (gVisor/Firecracker-style) with least privilege: no network unless required, read-only FS where possible, CPU/mem/time limits, and per-job teardown. Scoped, short-lived credentials, egress allow-lists, and full audit logging. Pool warm sandboxes for latency but never reuse state across tenants."

---

## Coding / systems round (ML Engineer)
- **Systems design of infra:** a model-serving gateway, a RAG ingestion pipeline, an autoscaling worker pool with a queue, a caching layer.
- **Concrete coding:** a token-bucket rate limiter, retry-with-backoff + circuit breaker, an LRU/semantic cache, batching/queue logic, a streaming response handler.
- **Data structures/algorithms** at a normal SWE bar, plus back-of-envelope **capacity math** (QPS, tokens, GPUs, memory, $).

## Signals they grade
✅ Knows batching/KV-cache/quantization/speculative decoding · reasons about SLOs and capacity math · designs idempotent, backpressured, observable systems · plans reindex/rollout/rollback · separates interactive vs. batch traffic.
🚩 Blocks a thread for a long agent run · no autoscaling/queue · ignores cold starts and rate limits · no rollback/canary · can't estimate GPUs or $/request · treats vector DB as "just a library."

## 1-week plan
- **D1–2:** Serving + inference Q&A above; learn vLLM/continuous batching/KV-cache/quantization/speculative decoding cold.
- **D3:** [07](07-system-design.md) async/queue/worker + capacity math; design a serving stack on paper.
- **D4:** [03](03-memory-and-context.md) + vector-infra Q&A; design a billion-scale retrieval + ingestion pipeline.
- **D5:** [08](08-production-evaluation-security.md) observability/reliability + drift/rollout Q&A; define SLOs.
- **D6:** Coding drills (rate limiter, backoff, cache, batching); [11 mock](11-mock-interview.md) round 3 (infra framing) + 4.
- **D7:** [09c §H](09c-followup-questions.md) + flashcards (production/security tags) + behavioral ops stories.
