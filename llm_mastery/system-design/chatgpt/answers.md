# System Design — Answer Key (Design ChatGPT)

> Full worked answers to [questions.md](questions.md). The bar: **lead with requirements + estimates**, identify the unique constraint (**stateful, multi-second, GPU-bound streaming**), go **deep on inference** (batching, paged KV, prefill/decode split), and weave **safety, reliability, and cost-per-token** throughout. Reference design: [README.md](README.md).
>
> Notation: $N$ = params, $D$ = tokens, $L$ = layers, $h_{kv}$ = KV heads, $d_{head}$ = head dim, TTFT = time-to-first-token, TPOT = time-per-output-token, MFU = model FLOPs utilization.

---

## 🟢 Fundamentals

**1. What makes serving an LLM different from a normal web service?**
The unit of work is a **stateful, multi-second, GPU-bound token stream**, not a cheap stateless CPU request. Consequences: latency is measured in two numbers (TTFT and TPOT) over a long-lived stream; the expensive resource is **GPU-seconds**, so the whole design optimizes **GPU utilization (MFU) and tail latency**; requests carry growing state (the KV cache); and capacity is gated by scarce, slow-to-provision accelerators. Everything else (gateway, storage, routing) is conventional — the inference tier is what's special.

**2. TTFT vs TPOT.**
**TTFT** (time to first token) is the prefill/queueing latency — it governs *perceived responsiveness* ("did it start?"). **TPOT** (time per output token) is the decode speed — it governs *how fast the answer flows* and must beat reading speed (~15–40 tok/s). They have different bottlenecks (prefill is compute-bound, decode is bandwidth-bound), so you optimize and even **scale them separately**.

**3. Why a stateless data plane?**
Stateless gateways/orchestrators scale horizontally, fail over trivially, and deploy without draining state. Conversation state lives in a **replicated store** and is loaded per request; the only unavoidable in-GPU state (the KV cache for an in-flight stream) is pinned to its replica for the duration of that turn. Statelessness everywhere else is what makes 100M-user horizontal scaling tractable.

**4. Why stream tokens (SSE)?**
A full 500-token answer takes many seconds; returning it all at once means staring at a spinner. Streaming shows the first token in ~hundreds of ms and the rest as generated, which both *feels* fast and matches how decode actually works (one token at a time). **SSE** is the right transport: unidirectional, HTTP-native, proxy/firewall-friendly, auto-reconnect. Bonus: you can run **incremental output moderation** and let users cancel early (freeing GPU/KV).

**5. Prefill vs decode.**
**Prefill** runs the whole prompt through the model in one parallel pass to build the KV cache → **compute-bound**, sets TTFT. **Decode** generates one token per step, re-reading all weights each time → **memory-bandwidth-bound**, sets TPOT. This split is the central fact of LLM serving: different bottlenecks → different optimizations → motivation for **disaggregated** prefill/decode pools.

**6. Continuous (in-flight) batching.**
Instead of forming a fixed batch and waiting for the slowest sequence, the scheduler works at **token granularity**: each step, finished sequences exit and queued ones join. This keeps the GPU saturated regardless of varied output lengths and is the **single biggest throughput lever** over static batching (often several-fold).

**7. KV cache and why it dominates memory.**
It stores the keys/values of all prior tokens so decode computes K/V only for the new token. It grows **linearly with context × batch × layers × heads**, so at long context and high concurrency it — not the weights — becomes the memory bottleneck, capping how many requests fit on a GPU. Hence GQA/MQA, KV quantization, and PagedAttention.

**8. Role of the API gateway.**
Edge of the data plane: TLS termination, **authentication/authorization**, **rate limiting and quotas** (RPS *and* tokens/min), request-ID injection for tracing, SSE plumbing/backpressure, and routing to the orchestrator. It protects the expensive backend and enforces per-tenant fairness.

**9. Why route between model sizes?**
Most queries are easy; sending them all to the flagship wastes GPU-seconds. A **router** sends easy/short turns to a small cheap model and reserves the large (or reasoning) model for hard ones — often the single biggest **cost** lever — while also enabling A/B tests and graceful degradation (fall back to a smaller model under load).

**10. Layers from client to GPU and back.**
Client → CDN/WAF → API gateway (auth/limit) → orchestrator → input safety → context builder (history + memory + RAG) → model router → inference scheduler/queue → GPU inference servers (prefill→decode) → output safety (streaming) → back through gateway as SSE → client. Async side: events → flywheel.

---

## 🟡 Core design

**11. End-to-end lifecycle of a streaming request.**
(1) Gateway authenticates, checks rate/quota, assigns a request ID. (2) Orchestrator runs **input moderation**. (3) Context builder loads history + user memory, runs **RAG retrieval**, and assembles a prompt that fits the context window. (4) Router picks a model. (5) Scheduler admits and batches it; **prefill** fills the KV cache (first token). (6) **Decode** streams tokens via continuous batching; each chunk passes **incremental output moderation** and is forwarded as SSE. (7) If the model emits a **tool call**, pause, run the sandboxed tool, append the observation, resume decode. (8) On completion, persist the turn durably and emit an event to the flywheel.

**12. Prompt assembly over the context budget.**
Keep recent turns verbatim, **summarize older turns** into a rolling summary, and add the top retrieved memory/RAG chunks — all prioritized by relevance and recency until the token budget is hit. Reserve room for the system prompt and the expected completion. Stable prefixes (system prompt, shared context) are **prompt-cached** as KV to skip re-prefill. The Context Builder owns truncation/compression so the model never overflows.

**13. How the router decides.**
Inputs: requested model/tier, task difficulty (heuristics or a small classifier), user plan/SLA, current fleet load/cost, and A/B assignment. Policy: default tier per plan, **escalate** hard prompts to the large/reasoning model, **downgrade** under saturation, honor pinned models for "custom GPTs." It also implements fallback chains (flagship → smaller → cached) for resilience.

**14. Design the inference scheduler.**
**Admission control** (reject/queue when saturated) → **priority queues** (paid > free, interactive > batch) → **continuous batcher** that forms per-step batches respecting KV-cache memory limits → **length-aware** placement so long contexts don't dominate a step → **GPU placement/bin-packing** across replicas. It exposes queue depth and TTFT as autoscaling signals and supports **chunked prefill** to interleave big prompts with ongoing decode.

**15. Many fine-tunes without a GPU each.**
Keep **one frozen base model resident** and apply per-request **LoRA adapters** (small low-rank deltas) batched together (S-LoRA / punica style): adapters are paged in/out of GPU memory and the kernel applies the right one per sequence. This serves thousands of tenant fine-tunes at near base-model cost instead of one deployment per tenant.

**16. PagedAttention.**
The KV cache is normally a contiguous per-request block, causing **fragmentation** and over-reservation. PagedAttention stores it in fixed-size **non-contiguous pages** with a lookup table (like OS virtual memory). Benefits: near-zero fragmentation → higher batch occupancy; **prefix sharing** (identical system prompts/prefixes stored once); and easy copy-on-write for parallel samples. It's the core trick behind vLLM's throughput.

**17. Streaming layer for millions of connections.**
The constraint is **concurrency, not bandwidth** (each stream is a trickle). Use **async/event-driven** gateways (epoll/Go/Netty) that hold connections cheaply — never thread-per-connection. A stream is **sticky** to the replica generating it; the gateway proxies token deltas. Enforce **max generation time** and bounded buffers so a slow client can't pin KV-cache memory. On replica death, fail the stream cleanly (client retries with an idempotency key) or resume from the last checkpointed token.

**18. RAG in the chat path.**
Embed the query → **hybrid retrieve** (dense + BM25) over a sharded ANN index → **cross-encoder rerank** the top-N → assemble a grounded prompt with **citations** → generate, abstaining if unsupported. **Evaluate retrieval and generation separately**: retrieval via recall@k / MRR / nDCG against gold docs; generation via faithfulness + answer-relevance. Localizing failures this way tells you whether to fix embeddings/chunking/rerank or the grounding prompt.

**19. Tool / function calling and bounding it.**
The model emits a **typed, schema-validated** call; the orchestrator runs it in a **sandboxed Tool Runtime** (network-isolated code exec, allow-listed APIs) and appends the observation, looping ReAct-style. Bound it with a hard **step/time/cost budget**, loop detection, argument validation, **least-privilege** scopes, and human confirmation for high-impact actions. Treat all tool/retrieved content as **untrusted** (injection surface).

**20. Safety placement (defense in depth).**
(1) **Input:** moderation + prompt-injection/jailbreak detection + PII redaction (pre-logging). (2) **Retrieval/tool boundary:** sanitize and delimit untrusted content (indirect-injection front line). (3) **Generation:** aligned model + policy system prompt + scoped tools. (4) **Output:** stream-time moderation (stop/redact) and **insecure-output-handling** guards (never `eval/SQL/shell` raw text). (5) **Cross-cutting:** abuse rate-limits, redacted audit logs, red-team + attack-success-rate release gate.

**21. Storage layer.**
Conversations → **wide-column KV** (Cassandra/Dynamo/Bigtable-class) partitioned by user for write throughput; memory/RAG vectors → **sharded vector DB** (HNSW/IVF-PQ) with per-tenant namespaces; documents → object storage + hybrid search index; uploads/images → object storage + CDN; accounts/billing → **relational** (Postgres) for transactions; sessions/quotas → **Redis**; events/logs → **Kafka** → data lake; weights/adapters → object storage + **Model Registry**. Each store honors TTLs and right-to-be-forgotten deletes.

**22. The data flywheel.**
Log prompts/outputs/feedback (thumbs, regenerate, edits, completion) → curate/dedup/PII-scrub/sample → human + AI **labeling/preferences** → **SFT → reward model → RLHF/DPO** (RL on verifiable rewards for reasoning) → **offline eval + red-team gate** → **canary → A/B → gradual rollout** via the registry, with one-click rollback. Governance: consent/opt-out, contamination checks vs eval sets, dataset versioning. The rule: **never ship on offline metrics alone**.

---

## 🔴 Senior / Staff deep dives

**23. Why disaggregate prefill and decode?**
They have opposite bottlenecks — prefill is **compute-bound**, decode is **bandwidth-bound** — and they interfere: a big prefill stalls everyone's decode (head-of-line blocking), wrecking TPOT/p99. Running them on **separate pools** lets each be sized, batched, and even hardware-matched independently, and you can scale them on different signals. Cost: added complexity and a **KV-cache handoff** (transfer or recompute) between pools, plus cross-pool networking. Worth it at scale; **chunked prefill** is the lighter-weight alternative on a single pool.

**24. Hitting p99 TTFT < 2 s under load.**
Levers: (a) **admission control + priority queues** so interactive traffic isn't stuck behind batch; (b) **disaggregated/chunked prefill** to kill head-of-line blocking; (c) **prompt/prefix caching** to skip re-prefill of system prompts; (d) **warm pools + predictive autoscaling** (GPUs are slow to start); (e) **length caps** and length-aware scheduling; (f) **smaller/quantized model** routing under pressure; (g) **regional routing** to cut network RTT; (h) keep queue depth low via headroom. Always measure the **p99**, and attack queueing + prefill first since they dominate TTFT.

**25. Long-context requests tanking p99.**
A few long sequences hog KV-cache memory (shrinking the batch) and dominate each compute step (head-of-line blocking). Fixes: **length-aware scheduling** and a separate pool/queue for long contexts; **chunked prefill** so long prompts interleave; **KV quantization / GQA / paging** to fit more; per-request **context caps** with summarization/RAG instead of stuffing; admission limits on simultaneous long requests. Isolate them so the median user is unaffected.

**26. Autoscaling a GPU fleet.**
GPUs provision slowly (minutes) and are scarce/expensive, so reactive CPU-style autoscaling fails. Use: **warm/standby pools**, **predictive scaling** on historical diurnal/weekly patterns, scale on **queue depth / TTFT / MFU** (not CPU), **bin-pack** models onto nodes, hold **reserved baseline + burst/spot** capacity, and **degrade gracefully** (smaller model, longer queue with honest wait) when capacity is exhausted rather than hard-failing. Capacity *reservations* are part of the design, not just replica counts.

**27. Speculative decoding end-to-end.**
A small **draft** model autoregressively proposes $k$ tokens; the **target** model scores all $k$ in **one** forward pass. Accept each draft token with probability $\min(1, p_\text{target}/p_\text{draft})$; on first rejection, **resample** from the normalized residual $\max(0, p_\text{target}-p_\text{draft})$ and stop, then append one bonus target token. Output distribution is provably identical to plain target sampling. It **fails to help** when acceptance is low (draft poorly aligned), the draft is too costly, or batches are already large (the target pass isn't the bottleneck) — you can add latency for no gain.

**28. Multi-region with residency + degradation.**
Independent **regional cells** (gateways + GPU fleet + storage) behind geo-DNS/global LB route users to the nearest healthy region honoring **data residency**. Conversation state replicates async within a residency boundary; the **Model Registry** distributes weights globally. **Graceful degradation** under regional GPU shortage: fall back to a smaller/quantized model, shorten context, disable non-essential tools, queue with an honest wait — never hard-fail if a degraded answer is possible. Game-day the cross-region failover.

**29. Surviving mid-stream replica failure.**
The hard case (stateful GPU work). Options: (a) **checkpoint the turn** — persist generated tokens incrementally so a new replica can resume from the last token (needs KV recompute or transfer); (b) if resume is too costly, **fail the stream cleanly** and let the client **retry with an idempotency key**, deduping so the user doesn't get a double turn; (c) keep the conversation write **durable before ack** so history stays consistent. Most systems do (b) + idempotency for simplicity, reserving (a) for very long generations.

**30. Cut cost per token 2×, in order.**
(1) **Router** easy queries to a small model (biggest lever). (2) **Quantize** (fp8/int4) and adopt **MoE** to cut active FLOPs/bandwidth. (3) Maximize **MFU** — continuous batching, paged KV, disaggregation, speculative decoding. (4) **Caching** — prefix/semantic/exact. (5) **Cap** output length and context; summarize history. (6) **Capacity mix** — reserved + spot for batch, off-peak scheduling. Validate each step against a quality eval so savings don't silently degrade users; track **$ / 1M tokens** per route.

**31. Defending against indirect prompt injection.**
Retrieved/tool content is **untrusted input** that can carry hidden instructions. Layered defense: **isolate and delimit** untrusted content (clear boundaries, never blend into the instruction channel), strip/escape embedded instructions, apply an **instruction hierarchy** (system > user > data), enforce **least-privilege** scoped tools with confirmation for high-impact actions, validate tool **outputs** before use, and run **injection red-teaming** as a gate. Assume no single filter is perfect — defense in depth.

**32. Safe rollout to 100M users.**
**Offline eval + safety red-team gate** → **shadow/canary** on a small % of live traffic → **A/B** measuring online quality + latency + cost + safety → **gradual ramp** with automatic rollback triggers (quality drop, refusal spike, latency/cost regression) → full rollout, all versioned in the **Model Registry** for one-click revert. Pin previous version warm for instant fallback. Never flip 100% on offline numbers.

---

## 🧮 Math & estimation

**33. Peak RPS and token throughput (100M DAU).**
$100\text{M} \times 10\ \text{msgs/day} = 1\text{B msgs/day}$. Average $= 1\text{B}/86400 \approx 11.6\text{K req/s}$; **peak ≈ 3–4× → ~40K req/s**. With ~500 output tokens/msg: peak generation $\approx 40\text{K} \times 500 = \mathbf{20\text{M output tok/s}}$ (plus a one-time ~1K-token prefill per request). State assumptions; the method matters more than the exact figure.

**34. GPU fleet for a 70B model.**
Decode is bandwidth-bound: per token-step read ~70 GB of fp8 weights. An 8×H100 node has ~26 TB/s aggregate HBM bandwidth; with continuous batching a tuned node sustains on the order of $\sim10\text{K}$ output tok/s. $\frac{20\text{M}}{10\text{K}} \approx 2\text{K nodes} = \mathbf{\sim16\text{K H100s}}$ for peak decode of one model — before prefill, redundancy, smaller models, multi-region, and headroom → realistically **tens of thousands of GPUs**. MoE shrinks this several-fold.

**35. KV-cache size, and for 1M sessions.**
$$\text{KV bytes} = 2 \times L \times h_{kv} \times d_{head} \times \text{seq} \times \text{batch} \times \text{dtype}$$
Example (Llama-2-13B-ish: $L{=}40$, $h_{kv}{=}40$, $d_{head}{=}128$, fp16): per token $= 2 \times 40 \times 40 \times 128 \times 2 \approx 0.82\ \text{MB}$. At 4K context $\approx 3.3$ GB **per sequence**. For **1M concurrent** 4K sessions that's ~3.3 PB of KV — obviously impossible to keep hot, which is *why* you need GQA (cut $h_{kv}$), KV quantization, paging/eviction, and capacity-bounded admission. (GQA with 8 KV heads cuts this ~5×.)

**36. TTFT budget and decode lower bound.**
TTFT ≈ auth/route (~10 ms) + input safety (~30 ms) + RAG retrieve+rerank (~80 ms) + queue + prefill (~200 ms) ≈ **~350 ms p50**. Decode lower bound per token $\approx \frac{\text{weight bytes read}}{\text{HBM bandwidth}}$: e.g. 70 GB / 26 TB/s ≈ **~2.7 ms/token** at small batch (~370 tok/s/stream), bounded by bandwidth not FLOPs.

**37. Storage per year.**
Conversations: $1\text{B msgs/day} \times \sim1\text{KB} \approx 1\ \text{TB/day} \to \sim0.4\ \text{PB/yr}$ raw; with metadata, indexes, and multi-year retention → **multi-PB**. Vectors: billions of chunks × ~1.5–3 KB → **multi-PB** sharded ANN. Blobs (uploads/images): **tens of PB** in object storage. Apply TTLs and tiering (hot → cold) to control cost.

**38. Cost per 1M tokens and what dominates.**
Dominated by **GPU-seconds**: $\text{cost} \approx \frac{\$/\text{GPU-hour} \times \text{GPUs}}{\text{tokens/hour}}$. Because decode is bandwidth-bound and re-reads all weights per token, **low MFU** is the enemy — so the cost levers are exactly the throughput levers (batching, quant, MoE, speculative decoding) plus routing to smaller models and caching. Network/storage are rounding errors next to GPUs.

**39. MoE's effect on fleet size.**
An MoE stores many experts but activates only top-$k$ per token, so **active params/FLOPs and bandwidth per token** drop several-fold for matched quality — directly shrinking the decode fleet (§34). Costs: **all experts must reside in GPU memory** (high VRAM), dynamic routing causes load imbalance, and **expert-parallel all-to-all** adds communication. Net: cheaper compute, harder serving — a deliberate trade frontier systems make.

**40. Concurrent streams per gateway node.**
With async/event-driven I/O each idle-ish SSE connection costs ~a few KB of memory + an fd, so a node holds **~100K–1M** connections, limited by **RAM, file descriptors, and ephemeral ports** — not CPU/bandwidth (each stream is a trickle). Thread-per-connection would cap you orders of magnitude lower. Scale out horizontally and pin streams to their generating replica.

---

## 🏗️ Design variations

**41. Reasoning ("thinking") model.**
It spends heavy **test-time compute** (long internal chains, sampling/search/verification), so per-request GPU time and output tokens balloon. Changes: much higher cost/latency → expose it as a premium tier behind explicit routing; **stream "thinking" progress** for UX; longer max-generation and KV budgets; possibly a **verifier/PRM** in the loop and best-of-n/self-consistency; separate capacity pool. The serving primitives are the same; the **economics and scheduling** shift toward fewer, longer, costlier requests.

**42. Add multimodal.**
Ingest path gains a **vision encoder → projector** that turns images into tokens prepended to the prompt; blobs go to object storage + CDN; prefill cost rises (more tokens). Output images require a **diffusion/generation service** as a tool. Safety expands to **image moderation** (in and out). Storage/bandwidth grow for media. Eval adds visual-hallucination/grounding checks. The text LLM core and serving stack are largely unchanged.

**43. Developer API product.**
Add API-key management, **per-key rate limits** (RPS + tokens/min) and quotas, **usage metering → billing**, tiered SLAs, **batch/async** endpoints (cheaper, off-peak, non-streaming) with webhooks, idempotency keys, and strong **multi-tenant isolation** + abuse/extraction defenses. Reuse the same inference fleet with separate priority classes so the API doesn't starve the consumer product (or vice versa).

**44. Per-user long-term memory.**
Extract salient facts from conversations (async, via an LLM), embed them, and store in a **per-user vector namespace** + a structured profile. At request time the Context Builder retrieves relevant memories and injects them under the token budget. Must be **user-inspectable, editable, and deletable** (privacy/GDPR), scoped strictly per user (no cross-tenant leakage), and decay/dedup stale facts. Treat retrieved memory as lower-trust than the system prompt.

**45. On-device / edge tier.**
Run a small **quantized (int4 GGUF)** model on the client for latency-/privacy-sensitive or offline short turns; route to the cloud flagship for hard queries. The router policy weighs difficulty, privacy flags, connectivity, and battery. Benefits: zero network RTT, privacy, server cost offload. Costs: weaker model, update/distribution logistics, device fragmentation. Hybrid: draft on-device, verify/escalate in cloud.

**46. Semantic caching.**
Cache answers keyed by **embedding similarity** of the query; on a near-hit, serve the cached completion. Great for **high-repetition** traffic (FAQs, free tier) → big latency/cost wins. Dangerous because semantically "similar" prompts can need **different** answers (context, recency, personalization), risking wrong/stale responses — so gate by a high similarity threshold, scope per tenant/context, exclude personalized/tool/RAG queries, and set TTLs. Combine with exact and prefix caches.

---

## 🐞 Debugging & ops

**47. TTFT p99 spiked 3× with flat traffic.**
Suspect the **queue/prefill** path, not the request rate: a capacity loss (unhealthy replicas, failed autoscale, a region down) shrinking the fleet; **head-of-line blocking** from a surge of long prompts; prefix-cache hit-rate drop; a noisy-neighbor batch job stealing priority; or network/RAG latency upstream of generation. Check queue depth, healthy-replica count, prompt-length distribution, and per-stage traces; mitigate with admission control + scaling + length isolation.

**48. High GPU utilization but low tok/s/GPU.**
"Utilization" can be high while doing **low-value work** → low **MFU**. Likely small effective **batch size** (memory-bound decode underfed), KV-cache pressure capping batch, no continuous batching, recompute/overhead, or no speculative decoding. Also check for stragglers and poor kernel use (no FlashAttention). Fix: raise effective batch (continuous batching, paging), quantize to fit more, add speculative decoding — optimize **tokens/sec/GPU and MFU**, not raw utilization.

**49. Responses cut off mid-sentence.**
Enumerate: **max-tokens / generation-time cap** hit; **stream timeout** or gateway/proxy idle-timeout dropping the SSE connection; client read timeout; a **replica failure** mid-stream without resume; **output moderation** truncating on a (false-positive) flag; or an EOS/stop-sequence bug. Check logs for finish-reason, replica health during the stream, and moderation hits; fix the matching layer (raise limits, lengthen proxy timeouts, add resume/idempotent retry).

**50. Cost per token crept up 30%, no deploy.**
Look for **MFU erosion**: shifting traffic mix (more long-context/RAG → bigger prefills, lower batch density), falling **cache hit rates**, growing average context (history bloat), **router drift** sending more traffic to the big model, lower batch occupancy from latency tuning, or fragmentation. Instrument **$ / 1M tokens per route**, context-length and cache-hit trends; rebalance routing, fix caching, cap context.

**51. OOM under long-context load.**
The KV cache outgrew GPU memory (it scales with context × batch). Immediate: tighten **admission control** and per-request context caps, reduce max batch for long requests, enable **KV quantization** and **paging/eviction**. Structural: **GQA/MQA** to shrink KV, length-aware scheduling, a dedicated long-context pool, and retrieval/summarization instead of stuffing full context. Add memory-pressure backpressure so the scheduler never over-commits HBM.

**52. Silent quality regression after a deploy.**
Detect via **continuous online evals** on sampled traffic (LLM-judge + human spot-checks), guardrail metrics (thumbs-down rate, regenerate rate, refusal rate), and A/B vs the previous version — with **alerts on drift**. Because you deployed via **canary → A/B**, the regression should trip a threshold and **auto-roll-back** through the registry to the warm previous version. Then root-cause offline (data, prompt template, tokenizer, eval gap) before re-shipping. The lesson: ship behind canaries with online quality monitoring, never on offline metrics alone.

---

## What strong answers share
- **Top-down discipline:** requirements → estimates → architecture → deep dive → tradeoffs. Numbers stated with assumptions.
- **Name the unique constraint:** stateful, multi-second, **GPU-bound streaming** → optimize **MFU and the tail (p99)**, not averages.
- **Depth where it counts:** continuous batching, **paged KV cache**, **prefill/decode disaggregation**, speculative decoding, quantization, MoE, multi-LoRA, scheduling/autoscaling — not just a box diagram.
- **Streaming as concurrency**, **safety as defense-in-depth**, and a **data flywheel** that never ships on offline metrics alone.
- **Cost-per-token thinking** throughout, and an honest list of **bottlenecks and failure modes** (tail latency, GPU scarcity, long-context KV blowup, hallucination, indirect injection).

---
Back to [questions](questions.md) · [HLD](README.md) · [Index](../../README.md)
