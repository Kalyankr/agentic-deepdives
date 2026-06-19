# LLM Inference Service — Answer Key

> Full worked answers to [questions.md](questions.md). The bar: **start from the bottleneck** (decode is bandwidth-bound, intensity ≈ 1 → batching is mandatory), go **deep on the engine + scheduler** (continuous batching, paged KV, prefill/decode split, speculation, quantization), handle **multi-model/multi-LoRA + cold starts + autoscaling**, and headline **tail latency (TTFT/TPOT)** and **$/1M tokens**. Reference design: [README.md](README.md).
>
> Notation: $L$ = layers, $h_{kv}$ = KV heads, $d_{head}$ = head dim, $B$ = batch, TTFT = time-to-first-token, TPOT = time-per-output-token, MFU = model FLOPs utilization, TP/PP/EP = tensor/pipeline/expert parallelism.

---

## 🟢 Fundamentals

**1. What does the service do, and why its own system?**
It takes completion/chat/embedding requests and runs them on a **GPU fleet at maximum tokens/s/GPU within TTFT/TPOT SLOs** — the serving substrate that products (ChatGPT, RAG, copilots) call into. It's its own system because LLM serving has unique mechanics absent from normal web services: **stateful multi-second token streams**, **GPU-bandwidth-bound decode**, **KV-cache memory management**, and **batching/scheduling for utilization**. Factoring it out lets many products share one optimized, autoscaled, multi-model fleet.

**2. Why is decode bandwidth-bound and prefill compute-bound?**
**Prefill** processes all prompt tokens in **one parallel pass** — lots of matmul work reusing loaded weights → **compute-bound** (sets TTFT). **Decode** emits **one token per step**, and to produce that single token it must **re-read every weight from HBM** while doing comparatively little math → **memory-bandwidth-bound** (sets TPOT). So a single decode stream is starved on bandwidth, not compute — the central fact that motivates batching.

**3. Arithmetic intensity and why batch.**
Arithmetic intensity = $\frac{\text{FLOPs}}{\text{bytes moved}}$. For decode at batch 1 it's ≈ **1** (you move ~all weights to compute one token's worth of math) → far below the GPU's compute:bandwidth ratio → **memory-bound, low MFU**. **Batching $B$ requests reuses each weight load across $B$ tokens**, so intensity ≈ $B$ and you move toward compute-bound, raising MFU and throughput. This is the **single most important lever** in LLM serving.

**4. TTFT vs TPOT.**
**TTFT** (time to first token) = prefill + queueing latency → governs *perceived responsiveness* ("did it start?"); SLO ~p99 < 1–2 s. **TPOT** (time per output token) = decode speed → governs *how fast the answer flows*, must beat reading speed (~15–40 tok/s, so ~30–80 ms/token). They have **different bottlenecks** (compute vs bandwidth) so you optimize — and sometimes **scale** — them separately.

**5. Continuous (in-flight) batching.**
Instead of forming a fixed batch and waiting for the slowest sequence to finish, the scheduler operates at **token granularity**: every step, completed sequences **exit** and queued ones **join** the running batch. This keeps the GPU busy despite wildly varying output lengths and is the **biggest throughput win** over static batching (often several-fold), because no GPU cycles are wasted waiting on the longest generation.

**6. KV cache and why it caps concurrency.**
The KV cache stores the keys/values of all prior tokens so each decode step computes K/V only for the new token (avoiding $O(T^2)$ recompute). It grows **linearly with context × batch × layers × KV-heads**, so at long context and high concurrency the **KV cache — not the weights — runs out of HBM first**, capping how many sequences fit. That's why batch formation is **KV-memory-bounded**, and why GQA/MQA, KV-quant, and paging exist.

**7. PagedAttention.**
Naive KV allocation reserves a contiguous max-length buffer per sequence → massive **internal fragmentation** and wasted HBM. **PagedAttention** stores KV in fixed-size **non-contiguous pages** with a per-sequence **block table** (like OS virtual memory): near-zero fragmentation, much higher occupancy/batch size, **prefix sharing** (identical prompt prefixes stored once), and copy-on-write for parallel samples. It's what lets engines like vLLM pack far more concurrent sequences per GPU.

**8. Why OpenAI-compatible API?**
Because the entire ecosystem of clients, SDKs, and tools already speaks `/v1/chat/completions` etc. A drop-in compatible API means **zero client rewrites**, easy migration, and instant tooling support — you compete on price/latency/models, not by forcing a new protocol. It also cleanly standardizes streaming (SSE), sampling params, function-calling, and embeddings across all your models.

**9. MFU and why it's the budget.**
**MFU** (model FLOPs utilization) = fraction of the GPU's peak FLOPs actually used for useful model compute. Cost is **GPU-seconds**, so **$/token is inversely proportional to MFU** — higher MFU = more tokens per GPU-second = lower cost. Decode's natural MFU is low (bandwidth-bound), so the whole engine — batching, paging, quantization, speculation — exists to **raise MFU at the SLO tail**. It's the headline efficiency metric.

**10. Components, gateway to GPU.**
Client → **API gateway** (auth, rate-limit, OpenAI API, SSE) → **router/load balancer** (model-aware, least-loaded, prefix-aware) → **scheduler** (admission, priority queues, batch former) → **engine replica** (model sharded TP/PP/EP, prefill→decode with paged KV) → tokens stream back through the gateway as SSE. Control plane (model registry, autoscaler, metrics) sits **off the hot path**, managing what the data plane runs.

---

## 🟡 Core design

**11. Streaming request lifecycle.**
Gateway authenticates + rate-limits → router picks a replica that has the model/adapter loaded (least-loaded, prefix-aware) → scheduler **admits** it (KV budget check) into a priority queue → **prefill** runs the prompt in parallel, building KV → **first token** emitted (TTFT) → **continuous-batched decode** emits one token/step (TPOT), streamed via SSE as deltas → on stop token / max tokens, KV is freed and the slot reused. Metrics/usage logged async.

**12. Design the scheduler.**
Three jobs: **(1) Admission control** — if the fleet is saturated (KV/compute), queue or shed (429 + `Retry-After`) rather than accept work that will miss SLOs. **(2) Priority** — multi-class queues (interactive > batch, paid > free) with **weighted fair queuing** for per-tenant fairness. **(3) Batch formation** — assemble the per-step batch bounded by **free KV pages** (not just count), with **chunked prefill** to interleave long prompts and **length-aware placement** to avoid head-of-line blocking. Support **preemption** (swap KV out) for high-priority arrivals.

**13. Multi-model packing.**
**Bin-pack** models onto GPUs by memory footprint + load: big models get dedicated multi-GPU replicas (TP across NVLink), small models **share** a GPU. A **model registry** drives placement and versioning; the router only sends a request to replicas holding that model. Balance utilization across the hot/cold model mix, use **scale-to-zero** for rarely-used models to reclaim GPUs, and keep popular models warm. The goal is to **avoid stranded GPU memory** while meeting each model's SLO.

**14. Multi-LoRA serving.**
Keep **one frozen base model resident** and apply per-request **LoRA adapters** (small low-rank deltas, MBs each) in-kernel, **batching requests with different adapters together** (S-LoRA / punica style). Adapters page in/out of GPU memory on demand. This serves **thousands of fine-tunes at roughly base-model cost** instead of a full deployment per tenant — the key efficiency trick for customization at scale. Cold adapters incur a small load latency; popular ones stay resident.

**15. KV-cache management under pressure.**
Use **PagedAttention** for fragmentation-free allocation; shrink KV with **GQA/MQA** (fewer KV heads) and **KV quantization** (int8). When HBM is tight: **evict/preempt** by priority (swap a low-priority sequence's KV to **CPU/NVMe**, resume later or recompute), apply **backpressure** to the scheduler so it never over-commits (preventing OOM), and **prefix-share** identical prompt prefixes. Admission must reserve KV up front so a running batch can't run out mid-step.

**16. How the router picks a replica.**
**Model-aware** (only replicas with the requested model/adapter loaded), then **least-loaded / least-queue** rather than round-robin (request cost varies wildly with prompt and output length), then **prefix-aware** (route requests sharing a long prefix — same system prompt or conversation — to the **same replica** to hit its prefix cache) and **session-affinity** (pin a streaming request to its generating replica). It honors per-replica **admission/backpressure** signals and spills elsewhere or queues when saturated.

**17. Autoscale a GPU fleet.**
**Don't scale on CPU.** Scale on **queue depth, TTFT, and MFU/GPU saturation**. Because GPUs provision in **minutes**, combine **predictive scaling** (diurnal/weekly patterns) with **warm/standby pools** so spikes don't blow the SLO; reactive-only autoscaling is too slow. Use a **capacity mix**: reserved baseline + autoscaled burst + **spot** for the batch tier; **scale-to-zero** cold models. Right-size parallelism (TP/PP) and replica count per model's latency target.

**18. Model loading and cold starts.**
Loading 100s of GB of weights from storage to HBM takes seconds-to-minutes. Mitigate with **warm pools** (pre-loaded replicas), **predictive preloading** of models about to be needed, **weight streaming / mmap / fast local cache**, and quantized weights (fewer bytes to move). For rarely-used models, accept **scale-to-zero** with a documented cold-start penalty while keeping popular models permanently warm. Health-check and warm up (a few dummy tokens) before adding a replica to the router.

**19. Multi-tenant fairness and quotas.**
Per-tenant **API keys** with **rate limits** (RPS *and* tokens/min) and **quotas**; **priority classes** so paid/interactive beats free/batch; **weighted fair queuing** in the scheduler so a heavy tenant can't monopolize a shared model. Cap per-request **max tokens/context** to prevent abuse, isolate batch jobs from interactive traffic, and **meter token usage per tenant** for billing and $/1M-token visibility. Optionally dedicated replicas for large/regulated tenants.

**20. Zero-downtime deploys/rollbacks.**
Drive deploys from the **model registry** with **rolling/canary** rollout: bring up new-version replicas, **warm** them, shift a small traffic fraction, watch quality/latency, then ramp — keeping the **previous version warm** for **instant rollback**. Use **graceful draining** (stop accepting new requests, finish in-flight streams) before retiring a replica so no stream is killed. Health checks gate replicas in/out of the router.

---

## 🔴 Senior / Staff deep dives

**21. Why disaggregate prefill and decode?**
They have **opposite bottlenecks** (prefill compute-bound, decode bandwidth-bound) and interfere: a big prefill stalls ongoing decodes, spiking everyone's TPOT/p99. **Disaggregation** runs them on **separate GPU pools**, each tuned and scaled independently (prefill pool sized for TTFT, decode pool for TPOT), and you can use different parallelism/hardware per pool. Cost: you must **transfer the KV cache** from prefill to decode nodes (network/complexity) and manage two pools. At scale the latency isolation is worth it; smaller setups use **chunked prefill** instead.

**22. Speculative decoding end to end.**
A small **draft** model proposes $k$ tokens cheaply; the **target** model verifies all $k$ in **one parallel forward pass**; accept the longest correct prefix via $\min(1, p_t/p_d)$ per token, resample the first rejected token from the residual, and continue. When acceptance is high you get the target's quality at **2–3× fewer target steps**. It **fails to help** when: draft↔target are poorly aligned (low acceptance), the **batch is already large** (the GPU is compute-saturated, so the "free" parallel verify isn't free), or output is highly unpredictable. Tune draft size and disable under heavy batching.

**23. Hit p99 TTFT under load — levers.**
TTFT = queue wait + prefill. Levers: **admission control + autoscaling** (don't let the queue grow unboundedly), **separate/short prefill path** (disaggregation or chunked prefill so long prompts don't block), **prefix caching** (skip re-prefill for shared system prompts), **priority queues** (interactive ahead of batch), **length-aware scheduling** (isolate huge prompts), **more/warm replicas** (GPUs start slow → warm pools), **quantization/smaller model** for the easy tier, and **cap context**. Measure and attack the dominant term (queue vs prefill).

**24. Long prompts tanking TPOT — diagnose and fix.**
A few long prefills/contexts are causing **head-of-line blocking** and hogging KV, so the shared decode batch slows for everyone. Fixes: **chunked prefill** (split long prefills across steps so decode keeps progressing), **length-aware scheduling** (separate queue/pool for long requests), **disaggregate** prefill from decode, **cap/segment** max context, and **admission limits** on concurrent long requests. Monitor per-request length distribution and KV occupancy to confirm the culprit.

**25. Survive replica failure mid-generation.**
The stream is **stateful** (its KV lives on the failed replica). Two strategies: **(a) fail cleanly** — end the SSE with a well-defined error and let the client **idempotently retry** (idempotency key dedupes), accepting a restart; or **(b) resume** — periodically **checkpoint** progress (generated tokens / KV) so another replica recomputes the prefix and continues. Most services do (a) for simplicity plus health-check-driven fast failover; (b) is for long, expensive generations. Either way, **draining** handles planned restarts without dropping streams.

**26. Cut $/1M tokens by 2× — in order.**
(1) **Maximize batching/MFU** — continuous batching, paged KV, prefix sharing (biggest, free quality-wise). (2) **Quantize** weights/KV (fp8/int4) and use **MoE** — fewer bytes/FLOPs per token. (3) **Speculative decoding** where acceptance is high. (4) **Route to a smaller model** for easy requests (tiered). (5) **Multi-LoRA** instead of per-tenant deployments. (6) **Scale-to-zero idle models + spot for batch + bin-pack** to kill stranded GPUs. (7) **Cap** output/context and **cache**. Most steps are quality-neutral; ordering does the cheap, safe wins first.

**27. Prefix-aware routing.**
Many requests share a long identical prefix (a big shared **system prompt**, or successive turns of one **conversation**). If you route them to the **same replica**, that replica's **prefix cache** already holds the prefix's KV, so it **skips re-prefilling** it — cutting TTFT and prefill compute dramatically. The router hashes/looks up the prefix and prefers the replica that cached it (with load balancing to avoid hotspots). Big win for shared-prompt and multi-turn workloads; requires cache-aware routing logic.

**28. Scale-to-zero cold models without hurting warm SLOs.**
Classify models by traffic: **hot** models stay pinned/warm with reserved capacity; **cold** models **scale to zero** and load on demand, accepting a **cold-start latency** (set client expectations / async for those). Keep a **shared warm pool** of spare GPUs to absorb cold-start loads quickly, **predictively preload** models with known schedules, and **isolate** cold-start loading from warm replicas' GPUs so a load spike doesn't steal bandwidth/memory from latency-critical traffic. Bin-pack so reclaimed GPUs are actually reusable.

---

## 🧮 Math & estimation

**29. Single-stream decode latency, 70B on 8×H100.**
Decode is bandwidth-bound, so latency/token ≈ **weight-bytes-read ÷ aggregate HBM bandwidth**. 70B in fp8 ≈ 70 GB weights; 8×H100 ≈ **~26 TB/s** aggregate. $\frac{70\text{ GB}}{26\text{ TB/s}} \approx 2.7$ ms/token → **~370 tok/s single stream** (floor; real is a bit higher with overhead). The GPU is mostly idle on compute here — which is exactly why **continuous batching** is needed to turn that spare compute into aggregate throughput (~10K+ tok/s/node batched).

**30. KV bytes per sequence and batch.**
$$\text{KV bytes} = 2 \cdot L \cdot h_{kv} \cdot d_{head} \cdot \text{seq} \cdot \text{dtype}$$
(the 2 = K and V). E.g. $L{=}80$, $h_{kv}{=}8$ (GQA), $d_{head}{=}128$, fp16 (2 B), seq 4096: per token $= 2\cdot80\cdot8\cdot128\cdot2 \approx 327$ KB, ×4096 ≈ **~1.3 GB/sequence**. If ~**60 GB** HBM is free for KV after weights, max batch ≈ $60/1.3 \approx$ **~45 concurrent 4K sequences**. Longer context or fewer GQA groups → fewer sequences. **KV — not weights — caps batch.**

**31. GPUs to serve 2M output tok/s.**
With ~**10K output tok/s/node** (8×H100, continuous batching), nodes $= \frac{2\text{M}}{10\text{K}} =$ **200 nodes = 1,600 H100s** for steady decode — plus prefill capacity, redundancy/HA, and burst headroom (call it ~1.3–1.5×). **Quantization and MoE** raise per-node throughput and shrink this; long-context workloads (KV-limited) raise it. The estimate frames the fleet as **throughput ÷ per-node throughput + overhead**.

**32. Speculative acceptance criterion.**
For a draft token with target prob $p_t$ and draft prob $p_d$, **accept with probability** $\min\!\left(1, \frac{p_t}{p_d}\right)$; if rejected, **resample** from the normalized residual $\propto \max(0, p_t - p_d)$. This **provably preserves the target model's output distribution** (no quality loss) while letting you verify $k$ drafted tokens in one target pass. Expected accepted tokens per pass rises with draft↔target alignment, which is what makes it a speedup.

**33. Batch size vs TTFT/TPOT.**
Bigger batch → **higher throughput/MFU** (weights reused more) and lower **$/token**, but each step does more work so **per-step latency rises** → **TPOT increases**, and a fuller pipeline + KV pressure can **raise TTFT/queueing**. So batch is tuned to the **SLO**, not maxed: grow it until TPOT/TTFT approach their p99 budgets, then stop. Interactive traffic runs smaller batches (latency); batch/offline traffic runs huge batches (throughput) — hence separate pools.

**34. Continuous vs static batching gain.**
Static batching wastes GPU time **waiting for the longest sequence** in each fixed batch (short ones finish and idle their slots). Continuous batching refills those slots **every step**, so utilization stays high regardless of length variance — typically a **~2–4× (often more) throughput improvement** on realistic mixed-length traffic, with the gain growing as output-length variance grows. It's the highest-ROI engine feature after the KV cache itself.

---

## 🏗️ Design variations

**35. Serving reasoning models (long test-time compute).**
Reasoning models emit **very long hidden chains** before the answer, so generations are long, KV-heavy, and latency-variable. Design changes: **schedule for long, bursty generations** (length-aware, generous KV budgets, preemption), expose **TTFT-to-final-answer** SLOs distinct from token flow, **cap/stream** thinking budgets, and **route** between "think hard" and "fast" tiers by query difficulty. Throughput accounting shifts to **tokens including reasoning**, and prefix caching of the prompt helps less (output dominates). Disaggregation and KV offload matter more.

**36. Embeddings-serving service.**
Embeddings are **single forward pass, no autoregressive decode** → **compute-bound, no KV cache, no streaming**. So it's mostly a **prefill/encode** problem: maximize throughput with **large batches**, short fixed-ish sequences, and high MFU; latency is one-shot (no TPOT). You can pack **huge batches** and use smaller/cheaper GPUs, autoscale on QPS/queue, and heavily **cache** identical inputs. Much simpler than chat serving — the hard parts (KV, continuous batching, decode) largely disappear.

**37. Batch / offline inference tier.**
For non-interactive jobs (bulk scoring, dataset generation), run a **separate low-priority tier** optimized for **throughput, not latency**: **huge batch sizes**, **spot/preemptible GPUs** (checkpoint + requeue on preemption), aggressive packing, and no SSE. Submit via an **async job API** (file in → file out), schedule it to **backfill** spare capacity behind interactive traffic (preempt it first), and price it cheaper. Isolation from the interactive tier protects p99.

**38. Heterogeneous-hardware serving.**
Different GPU types (e.g. H100 vs A100 vs L4) have different memory/bandwidth/cost. **Tier by workload**: big/latency-critical models on top-end multi-GPU nodes, small/embedding/batch models on cheaper GPUs. The router/scheduler must be **hardware-aware** (capabilities, per-type throughput, KV capacity) and place models where they're cost-efficient while meeting SLOs. Maintain **per-hardware perf profiles** for autoscaling math. Adds placement complexity but improves $/token by matching work to the cheapest GPU that meets the SLO.

---

## 🐞 Debugging & ops

**39. Low MFU but GPUs look "busy."**
"Busy" (high utilization %) ≠ high MFU — the GPU can be **stalled on memory bandwidth** (decode at small batch) or **launch/overhead-bound**, doing little useful FLOPs. Diagnose with profiling: check **batch size / KV occupancy** (too small → memory-bound → increase batching), **prefill/decode interference**, kernel efficiency (use **CUDA graphs / fused kernels**), and whether you're **bandwidth-bound** (then quantize / raise batch / speculate). The fix is almost always **bigger effective batches** (continuous batching, paged KV) or reducing bytes moved (quantization).

**40. TPOT degrades as load rises.**
As more sequences join, the **decode batch grows** (more per-step work) and **KV pressure** rises, so each step takes longer → TPOT climbs. Also **long requests** in the batch slow everyone (head-of-line). Mitigate: **cap batch size to the TPOT SLO** (don't over-batch interactive traffic), **length-aware scheduling / chunked prefill**, **separate interactive vs batch pools**, autoscale out to spread load, and shed/queue via admission control rather than degrading the running batch. There's a real **throughput↔TPOT tradeoff** to tune.

**41. OOM under long-context bursts — fix without new GPUs.**
Long contexts blow up **KV memory** (linear in seq×batch), overflowing HBM. Without buying GPUs: **admission control** that reserves KV up front and **bounds concurrent long requests**, **PagedAttention** (kill fragmentation), **GQA/MQA + KV quantization** (shrink KV), **KV offload/eviction** to CPU/NVMe under pressure, **chunked prefill**, and **cap max context** per request/tenant. The root cause is over-committing KV — the scheduler must treat **free KV pages** as a hard admission constraint so it never OOMs mid-step.

**42. Speculative decoding made throughput worse.**
Speculation adds draft-model cost and a verify pass; it only pays off when **acceptance is high** *and* the GPU has **spare compute**. It backfires when: the **batch is already large** (GPU compute-saturated → the parallel verify isn't free, and you've added draft overhead), **draft↔target alignment is poor** (low acceptance → wasted drafts), or the draft model is too big. Fix: **disable speculation under heavy batching** (gate on batch size/load), use a **smaller, better-aligned draft**, and tune $k$. It's a **low-batch latency optimization**, not a high-throughput one.

---

## What strong answers share
- **Start from the bottleneck:** decode is **bandwidth-bound** (intensity ≈ 1) → **continuous batching** is mandatory; prefill is compute-bound → split/chunk/disaggregate it.
- **Go deep on engine + scheduler:** paged KV, **KV-memory-bounded** batch formation, chunked prefill, speculative decoding, quantization, model parallelism.
- **Handle multi-model reality:** **multi-LoRA** on a shared base, bin-packing, **cold starts**, scale-to-zero, **prefix-aware routing**.
- **Autoscale on the right signals** (queue/TTFT/MFU + warm pools) — never CPU — because GPUs provision slowly and dominate cost.
- **Headline metrics:** **tail TTFT/TPOT** and **$/1M tokens**, with explicit throughput↔latency and KV-memory tradeoffs.

---
Back to [questions](questions.md) · [HLD](README.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
