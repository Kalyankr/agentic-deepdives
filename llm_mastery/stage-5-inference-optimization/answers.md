# Stage 5 — Answer Key (Inference Optimization)

> Full worked answers to [interview-questions.md](interview-questions.md). The bar: identify the **correct bottleneck** (memory bandwidth vs compute) *before* optimizing, do KV-cache/latency math live, know the tradeoff behind every speedup, and think in **p50/p99 and cost-per-token**.

---

## 🟢 Fundamentals

**1. Two phases of autoregressive generation.**
**Prefill** — process the whole prompt in one parallel forward pass, building the KV cache (compute-bound). **Decode** — generate one token at a time, each step attending over the cached KV (memory-bandwidth-bound). They have completely different performance characteristics.

**2. KV cache — what and why.**
During decode, attention needs the keys/values of all *previous* tokens. The KV cache stores them so each new step only computes K/V for the **one new token** and reuses the rest — turning per-step cost from recomputing $O(T^2)$ into $O(T)$. The price is memory that grows linearly with context × batch × layers × heads.

**3. Greedy vs sampling vs beam.**
- **Greedy:** take argmax each step — deterministic, fast, but repetitive; good for short factual/extraction.
- **Sampling** (temperature/top-p): draws from the distribution — diverse/creative, the default for chat.
- **Beam search:** keeps the top-$b$ partial sequences by total probability — good for **closed-ended** tasks (translation, where one best answer exists), bad for open-ended (bland, repetitive) and slower.

**4. Temperature, top-k, top-p.**
- **Temperature** $T$ scales logits before softmax: $T{<}1$ sharpens (more deterministic), $T{>}1$ flattens (more random).
- **Top-k** restricts sampling to the $k$ highest-probability tokens.
- **Top-p (nucleus)** restricts to the smallest set of tokens whose cumulative probability ≥ $p$ — an adaptive cutoff that adjusts to how peaked the distribution is.

**5. Quantization — what and why faster.**
Represent weights (and sometimes activations) in lower precision (int8/int4) instead of fp16. Since decode is **memory-bandwidth-bound**, moving 4-bit weights instead of 16-bit means ~4× less data off HBM per token → ~proportional speedup, plus ~4× smaller memory footprint (bigger models/batches fit).

**6. What FlashAttention solves.**
The naive attention materializes the $T\times T$ score matrix in HBM, making it **memory-IO-bound** and $O(T^2)$ memory. FlashAttention computes attention in **tiles in on-chip SRAM** with an online softmax, never writing the full score matrix — same math, far fewer HBM reads/writes, $O(T)$ memory, and big speedups at long context.

**7. Continuous (in-flight) batching.**
Instead of waiting for a whole batch to finish (static batching wastes GPU while short requests wait for long ones), the scheduler **adds and removes requests at the token level** every step, keeping the GPU saturated. Dramatically higher throughput under mixed-length, streaming traffic (vLLM/TGI).

**8. Speculative decoding (high level).**
A small **draft** model proposes several tokens cheaply; the large **target** model verifies them in **one parallel forward pass** and accepts the longest correct prefix. Produces the *exact same distribution* as the target while running fewer expensive target steps — latency win when the draft is accurate.

**9. Three serving frameworks + use case.**
- **vLLM:** PagedAttention + continuous batching — high-throughput GPU serving / online APIs.
- **TensorRT-LLM:** NVIDIA-optimized compiled kernels — lowest latency on NVIDIA hardware.
- **llama.cpp / GGUF:** CPU/edge/quantized — laptops, phones, local inference.
(Also TGI, SGLang.)

**10. TTFT, TPOT, throughput.**
- **TTFT** (time-to-first-token): latency of prefill — how long until the user sees anything.
- **TPOT** (time-per-output-token, a.k.a. inter-token latency): decode speed — how fast tokens stream after the first.
- **Throughput:** total tokens/sec across all concurrent requests — the cost-efficiency metric.

---

## 🟡 Core (L4–L5)

**11. Why decode is bandwidth-bound, prefill compute-bound.**
Prefill multiplies a big activation matrix ($T$ tokens) by the weights — large GEMMs with high **arithmetic intensity** (many FLOPs per byte loaded) → compute-bound. Decode processes **one token**: each matmul is a tiny GEMV that must still **read all the model weights (and KV) from HBM** to produce one token, doing very few FLOPs per byte → **bandwidth-bound**. You're paying to move weights, not to compute.

**12. KV-cache size formula.**
$$\text{bytes} = 2 \times L \times n_{kv} \times d_\text{head} \times T \times B \times \text{bytes/elem}$$
the leading **2** for K and V; $L$ layers, $n_{kv}$ KV heads, $d_\text{head}$ head dim, $T$ context length, $B$ batch, dtype size. Linear in context and batch — the thing that limits how many/long requests you can serve.

**13. MQA/GQA reduce KV cache; tradeoff.**
The cache scales with the number of **KV heads** $n_{kv}$. **MQA** uses one shared KV head ($n_{kv}=1$) → up to $h\times$ smaller cache and much higher throughput, but a measurable quality drop. **GQA** shares KV across **groups** of query heads ($1<n_{kv}<h$) → most of MQA's savings with nearly MHA quality. Tradeoff: fewer KV heads = less memory/bandwidth but lower capacity; GQA is the modern default.

**14. Why FlashAttention is faster without changing math.**
It reorders the *computation*, not the result. By tiling Q/K/V into SRAM and maintaining a running max + sum (online softmax), it produces the identical attention output while avoiding the HBM round-trip of the $T\times T$ matrix. The speedup is pure **IO reduction** — fewer/coalesced memory accesses — so outputs are bit-comparable (up to FP reordering).

**15. GPTQ vs AWQ vs GGUF.**
- **GPTQ:** post-training, layer-wise second-order (Hessian-based) weight quantization to 4-bit; accurate, GPU-focused.
- **AWQ (Activation-aware):** protects the **salient weight channels** (those multiplying large activations) by scaling, preserving accuracy especially where outliers matter; GPU.
- **GGUF:** a **file format** (llama.cpp) packaging various quant levels (Q4_K_M, etc.) for **CPU/edge/Metal** inference, with memory-mapping.
First two are methods; GGUF is a format/runtime.

**16. How speculative decoding preserves the exact distribution.**
It uses **modified rejection sampling**: accept draft token $x$ with probability $\min(1, p_\text{target}(x)/p_\text{draft}(x))$; on rejection, resample from the normalized residual $\max(0, p_\text{target}-p_\text{draft})$. This procedure provably yields samples distributed **exactly** as $p_\text{target}$ — the draft only affects *speed*, never the output distribution.

**17. PagedAttention.**
Stores the KV cache in fixed-size **blocks** (like OS virtual-memory pages) instead of one contiguous buffer per request. This eliminates internal/external **fragmentation** and over-reservation (you don't pre-allocate for max length), so you can pack many more concurrent requests → larger effective batch and ~2–4× throughput; also enables prefix/KV sharing.

**18. Latency ↔ throughput tradeoff; batch size.**
Bigger batches amortize the weight-loading cost across more requests, raising **throughput and GPU utilization** — but each individual request waits longer (higher latency). Small batches give low latency but waste bandwidth. Batch size is the main dial: pick it (with continuous batching) to meet a **latency SLA** while maximizing throughput.

**19. Weight-only vs weight+activation quant; why latter is harder.**
Weight-only (W4A16) just compresses stored weights and dequantizes for compute — easy, since weights are static and well-behaved. Quantizing **activations** too (W8A8/W4A4) is harder because activations are **dynamic and have large outliers** (a few channels with huge values) that wreck low-precision range; needs per-token/per-channel scaling, outlier handling (SmoothQuant/LLM.int8), and risks accuracy loss — but unlocks faster integer matmuls.

**20. When distillation/pruning over quantization.**
Quantization shrinks bytes-per-weight but keeps the same architecture/params. Reach for **distillation** when you want a genuinely **smaller/faster model** (train a small student to mimic a large teacher) — best quality-per-FLOP at small sizes. Use **structured pruning** to remove whole heads/layers/channels for real speedups when the model is over-parameterized for the task. Often combine: distill → prune → quantize.

---

## 🔴 Senior / Staff deep dives

**21. Serving stack: p99 < 300ms TTFT @ 1000 RPS, 13B.**
- **Model:** a **GQA** 13B, **quantized** (W4A16/AWQ or FP8 on H100) to cut weight bandwidth and footprint.
- **Engine:** **vLLM or TensorRT-LLM** with **continuous batching** + **PagedAttention**; **prefix caching** for shared system prompts (huge TTFT win).
- **Prefill/decode disaggregation:** separate pools so long prefills don't stall decode and you can size each for its bottleneck (compute vs bandwidth).
- **Scheduling:** length-aware batching, chunked prefill to bound TTFT; cap max prompt or route long ones separately.
- **Scale-out:** replicate behind a load balancer; **autoscale** on queue depth/TTFT; provision for p99 not mean.
- **Hardware:** H100/A100 for the math; right-size replicas to hit 1000 RPS with headroom. Continuously monitor **p50/p99 TTFT & TPOT** and cost/token.

**22. Why tokens/sec ≪ FLOPs roofline in decode (expected).**
Decode does a **GEMV per layer** (batch×1 token), so very few FLOPs, but it must still **stream all model weights (+ KV) from HBM** each token. The bottleneck is **HBM bandwidth**, not compute — arithmetic intensity is far below the roofline's ridge point, so you operate on the memory-bound side. You'll never hit FLOPs peak in decode; the fix is **batching** (reuse the loaded weights across many requests) to raise arithmetic intensity.

**23. Levers to halve cost-per-token + quality risk.**
- **Quantization** (W4/W8/FP8): ~2–4× cheaper; small quality risk (task-dependent, mitigated by AWQ/mixed precision).
- **Smaller / distilled model:** big cost win; quality risk if over-distilled.
- **Batching / continuous batching:** higher utilization; ~no quality risk, adds latency.
- **Speculative decoding:** fewer target steps; no quality change (exact), risk = no speedup if acceptance low.
- **KV-cache quantization / GQA:** more concurrency; small quality risk.
- **Prompt/context reduction + prefix caching:** fewer tokens processed; risk of dropping needed context.
- **Caching (exact/semantic) + routing** (small model for easy queries): big savings; risk = stale/wrong cache or misroute.
Order: free wins first (batching, caching, prefix), then quantization/spec-decoding, then model size.

**24. Speculative decoding in detail (accept/reject).**
1. Draft model autoregressively proposes $k$ tokens $x_1..x_k$ with probs $q(x_i)$.
2. Target runs **one parallel forward** over the prompt + $k$ drafts, giving $p(x_i)$ for each position.
3. For each $i$ in order: accept $x_i$ with prob $\min(1, p(x_i)/q(x_i))$; if rejected, **resample** that position from the normalized residual $\max(0,p-q)$ and **stop** (discard the rest).
4. If all $k$ accepted, sample one extra "free" token from the target.
Expected tokens per target pass rises with the **acceptance rate**; speedup ≈ accepted-tokens-per-verification, capped by draft cost. Output distribution is provably exactly the target's.

**25. Serve 100 LoRA adapters on one base — inference path.**
Keep **one frozen base** resident. Use a **multi-adapter serving system** (S-LoRA / punica-style) that stores all adapters compactly and applies the right one per request via **batched, adapter-aware kernels** (so requests with *different* adapters run in the same batch). **Dynamically load/evict** adapters (they're a few MB) with an LRU cache; account for adapter memory separately from base. This serves thousands of customer models at near single-model cost.

**26. 128K context blowing up memory — serving-layer options.**
- **GQA/MQA** to shrink KV per token.
- **KV-cache quantization** (int8/int4 KV) → 2–4× less KV memory.
- **PagedAttention** to avoid fragmentation/over-reservation.
- **Chunked prefill** to bound peak memory and TTFT.
- **Eviction / sliding window + attention sinks** for streaming (drop middle, keep sinks).
- **Prefix caching** for shared long contexts.
- Push for **retrieval** instead of full-context where possible.

**27. MFU — what, why low in decode, how to raise.**
**Model FLOPs Utilization** = achieved useful FLOPs ÷ hardware peak FLOPs. It's low in decode because decode is **bandwidth-bound** (tiny GEMVs), so the ALUs sit idle waiting on HBM — you might see single-digit % MFU. Raise it by **batching** (more tokens per weight load), **quantization** (less bytes to move), **fused/Flash kernels**, **speculative decoding**, and disaggregating prefill (high MFU) from decode.

**28. Disaggregated prefill/decode — why split.**
Prefill is **compute-bound** and bursty; decode is **bandwidth-bound** and long-running. Co-locating them causes interference — a big prefill stalls token streaming (hurting TPOT/p99). Splitting them onto **separate pools** lets you size and schedule each for its own bottleneck (compute-optimized for prefill, bandwidth/parallel-friendly for decode), transfer the KV cache between them, and meet TTFT and TPOT SLAs independently.

---

## 🧮 Math & derivations

**29. KV-cache bytes: 32 layers, 8 KV heads, head_dim 128, seq 8192, batch 16, fp16.**
$$2 \times L \times n_{kv} \times d_\text{head} \times T \times B \times 2\text{ bytes}$$
$= 2 \times 32 \times 8 \times 128 \times 8192 \times 16 \times 2$
Per token-per-layer KV: $2\times8\times128\times2 = 4096$ B. ×32 layers = 131072 B/token. ×8192 seq = ~1.07 GB/sequence. ×16 batch = **~17.2 GB**. (This is *just* the KV cache — often larger than the weights at long context.)

**30. Estimate decode latency from model size & HBM bandwidth.**
In the bandwidth-bound regime, time-per-token ≈ (bytes of weights moved) ÷ (HBM bandwidth). For a 13B model in fp16: ~26 GB of weights. On an A100 (~2 TB/s): $26\text{e9}/2\text{e12}\approx 13$ ms/token → ~77 tok/s (single stream, ignoring KV/overhead). Quantize to int4 (~6.5 GB): ~3.3 ms/token → ~300 tok/s. Shows why decode speed tracks **bytes moved**, and why quantization and batching matter.

**31. Arithmetic intensity of a decode step; why batching helps.**
Arithmetic intensity = FLOPs ÷ bytes moved. A single-token decode does ~$2N$ FLOPs but must read ~$N\cdot$bytes/param of weights → intensity ≈ $2N / (N\cdot b) = 2/b$ — tiny, far left of the roofline ridge → memory-bound. **Batching $B$ requests** does $2NB$ FLOPs for the *same* weight read ($N\cdot b$ bytes) → intensity ≈ $2B/b$, scaling with $B$. So bigger batches move you toward compute-bound and raise utilization/throughput.

**32. int4 vs fp16 memory-movement (speed) ratio for decode.**
Decode time ∝ bytes of weights streamed. int4 = 0.5 bytes/param vs fp16 = 2 bytes/param → **4× less data moved → ~4× faster** (and 4× less memory), in the ideal bandwidth-bound limit. Real speedup is a bit less due to dequant overhead and non-weight costs (KV, activations), but the dominant term is the 4× reduction.

---

## 💻 Coding / implementation

**33. Greedy + temperature + top-k + top-p sampling.**
```python
import torch, torch.nn.functional as F
def sample(logits, temperature=1.0, top_k=0, top_p=1.0, greedy=False):
    if greedy: return logits.argmax(-1)
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    if top_p < 1.0:
        s, idx = torch.sort(logits, descending=True)
        cum = F.softmax(s, dim=-1).cumsum(-1)
        mask = cum - F.softmax(s, -1) > top_p           # keep nucleus
        s = s.masked_fill(mask, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, idx, s)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)
```

**34. Incremental KV cache + decode loop.**
```python
def generate(model, prompt_ids, max_new):
    kv = [None] * model.n_layers
    logits, kv = model(prompt_ids, kv, start_pos=0)        # prefill
    out = [sample(logits[:, -1])]
    pos = prompt_ids.size(1)
    for _ in range(max_new - 1):
        logits, kv = model(out[-1][:, None], kv, start_pos=pos)  # 1 token
        out.append(sample(logits[:, -1]))
        pos += 1
    return torch.cat([t[:, None] for t in out], dim=1)
```
Each step feeds only the new token; `kv` is appended in-place inside attention.

**35. Speculative decoding accept/reject.**
```python
import torch
def speculative_step(target_p, draft_p, draft_tokens):
    # target_p, draft_p: prob over vocab at each drafted position
    accepted = []
    for i, x in enumerate(draft_tokens):
        r = torch.rand(())
        if r < min(1.0, (target_p[i, x] / draft_p[i, x]).item()):
            accepted.append(x)                              # accept
        else:
            residual = torch.clamp(target_p[i] - draft_p[i], min=0)
            residual /= residual.sum()
            accepted.append(torch.multinomial(residual, 1).item())
            return accepted                                 # stop at first reject
    # all accepted -> one bonus token from target
    accepted.append(torch.multinomial(target_p[len(draft_tokens)], 1).item())
    return accepted
```

**36. Benchmark harness: TTFT, TPOT, throughput under concurrency.**
```python
import time, asyncio, numpy as np
async def one(client, prompt):
    t0 = time.perf_counter(); first = None; n = 0
    async for tok in client.stream(prompt):
        if first is None: first = time.perf_counter() - t0   # TTFT
        n += 1
    total = time.perf_counter() - t0
    tpot = (total - first) / max(1, n - 1)                   # per output token
    return first, tpot, n

async def bench(client, prompts, concurrency):
    sem = asyncio.Semaphore(concurrency); res = []
    async def run(p):
        async with sem: res.append(await one(client, p))
    t0 = time.perf_counter(); await asyncio.gather(*(run(p) for p in prompts))
    wall = time.perf_counter() - t0
    ttft, tpot, toks = zip(*res)
    return dict(ttft_p50=np.percentile(ttft,50), ttft_p99=np.percentile(ttft,99),
                tpot_p50=np.percentile(tpot,50), tpot_p99=np.percentile(tpot,99),
                throughput=sum(toks)/wall)
```

**37. Naive int8 weight quantization + dequant for a linear.**
```python
import torch
def quantize_int8(W):                       # per-output-channel symmetric
    scale = W.abs().amax(dim=1, keepdim=True) / 127.0
    q = torch.clamp((W / scale).round(), -127, 127).to(torch.int8)
    return q, scale

def linear_int8(x, q, scale, bias=None):
    W = q.to(x.dtype) * scale               # dequant
    out = x @ W.T
    return out + bias if bias is not None else out
```
(Real kernels do int8×int8 GEMM then rescale; this shows the scale/round/dequant idea.)

---

## 🏗️ System design / applied

**38. Multi-tenant LLM API platform.**
- **Routing:** per-model/endpoint routing; route easy queries to a small model, hard to large (model cascade); sticky routing for prefix-cache hits.
- **Batching:** continuous batching + PagedAttention per replica; chunked prefill.
- **Autoscaling:** scale on queue depth / TTFT SLO; separate prefill and decode pools.
- **Quotas/fairness:** per-tenant rate limits, token budgets, priority classes; isolate noisy neighbors.
- **Observability:** p50/p99 TTFT & TPOT, throughput, MFU, GPU mem/util, cache hit-rate, error/timeout rates, cost/token per tenant.
- **Cost controls:** quantized models, caching (exact + semantic), max-token caps, autoscale-to-zero for idle models.

**39. Edge: 7B on laptop/phone.**
Switch to **GGUF 4-bit** (Q4_K_M) with **llama.cpp**; use **CPU + Metal/NEON** kernels and **memory-mapped** weights so it loads lazily. **Reduce context** (smaller KV), small batch (1), maybe a **smaller/distilled** model. Accept lower tokens/sec; optimize for footprint and load time. Possibly NPU/CoreML/ONNX-runtime backends on phones.

**40. Hardware for three workloads.**
- **Chat (low-latency, interactive):** **H100/A100** — need bandwidth for fast decode and headroom for p99; FP8/quantized.
- **Batch summarization (throughput, latency-tolerant):** **A100** (or L4 fleet) with large continuous batches — maximize tokens/sec/\$; can run quantized, offline.
- **Embeddings (small encoder, high QPS, cheap):** **L4** (or even CPU) — encoder forward is cheap and parallel; pick the most cost-efficient accelerator, no need for H100.
Justify by each workload's bottleneck: latency vs throughput vs cost.

---

## 🐞 Debugging

**41. Throughput collapses when long-context requests arrive.**
A few long requests consume huge **KV-cache** memory, causing **fragmentation/OOM** and forcing the scheduler to shrink the batch (or preempt), starving everyone. Fix: **PagedAttention** (no fragmentation), **length-aware scheduling** / separate pools for long requests, KV quantization, chunked prefill, and per-request context caps so one whale can't evict the fleet.

**42. int4 tanked accuracy on one task only.**
**Outlier channels / activation sensitivity** — that task exercises weights/activations with large-magnitude outliers that 4-bit can't represent, while other tasks don't. Diagnose by per-layer error analysis. Fix: **AWQ** (protect salient channels), **mixed precision** (keep sensitive layers/embeddings/lm-head in fp16), per-channel/group quant, or a higher-bit quant for that subnetwork.

**43. p50 fine, p99 terrible.**
Tail-latency causes: **head-of-line blocking** (a long prefill/long request stalls the batch), **scheduling/batch wait**, **cold caches** (prefix-cache or weights not warm), **paging/GC** or memory pressure, autoscaler cold starts, and **long-prompt prefill stalls**. Fix: chunked prefill, prefill/decode disaggregation, length-aware/priority scheduling, prewarming, and capping max prompt length.

**44. Speculative decoding gave no speedup.**
Most likely **low acceptance rate** — the draft is too weak or **mismatched** (different tokenizer/distribution), so few drafts survive verification and you pay draft+verify for little gain. Also: **verification overhead** dominates for a too-large draft or small $k$, **short outputs** (not enough tokens to amortize), or memory bandwidth already saturated. Fix: a better-aligned/cheaper draft, tune $k$, or use a draft distilled from the target.

---

## What strong answers share
Naming the **correct bottleneck** (bandwidth vs compute) before optimizing; doing **KV-cache and latency math** live; knowing the **quality/memory/complexity tradeoff** behind every speedup; and reasoning in **p50/p99 and cost-per-token**, not averages.

---
Back to [questions](interview-questions.md) · [Stage README](README.md) · [Index](../README.md)
