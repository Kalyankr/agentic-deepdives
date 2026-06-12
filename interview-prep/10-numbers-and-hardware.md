# 10 · Numbers & Hardware — The Reference Card

> "Reason in numbers" is the senior bar. In design and depth rounds you'll be expected to do
> **back-of-envelope** capacity math live: how much memory a model needs, how many GPUs to serve it,
> what limits throughput. This file is the **memorizable constants + the formulas that use them**.

> All numbers are **order-of-magnitude / approximate** and shift with each hardware generation and
> vendor. Interviewers want a *defensible estimate with stated assumptions*, not a spec sheet.

---

## GPU cheat-sheet (datacenter, approximate)

| GPU | HBM | Bandwidth | bf16/fp16 dense | FP8 dense | Interconnect |
|-----|-----|-----------|-----------------|-----------|--------------|
| A100 40GB (SXM) | 40 GB | ~1.6 TB/s | ~312 TFLOPS | — | NVLink ~600 GB/s |
| A100 80GB (SXM) | 80 GB | ~2.0 TB/s | ~312 TFLOPS | — | NVLink ~600 GB/s |
| H100 (SXM) | 80 GB | ~3.35 TB/s | ~990 TFLOPS | ~1980 TFLOPS | NVLink ~900 GB/s |
| H200 | 141 GB | ~4.8 TB/s | ~990 TFLOPS | ~1980 TFLOPS | NVLink ~900 GB/s |
| B200 (Blackwell) | ~192 GB | ~8 TB/s | ~2.2 PFLOPS | ~4.5 PFLOPS | NVLink ~1.8 TB/s |

> Memorize the shapes: **A100 ≈ 80GB / 2 TB/s / 312 bf16 TFLOPS**, **H100 ≈ 80GB / 3.3 TB/s / ~1 PFLOPS bf16**.
> Real achieved FLOPs is **MFU ~40–55%** of peak. Sparse/marketing numbers are ~2× dense — use dense.

## Precision & bytes per parameter

| dtype | Bytes/param | Use |
|-------|-------------|-----|
| fp32 | 4 | master weights, optimizer moments |
| bf16 / fp16 | 2 | training compute, weights at inference |
| fp8 | 1 | cutting-edge training/inference |
| int8 | 1 | weight-only quant (~lossless) |
| int4 | 0.5 | aggressive quant (GPTQ/AWQ; eval after) |

## Model weights (inference, bf16 ≈ 2 B/param)

| Params | bf16 weights | int8 | int4 |
|--------|--------------|------|------|
| 7B | ~14 GB | ~7 GB | ~3.5 GB |
| 13B | ~26 GB | ~13 GB | ~6.5 GB |
| 70B | ~140 GB | ~70 GB | ~35 GB |
| 175B | ~350 GB | ~175 GB | ~88 GB |
| 405B | ~810 GB | ~405 GB | ~203 GB |

> Quick rule: **bf16 GB ≈ 2 × (params in B)**. A 70B model needs **≥2× H100 (80GB)** just for weights —
> add KV cache + activations, so realistically more.

## Training memory (mixed-precision Adam)

Per parameter you store: fp16 weights (2) + fp16 grads (2) + fp32 master copy (4) + Adam `m`,`v` (4+4)
= **~16 bytes/param**, often **~18–20** with activations/overhead.

- **70B training:** ~70B × 18 ≈ **~1.26 TB** of state → impossible on one GPU → **ZeRO/FSDP sharding**
  across many GPUs.
- **Activation memory** scales with batch × seq × layers × `d`; tame it with **gradient checkpointing**
  (recompute in backward) and sequence/tensor parallelism.

## KV cache (the inference memory driver)

$$\text{KV bytes} = 2 \times L \times n_{kv} \times d_{head} \times \text{bytes} \times T \times B$$

(`2` = K and V; `L` layers; `n_kv` KV heads; `T` tokens; `B` batch.)

- Worked example — 70B-class (`L=80`, `n_kv=8` GQA, `d_head=128`, bf16), per token:
  `2 × 80 × 8 × 128 × 2 ≈ 328 KB/token`. At `T=8192`, `B=1` ≈ **~2.7 GB** for one sequence.
- **GQA/MQA** shrink `n_kv` → the main lever on KV size and thus max batch/throughput.
- KV cache grows **linearly** with batch × context — at scale it can exceed the weights.

## What limits throughput (roofline intuition)

- **Prefill** (process the prompt) is **compute-bound** → governs **TTFT**.
- **Decode** (one token at a time) is **memory-bandwidth-bound** → governs **TPOT**.
- Decode throughput rule: $\text{tok/s} \approx \dfrac{\text{HBM bandwidth}}{\text{bytes read per token}}$.
  Bigger batches amortize weight reads → more tok/s (but higher per-request latency).
- **Arithmetic intensity** = FLOPs / byte. Below the GPU's ratio (~peak FLOPs ÷ bandwidth, ~300+ for
  H100) you're bandwidth-bound — which is why decode loves batching and quantization.

## Latency ladder (orders of magnitude)

| Operation | ~Time |
|-----------|-------|
| L1 cache reference | ~1 ns |
| Main memory (RAM) reference | ~100 ns |
| Read 1 MB sequentially from RAM | ~3 µs |
| SSD random read | ~16 µs |
| Same-datacenter round trip | ~0.5 ms |
| Read 1 MB from SSD | ~1 ms |
| Cross-continent network round trip | ~150 ms |
| **LLM TTFT (served)** | ~0.1–1 s |
| **LLM per-token (decode)** | ~10–50 ms (≈ 20–100 tok/s) |

> The point: **network + disk dominate** app latency, and an LLM call is *enormous* next to a cache hit.
> Design around it: stream tokens, cache prompts/embeddings, batch, and put compute near data.

## Capacity-planning quickref

- **Average QPS** = daily requests ÷ 86,400. **Peak QPS** ≈ 2–5× average.
- **GPUs needed** ≈ peak throughput required ÷ throughput per GPU (after MFU + KV-cache headroom).
- Always size for **peak**, leave **~30–40% headroom**, and plan **redundancy** (N+1) + autoscaling.
- **Cost levers (biggest first):** smaller/quantized model, batching, prompt/prefix caching, model
  **cascade** (cheap model first), speculative decoding, caching identical requests.

## Cost ballparks (vary wildly — state assumptions)

- Datacenter GPU rental: **~$2–4 / H100-hour** (cloud, on-demand; reserved is less).
- Frontier API pricing: **~$1–15 / million input tokens**, **~$5–75 / million output tokens**
  (output costs more; small/cheap models are ~10–50× less).
- Rule of thumb: **output tokens dominate cost** — trim max_tokens and prompt size first.

---

## Numbers to know cold (recite these)

- bf16 = **2 B/param**, fp32 = **4 B/param**; bf16 weights GB ≈ **2 × params(B)**.
- Training state ≈ **16–20 B/param** (Adam mixed precision).
- Training FLOPs ≈ **6 · N · D**; inference ≈ **2 · N** FLOPs/token.
- Chinchilla: **~20 tokens / param** compute-optimal.
- AdamW optimizer state = **8 B/param** (two fp32 moments).
- A100 ≈ **80GB / 2 TB/s / 312 bf16 TFLOPS**; H100 ≈ **80GB / 3.3 TB/s / ~1 PFLOPS bf16**.
- MFU good ≈ **40–55%**; KV bytes/token = **2 · L · n_kv · d_head · bytes**.
- `perplexity = exp(CE_loss)`; random-init LM loss ≈ `ln(vocab)`.
- Decode = **bandwidth-bound**; prefill = **compute-bound**.
- Peak ≈ **2–5×** average traffic.

## Night-before cheat sheet (one block)

```
MEMORY     bf16=2B/param  fp32=4B/param  weightsGB≈2×params(B)  train≈16–20B/param
FLOPs      train≈6ND      infer≈2N/token      Chinchilla D≈20N
KV CACHE   2·L·n_kv·d_head·bytes·T·B   (GQA/MQA shrink n_kv)   grows w/ batch×ctx
SERVE      prefill=compute→TTFT   decode=bandwidth→TPOT   tok/s≈BW/bytes_per_token
HW         A100 80GB/2TBs/312TF   H100 80GB/3.3TBs/~1PF   MFU 40–55%
CAPACITY   QPS=daily/86400   peak=2–5×avg   GPUs=peakThroughput/perGPU + 30–40% headroom
COST       output≫input tokens   levers: smaller model, batch, cache, cascade, spec-decode
EVAL MATH  ppl=exp(loss)   randomLoss≈ln(vocab)   always report a CI on scores
```

> Drill: given "serve a 70B model at 10k QPS, 500-token outputs, p95 < 2 s," produce GPU count + cost
> in under 10 minutes using only this card. Then check yourself against [03-system-design](03-system-design.md).
