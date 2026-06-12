"""Build NB06 — Inference & serving."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 06 · Inference & Serving

> Module: **04 · GPU Architecture & Inference**.

**Goal:** master how LLMs actually run at serve time — the **prefill/decode** split, the
**KV cache**, **sampling**, **quantization**, **speculative decoding**, **batching**, and the
**latency/throughput** math you'll be asked to do live.

### Learning objectives
1. Explain prefill (compute-bound) vs decode (memory-bandwidth-bound).
2. Implement samplers (greedy, temperature, top-k, top-p).
3. Quantify KV-cache memory and the roofline that bounds decode speed.
4. Simulate **speculative decoding** and reason about **continuous batching**.
"""),
    md(r"""
## 1. Two phases: prefill vs decode

| Phase | What | Bound by | Metric |
|-------|------|----------|--------|
| **Prefill** | process the whole prompt in parallel, fill KV cache | **compute** (big matmuls) | **TTFT** (time to first token) |
| **Decode** | generate one token at a time | **memory bandwidth** (read all weights + KV each step) | **TPOT** (time per output token) |

This split explains almost every serving decision: prefill loves big GEMMs; decode is starved
for bandwidth, so we **batch** many sequences to reuse each weight read.
"""),
    md(r"""
## 2. Sampling — turning logits into tokens

Decoding controls the quality/diversity trade-off:
- **greedy / argmax** — deterministic, can be repetitive.
- **temperature** — divide logits by $T$; >1 more random, <1 sharper.
- **top-k** — keep the k highest-prob tokens.
- **top-p (nucleus)** — keep the smallest set whose cumulative prob ≥ p.
"""),
    code(r"""
import numpy as np
rng = np.random.default_rng(0)

def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def sample(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits.astype(float).copy()
    if temperature != 1.0:
        logits /= max(temperature, 1e-6)
    if top_k is not None:
        kth = np.sort(logits)[-top_k]
        logits[logits < kth] = -np.inf
    if top_p is not None:
        probs = softmax(logits)
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        cutoff = order[cum > top_p]
        if len(cutoff):
            logits[cutoff[1:]] = -np.inf   # keep tokens up to and including crossing p
    p = softmax(logits)
    return int(rng.choice(len(p), p=p)), p

logits = rng.standard_normal(10) * 2
print("greedy       :", int(logits.argmax()))
print("temp=0.7 draw:", sample(logits, temperature=0.7)[0])
print("top_k=3 draw :", sample(logits, top_k=3)[0])
print("top_p=0.9 draw:", sample(logits, top_p=0.9)[0])
"""),
    md(r"""
## 3. KV-cache memory — usually the real bottleneck

Per token the cache stores $K$ and $V$ for every layer:

$$\text{KV bytes} = 2 \times L \times T \times d_{kv} \times \text{batch} \times \text{bytes}$$

This grows **linearly** with context length *and* batch size, and it competes with the weights
for HBM. **GQA** (fewer KV heads) and **KV quantization** shrink it; **PagedAttention** stops it
from fragmenting so you can pack more sequences.
"""),
    code(r"""
def kv_cache_gb(L, d_kv, T, batch, bytes_per=2):
    return 2 * L * T * d_kv * batch * bytes_per / 1e9

# 7B-ish: 32 layers, full d_kv=4096 (MHA) vs GQA with 8 KV heads (d_kv=1024)
for label, d_kv in [("MHA d_kv=4096", 4096), ("GQA d_kv=1024", 1024)]:
    gb = kv_cache_gb(L=32, d_kv=d_kv, T=8192, batch=32)
    print(f"{label}: {gb:.1f} GB for batch=32 @ 8k context")
print("\n-> GQA cuts KV memory ~4x here, letting you 4x the batch (=throughput).")
"""),
    md(r"""
## 4. The roofline: why decode is bandwidth-bound

For batch=1 decode, each step must read the **entire weight set** from HBM to produce one token.
So an upper bound on tokens/sec is `HBM_bandwidth / bytes_read_per_token`. **Batching** amortizes
that weight read across many sequences — the single biggest throughput lever.
"""),
    code(r"""
def decode_tokps(weight_bytes, hbm_bw=3.35e12, batch=1):
    # naive bound: each step reads weights once; batching reuses that read across `batch` seqs
    per_step_seconds = weight_bytes / hbm_bw
    return batch / per_step_seconds

W = 13e9 * 2   # 13B params, fp16
for b in [1, 8, 32, 128]:
    print(f"batch={b:3d}: ~{decode_tokps(W, batch=b):,.0f} tok/s (bandwidth-bound upper bound)")
print("\nReal servers approach this with continuous batching + paged KV cache.")
"""),
    md(r"""
## 5. Quantization — fewer bits per weight

Storing weights in 8/4 bits cuts memory and (for memory-bound decode) boosts speed, at some
quality cost. Know the families:
- **Weight-only PTQ:** GPTQ, AWQ (4-bit) — popular, near-lossless for many models.
- **Weight+activation:** SmoothQuant, FP8 (H100+) — faster matmuls too.
- **KV-cache quantization** — shrink the cache (often the bigger win at long context).
"""),
    code(r"""
def quantize_dequantize(x, bits=8):
    qmax = 2**(bits-1) - 1
    scale = np.abs(x).max() / qmax
    q = np.round(x / scale).clip(-qmax-1, qmax)
    return q * scale, scale

w = rng.standard_normal(10000) * 0.1
for bits in [8, 4]:
    deq, _ = quantize_dequantize(w, bits)
    err = np.sqrt(np.mean((w - deq)**2)) / np.sqrt(np.mean(w**2))
    print(f"{bits}-bit: relative RMS error = {err:.3%}, memory = {bits/16:.0%} of fp16")
"""),
    md(r"""
## 6. Speculative decoding — a free* lunch

A small, cheap **draft** model proposes several tokens; the big **target** model verifies them
in **one parallel forward pass**. Accepted tokens are kept (and the distribution is provably
unchanged); on rejection we fall back. If the draft is right most of the time, you get a 2–3×
speedup for *identical* output. Let's simulate the acceptance dynamics.
"""),
    code(r"""
def speculative_sim(accept_prob=0.7, gamma=4, steps=1000):
    # gamma = tokens proposed per round. Accepted run length is geometric-ish, capped at gamma,
    # plus 1 bonus token from the target on each round.
    tokens = 0
    target_calls = 0
    for _ in range(steps):
        target_calls += 1                      # one (parallel) target verification per round
        accepted = 0
        for _ in range(gamma):
            if rng.random() < accept_prob:
                accepted += 1
            else:
                break
        tokens += accepted + 1                  # +1 bonus token
    speedup = tokens / target_calls             # tokens per (expensive) target call
    return speedup

for ap in [0.5, 0.7, 0.9]:
    print(f"draft accept rate {ap:.0%} -> ~{speculative_sim(ap):.2f}x tokens per target call")
print("\n*Lossless: output distribution matches sampling from the target alone.")
"""),
    md(r"""
## 7. Continuous batching & PagedAttention (vLLM)

- **Static batching** wastes the GPU: fast requests wait for slow ones to finish.
- **Continuous (in-flight) batching** adds/removes requests every step, keeping the GPU full —
  huge throughput win for mixed-length traffic.
- **PagedAttention** stores the KV cache in fixed-size **pages** (like OS virtual memory), so
  there's no fragmentation and you can run far higher batch sizes. This is the core of **vLLM**.

### The serving metrics you must name
**TTFT, TPOT/ITL, end-to-end p50/p95/p99, throughput (tok/s, req/s), goodput under SLA.**

## 8. Hands-on
Use the **lab benchmark** to measure a real server and find the latency–throughput knee:
```bash
uvx vllm serve facebook/opt-125m --port 8000
uv run --project labs python -m lab04_inference_bench.benchmark \
    --url http://localhost:8000 --model facebook/opt-125m --concurrency 8
```

## Exercises
1. Sweep `--concurrency` on the lab benchmark; plot throughput & p95 latency; explain the knee.
2. Compute KV-cache memory for Llama-3-70B (GQA) at 32k context, batch 16.
3. Quantize a model with AWQ/GPTQ; tabulate perplexity vs latency vs memory.
4. Vary the draft acceptance rate in the sim; at what rate does speculation stop helping?

## Resources
- *PagedAttention / vLLM* (Kwon 2023); *Orca* continuous batching (Yu 2022).
- *Speculative Decoding* (Leviathan 2023); *Medusa*, *EAGLE*.
- *GPTQ*, *AWQ*, *SmoothQuant*; Horace He — *Making Deep Learning Go Brrrr From First Principles*.
"""),
]

if __name__ == "__main__":
    write(cells, "06_inference_and_serving.ipynb")
