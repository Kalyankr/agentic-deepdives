# Chapter 10 — Inference Optimization · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-3-llm-stack/10-inference-optimization.md)

---

## Interview answers

### Q: "Why is LLM decoding slow / memory-bound?"

Autoregressive decoding generates **one token at a time**, and each token requires reading the **entire model's weights** from HBM to do a relatively small amount of arithmetic (a few matrix-vector products). So you're bottlenecked by **memory bandwidth**, not compute — the GPU's FLOPs sit mostly idle while it waits for weights to stream in. Contrast with **prefill** (processing the prompt), which does one big matmul over all prompt tokens at once → high arithmetic intensity → **compute-bound**. So: prefill compute-bound, decode memory-bound, and decode is the bottleneck for generation.

### Q: "What does the KV cache store and what's its downside?"

It stores the **keys and values for every past token at every layer**, so that when generating token $t$ you don't recompute attention over the whole prefix — you reuse cached K/V and only compute the new token's contribution. That turns generation from $O(t^2)$ to $O(t)$ per token. **Downside:** the cache grows linearly with sequence length × batch size × layers × KV-heads, and at long context / large batch it can **exceed the size of the model weights**, becoming the dominant memory cost and the limiter on how many requests you can batch.

### Q: "Explain PagedAttention."

The naive KV cache pre-allocates one big contiguous block per request for the max possible length → massive **internal fragmentation** (wasted reserved memory) and no sharing. **PagedAttention** (from vLLM) borrows **OS virtual-memory paging**: it splits the KV cache into fixed-size **blocks** allocated **on demand** and tracked by a block table, so memory is used only as the sequence actually grows. Benefits: near-zero fragmentation (pack many more requests), and **prefix sharing** (multiple requests with the same prompt prefix share KV blocks copy-on-write). It's the core reason vLLM gets such high throughput.

### Q: "How does quantization speed up inference?"

Decode is **memory-bound**, so the cost is dominated by moving weights from HBM. Quantizing weights from 16-bit to 8- or 4-bit **halves or quarters the bytes to move** (and the footprint), directly speeding up memory-bound decode and letting bigger models fit on a GPU. The challenge is **outliers** — a few large-magnitude weights/activations that, if naively quantized, wreck accuracy; methods like **GPTQ, AWQ, SmoothQuant** handle them (per-channel scales, keeping salient weights in higher precision). Quality loss is small when done well.

### Q: "How do you cut latency without hurting quality?"

- **Speculative decoding** — a small draft model proposes several tokens, the big model verifies them in one parallel pass; accepted tokens are *provably* from the target distribution, so **quality is identical** with a 2–3× speedup.
- **Prefix caching** — reuse KV for shared system prompts/prefixes so you skip re-prefilling them.
- **FlashAttention** — same math, fewer HBM round-trips → faster prefill (lower TTFT).

These are "free lunch" wins: faster without changing outputs.

### Q: "How do you raise throughput?"

In order of impact: **continuous batching** first (the single biggest win — dynamically add/remove requests each step so the GPU never idles between requests of different lengths), then **PagedAttention** (pack more concurrent requests), then **quantization** (smaller weights → more room for batch/KV), then simply **larger batches** within your latency SLA. Throughput and latency trade off, so you tune batch size to maximize **goodput** (throughput subject to the SLA).

### Q: "TTFT vs TPOT — what affects each?"

- **TTFT** (time to first token) ≈ **prefill** cost → driven by **prompt length** and prefill efficiency; it's compute-bound. Long prompts and cold caches hurt TTFT.
- **TPOT** (time per output token) ≈ **decode** cost → driven by **model size** and memory bandwidth; it's memory-bound. Quantization and GQA help TPOT.

Users feel TTFT as "responsiveness" (mitigated by streaming) and TPOT as "typing speed." Optimize them separately because they have different bottlenecks.

---

## Exercise solutions

### Exercise 1 — KV cache: speedup with identical outputs

```python
import torch, torch.nn.functional as F, time

torch.manual_seed(0)
d, V, T = 64, 100, 200
Wqkv = torch.randn(d, 3*d) * 0.02
Wout = torch.randn(d, V) * 0.02
emb  = torch.randn(V, d)

def attn(x, cache=None):
    qkv = x @ Wqkv
    q, k, v = qkv.split(d, dim=-1)
    if cache is not None:
        k = torch.cat([cache[0], k], 0); v = torch.cat([cache[1], v], 0)
    w = F.softmax(q @ k.T / d**0.5, -1)
    return w @ v, (k, v)

def generate_nocache(prompt, n):
    seq = list(prompt)
    for _ in range(n):
        x = emb[torch.tensor(seq)]
        h, _ = attn(x)                          # recompute over WHOLE sequence
        seq.append(int((h[-1] @ Wout).argmax()))
    return seq

def generate_cache(prompt, n):
    seq = list(prompt)
    x = emb[torch.tensor(seq)]
    h, cache = attn(x)                          # prefill once
    seq.append(int((h[-1] @ Wout).argmax()))
    for _ in range(n - 1):
        x = emb[torch.tensor([seq[-1]])]        # only the NEW token
        h, cache = attn(x, cache)
        seq.append(int((h[-1] @ Wout).argmax()))
    return seq

prompt = [1, 2, 3]
t0 = time.perf_counter(); a = generate_nocache(prompt, 100); t1 = time.perf_counter()
t2 = time.perf_counter(); b = generate_cache(prompt, 100);   t3 = time.perf_counter()
print("identical outputs:", a == b)                         # True
print(f"no-cache: {(t1-t0)*1000:.1f} ms   cache: {(t3-t2)*1000:.1f} ms")
print(f"speedup: {(t1-t0)/(t3-t2):.1f}x")
```

**Result:** outputs are **identical** (the cache is a pure efficiency optimization, not an approximation), but the cached version is much faster because it does $O(t)$ work per token instead of re-attending over the whole prefix ($O(t^2)$ total). The gap widens as the sequence grows — which is the whole point.

### Exercise 2 — KV-cache size for 7B vs 70B (why GQA matters)

```python
def kv_gb(layers, kv_heads, d_head, seq, batch=1, bytes_=2):
    return 2 * layers * kv_heads * d_head * seq * batch * bytes_ / 1e9

configs = {
    "7B  MHA (32 kv)":  dict(layers=32, kv_heads=32, d_head=128),
    "7B  GQA (8 kv)":   dict(layers=32, kv_heads=8,  d_head=128),
    "70B MHA (64 kv)":  dict(layers=80, kv_heads=64, d_head=128),
    "70B GQA (8 kv)":   dict(layers=80, kv_heads=8,  d_head=128),
}
for name, c in configs.items():
    for seq in (4096, 32768):
        print(f"{name:16s} seq={seq:6d}: {kv_gb(seq=seq, **c):6.2f} GB/request")
```

**Result:** the KV cache is **per request** and grows with sequence length and KV-head count. A 70B MHA model at 32k context needs tens of GB **per request** just for KV — quickly dwarfing batch capacity. GQA (8 KV heads) cuts that 4–8×, which is what makes long-context, multi-request serving feasible. This is the quantitative reason GQA exists.

### Exercise 3 — int8 weight quantization (savings vs accuracy)

```python
import torch

torch.manual_seed(0)
W = torch.randn(1024, 1024)

def quantize_int8(W):
    scale = W.abs().max() / 127.0          # symmetric per-tensor scale
    q = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.float() * scale

q, scale = quantize_int8(W)
W_hat = dequantize(q, scale)

err = (W - W_hat).abs().mean().item()
mem_fp32 = W.numel() * 4
mem_int8 = q.numel() * 1 + 4              # int8 weights + one fp32 scale
print(f"mean abs error: {err:.5f}")
print(f"memory: fp32={mem_fp32/1e6:.2f} MB  int8={mem_int8/1e6:.2f} MB  ({mem_fp32/mem_int8:.1f}x)")

# Effect on a matmul (proxy for layer output quality):
x = torch.randn(8, 1024)
rel = ((x @ W - x @ W_hat).norm() / (x @ W).norm()).item()
print(f"relative output error: {rel*100:.2f}%")
```

**Result:** int8 cuts weight memory ~4× vs fp32 (~2× vs fp16) with small reconstruction error and ~1–2% relative output error from a simple per-tensor scheme. Because decode is memory-bound, that 2–4× reduction in bytes-to-move translates fairly directly into speed. **Per-channel** scales and outlier handling (GPTQ/AWQ) shrink the error further — the gap between this toy and production quantizers.

### Exercise 4 — Continuous vs static batching (GPU idle time)

```python
import numpy as np

rng = np.random.default_rng(0)
req_lens = rng.integers(5, 50, size=16)         # tokens each request needs

def static_batches(lens, bs=4):
    """Static: a batch finishes only when its LONGEST request finishes."""
    busy = idle = 0
    for i in range(0, len(lens), bs):
        batch = lens[i:i+bs]; longest = batch.max()
        busy += batch.sum()                      # useful token-steps
        idle += (longest - batch).sum()          # padding/idle token-steps
    return busy, idle

def continuous(lens, bs=4):
    """Continuous: finished requests are immediately replaced -> ~no idle."""
    return lens.sum(), 0

b_s, i_s = static_batches(req_lens)
b_c, i_c = continuous(req_lens)
print(f"static:     useful={b_s} idle={i_s}  util={b_s/(b_s+i_s)*100:.0f}%")
print(f"continuous: useful={b_c} idle={i_c}  util=100%")
```

**Result:** with **static** batching, short requests sit idle waiting for the longest request in their batch to finish — wasted GPU cycles (often 30–50% with variable lengths). **Continuous** batching evicts a finished request and slots in a new one every step, keeping utilization near 100%. This is why continuous batching (vLLM/TGI) is the single biggest throughput win in LLM serving.

### Exercise 5 — Toy speculative decoding (acceptance rate & speedup)

```python
import numpy as np

rng = np.random.default_rng(0)
V = 100

def target_dist(ctx):  # the "expensive" model's next-token distribution
    p = np.ones(V); p[(ctx * 7) % V] += 10; return p / p.sum()

def draft_dist(ctx):   # the "cheap" model: similar but imperfect
    p = np.ones(V); p[(ctx * 7) % V] += 8; p[(ctx * 13) % V] += 2; return p / p.sum()

def speculative_step(ctx, k=4):
    """Draft proposes k tokens; target verifies with the standard accept/reject rule."""
    accepted = []
    for _ in range(k):
        q = draft_dist(ctx); tok = rng.choice(V, p=q)
        p = target_dist(ctx)
        # accept with prob min(1, p[tok]/q[tok]) -> guarantees target distribution
        if rng.random() < min(1.0, p[tok] / q[tok]):
            accepted.append(tok); ctx = tok
        else:
            # reject: resample from the adjusted residual distribution, stop
            resid = np.clip(p - q, 0, None); resid /= resid.sum()
            accepted.append(rng.choice(V, p=resid)); ctx = accepted[-1]
            break
    return accepted, ctx

ctx, total_tokens, target_calls = 1, 0, 0
for _ in range(200):
    toks, ctx = speculative_step(ctx, k=4)
    total_tokens += len(toks)
    target_calls += 1                  # one parallel target verification per step
avg_accept = total_tokens / target_calls
print(f"tokens/target-call: {avg_accept:.2f}   (>1 means speedup)")
print(f"approx speedup vs 1 token/call: {avg_accept:.1f}x")
```

**Result:** each expensive **target** verification yields more than one accepted token (here ~2–3), because the cheap draft guesses many tokens correctly. The accept/reject rule guarantees the output is **distributed exactly like the target model** — so it's faster with **zero quality change**. The closer the draft is to the target, the higher the acceptance rate and the bigger the speedup.

---

[← Chapter 9 solutions](09-alignment-solutions.md) · [Solutions index](README.md) · [Next: Chapter 11 solutions →](11-fine-tuning-solutions.md)
