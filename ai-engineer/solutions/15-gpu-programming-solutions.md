# Chapter 15 — GPU Programming · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-4-systems/15-gpu-programming.md)

---

## Interview answers

### Q: "Why are GPUs good for deep learning?"

Deep learning is **massively parallel identical arithmetic** — the same matmul applied to thousands of elements. GPUs are **throughput machines**: thousands of **SIMT** cores running the same instruction over different data, plus **tensor cores** that do low-precision matrix-multiply-accumulate extremely fast, plus high-bandwidth memory to feed them. A CPU optimizes **latency** for a few sequential threads; a GPU optimizes **throughput** for thousands of parallel ones — which is exactly the shape of neural-network compute. You write one thread's work and launch a million of them.

### Q: "What is memory coalescing / warp divergence?"

Both are about how a **warp** (32 threads executing in lockstep) behaves:

- **Memory coalescing**: when the 32 threads access **consecutive** addresses, the hardware combines them into **one** wide memory transaction (fast). Strided/scattered accesses become many transactions (slow). Coalesce your accesses — it's the #1 memory rule.
- **Warp divergence**: when threads in a warp take **different branches** (`if/else`), the warp executes **both** paths serially with threads masked off — so divergence wastes cycles. Minimize data-dependent branching inside a warp.

### Q: "Explain the GPU memory hierarchy and its optimization implication."

From fast/small to slow/big: **registers** (per-thread) → **shared memory / L1** (per-block, on-chip, ~TB/s) → **L2** → **HBM / global memory** (GBs, but ~100s GB/s and high latency). The implication: **HBM traffic is the bottleneck**, so the winning pattern is **load a tile from HBM into shared memory once, do as much compute there as possible, write the result back once** — minimizing round-trips to slow global memory. This single idea drives tiling, kernel fusion, and FlashAttention.

### Q: "Why is FlashAttention faster if it computes the same thing?"

Because it's **IO-aware**, not because it does less math. Standard attention materializes the full $n\times n$ score matrix in **HBM** (write it, read it back for softmax, read again for the $V$ multiply) — that HBM traffic dominates and scales as $O(n^2)$ memory. FlashAttention **tiles** Q/K/V into shared memory and uses **online softmax** (running max & sum) to compute attention block-by-block **without ever writing the $n\times n$ matrix to HBM**. Same output (it's **exact**, not approximate), but memory-linear and far fewer HBM round-trips → much faster and longer-context-capable. It's the canonical "respect the memory hierarchy" result.

### Q: "Compute-bound vs memory-bound — how do you tell and what do you do?"

Compute the **arithmetic intensity** = FLOPs ÷ bytes moved, and place it on the **roofline**: if intensity is below the ridge point, you're **memory-bound** (limited by bandwidth); above it, **compute-bound** (limited by FLOPs). **Memory-bound** (most elementwise/norm/attention-glue ops) → **fuse kernels** and cut HBM traffic. **Compute-bound** (big matmuls) → use **tensor cores**, better tiling, higher precision utilization. The roofline tells you which knob will actually help so you don't optimize the wrong thing.

### Q: "What is Triton?"

**Triton** is a Python-embedded language for writing GPU kernels at the **block level**: you express what a *block* of threads computes, and the compiler handles the hard low-level details — thread scheduling, memory coalescing, shared-memory management. It gives you most of CUDA's performance with a fraction of the complexity, and it's what powers many `torch.compile` kernels and hand-written ops (including FlashAttention implementations). A benchmarked custom Triton kernel is a standout portfolio piece.

---

## Exercise solutions

### Exercise 1 — Triton vector-add kernel (benchmark + correctness)

```python
# pip install triton  (Linux + NVIDIA GPU)
import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # this block's indices
    mask = offs < n                              # guard the tail
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def triton_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out

x = torch.randn(2**22, device="cuda"); y = torch.randn(2**22, device="cuda")
out = triton_add(x, y)
print("correct:", torch.allclose(out, x + y))     # True

import triton.testing
for name, fn in [("triton", lambda: triton_add(x, y)), ("torch", lambda: x + y)]:
    ms = triton.testing.do_bench(fn)
    gbps = 3 * x.numel() * 4 / ms * 1e-6           # 3 arrays (2 read, 1 write) × 4 bytes
    print(f"{name}: {ms:.3f} ms  {gbps:.0f} GB/s")
```

**Result:** the Triton kernel matches PyTorch numerically and reaches a similar **memory bandwidth** (vector add is memory-bound, so both saturate HBM and the GB/s figure approaches the GPU's peak). The lesson: for a bandwidth-bound op you can't beat a good library, but you *can* match it in a few lines of Triton — and the real wins come from **fusion** (next exercise).

### Exercise 2 — Fused SiLU kernel (less HBM traffic)

```python
import torch, triton, triton.language as tl

@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * tl.sigmoid(x), mask=mask)   # fused: one read, one write

def fused_silu(x):
    out = torch.empty_like(x); n = x.numel()
    silu_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out

x = torch.randn(2**22, device="cuda")
print("correct:", torch.allclose(fused_silu(x), x * torch.sigmoid(x), atol=1e-6))

# Unfused does it in two ops: sigmoid (read x, write s) then multiply (read x, read s, write o)
# Fused reads x once and writes once -> ~2-3x less HBM traffic on this op.
```

**Result:** the fused kernel reads `x` **once** and writes the output **once**. The unfused `x * torch.sigmoid(x)` materializes an intermediate `sigmoid(x)` in HBM — extra write + read. Since SiLU is **memory-bound**, cutting HBM traffic ~2–3× makes the fused version proportionally faster. This is the essence of **kernel fusion**: combine memory-bound ops so data stays in registers/shared memory instead of round-tripping to HBM.

### Exercise 3 — Arithmetic intensity & the roofline

```python
def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved

n = 4096
ops = {
    # elementwise add of two n-vectors: n FLOPs, 3n floats moved (2 read + 1 write)
    "elementwise add": (n, 3 * n * 4),
    # LayerNorm over n elements: ~5 FLOPs/elem, ~2n floats moved (read + write)
    "LayerNorm":       (5 * n, 2 * n * 4),
    # n×n matmul: 2 n^3 FLOPs, 3 n^2 floats moved
    "matmul (n×n)":    (2 * n**3, 3 * n**2 * 4),
}
RIDGE = 50      # example: peak_FLOPs / peak_bandwidth for a GPU (FLOP per byte)
for name, (f, b) in ops.items():
    ai = arithmetic_intensity(f, b)
    bound = "compute-bound" if ai > RIDGE else "memory-bound"
    print(f"{name:18s} AI={ai:8.2f} FLOP/byte -> {bound}")
```

**Result:**

| Op | Arithmetic intensity | Bottleneck |
|---|---|---|
| elementwise add | ~0.08 | memory-bound |
| LayerNorm | ~0.6 | memory-bound |
| matmul (4096³) | ~683 | compute-bound |

The elementwise and norm ops have tiny intensity (far left of the roofline ridge) → **memory-bound**, so fuse them. The large matmul has huge intensity → **compute-bound**, so it's the thing tensor cores accelerate. The roofline tells you *which* optimization matters for *which* op — the core skill of a performance engineer.

### Exercise 4 — Profile the Chapter 6 GPT

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = GPT().cuda()                     # your Chapter 6 model
x = torch.randint(0, V, (8, 64), device="cuda")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for _ in range(10):
        logits, loss = model(x, x)
        loss.backward()
torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# Inspect in detail with: prof.export_chrome_trace("trace.json")  (open in chrome://tracing)
```

**Result:** the table, sorted by CUDA time, typically shows the **matmuls** (attention QK/AV and the FFN linears, often via `aten::mm`/`addmm` or cuBLAS GEMM kernels) dominating compute time — these are **compute-bound**. The softmax, LayerNorm, and elementwise ops show up as many small **memory-bound** kernels whose launch overhead adds up (motivating fusion / `torch.compile`). Profiling tells you *where the time actually goes* before you optimize — never guess.

### Exercise 5 — Online softmax (the FlashAttention trick) in NumPy

```python
import numpy as np

def standard_softmax(x):
    m = x.max(); e = np.exp(x - m); return e / e.sum()

def online_softmax(x, block=4):
    """One streaming pass tracking running max m and running sum l, rescaling on the fly."""
    m, l = -np.inf, 0.0
    for i in range(0, len(x), block):
        blk = x[i:i+block]
        m_new = max(m, blk.max())
        l = l * np.exp(m - m_new) + np.exp(blk - m_new).sum()   # rescale old sum + add new
        m = m_new
    return np.exp(x - m) / l                                    # final normalize with global m,l

rng = np.random.default_rng(0)
x = rng.standard_normal(37) * 5          # wide range -> stability matters
print("matches standard softmax:", np.allclose(online_softmax(x), standard_softmax(x)))  # True
```

**Result:** processing the input in **blocks** while maintaining a running max `m` and running normalizer `l` (rescaling the accumulated sum by $e^{m_\text{old}-m_\text{new}}$ each time a new block raises the max) yields **exactly** the standard softmax. This is the mathematical heart of FlashAttention: it lets you compute softmax **incrementally** over tiles, so you never need the whole row in memory at once — which is what avoids materializing the $n\times n$ attention matrix in HBM. Same math, memory-aware execution.

---

[← Chapter 14 solutions](14-distributed-training-solutions.md) · [Solutions index](README.md) · [Next: Chapter 16 solutions →](16-frameworks-solutions.md)
