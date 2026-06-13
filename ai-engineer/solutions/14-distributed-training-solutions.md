# Chapter 14 — Distributed Training · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-4-systems/14-distributed-training.md)

---

## Interview answers

### Q: "Data vs tensor vs pipeline parallelism?"

Three ways to split a training job, attacking different walls:

- **Data parallelism (DP)** — **replicate** the whole model on each GPU, split the **batch**, all-reduce gradients. Scales **throughput** (the time wall) but every GPU holds the full model, so it gives **no memory relief**.
- **Tensor parallelism (TP)** — split individual **matrices** (e.g., a big linear) across GPUs, so each holds a slice of every layer. Communicates **every layer**, so it needs **NVLink** bandwidth → keep it **within a node**.
- **Pipeline parallelism (PP)** — split the model's **layers** into stages across GPUs. Communicates only at **stage boundaries**, so it tolerates **slower inter-node** links, but introduces a **bubble** (idle time) fixed by micro-batching.

### Q: "Explain ZeRO / FSDP stages."

ZeRO (and PyTorch **FSDP**) eliminate the memory redundancy of plain DP by **sharding** training state across GPUs instead of replicating it:

- **Stage 1**: shard the **optimizer state** (Adam $m,v$, master weights) — the biggest chunk.
- **Stage 2**: also shard the **gradients**.
- **Stage 3 / FSDP**: also shard the **parameters** — each GPU stores only a slice and **gathers** the full weights layer-by-layer just-in-time for compute, then frees them.

It trades extra **communication** (all-gather weights, reduce-scatter gradients) for ~**linear memory savings** with GPU count — the standard way to train models that don't fit.

### Q: "How do you train a model that doesn't fit on one GPU?"

Escalate by complexity: (1) **FSDP / ZeRO-3** first — simplest, shards everything, often enough; (2) if a **single layer** is too big even sharded, add **tensor parallelism** within a node; (3) if the model is very deep, add **pipeline parallelism** across nodes; (4) combine all three as **3D parallelism** for frontier scale. Also stack **gradient checkpointing** (trade compute for activation memory) and **mixed precision**. Start simple — most "doesn't fit" problems are solved by FSDP + checkpointing.

### Q: "What's the pipeline bubble and how do you reduce it?"

In pipeline parallelism the stages run sequentially, so at the start (fill) and end (drain) some GPUs sit **idle** waiting for data to reach/leave them — that idle time is the **bubble**. You shrink it by splitting each batch into many **micro-batches** (GPipe / 1F1B scheduling) so all stages stay busy most of the time. The bubble fraction is $\frac{p-1}{m+p-1}$ for $p$ stages and $m$ micro-batches — more micro-batches → smaller bubble (Exercise 3).

### Q: "Why is TP within a node but PP across nodes?"

It's about **communication frequency vs. interconnect speed**. **TP** communicates **every layer, every step** (all-reduce activations) — that only stays efficient over very fast links like **NVLink** inside a node. **PP** communicates only at **stage boundaries** (a few times per step), so it tolerates the **slower inter-node** network (InfiniBand/Ethernet). You match each parallelism to the interconnect it can saturate without stalling.

### Q: "What is MFU and why care?"

**Model FLOPs Utilization** = (FLOPs your model actually does) ÷ (peak FLOPs the hardware *could* do). It's the efficiency/cost metric for a training run — a frontier run might hit 40–55% MFU; below that you're burning money. The main drags are **communication overhead** (sync gradients/weights) and poor overlap/kernels. You care because at thousand-GPU scale, a few points of MFU is millions of dollars and days of wall-clock — so the job is to **hide communication behind compute** and keep MFU high.

---

## Exercise solutions

### Exercise 1 — DDP with identical weights after all-reduce (CPU `gloo`)

Runnable on a single machine with the `gloo` backend (no GPUs needed).

```python
# run with: torchrun --nproc_per_node=2 ddp_demo.py
import os, torch, torch.distributed as dist, torch.nn as nn

def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)                          # SAME init on every rank
    model = nn.Linear(10, 1)

    # each rank sees a DIFFERENT data shard
    torch.manual_seed(rank)
    x, y = torch.randn(8, 10), torch.randn(8, 1)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()

    # manual all-reduce: average gradients across ranks (what DDP does internally)
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= world

    with torch.no_grad():
        for p in model.parameters(): p -= 0.1 * p.grad

    # verify every rank ended with identical weights
    w = next(model.parameters()).clone()
    gathered = [torch.zeros_like(w) for _ in range(world)]
    dist.all_gather(gathered, w)
    if rank == 0:
        print("all replicas identical:", all(torch.allclose(w, g) for g in gathered))
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

**Result:** despite each rank computing gradients on a **different** data shard, the **all-reduce averages them** so every replica applies the identical update and stays bit-identical. That synchronization invariant is what makes data-parallel training correct; real `DistributedDataParallel` just overlaps this all-reduce with the backward pass.

### Exercise 2 — Toy ZeRO-1: shard optimizer state across 4 ranks

```python
import numpy as np

P, R = 16, 4                              # 16 params, 4 "ranks"
np.random.seed(0)
weights = np.random.randn(P)
grads   = np.random.randn(P)

# Each rank OWNS a slice of the optimizer state (Adam m, v) -> 1/R the memory
shard = P // R
opt_state = {r: {"m": np.zeros(shard), "v": np.zeros(shard)} for r in range(R)}

lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
new_w = weights.copy()
for r in range(R):
    sl = slice(r*shard, (r+1)*shard)
    g = grads[sl]
    st = opt_state[r]
    st["m"] = b1*st["m"] + (1-b1)*g        # rank updates ONLY its slice's state
    st["v"] = b2*st["v"] + (1-b2)*g*g
    new_w[sl] = weights[sl] - lr * st["m"]/(np.sqrt(st["v"]) + eps)
# all-gather the updated parameter slices (each rank broadcasts its piece)

# Reference: full Adam on one device
m = (1-b1)*grads; v = (1-b2)*grads**2
ref_w = weights - lr * m/(np.sqrt(v) + eps)

print("matches single-device Adam:", np.allclose(new_w, ref_w))   # True
print(f"optimizer memory per rank: {shard*2} floats vs {P*2} replicated  ({R}x less)")
```

**Result:** each rank stores Adam state for only its **1/R slice** of parameters yet the combined update **exactly matches** single-device Adam — that's ZeRO-1. Optimizer state is the biggest memory consumer (8 bytes/param), so sharding it across $R$ GPUs cuts that cost $R\times$ for the price of an extra all-gather of the updated weights.

### Exercise 3 — Pipeline bubble fraction vs stages and micro-batches

```python
import numpy as np
import matplotlib.pyplot as plt

def bubble_fraction(p, m):
    return (p - 1) / (m + p - 1)          # GPipe bubble

for p in (4, 8, 16):
    print(f"\n{p} stages:")
    for m in (1, 4, 16, 64):
        print(f"  micro-batches={m:3d}: bubble={bubble_fraction(p, m)*100:5.1f}%")

ms = np.arange(1, 65)
for p in (4, 8, 16):
    plt.plot(ms, [bubble_fraction(p, m) for m in ms], label=f"{p} stages")
plt.xlabel("micro-batches (m)"); plt.ylabel("bubble fraction"); plt.legend()
plt.title("Pipeline bubble shrinks with more micro-batches"); plt.show()
```

**Result:** with micro-batches = 1 the bubble is huge (e.g., 16 stages → 94% idle!), but it falls off as $1/m$: at $m=64$, 16 stages drops to ~19%. This is exactly why GPipe/1F1B use **many micro-batches** — you keep all pipeline stages busy by always having work in flight. The tradeoff is micro-batches must be large enough for good per-GPU efficiency.

### Exercise 4 — Manual column-parallel linear (matches single-GPU after all-gather)

```python
import numpy as np

np.random.seed(0)
d_in, d_out = 8, 6
X = np.random.randn(4, d_in)
W = np.random.randn(d_in, d_out)          # full weight

# Split output columns across 2 "GPUs": each computes a slice of the output
half = d_out // 2
W0, W1 = W[:, :half], W[:, half:]
Y0 = X @ W0                                # GPU 0 -> first 3 output cols
Y1 = X @ W1                                # GPU 1 -> last 3 output cols
Y_parallel = np.concatenate([Y0, Y1], axis=1)   # all-gather along the column dim

Y_single = X @ W                          # reference
print("column-parallel == single GPU:", np.allclose(Y_parallel, Y_single))   # True
```

**Result:** splitting the weight matrix **by output columns**, computing each slice on a different "GPU," and **all-gathering** the partial outputs reproduces the single-GPU result exactly. This is Megatron-style **tensor parallelism**: no GPU holds the full matrix, but together they compute the full output. (Row-parallel splits the input dim and needs an all-*reduce* instead of all-gather.)

### Exercise 5 — Memory per GPU: 13B model, DDP vs FSDP

```python
def per_gpu_gb(n_params, n_gpus, method, bytes_per_param=16):
    # mixed-precision AdamW ~16 bytes/param: fp16 w(2)+grad(2)+fp32 master(4)+m(4)+v(4)
    total = n_params * bytes_per_param
    if method == "DDP":
        return total / 1e9                # replicated: same on EVERY gpu
    elif method == "FSDP":
        return total / n_gpus / 1e9       # sharded across gpus

N = 13e9
for gpus in (8, 64):
    print(f"{gpus:2d} GPUs:  DDP = {per_gpu_gb(N, gpus, 'DDP'):6.1f} GB/GPU"
          f"   FSDP = {per_gpu_gb(N, gpus, 'FSDP'):5.2f} GB/GPU")
```

**Result:**

| GPUs | DDP per GPU | FSDP per GPU |
|---|---|---|
| 8 | ~208 GB (won't fit!) | ~26 GB |
| 64 | ~208 GB (won't fit!) | ~3.25 GB |

DDP **replicates** the full ~208 GB of training state on **every** GPU regardless of count — so a 13B model can't even start under DDP on 80 GB cards. **FSDP shards** that state, so per-GPU memory falls **linearly** with GPU count (26 GB at 8, 3.25 GB at 64, plus activations). This is the concrete reason FSDP/ZeRO is the default for large models.

---

[← Chapter 13 solutions](13-evaluation-solutions.md) · [Solutions index](README.md) · [Next: Chapter 15 solutions →](15-gpu-programming-solutions.md)
