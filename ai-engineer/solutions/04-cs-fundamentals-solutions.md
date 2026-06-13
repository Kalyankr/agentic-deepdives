# Chapter 4 — CS Fundamentals · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-1-foundations/04-cs-fundamentals.md)

---

## Interview answers

### Q: "What's the time complexity of self-attention and why does it matter?"

Self-attention is $O(n^2 d)$ in compute and $O(n^2)$ in memory for the attention matrix, where $n$ is sequence length and $d$ the head dimension — every token attends to every other token, so the score matrix is $n \times n$. **Why it matters:** doubling the context **quadruples** the attention cost. That single fact drives a huge amount of modern systems work: KV caching, **FlashAttention** (which keeps the cost but avoids materializing the $n^2$ matrix in HBM), sparse/sliding-window attention, and the general expense of long-context models. When someone asks "why is 128k context so much more expensive than 4k," this is the answer.

### Q: "Is LLM inference compute-bound or memory-bound?"

It depends on the phase and batch size:

- **Prefill** (processing the prompt) is **compute-bound** — it's a big matmul over all prompt tokens at once, with high arithmetic intensity.
- **Decode** (generating one token at a time) at small batch is **memory-bound** — for each token you must reload the entire weight matrix from HBM to do very little arithmetic, so you're limited by memory bandwidth, not FLOPs.

This is *the* key inference insight: because decode is memory-bound, **quantization** (less data to move) and **operator fusion** (fewer HBM round-trips) speed it up, while raw FLOPs are often idle. Batching raises arithmetic intensity and pushes you back toward compute-bound (Chapter 10).

### Q: "bf16 vs fp16 for training?"

Both are 16-bit, but they split the bits differently:

- **fp16**: 1 sign / **5 exponent** / 10 mantissa → more precision but a **narrow dynamic range**; gradients underflow/overflow easily, so it needs **loss scaling**.
- **bf16**: 1 sign / **8 exponent** / 7 mantissa → the **same exponent range as fp32**, so it rarely overflows and needs **no loss scaling**, at the cost of less precision.

For training, **bf16 wins** (on hardware that supports it) because range matters more than mantissa precision for gradients — fewer numerical headaches, no loss-scaling bookkeeping. It's the standard for modern LLM training. The whole answer comes straight from the bit layout.

### Q: "Explain all-reduce and where it's used."

All-reduce is a collective operation that **combines a value across all workers (typically by sum) and distributes the result back to every worker**, so all end with the identical reduced value. In **data-parallel training**, each GPU computes gradients on its own shard of the batch; an all-reduce **averages** those gradients so every replica applies the same update and the weights stay in sync. It's implemented efficiently as **ring-all-reduce** (bandwidth-optimal) in NCCL, and it's frequently the **communication bottleneck** at scale — which is why we overlap it with backward compute (Chapter 14).

### Q: "Why does your DataLoader have multiple workers?"

To keep the GPU fed. Data loading/preprocessing (decode images, tokenize, augment) is CPU work; if the main process does it inline, the GPU **starves** waiting for the next batch. Multiple worker **processes** (separate processes to escape the GIL) preprocess batches **in parallel** and prefetch them into a queue, so the next batch is ready the instant the GPU finishes the current one. Tune `num_workers` and `prefetch_factor` so data prep is never the bottleneck — an idle GPU is wasted money.

---

## Exercise solutions

### Exercise 1 — top-k (heap) and top-p (binary search) sampling

```python
import numpy as np, heapq

def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def top_k_sample(logits, k, rng):
    # heapq.nlargest finds the k largest in O(n log k)
    idx = heapq.nlargest(k, range(len(logits)), key=lambda i: logits[i])
    p = softmax(logits[idx])
    return int(rng.choice(idx, p=p))

def top_p_sample(logits, p_threshold, rng):
    order = np.argsort(logits)[::-1]            # sort descending: O(n log n)
    probs = softmax(logits[order])
    cum = np.cumsum(probs)
    # binary search for the smallest prefix whose cumulative prob >= threshold: O(log n)
    cutoff = np.searchsorted(cum, p_threshold) + 1
    keep = order[:cutoff]
    p = softmax(logits[keep])
    return int(rng.choice(keep, p=p))

rng = np.random.default_rng(0)
logits = rng.standard_normal(50_000)
print("top-k :", top_k_sample(logits, k=40, rng=rng))
print("top-p :", top_p_sample(logits, p_threshold=0.9, rng=rng))
```

**Complexity:** top-k via a heap is $O(n\log k)$ (cheaper than a full sort when $k \ll n$). top-p needs a sorted order ($O(n\log n)$) and then a **binary search** ($O(\log n)$) over the cumulative distribution to find the nucleus cutoff. Both then sample from the renormalized survivors. These are the exact algorithms behind the `top_k`/`top_p` API knobs.

### Exercise 2 — Topological sort of a DAG (reused in Chapter 5 autograd)

```python
def topo_sort(graph):
    """graph: {node: [children]}. Returns nodes so every node precedes its children."""
    visited, order = set(), []
    def dfs(u):
        if u in visited: return
        visited.add(u)
        for v in graph.get(u, []):
            dfs(v)
        order.append(u)          # post-order
    for node in graph:
        dfs(node)
    return order[::-1]           # reverse post-order = topological order

dag = {'a': ['b', 'c'], 'b': ['d'], 'c': ['d'], 'd': ['e'], 'e': []}
print(topo_sort(dag))            # e.g. ['a', 'c', 'b', 'd', 'e']
```

**Result:** every node appears before its children. This is *exactly* the algorithm autograd uses: build the graph of operations, topologically sort it, then traverse in **reverse** to apply the chain rule (each node's gradient is ready before its parents need it). You'll reuse this verbatim in `Value.backward()` in Chapter 5.

### Exercise 3 — Sequential vs strided memory access (cache locality)

```python
import numpy as np, time

n = 64 * 1024 * 1024
a = np.ones(n, dtype=np.float64)

def walk(stride):
    t0 = time.perf_counter()
    s = a[::stride].sum()         # touch every `stride`-th element
    return time.perf_counter() - t0, s

for stride in (1, 16, 256):
    dt, _ = walk(stride)
    print(f"stride {stride:4d}: {dt*1000:7.2f} ms  (elements touched: {n//stride})")
```

**Result:** even though larger strides touch **fewer** elements, the cost per element rises sharply. Sequential access (stride 1) uses every byte of each 64-byte cache line and lets the hardware **prefetcher** stream data; large strides waste most of each fetched cache line and defeat prefetching, so you pay full memory latency per access. **This is why memory layout matters** — coalesced/contiguous access is the same principle that makes GPU memory coalescing (Chapter 15) and array-of-structs vs struct-of-arrays decisions matter.

### Exercise 4 — Simulate all-reduce across 4 "GPUs"

```python
import numpy as np

# Each "GPU" computed gradients on a different data shard:
grads = [np.array([1.0, 2.0]),
         np.array([3.0, 0.0]),
         np.array([1.0, 4.0]),
         np.array([3.0, 2.0])]

def all_reduce_mean(grads):
    total = np.sum(grads, axis=0)          # the "reduce" (sum)
    avg = total / len(grads)               # average
    return [avg.copy() for _ in grads]     # "distribute" identical result to all

reduced = all_reduce_mean(grads)
print("averaged gradient:", reduced[0])    # [2. 2.]
print("all identical    :", all(np.allclose(reduced[0], g) for g in reduced))  # True

# Apply the synced update on each replica with identical starting weights:
w = np.zeros(2); lr = 0.1
weights = [w - lr * g for g in reduced]
print("replicas in sync :", all(np.allclose(weights[0], x) for x in weights))  # True
```

**Result:** after the all-reduce, every replica holds the identical averaged gradient `[2, 2]`, so applying the update leaves all replicas with **identical weights** — the invariant that makes data-parallel training correct. Real NCCL does this as a bandwidth-optimal ring, but the semantics are exactly this sum-and-broadcast.

### Exercise 5 — Classify ten ops as compute- vs memory-bound

Arithmetic intensity = FLOPs ÷ bytes moved. **High intensity → compute-bound; low intensity → memory-bound.**

| Op | Bound | Why |
|---|---|---|
| Large dense matmul / GEMM (big $N$) | **Compute** | $O(n^3)$ FLOPs vs $O(n^2)$ data → reuses each loaded value many times |
| Conv with many channels | **Compute** | High data reuse, maps to GEMM |
| Attention $QK^\top$ (large, batched) | **Compute** | Matmul-heavy at scale |
| Elementwise add / scale | **Memory** | 1 FLOP per element loaded → ~zero reuse |
| GELU / ReLU / SiLU activation | **Memory** | A few FLOPs per element, then write back |
| LayerNorm / RMSNorm | **Memory** | Reductions + scale over each element, little reuse |
| Softmax | **Memory** | max/exp/sum/divide passes, low intensity |
| Embedding lookup | **Memory** | Pure gather from HBM, no math |
| Dropout / mask | **Memory** | Read, multiply by mask, write |
| **Decode-phase** matvec (batch 1) | **Memory** | Reload whole weight matrix to produce one token |

**Takeaway:** the matmuls (and convs) are compute-bound; almost everything else — the "glue" ops and single-token decode — is memory-bound. That's precisely why **kernel fusion** (do all the memory-bound ops in one HBM round-trip) is such a big win, and why FlashAttention restructures attention to be memory-efficient (Chapter 15).

---

[← Chapter 3 solutions](03-programming-solutions.md) · [Solutions index](README.md) · [Next: Chapter 5 solutions →](05-neural-networks-solutions.md)
