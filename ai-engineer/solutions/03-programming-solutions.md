# Chapter 3 — Programming Mastery · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-1-foundations/03-programming.md)

---

## Interview answers

### Q: "asyncio vs threading vs multiprocessing?"

All three give "concurrency," but they solve different bottlenecks:

| Tool | Parallelism | Best for | Why |
|---|---|---|---|
| **asyncio** | Concurrent, single-threaded | High-volume **I/O-bound** (thousands of network calls) | Cooperative `await`; one thread juggles many waits with tiny overhead |
| **threading** | Concurrent, not parallel (CPython) | **I/O-bound**, blocking libraries | The GIL is released during I/O and many C extensions, so threads overlap waits |
| **multiprocessing** | Truly parallel | **CPU-bound** (number crunching in pure Python) | Separate processes = separate interpreters = no shared GIL |

The deciding question is **"am I waiting or computing?"** Waiting on the network/disk → asyncio (or threads). Burning CPU in Python → multiprocessing. For an **inference server** you want asyncio: each request spends most of its time awaiting the GPU/model, so one event loop can handle thousands of concurrent requests cheaply.

### Q: "How would you speed up this slow Python loop?"

In order:

1. **Profile first** (`cProfile`, `line_profiler`) — never optimize by guessing; find the actual hot line.
2. **Vectorize** with NumPy/PyTorch — replace the Python-level loop with array ops that run in optimized C. This is usually a 10–100× win.
3. **Use better algorithms/data structures** — an $O(n^2)$ loop that should be $O(n\log n)$ won't be saved by vectorization.
4. **Drop to a compiled language** — Numba/Cython, or a custom C++/CUDA kernel — only for the hot path that's left after the above.

The headline: *measure, then vectorize, then compile.* Most "slow Python" is just un-vectorized array math.

### Q: "How do you make a training run reproducible?"

- **Seed every RNG**: Python `random`, NumPy, and the framework (`torch.manual_seed`, plus `torch.cuda.manual_seed_all`).
- **Pin dependencies**: lockfile (`uv.lock`/`requirements.txt` with hashes), and record the CUDA/cuDNN versions.
- **Log the full config**: hyperparameters, git commit, data version/hash — ideally auto-logged to W&B/MLflow.
- **Control nondeterministic ops**: `torch.use_deterministic_algorithms(True)`, set `cudnn.deterministic=True`/`benchmark=False`, and seed the DataLoader workers. Accept that some GPU ops trade a little speed for determinism.

Perfect bitwise reproducibility across different hardware is often impossible; the realistic goal is *same machine + same seed + pinned env → same result.*

### Q: "Your model trains but performs poorly — debug it."

A concrete checklist, cheapest first:

1. **Overfit one batch.** Can the model drive loss to ~0 on a single batch? If not, there's a bug (wiring, loss, labels), not a tuning problem.
2. **Check shapes & dtypes** at every boundary — a silent broadcast bug is the classic culprit.
3. **Inspect gradient norms** — exploding (→ clip/lower LR) or vanishing (→ init, residuals, normalization)?
4. **Sanity-check the data** — labels aligned? normalization correct? train/val leakage?
5. **Bisect** — remove augmentations/regularization, shrink the model, compare against a known-good reference implementation.

### Q: "What does the GIL prevent and how do you work around it?"

The **Global Interpreter Lock** lets only one thread execute Python **bytecode** at a time, so pure-Python code can't run in parallel across threads — multithreading won't speed up CPU-bound Python. Workarounds:

- **multiprocessing** — separate processes each have their own GIL → true parallelism (at the cost of IPC/memory).
- **Push compute into C/CUDA** — NumPy, PyTorch, and many extensions **release the GIL** during heavy work, so the real math runs in parallel even from threads.
- **asyncio** for I/O — sidesteps the issue entirely since you're waiting, not computing.

(Note: "free-threaded" CPython, PEP 703, is making the GIL optional — worth mentioning to show you follow the language.)

---

## Exercise solutions

### Exercise 1 — Vectorize a triple-nested loop

A naive matrix multiply is three nested loops; NumPy collapses it into one C call.

```python
import numpy as np, time

A = np.random.rand(200, 200); B = np.random.rand(200, 200)

def matmul_loops(A, B):
    n, k = A.shape; k2, m = B.shape
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += A[i, p] * B[p, j]
            C[i, j] = s
    return C

t0 = time.perf_counter(); C_loop = matmul_loops(A, B); t1 = time.perf_counter()
C_vec = A @ B;                                          t2 = time.perf_counter()

print("loops :", t1 - t0, "s")
print("numpy :", t2 - t1, "s")
print("speedup:", (t1 - t0) / (t2 - t1), "x")
print("match :", np.allclose(C_loop, C_vec))
```

**Result:** identical numerically (`match = True`), but NumPy is typically **hundreds to thousands of times faster** — it dispatches to a tuned BLAS kernel instead of running 8M Python-interpreter iterations. This is the single most important performance habit in ML Python.

### Exercise 2 — Concurrent URL fetches with asyncio

Self-contained version (simulates network latency with `asyncio.sleep`; swap in `aiohttp` for real URLs).

```python
import asyncio, time

async def fetch(url):
    await asyncio.sleep(0.1)        # stands in for network latency
    return f"{url}: 200"

async def fetch_all(urls):
    return await asyncio.gather(*(fetch(u) for u in urls))   # all in flight at once

def fetch_sequential(urls):
    async def run():
        return [await fetch(u) for u in urls]                # one at a time
    return asyncio.run(run())

urls = [f"https://example.com/{i}" for i in range(100)]

t0 = time.perf_counter(); fetch_sequential(urls); t1 = time.perf_counter()
t2 = time.perf_counter(); asyncio.run(fetch_all(urls)); t3 = time.perf_counter()
print("sequential:", round(t1 - t0, 2), "s")   # ~10 s  (100 × 0.1)
print("concurrent:", round(t3 - t2, 2), "s")   # ~0.1 s (all overlap)
```

Real version:

```python
import aiohttp, asyncio
async def fetch(session, url):
    async with session.get(url) as r:
        return r.status, await r.text()
async def fetch_all(urls):
    async with aiohttp.ClientSession() as s:
        return await asyncio.gather(*(fetch(s, u) for u in urls))
```

**Result:** sequential takes ~10 s (the 100 waits happen back-to-back); concurrent takes ~0.1 s (all waits overlap on one thread). This is exactly why I/O-bound servers use asyncio — the speedup is the degree of concurrency.

### Exercise 3 — Profile, find the bottleneck, fix, re-measure

```python
import cProfile, pstats, io

def slow():
    # Accidentally O(n^2): building a string by repeated concatenation
    s = ""
    for i in range(100_000):
        s += str(i)
    return s

def fast():
    return "".join(str(i) for i in range(100_000))   # O(n)

def profile(fn):
    pr = cProfile.Profile(); pr.enable(); fn(); pr.disable()
    st = pstats.Stats(pr, stream=io.StringIO()); st.sort_stats('cumulative')
    return st.total_tt

print("slow:", profile(slow), "s")   # bottleneck: the += string concat
print("fast:", profile(fast), "s")   # join is dramatically faster
```

**Diagnosis & fix:** `cProfile` (sorted by cumulative time) shows the time concentrated in the loop doing `s += str(i)`. Python strings are immutable, so each `+=` copies the whole accumulated string → quadratic. Replacing it with `"".join(...)` makes it linear and much faster. **The lesson is the workflow:** profile → identify the hot line → fix the algorithm → re-measure to confirm.

### Exercise 4 — Retry decorator with exponential backoff

```python
import time, functools, random

def retry(max_attempts=5, base_delay=0.1, backoff=2.0, jitter=True, exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise                      # exhausted -> re-raise
                    sleep = delay + (random.random() * delay if jitter else 0)
                    print(f"attempt {attempt} failed ({e!r}); retrying in {sleep:.2f}s")
                    time.sleep(sleep)
                    delay *= backoff               # exponential growth
        return wrapper
    return decorator

calls = {"n": 0}

@retry(max_attempts=5, base_delay=0.05)
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("transient")
    return "ok"

print(flaky())          # succeeds on the 3rd attempt -> "ok"
```

**Result:** the function fails twice with growing delays (0.05s, 0.10s, …) then succeeds on attempt 3. Exponential backoff + jitter is the standard pattern for transient failures (rate limits, flaky networks) — it avoids hammering a struggling service and prevents synchronized "thundering herd" retries. `functools.wraps` preserves the wrapped function's name/docstring.

### Exercise 5 — A tiny `attention()` with unit tests

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V):
    """Q:(t,d) K:(s,d) V:(s,dv) -> (t,dv). Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)      # (t, s)
    weights = softmax(scores, axis=-1)   # rows sum to 1
    return weights @ V, weights

# --- unit tests ---
def test_attention():
    t, s, d, dv = 3, 5, 8, 4
    Q = np.random.rand(t, d); K = np.random.rand(s, d); V = np.random.rand(s, dv)
    out, w = attention(Q, K, V)
    assert out.shape == (t, dv), f"bad output shape {out.shape}"
    assert w.shape == (t, s),    f"bad weight shape {w.shape}"
    assert np.allclose(w.sum(axis=-1), 1.0), "attention rows must sum to 1"
    assert (w >= 0).all(), "attention weights must be non-negative"
    print("all attention tests passed")

test_attention()
```

**Result:** all assertions pass. The two invariants that matter — **output shape** `(t, dv)` and **each attention row summing to 1** (it's a probability distribution over the keys) — are exactly the checks that catch the most common attention bugs (wrong transpose, wrong softmax axis). You'll reuse this `attention` in Chapter 6.

---

[← Chapter 2 solutions](02-mathematics-solutions.md) · [Solutions index](README.md) · [Next: Chapter 4 solutions →](04-cs-fundamentals-solutions.md)
