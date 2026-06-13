# Chapter 3 — Programming Mastery

> Anthropic, OpenAI, and DeepMind hire **exceptional software engineers who learned ML** — not the other way around. Your code quality, speed, and systems sense are the floor you must clear. ML is what you build *on top* of that floor.

This chapter covers the two languages that matter (Python deeply, one systems language seriously), plus the engineering practices — profiling, testing, debugging — that distinguish a professional from a script-writer.

---

## 3.1 Why engineering still wins

A frontier model is not a Jupyter notebook. It's a distributed system with data pipelines, training loops, checkpointing, evaluation harnesses, and serving infrastructure — hundreds of thousands of lines of code that must be **correct, fast, and maintainable**. The ML insight is necessary but not sufficient; the engineering is what ships it.

> Concretely: a research idea that takes you 3 days to implement cleanly beats the same idea that takes a sloppier engineer 3 weeks and produces unreproducible results. Speed and rigor compound.

---

## 3.2 Python mastery (not just literacy)

Everyone "knows Python." Few *master* it. Here's the gap.

### Idiomatic, readable code

```python
# Junior: works, but noisy
result = []
for i in range(len(items)):
    if items[i].score > 0.5:
        result.append(items[i].name)

# Senior: intent is obvious at a glance
result = [item.name for item in items if item.score > 0.5]
```

### Type hints — non-negotiable in real codebases

Type hints catch bugs before runtime and make code self-documenting. Every serious ML library (PyTorch, HF Transformers, vLLM) uses them.

```python
from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 32
    max_steps: int = 100_000
    warmup_steps: int = 2_000

def make_optimizer(params: list, cfg: TrainConfig) -> "torch.optim.Optimizer":
    ...
```

`@dataclass` is the idiomatic way to hold config — concise, typed, with free `__init__`/`__repr__`/equality. You'll see it constantly.

### Generators & iterators — essential for data pipelines

You cannot load a 2TB dataset into RAM. You **stream** it. Generators yield items lazily, one at a time, with constant memory.

```python
def stream_tokens(file_path: str):
    """Yield tokens one line at a time — constant memory, even for huge files."""
    with open(file_path) as f:
        for line in f:
            for token in line.split():
                yield token

# Memory stays flat whether the file is 1MB or 1TB.
for i, tok in enumerate(stream_tokens("corpus.txt")):
    if i >= 5:
        break
    print(tok)
```

> **Real-world:** Every data loader in large-scale training is a streaming pipeline. `IterableDataset` in PyTorch and the HF `datasets` streaming mode are generators under the hood. Understanding lazy evaluation is the difference between "OOM-killed" and "trains fine."

### Decorators — used everywhere in ML frameworks

```python
import time
from functools import wraps

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        print(f"{fn.__name__} took {time.perf_counter() - t0:.3f}s")
        return out
    return wrapper

@timed
def train_step(batch):
    ...
```

You'll meet `@torch.no_grad()`, `@torch.compile`, `@jax.jit`, `@property`, `@staticmethod` constantly. Knowing how decorators work demystifies all of them.

### Context managers — resource safety

```python
import torch

with torch.no_grad():          # disable gradient tracking inside this block
    logits = model(inputs)     # saves memory during inference/eval

# Custom one with contextlib:
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - t0:.3f}s")

with timer("forward pass"):
    out = model(x)
```

### Concurrency: async, threads, and processes — know which to use

This is a top interview topic. The decision hinges on **I/O-bound vs CPU-bound** and Python's **GIL** (Global Interpreter Lock, which prevents true parallel execution of Python bytecode across threads).

| Tool | Use when | Why |
|------|----------|-----|
| `asyncio` | Many concurrent I/O waits (API calls, DB) | Single thread, cooperative; perfect for serving many LLM requests |
| `threading` | I/O-bound, blocking libraries | GIL released during I/O, so threads help here |
| `multiprocessing` | CPU-bound Python work | Separate processes sidestep the GIL for true parallelism |

```python
import asyncio

async def call_model(prompt: str) -> str:
    await asyncio.sleep(0.5)          # stand-in for an async API/network call
    return f"response to {prompt!r}"

async def main():
    prompts = ["a", "b", "c", "d"]
    # All four "requests" run concurrently → ~0.5s total, not 2s.
    results = await asyncio.gather(*(call_model(p) for p in prompts))
    print(results)

asyncio.run(main())
```

> **Why this matters for AI infra:** An LLM inference server (Chapter 17) handles thousands of concurrent requests, each mostly *waiting* on the GPU. `asyncio` lets one process juggle them all efficiently. Picking `multiprocessing` here would be a costly mistake; picking it for CPU-bound data preprocessing would be correct. Interviewers love this distinction.

### NumPy vectorization — think in arrays, not loops

```python
import numpy as np

# Slow: Python loop (interpreted, element by element)
def slow_normalize(x):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = (x[i] - x.mean()) / x.std()
    return out

# Fast: vectorized (runs in optimized C, uses SIMD)
def fast_normalize(x):
    return (x - x.mean()) / x.std()
```

The vectorized version can be **10–100×** faster because it dispatches to compiled, SIMD-optimized routines instead of the Python interpreter. **Thinking in tensors instead of loops is the core skill** carried straight into PyTorch and JAX.

---

## 3.3 A systems language: C++, CUDA, or Rust

To be *cracked* (not just competent), you need to drop below Python when performance demands it. Python is the orchestration layer; the hot loops are compiled.

### Why you need this

PyTorch's fast ops are C++/CUDA. vLLM's PagedAttention is CUDA. The moment you want a custom kernel, a faster tokenizer, or to understand *why* something is slow, you're in systems-language territory.

**Pick one to start:**

- **CUDA / C++** — the direct path to GPU performance and the internals of every DL framework. Highest leverage for the inference/systems track (Chapter 15).
- **Rust** — increasingly the language of fast, safe ML tooling (HF's `tokenizers`, many inference servers, data tools). Memory-safe without a garbage collector.

### A taste of CUDA thinking (full treatment in Chapter 15)

```cpp
// Each GPU thread handles ONE element. Thousands run in parallel.
__global__ void add_vectors(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's global index
    if (i < n) c[i] = a[i] + b[i];                  // guard against overrun
}
```

The mental shift: instead of "loop over n elements," you write "what does *one* thread do?" and launch thousands at once. That **data-parallel** mindset is the foundation of GPU programming.

### A taste of Rust

```rust
// Memory-safe, no garbage collector, C-like speed.
fn normalize(xs: &[f32]) -> Vec<f32> {
    let mean: f32 = xs.iter().sum::<f32>() / xs.len() as f32;
    xs.iter().map(|x| x - mean).collect()
}
```

> **You don't need to master all three.** Pick CUDA if you want the inference/systems track; Rust if you love tooling and safety; C++ if you want framework-internals depth. One, done well, is plenty to start.

---

## 3.4 Profiling — measure, never guess

> "Premature optimization is the root of all evil" — but *informed* optimization wins. The rule: **profile first, optimize the actual bottleneck.**

```python
import cProfile, pstats

def workload():
    total = 0
    for i in range(1_000_000):
        total += i ** 2
    return total

profiler = cProfile.Profile()
profiler.enable()
workload()
profiler.disable()
pstats.Stats(profiler).sort_stats("cumulative").print_stats(5)
```

For GPU work you'll graduate to the **PyTorch profiler** and **NVIDIA Nsight** (Chapter 15). The discipline is identical: find where the time actually goes, fix *that*, re-measure.

```python
# PyTorch profiler sketch — find whether you're compute- or memory-bound
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    out = model(inputs)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 3.5 Testing & reproducibility — what separates research from "it worked once"

Unreproducible results are worthless. Two habits make you trustworthy:

### Seed everything

```python
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

### Test the tricky parts

ML code has subtle bugs that don't crash — they just make your model slightly worse, silently. Test shapes, invariants, and numerics.

```python
import torch

def test_attention_output_shape():
    q = k = v = torch.randn(2, 8, 16)        # (batch, seq, dim)
    out = attention(q, k, v)
    assert out.shape == (2, 8, 16), f"bad shape {out.shape}"

def test_softmax_sums_to_one():
    probs = torch.softmax(torch.randn(4, 10), dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-6)
```

> **Real-world:** A famous class of ML bugs is the "silent shape broadcast" — your loss still goes down, but a `(B, 1)` accidentally broadcasts against `(B, T)` and you're training on garbage. A three-line shape assertion catches it. Senior ML engineers litter their code with these.

---

## 3.6 Debugging the un-Googleable

When the bug is in *your* training run, you debug from first principles:

1. **Overfit a single batch.** A correct model can drive the loss to ~0 on one batch. If it can't, your model or loss is broken — not your data.
2. **Check shapes at every boundary.** Print or assert. Most bugs are shape bugs.
3. **Watch gradient norms.** Exploding → lower LR or clip. Vanishing → check init/activations.
4. **Bisect.** Disable half the pipeline; see if the bug persists. Repeat.
5. **Compare against a reference.** Reproduce a known-good result first, then change one thing at a time.

```python
# The single most useful ML debugging trick: can your model overfit ONE batch?
batch = next(iter(dataloader))
for step in range(200):
    loss = model(batch).loss
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    if step % 20 == 0:
        print(step, loss.item())     # should march toward ~0
```

---

## 3.7 Tooling that signals professionalism

| Tool | Purpose |
|------|---------|
| `uv` / `poetry` | Fast, reproducible dependency & environment management |
| `ruff` | Lightning-fast linter + formatter |
| `mypy` / `pyright` | Static type checking |
| `pytest` | Testing framework |
| `git` (fluently) | Branching, rebasing, bisecting, clean history |
| `pre-commit` | Auto-run lint/format/type checks before each commit |

> Using these isn't bureaucracy — it's how you keep a fast-moving codebase from rotting. A clean, typed, tested repo is itself a hiring signal when someone browses your GitHub.

---

## Interview signal

- **Q: "asyncio vs threading vs multiprocessing?"** → I/O-bound async/threads (GIL released on I/O), CPU-bound multiprocessing (sidestep GIL). Inference servers → asyncio.
- **Q: "How would you speed up this slow Python loop?"** → Profile first, then vectorize with NumPy, then push the hot path to C++/CUDA if needed.
- **Q: "How do you make a training run reproducible?"** → Seed all RNGs, pin dependencies, log configs, control nondeterministic CUDA ops.
- **Q: "Your model trains but performs poorly — debug it."** → Overfit one batch, check shapes, inspect gradient norms, bisect, compare to a reference.
- **Q: "What does the GIL prevent and how do you work around it?"** → Prevents parallel Python bytecode execution across threads; use multiprocessing or push compute into C/CUDA that releases the GIL.

---

## Exercises

1. Rewrite a triple-nested Python loop as a single vectorized NumPy expression; benchmark the speedup.
2. Write an `asyncio` script that fetches 100 URLs concurrently and compare wall-clock time to a sequential version.
3. Profile a slow function with `cProfile`, identify the bottleneck, fix it, and re-measure.
4. Write a custom decorator that retries a flaky function with exponential backoff.
5. Implement and unit-test a tiny `attention()` function: assert output shape and that attention weights sum to 1.

## Key takeaways

- Labs hire great engineers who learned ML — keep your engineering sharp.
- Master Python *deeply*: types, generators, decorators, context managers, async, vectorization.
- Learn one systems language (CUDA/C++/Rust) to drop below Python when it counts.
- Profile before optimizing; test shapes and numerics; seed everything.
- The "overfit one batch" trick is your fastest model-debugging tool.

**Next:** [Chapter 4 — CS Fundamentals](04-cs-fundamentals.md)
