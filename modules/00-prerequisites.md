# Module 00 · Prerequisites

> **Goal:** Build the mathematical, programming, and systems foundation so that nothing in later modules is a black box.

**Duration:** ~3–4 weeks (skip what you already know — use the checklist to test out).

---

## 0.1 Mathematics for ML

You don't need a PhD, but you must be *fluent* — able to read equations in papers and translate them to code.

### Linear algebra
- Vectors, matrices, tensors; shapes and broadcasting
- Matrix multiplication, transpose, inverse, rank
- Dot products, norms (L1, L2, cosine similarity)
- Eigenvalues/eigenvectors, SVD (used in LoRA, PCA)
- **Why it matters:** every layer is a matmul; attention is a similarity computation.

### Calculus & optimization
- Derivatives, partial derivatives, gradients, the chain rule (= backprop)
- Jacobians and Hessians (conceptually)
- Convexity, local vs. global minima, saddle points
- Gradient descent, momentum, learning-rate intuition

### Probability & statistics
- Random variables, distributions (Gaussian, Bernoulli, categorical)
- Expectation, variance, covariance
- Conditional probability, Bayes' rule
- Maximum likelihood estimation (MLE), KL divergence, cross-entropy, entropy
- **Why it matters:** the LM loss *is* cross-entropy; RLHF and DPO are built on KL terms.

### Information theory (light)
- Entropy, perplexity, mutual information

> **Build:** Implement in NumPy (no autograd) — linear regression with gradient descent, logistic regression, softmax + cross-entropy, and PCA via SVD. Plot the loss curves.

---

## 0.2 Programming & engineering

### Python (expert level)
- Idiomatic Python, typing, dataclasses, generators, context managers, decorators
- `async`/`await` and `asyncio` (critical for agent/inference servers)
- Multiprocessing vs. threading vs. async; the GIL
- Profiling: `cProfile`, `py-spy`, `line_profiler`, memory profiling
- Packaging & envs: `uv`, `pyproject.toml`, virtual environments

### NumPy / tensor mechanics
- Vectorization, broadcasting rules, `einsum`, strides, views vs. copies
- Why loops are slow and vectorized ops are fast (memory locality)

### Software engineering
- Git (branching, rebase, bisect), code review discipline
- Testing: `pytest`, fixtures, property-based testing (`hypothesis`)
- Clean code, design docs, reproducibility (seeds, config management with `hydra`/`pydantic`)
- Containers: Docker fundamentals

> **Build:** A small typed Python package with tests, CI (GitHub Actions), and a CLI. Profile a slow function and make it 10× faster with vectorization.

---

## 0.3 Systems fundamentals

These separate "ML hobbyist" from "AI systems engineer."

### Computer architecture
- CPU vs. GPU execution models (latency- vs. throughput-optimized)
- Memory hierarchy: registers → L1/L2 → DRAM → disk; latency numbers
- Cache lines, locality, the **roofline model** (compute-bound vs. memory-bound)
- Floating point: FP32, TF32, FP16, BF16, FP8, INT8 — ranges and precision

### Operating systems & networking
- Processes, threads, scheduling, virtual memory, page faults
- File I/O, memory mapping, zero-copy
- TCP/IP basics, latency vs. bandwidth, RPC, gRPC, HTTP/2
- Serialization (protobuf, JSON), backpressure, queues

### "Numbers every engineer should know"
Memorize orders of magnitude:

| Operation | ~Latency |
|-----------|----------|
| L1 cache reference | ~1 ns |
| Branch mispredict | ~3 ns |
| L2 cache reference | ~4 ns |
| Main memory (DRAM) reference | ~100 ns |
| Read 1 MB sequentially from RAM | ~3 µs |
| SSD random read | ~16 µs |
| Round trip within same datacenter | ~500 µs |
| Read 1 MB sequentially from SSD | ~49 µs |
| Round trip CA ↔ Netherlands | ~150 ms |

> These power back-of-envelope estimates in [Module 11](11-system-design-and-capacity-planning.md).

---

## 0.4 GPU & environment setup

- Install CUDA toolkit + drivers; verify with `nvidia-smi`
- PyTorch with CUDA; confirm `torch.cuda.is_available()`
- Learn `nvidia-smi`, `nvtop`, and basic GPU memory monitoring
- Options if you lack a GPU: Google Colab, Kaggle (free T4/P100), Lambda, RunPod, vast.ai, Modal
- Mixed precision basics: `torch.autocast`, `bfloat16`

> **Build:** Benchmark a matmul on CPU vs. GPU across sizes. Plot GFLOP/s and find where the GPU wins. Measure the memory-bandwidth roofline.

---

## Module 00 capstone

A repo containing:
1. From-scratch (NumPy) linear & logistic regression + softmax classifier with training curves.
2. A profiling write-up: a function you sped up 10×, with before/after flamegraphs.
3. A GPU vs. CPU matmul benchmark with a roofline plot and a short analysis.

## Exit criteria (you can move on when…)
- [ ] You can derive the gradient of softmax + cross-entropy by hand.
- [ ] You can explain compute-bound vs. memory-bound using the roofline model.
- [ ] You can write async Python and explain the GIL.
- [ ] You know latency numbers to an order of magnitude.

## Core resources
- *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong (free PDF)
- 3Blue1Brown — Linear Algebra & Neural Network series
- *Designing Data-Intensive Applications* — Kleppmann (Ch. 1–4 now; rest in Module 10)
- CS231n notes (optimization, backprop)
- "Latency Numbers Every Programmer Should Know" (Jeff Dean)
