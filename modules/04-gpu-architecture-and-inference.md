# Module 04 · GPU Architecture & Inference

> **Goal:** Understand how GPUs actually execute deep learning, write/read CUDA-level kernels, and master single-node LLM **inference** optimization — quantization, KV-cache management, batching, and high-performance serving with vLLM/TensorRT-LLM.

**Duration:** ~6 weeks. **Prereqs:** [Module 02](02-transformer-internals.md).

---

## 4.1 GPU architecture

- SIMT execution model: threads → warps (32) → blocks → grid
- Streaming Multiprocessors (SMs), warp schedulers, occupancy
- **Memory hierarchy:** registers → shared memory/L1 → L2 → **HBM** (global). Bandwidth and latency of each.
- **Tensor Cores** — matrix-multiply-accumulate units; supported dtypes (FP16/BF16/FP8/INT8); why they dominate throughput
- Memory coalescing, bank conflicts, warp divergence
- **Roofline model** for GPUs: arithmetic intensity (FLOP/byte) vs. peak compute vs. peak bandwidth — *the* tool for deciding if a kernel is compute- or memory-bound
- Know the spec sheet: A100, H100, H200, B200 — FLOPs (per dtype), HBM capacity & bandwidth, NVLink

## 4.2 CUDA & kernels (enough to be dangerous)

- CUDA programming model: kernels, threads, blocks, grids, `__global__`/`__device__`
- Shared memory, synchronization (`__syncthreads`), tiling
- Write a few kernels: vector add → matrix multiply (naive → tiled) → a fused operation
- **Triton** — Python-like kernel authoring; write a fused softmax and a simplified attention kernel
- Why **fusion** matters (memory-bound ops dominate; fewer round-trips to HBM)
- `torch.compile`, CUDA graphs, and how PyTorch dispatches to kernels
- Profiling: Nsight Systems/Compute, `torch.profiler`, reading a kernel timeline

> **Build:** A tiled matmul CUDA (or Triton) kernel that approaches cuBLAS for a given size; plot it on a roofline. A fused softmax in Triton benchmarked vs. PyTorch eager.

## 4.3 LLM inference, deeply

### The two phases (memorize this)
- **Prefill** — process the whole prompt in parallel; **compute-bound**; fills the KV cache.
- **Decode** — generate one token at a time; **memory-bandwidth-bound** (must read all weights + KV cache per token).

### The latency metrics that matter
- **TTFT** (Time To First Token) — dominated by prefill
- **TPOT / ITL** (Time Per Output Token / Inter-Token Latency) — dominated by decode
- **Throughput** (tokens/sec, requests/sec) vs. per-request latency — the fundamental tension
- Goodput under SLA

### KV cache management
- Why the KV cache grows linearly with sequence length and batch
- It is usually the **memory bottleneck** at inference, not the weights
- **PagedAttention (vLLM)** — virtual-memory-style paging to eliminate fragmentation and enable high batch sizes
- Prefix/prompt caching (share KV for common prefixes), radix attention (SGLang)

### Batching
- Static batching vs. **continuous (in-flight) batching** — add/remove requests mid-flight to keep the GPU busy
- Chunked prefill, prefill/decode disaggregation (split the two phases onto different workers)

## 4.4 Quantization & compression

- Precision formats recap: FP16/BF16/FP8/INT8/INT4 and their ranges
- **Post-training quantization:** GPTQ, AWQ, SmoothQuant; weight-only vs. weight+activation
- **FP8** inference (H100+), KV-cache quantization
- Quantization-aware training (concept)
- **Speculative decoding** — a small draft model proposes tokens, the big model verifies in parallel; lossless speedup. Variants: Medusa, EAGLE, lookahead, n-gram.
- Distillation & pruning (overview)

> **Build:** Quantize your Module 02/03 model to INT8/INT4 (GPTQ or AWQ). Measure quality drop (perplexity), latency, and memory. Then add speculative decoding with a small draft model and measure the speedup.

## 4.5 Serving stacks

- **vLLM** — PagedAttention + continuous batching; the de facto OSS server. Learn its architecture and OpenAI-compatible API.
- **TensorRT-LLM / Triton Inference Server** — NVIDIA's high-performance path
- **SGLang** — RadixAttention, structured generation
- **TGI** (Text Generation Inference), **llama.cpp / GGUF** (CPU/edge), **MLC**
- Structured output / constrained decoding (JSON, grammars) — Outlines, XGrammar
- Request scheduling, admission control, priority

> **Build:** Stand up vLLM serving an open model. Load-test it (e.g., with a locust/k6 script or vLLM's benchmark). Produce a **latency–throughput curve**: sweep concurrency and plot TTFT, TPOT, and tokens/sec. Find the knee of the curve.

---

## Module 04 capstone — **A measured inference stack**

1. A Triton/CUDA fused kernel benchmarked on a roofline.
2. Your model quantized (INT8/INT4) with a quality-vs-speed-vs-memory table.
3. Speculative decoding integrated and measured.
4. A vLLM deployment with a full latency–throughput characterization and a short report on where the bottleneck is (compute vs. bandwidth vs. KV cache) and how you'd fix it.

## Exit criteria
- [ ] You can explain prefill vs. decode and which resource bounds each.
- [ ] You can compute KV-cache memory and explain PagedAttention and continuous batching.
- [ ] You can read a roofline plot and classify a kernel.
- [ ] You can quantize a model and reason about the quality/latency trade-off.
- [ ] You can produce and interpret a latency–throughput curve.

## Core papers / sources
- *FlashAttention v1/v2/v3* — Dao et al.
- *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM) — Kwon et al., 2023
- *Orca* (continuous batching) — Yu et al., 2022
- *GPTQ*, *AWQ*, *SmoothQuant*
- *Speculative Decoding* — Leviathan et al., 2023; *Medusa*; *EAGLE*
- NVIDIA CUDA C++ Programming Guide; Triton docs/tutorials
- *Programming Massively Parallel Processors* (PMPP) — Kirk & Hwu
- "Making Deep Learning Go Brrrr From First Principles" — Horace He
