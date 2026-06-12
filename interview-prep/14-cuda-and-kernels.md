# 14 · CUDA & GPU Kernels (ML/Research & Performance Track)

> For **ML Systems / Performance / Research Engineer** loops (and any role touching training or
> inference infra), expect a round on **how GPUs actually compute**: the execution & memory model, why
> a kernel is slow, and sometimes "write/optimize this kernel." You rarely need to write perfect CUDA
> live — you need to **reason about memory traffic, parallelism, and the roofline** out loud.

Pairs with [10-numbers-and-hardware](10-numbers-and-hardware.md) (specs & roofline) and the
FlashAttention/serving entries in [09-papers](09-papers.md).

---

## The execution model (say this fluently)

- **Thread → warp → block → grid.** A kernel launches a **grid** of **thread blocks**; each block runs
  on one **SM** (streaming multiprocessor); threads execute in **warps of 32** in lockstep (SIMT).
- **Blocks are independent** and scheduled in any order across SMs — that's how CUDA scales across GPUs
  of different sizes. Threads **within a block** can cooperate via **shared memory** + `__syncthreads()`.
- **Warp divergence:** if threads in a warp take different branches (`if/else`), the warp executes both
  paths serially with masking → slowdown. Keep branches warp-aligned.
- **Occupancy:** active warps per SM ÷ max possible. Higher occupancy hides memory latency (more warps
  to swap in), but it's a means, not the goal — limited by registers/shared memory per block.

## The memory hierarchy (this is where perf lives)

| Memory | Scope | Latency | Notes |
|--------|-------|---------|-------|
| Registers | per-thread | ~1 cycle | fastest; spilling to "local" memory is slow |
| Shared memory / L1 | per-block | ~20–30 cycles | programmer-managed scratchpad; key to tiling |
| L2 cache | device-wide | ~200 cycles | shared across SMs |
| Global (HBM) | device-wide | ~400–800 cycles | huge but slow; **minimize traffic** |
| Host (CPU RAM) | over PCIe/NVLink | very slow | avoid in hot loops |

> The #1 optimization theme: **move data up the hierarchy and reuse it.** Most ML kernels are
> **memory-bandwidth bound**, so the win is cutting HBM traffic, not FLOPs.

## Coalescing & bank conflicts

- **Coalesced global access:** consecutive threads in a warp should read consecutive addresses → the
  hardware merges them into a few wide transactions. Strided/random access wastes bandwidth.
- **Shared-memory bank conflicts:** shared memory has 32 banks; if multiple threads in a warp hit the
  same bank (different address), accesses serialize. Fix with padding (e.g. `tile[32][33]`).

## Roofline: is my kernel compute- or memory-bound?

**Arithmetic intensity** = FLOPs ÷ bytes moved. Compare to the GPU's ratio (peak FLOPs ÷ bandwidth,
~300+ FLOP/byte for an H100):
- Below it → **memory-bound** (most elementwise/attention decode ops) → cut bytes: fuse kernels, use
  lower precision, improve reuse.
- Above it → **compute-bound** (big GEMMs) → use **tensor cores**, better tiling, higher precision throughput.

Worked intuition: elementwise `y = a*x + b` reads ~8 bytes, does ~2 FLOPs → intensity ~0.25 →
hopelessly memory-bound → only HBM bandwidth matters.

---

## The canonical example: matmul, naive → tiled

**Naive** — every output element re-reads a full row and column from global memory (`O(N)` global reads
per output, terrible reuse):
```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col];   // repeated global loads
        C[row * N + col] = acc;
    }
}
```

**Tiled** — cooperatively load `TILE×TILE` blocks into **shared memory** once, reuse across the block
(cuts global traffic by ~`TILE×`):
```cuda
#define TILE 16
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < N / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();                              // all loads done
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                              // before overwriting tiles
    }
    C[row * N + col] = acc;
}
```
*The point you make in the room:* same FLOPs, far less HBM traffic → it's faster because matmul is
**memory-bound until you get reuse**. Production GEMMs (cuBLAS/CUTLASS) push this further with register
tiling, double-buffering, and tensor cores.

## Reduction (the other classic)

Summing an array tests your grasp of tree reduction + shared memory + avoiding divergence:
```cuda
__global__ void reduce_sum(const float* in, float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {  // tree, no warp divergence
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];             // block partial; reduce partials again
}
```
*Talking points:* sequential addressing (`tid < stride`) avoids bank conflicts/divergence; for the last
warp you can use **warp-shuffle** (`__shfl_down_sync`) to skip `__syncthreads()`.

---

## Why FlashAttention is fast (a favorite question)

Standard attention materializes the `T×T` scores matrix in HBM (read+write `O(T²)` bytes) — pure
bandwidth waste. **FlashAttention**:
1. **Tiles** Q, K, V into SRAM (shared memory) blocks.
2. Computes attention block-by-block with an **online softmax** (running max + running denominator) so
   it never stores the full `T×T` matrix.
3. **Fuses** softmax + matmuls into one kernel → one pass over HBM.

Result: **exact** attention, ~linear memory in `T`, big speedups — **without changing the FLOPs**. The
lesson interviewers want: *attention is memory-bound; kill the HBM round-trips.* (See [09-papers](09-papers.md).)

## Kernel fusion (the everyday win)

Each separate kernel reads inputs from and writes outputs to HBM. **Fusing** `x → LayerNorm → GELU`
into one kernel keeps intermediates in registers/shared memory → fewer HBM round-trips. This is exactly
what `torch.compile`/Triton/XLA do automatically.

## Triton (the modern, interview-friendly path)

Most teams write custom kernels in **Triton** (Python) now, not raw CUDA — you reason in **blocks/tiles**
and it handles coalescing/scheduling:
```python
import triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask)
                           + tl.load(y_ptr + offs, mask=mask), mask=mask)
```
*Say this:* "I'd prototype in Triton for a fused kernel, profile, and only drop to CUDA/CUTLASS if I
need the last 10–20% or tensor-core control." That's the pragmatic senior answer.

## Tensor cores & precision

- **Tensor cores** do small matrix-multiply-accumulate (e.g. 16×16) per instruction — the bulk of bf16/
  fp16/fp8 FLOPs. GEMMs must be shaped/aligned (multiples of 8/16) to use them.
- **Mixed precision:** compute in bf16/fp16, accumulate in fp32 for stability. fp8 (H100+) for the
  fastest training/inference, with care around scaling.

## Profiling & debugging (name the tools)

- **Nsight Systems** (timeline: are you kernel-bound, memory-bound, or stalling on the CPU/PCIe?) and
  **Nsight Compute** (per-kernel: occupancy, memory throughput, stalls).
- **`compute-sanitizer`** for race conditions / out-of-bounds.
- First questions on a slow kernel: *Is it memory- or compute-bound? Are accesses coalesced? What's the
  occupancy? Can I fuse to cut HBM traffic?*

---

## Likely tasks & questions

- "Write a CUDA/Triton kernel for **vector add / reduction / softmax**." → show coalescing, masking, shared memory.
- "**Optimize** this kernel." → identify memory- vs compute-bound, add tiling/fusion, fix coalescing/divergence.
- "Why is **FlashAttention** faster if FLOPs are unchanged?" → HBM traffic / online softmax / fusion.
- "Explain **warp divergence / bank conflicts / occupancy / coalescing**."
- "When **tensor parallel vs pipeline** (across GPUs)?" → cross-link to [02](02-ml-and-llm-depth.md)/[03](03-system-design.md).
- "How would you **profile** a slow training step?" → Nsight, check overlap of compute/comm/data-loading.

## Rapid-fire (CUDA)

1. **Q:** Warp size? **A:** 32 threads, SIMT lockstep.
2. **Q:** Shared memory scope? **A:** Per-block, programmer-managed, ~L1-speed scratchpad.
3. **Q:** Coalescing? **A:** Consecutive threads → consecutive addresses → merged wide transactions.
4. **Q:** Bank conflict fix? **A:** Pad shared arrays (e.g. `[32][33]`) so a warp hits distinct banks.
5. **Q:** Occupancy? **A:** Active warps/SM ÷ max; helps hide latency; capped by regs/shared mem.
6. **Q:** Most ML kernels are bound by? **A:** Memory bandwidth, not FLOPs.
7. **Q:** Biggest single optimization? **A:** Cut HBM traffic — tile into shared memory + **fuse** kernels.
8. **Q:** `__syncthreads()` does? **A:** Barrier for all threads in a block (e.g. after loading a tile).
9. **Q:** Tensor cores do? **A:** Small fixed-size MMA per instruction (bf16/fp16/fp8) → most FLOPs.
10. **Q:** Triton vs CUDA? **A:** Triton = Python, tile-level, auto coalescing/scheduling; CUDA = full control.
11. **Q:** Warp divergence cost? **A:** Both branch paths run serially under masking.
12. **Q:** Profilers? **A:** Nsight Systems (timeline) + Nsight Compute (per-kernel).

> Interview reflex: for **any** kernel question, start with *"Is this memory- or compute-bound?"* (the
> roofline), then talk **data reuse** (tiling/shared memory) and **fusion** (fewer HBM round-trips).
> That single framing answers 80% of GPU-perf questions.
