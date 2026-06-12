"""Build NB07 — Distributed training & inference."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 07 · Distributed Training & Inference

> Module: **05 · Distributed Systems & Inference**.

**Goal:** scale beyond one GPU. Understand **collective communication**, the parallelism
strategies (**DP, TP, PP, ZeRO/FSDP, EP**), the **training memory math**, and how distributed
**inference** differs.

### Learning objectives
1. Name the collectives (all-reduce, all-gather, reduce-scatter, all-to-all) and where each is used.
2. Compute the **memory breakdown** of a training run and why it dwarfs the weights.
3. Choose a parallelism strategy from cluster topology + model size.
4. Reason about distributed **serving** (TP across a node, replicas for QPS, P/D disaggregation).
"""),
    md(r"""
## 1. Collective communication

All parallelism is built on a handful of primitives (NCCL on NVIDIA):

| Collective | Does | Used by |
|------------|------|---------|
| **All-Reduce** | sum a tensor across GPUs, everyone gets the result | gradient sync in **data parallel** |
| **All-Gather** | each GPU collects all shards | gather params/activations (FSDP, TP) |
| **Reduce-Scatter** | sum then split shards across GPUs | FSDP gradient reduction |
| **All-to-All** | each GPU sends a piece to every other | **MoE** expert routing |
| **Broadcast** | one GPU sends to all | weight/init distribution |

**Topology matters:** intra-node **NVLink** (~hundreds of GB/s+) vs inter-node **InfiniBand** —
put communication-heavy parallelism (tensor) *inside* a node.
"""),
    code(r"""
import numpy as np

# Ring all-reduce: the bandwidth-optimal algorithm DDP uses. Each GPU sends ~2*(N-1)/N of its
# data total, independent of GPU count -> scales well. We simulate the *result* + the cost model.
def ring_allreduce(grads_per_gpu):
    n = len(grads_per_gpu)
    summed = np.sum(grads_per_gpu, axis=0)          # the value everyone ends up with
    size = grads_per_gpu[0].size
    bytes_moved_per_gpu = 2 * (n - 1) / n * size * 4 # fp32; ~2x param bytes regardless of n
    return summed, bytes_moved_per_gpu

grads = [np.ones(1_000_000) * (i + 1) for i in range(4)]   # 4 GPUs
summed, moved = ring_allreduce(grads)
print("all-reduced value (should be 1+2+3+4=10):", summed[0])
print(f"bytes moved per GPU: {moved/1e6:.1f} MB  (~2x param bytes, independent of #GPUs)")
"""),
    md(r"""
## 2. The training memory math (why you need many GPUs)

To train, each parameter needs more than its own bytes. With Adam in mixed precision, the
classic accounting per parameter is roughly:

- fp16 **weights**: 2 B   +  fp16 **gradients**: 2 B
- fp32 optimizer states (master weight + Adam m + v): 4 + 4 + 4 = 12 B
- → **~16 bytes/param** *before activations*.

Plus **activations**, which scale with batch × sequence × layers (often the largest term, and
the reason for **gradient checkpointing**).
"""),
    code(r"""
def training_memory_gb(N_params, batch, seq, d, L):
    model_states = 16 * N_params                       # weights+grads+Adam (mixed precision)
    # rough activation memory (bytes): ~ a few * batch*seq*d*L; constant absorbs attention/FFN buffers
    activations = 12 * batch * seq * d * L
    return model_states/1e9, activations/1e9

N = 7e9
ms, act = training_memory_gb(N, batch=4, seq=4096, d=4096, L=32)
print(f"7B model states (weights+grads+optimizer): {ms:.1f} GB")
print(f"activations (batch=4, 4k seq)            : {act:.1f} GB  <- gradient checkpointing cuts this")
print(f"total                                    : {ms+act:.1f} GB  (>> one 80GB GPU -> must shard)")
"""),
    md(r"""
## 3. The parallelism strategies

| Strategy | Splits | Communication | Use when |
|----------|--------|---------------|----------|
| **Data Parallel (DDP)** | the *batch* (full model replicated) | all-reduce grads | model fits on 1 GPU; scale throughput |
| **ZeRO / FSDP** | optimizer states → grads → params | reduce-scatter + all-gather | model too big for 1 GPU, keep DP simplicity |
| **Tensor Parallel (TP)** | individual matmuls | all-reduce per layer (heavy) | within a node (NVLink); shrink per-GPU weights |
| **Pipeline Parallel (PP)** | layers into stages | point-to-point activations | across nodes; micro-batch to fill the "bubble" |
| **Expert Parallel (EP)** | MoE experts | all-to-all routing | MoE models |

Frontier training composes them: **3D/nD parallelism** = data × tensor × pipeline (× expert).
"""),
    code(r"""
# ZeRO stages: how much does sharding model states across `n` GPUs save per GPU?
def zero_per_gpu_gb(N_params, n, stage):
    weights, grads, opt = 2*N_params, 2*N_params, 12*N_params
    if stage == 0:   per = weights + grads + opt                 # DDP: full replica
    elif stage == 1: per = weights + grads + opt/n               # shard optimizer
    elif stage == 2: per = weights + grads/n + opt/n             # + shard grads
    elif stage == 3: per = weights/n + grads/n + opt/n           # + shard params (full FSDP)
    return per/1e9

N, n = 7e9, 8
for s in range(4):
    print(f"ZeRO stage {s}: {zero_per_gpu_gb(N, n, s):6.1f} GB/GPU across {n} GPUs")
print("\n-> stage 3 (FSDP) makes per-GPU memory ~ 1/n of the model states.")
"""),
    md(r"""
## 4. Distributed inference

Serving differs from training (no optimizer states, latency-sensitive, stateful KV cache):

- **Tensor parallel** a model too big for one GPU **within a node** (NVLink); shard attention/FFN.
- **Pipeline parallel** across nodes; manage the decode-time bubble.
- **Replicas + router** for **QPS**: stateless requests load-balanced across many model copies.
- **Prefill/decode disaggregation** (DistServe, Mooncake): run compute-bound prefill and
  bandwidth-bound decode on separate, independently-scaled pools; transfer the KV cache between them.
- **KV-cache-aware routing** + global **prefix caching** for shared system prompts.
"""),
    code(r"""
# Tiny "choose a strategy" helper combining the rules above.
def recommend(params_b, gpu_mem_gb=80, gpus_per_node=8):
    weights_gb = params_b * 2
    fits_one = weights_gb < 0.6 * gpu_mem_gb      # leave room for KV cache
    if fits_one:
        return "Single GPU + replicate (DDP for training; N replicas for serving)."
    tp = int(np.ceil(weights_gb / (0.6 * gpu_mem_gb)))
    if tp <= gpus_per_node:
        return f"Tensor-parallel across {tp} GPUs in one node (NVLink); replicate nodes for QPS."
    return f"Need >1 node: tensor-parallel within node + pipeline-parallel across nodes (tp~{gpus_per_node})."

for p in [7, 13, 70, 405]:
    print(f"{p:3d}B params -> {recommend(p)}")
"""),
    md(r"""
## 5. Practical notes
- **Overlap** communication with computation (prefetch next layer's params while computing current).
- **Gradient accumulation** simulates a big batch without the memory.
- **Activation/gradient checkpointing** trades compute for memory (recompute in backward).
- **Stragglers & failures**: at 1000s of GPUs, something always fails — checkpoint often, design for restart.

## Exercises
1. Train with PyTorch **DDP** then **FSDP** on 2+ GPUs; plot scaling efficiency vs #GPUs.
2. Compute the memory breakdown for a 70B model and pick ZeRO stage + TP degree for 8×80GB.
3. Explain why TP wants NVLink but DP tolerates slower interconnect.
4. Serve a model with TP=2 in vLLM; add a 2nd replica behind a router; measure p99.

## Resources
- *Megatron-LM* (Shoeybi 2019); *ZeRO* (Rajbhandari 2020); PyTorch **FSDP** docs.
- *GPipe* (Huang 2019), 1F1B/*PipeDream*; *Ring Attention* (2023).
- *DistServe* / *Mooncake* (P/D disaggregation); HF *Ultra-Scale Playbook*; DeepMind *How to Scale Your Model*.
"""),
]

if __name__ == "__main__":
    write(cells, "07_distributed_training_and_inference.ipynb")
