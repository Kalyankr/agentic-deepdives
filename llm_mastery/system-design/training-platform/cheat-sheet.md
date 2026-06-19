# 🃏 LLM Training & Fine-Tuning Platform — One-Page Cheat-Sheet

> Last-minute recall card for the [full HLD](README.md). Drill the bold bits.

## The one idea
**Turn a declarative job spec into a trained, evaluated, registered model** across thousands of GPUs. Two hard parts: **fitting a model that doesn't fit one GPU** (parallelism + sharding) and **keeping a multi-week job alive on flaky hardware** (checkpoint + auto-resume + elasticity). Train = **throughput/durability**; serving = latency (mirror image).

## The two numbers (lead with these)
- **Compute:** $C \approx 6ND$. 70B × 15T tok = $6.3\times10^{24}$ FLOPs → ~**170K GPU-days** → ~21 days on 8,192 H100. GPUs buy wall-clock only if **MFU holds**.
- **Memory:** AdamW mixed precision = **~16 bytes/param** (fp16 w 2 + grad 2 + fp32 master 4 + Adam m 4 + v 4). 70B → **1.12 TB states** vs 80 GB/GPU → **must shard**.
- Inference contrast: ~$2N$/token, no backward, no optimizer states.

## Two planes (separate them)
- **Control plane** (cheap): scheduler/quota → orchestrator → experiment tracker → model registry.
- **Execution plane** (expensive): GPU mesh (DP×TP×PP) + data pipeline + checkpoint store + eval workers.
- **Health monitor** drives **auto-resume** from last checkpoint on failure.

## Parallelism = a toolkit (match comms to the network)
| Split | Splits | Comms | Place |
|---|---|---|---|
| **DP** | the batch (full replica) | all-reduce grads/step | everywhere |
| **TP** | matmuls in a layer | all-reduce **in** every layer (chatty) | **intra-node** NVLink |
| **PP** | layers into stages | activations at boundaries | across nodes (watch **bubbles**) |
| **EP** | MoE experts | all-to-all routing | MoE only |

Example 2048 GPUs: **TP=8 × PP=4 × DP=64**. **Overlap comms with compute.** Pick the **cheapest split that fits:** DP → ZeRO/FSDP → activation checkpointing → TP → PP → offload.

## Memory levers
- **ZeRO/FSDP** shards the 16 B/param across DP ranks → ~**16/P B/param** (stage 1 optim → 2 +grads → 3/FSDP +params, gather just-in-time).
- **Activation checkpointing:** store few, **recompute** rest. $O(L)\to O(\sqrt L)$ memory for **~33% extra compute**.
- **Offload** (CPU/NVMe) = last resort; PCIe-bound, MFU drops.

## Data pipeline (garbage in → garbage model)
`raw → clean → quality-filter → **dedup** (MinHash) → **decontaminate** → tokenize → **pack** → sharded shuffled store → streaming loader (mix + curriculum)`
- **Dedup + decontamination** = highest-leverage quality steps.
- **Pack** to seq_len (no padding waste); **mix weights** are a hyperparameter (e.g. 60/30/10).
- **Seeded resumable sampler** → deterministic order (reproducibility + clean resume).

## Fault tolerance (the hard part at scale)
- **Checkpoint** = weights + optimizer states + **data-loader position** + RNG + step.
- **Sharded** (each rank writes its slice) + **async** (snapshot→CPU, flush in background) → resume in seconds.
- **Auto-resume:** health monitor detects dead/slow rank → swap hot spare → relaunch from checkpoint.
- **Elastic:** resize world (2048→1920) and continue. **Stragglers:** sync moves at slowest rank → detect step-time outliers, fence + blocklist.
- **Preemption is safe because checkpoints** → high utilization.

## Scheduling (cluster economics)
**Gang** (all-or-nothing, avoid deadlock) · **topology-aware** (one InfiniBand island — fragmenting tanks MFU) · **quotas + fair-share** · **priority + preemption** · **backfill** small jobs into gaps · spot for restartable jobs.

## Fine-tuning ladder (most usage)
| Path | Cost | Note |
|---|---|---|
| **SFT** | full-model | instruction tuning |
| **LoRA/QLoRA** | **tiny** (adapters; QLoRA = 4-bit base) | multi-tenant, shared base |
| **RLHF/PPO** | **heaviest** — **4 models** (actor/critic/reward/ref) + **rollouts** | powerful, unstable |
| **DPO** | ~2 models, no reward model/rollouts | pragmatic default |

## MFU = the money metric
Healthy **40–55%**; **<30% = something's wrong** (exposed comms, loader stall, pipeline bubbles, fragmented placement). Levers: **FlashAttention** + fused kernels · **bf16/fp8** · **overlap comms** · **sequence packing** · big batch · shrink bubbles (1F1B) · right-size parallelism.

## Reproducibility
Pin code SHA / image / lib+CUDA versions · dataset **version + mix** · seeds (per rank) · **deterministic resumable order** · same **topology** (different GPU count changes reduction order). Track full lineage base→data→job→artifact→evals.

## Eval & registry
**In-loop** held-out loss/ppl (convergence) · **periodic** MMLU/HumanEval/domain on milestone ckpt (off critical path) · **gate**: beat incumbent, no regressions → promote artifact + eval card. Decontamination makes numbers real.

## Cost order
**Maximize MFU** → right-size compute (**Chinchilla ~20 tok/param**; over-train smaller if serving a lot) → **PEFT first** → **spot** for restartable jobs → preemption + backfill + topology packing → early-stop diverged/plateaued runs → bf16/fp8.

## Top tradeoffs / failure modes
loss spike/divergence (clip + bf16 + **rollback**) · **NaN** (loss scaling, halt+rollback, inspect batch) · low MFU (overlap/loader/placement) · stragglers (fence+blocklist) · hardware failure (ckpt+auto-resume+spares) · **checkpoint stall** (async+sharded) · data-loader bottleneck · OOM at long seq (activation ckpt / seq-parallel) · comm bottleneck at scale · non-reproducibility · pipeline bubbles · scheduler fragmentation.

---
[← HLD](README.md) · [Q&A](questions.md) · [Answers](answers.md) · [Index](../../README.md)
