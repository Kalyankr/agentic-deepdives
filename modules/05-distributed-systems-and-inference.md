# Module 05 · Distributed Systems & Distributed Inference

> **Goal:** Scale beyond one GPU. Master the parallelism strategies for training and serving large models across many GPUs and nodes, the collective communication that underpins them, and the distributed-systems fundamentals that production AI infrastructure relies on.

**Duration:** ~6 weeks. **Prereqs:** [Module 04](04-gpu-architecture-and-inference.md), DDIA Ch. 1–4 from [Module 00](00-prerequisites.md).

---

## 5.1 Distributed systems fundamentals

These are classic systems concepts that frontier-lab interviews assume.

- Why distribute: capacity, throughput, fault tolerance
- **CAP theorem**, consistency models (strong, eventual, causal), linearizability
- Replication, partitioning/sharding, consensus (Raft/Paxos — conceptual)
- Failure modes: partial failures, network partitions, stragglers, retries, idempotency
- Load balancing, backpressure, queueing theory (Little's Law — you'll use it for capacity planning)
- Caching layers, CDNs, message queues (Kafka), service discovery
- Observability primitives: metrics, logs, traces (deep dive in [Module 10](10-ai-infrastructure-and-production.md))

> **Read:** *Designing Data-Intensive Applications* — the rest of it. This is the canonical text.

## 5.2 Collective communication

The primitives all parallelism is built on:
- **All-Reduce** (sum gradients), All-Gather, Reduce-Scatter, Broadcast, All-to-All (used by MoE)
- Ring vs. tree algorithms; bandwidth- vs. latency-optimal
- **NCCL** (NVIDIA Collective Communications Library)
- Interconnects: **NVLink/NVSwitch** (intra-node), **InfiniBand/RoCE** (inter-node); why topology dictates which parallelism you use where

## 5.3 Parallelism strategies for training

- **Data Parallelism (DDP)** — replicate model, shard the batch, all-reduce gradients
- **ZeRO / FSDP** — shard optimizer states, gradients, and parameters across GPUs to fit huge models (ZeRO stages 1/2/3). Understand the memory math.
- **Tensor (model) Parallelism** — split individual matmuls across GPUs (Megatron-LM); heavy communication → keep within a node over NVLink
- **Pipeline Parallelism** — split layers across GPUs; micro-batching to fill the "bubble" (GPipe, 1F1B)
- **Sequence / context parallelism** — for very long sequences (Ring Attention)
- **Expert parallelism** — distribute MoE experts; All-to-All routing
- **3D / nD parallelism** — compose data × tensor × pipeline (× expert) for frontier-scale training
- Memory accounting: params + gradients + optimizer states (Adam = 2× params in FP32) + activations
- Overlapping communication with computation; gradient accumulation; checkpointing at scale

> **Build:** Train a model with **DDP** across ≥2 GPUs (or simulated), then with **FSDP**. Measure scaling efficiency (speedup vs. #GPUs) and explain where it falls short of linear (communication, stragglers).

## 5.4 Distributed inference

Serving models too big for one GPU, or serving at high QPS.

- **Tensor parallelism for inference** — shard attention/FFN across GPUs in a node
- **Pipeline parallelism for inference** — across nodes; managing the bubble at decode
- **Expert parallelism** for MoE serving
- **Prefill/decode disaggregation** — separate clusters optimized for compute-bound prefill vs. bandwidth-bound decode (e.g., DistServe, Mooncake)
- **Multi-node serving** with vLLM / TensorRT-LLM; KV-cache-aware routing
- Replication & autoscaling for QPS; routing, load balancing, request scheduling across replicas
- **KV-cache offload** (to CPU/NVMe), distributed/global prefix caching
- Cold starts, model loading time, weight streaming
- Reliability: health checks, draining, rolling upgrades, canaries

> **Build:** Serve a model that requires **tensor parallelism across 2+ GPUs** with vLLM. Then put ≥2 replicas behind a load balancer/router and load-test horizontal scaling. Report scaling efficiency and tail latency (p50/p95/p99).

## 5.5 Orchestration & infrastructure (bridge to Module 10)

- Kubernetes basics for GPU workloads, the NVIDIA device plugin, GPU scheduling/MIG
- Ray / Ray Serve for distributed Python and model serving
- Job schedulers (Slurm) for training clusters
- Storage for checkpoints/datasets (object storage, parallel FS)

---

## Module 05 capstone — **Scale it out**

1. **Training:** the same model trained with DDP and FSDP across multiple GPUs, with a scaling-efficiency plot and a memory-breakdown analysis.
2. **Inference:** a model served with tensor parallelism, then horizontally scaled across replicas behind a router, with a tail-latency + scaling report.
3. **Design write-up:** choose a parallelism strategy for a hypothetical 70B and a 400B-MoE model on a given cluster topology, and justify it with the memory/comm math.

## Exit criteria
- [ ] You can explain DP vs. TP vs. PP vs. ZeRO/FSDP and when to use each.
- [ ] You can do the memory math for training a model of size X (params+grads+optimizer+activations).
- [ ] You understand NCCL collectives and why interconnect topology matters.
- [ ] You can design a multi-GPU/multi-node serving setup and reason about tail latency.

## Core papers / sources
- *Megatron-LM* — Shoeybi et al., 2019; *Efficient Large-Scale Training* — Narayanan et al., 2021
- *ZeRO* — Rajbhandari et al., 2020; PyTorch **FSDP** paper/docs
- *GPipe* — Huang et al., 2019; *PipeDream* (1F1B)
- *Ring Attention* — Liu et al., 2023
- *DistServe* / *Mooncake* (prefill–decode disaggregation)
- *Designing Data-Intensive Applications* — Kleppmann
- "How to Scale Your Model" (DeepMind) and the *Ultra-Scale Playbook* (Hugging Face)
