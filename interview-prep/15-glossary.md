# 15 · Glossary & Quick-Reference (A–Z)

> A fast lookup for the vocabulary you'll hear and use across the loop. One-line definitions — when you
> need depth, follow the link to the file that covers it. Skim this the morning of an interview so the
> terms are *active*, not just recognizable.

> Reading guide: **bold term** — definition *(→ deeper file)*. Formulas are in
> [10-numbers-and-hardware](10-numbers-and-hardware.md); the cards are in [07-rapid-fire](07-rapid-fire.md).

---

## A

- **Activation checkpointing** — recompute activations in the backward pass to save memory *(→ [02](02-ml-and-llm-depth.md))*.
- **AdamW** — Adam optimizer with **decoupled** weight decay; the LLM default. State = 8 B/param.
- **ALiBi** — positional method adding a linear distance penalty to attention scores; good length extrapolation.
- **ANN** — Approximate Nearest Neighbor search (HNSW, IVF-PQ) for vectors *(→ [04](04-applied-llm.md), [lab06](../labs/lab06_rag/))*.
- **Arithmetic intensity** — FLOPs ÷ bytes moved; sets memory- vs compute-bound *(→ [14](14-cuda-and-kernels.md))*.
- **ASL** — AI Safety Levels; Anthropic's capability tiers in the RSP *(→ [05](05-safety-alignment.md))*.
- **Attention** — weighted mixing of values by query–key similarity; the Transformer's core *(→ [01](01-coding.md), [02](02-ml-and-llm-depth.md))*.
- **AUC (ROC/PR)** — threshold-free ranking quality; PR-AUC for rare positives *(→ [12](12-math-stats.md))*.

## B

- **Batch (continuous batching)** — swap finished/new sequences each decode step to keep GPUs full *(→ [03](03-system-design.md))*.
- **bf16** — brain float16: fp32 range, fewer mantissa bits; 2 B/param; usually no loss scaling.
- **BM25** — classic lexical retrieval ranker; fused with dense via RRF for hybrid search.
- **BPE** — Byte-Pair Encoding tokenizer; merges frequent pairs *(→ [01](01-coding.md))*.
- **Bradley–Terry** — pairwise preference model behind reward models / Elo *(→ [02](02-ml-and-llm-depth.md))*.

## C

- **Calibration (ECE)** — does predicted confidence match accuracy *(→ [12](12-math-stats.md))*.
- **Chinchilla** — compute-optimal scaling: ~20 tokens/param *(→ [09](09-papers.md), [10](10-numbers-and-hardware.md))*.
- **CoT (Chain-of-Thought)** — eliciting step-by-step reasoning to improve multi-step tasks.
- **Coalescing** — consecutive threads → consecutive addresses → merged memory transactions *(→ [14](14-cuda-and-kernels.md))*.
- **Constitutional AI** — RLAIF guided by a written constitution; Anthropic's harmlessness method *(→ [05](05-safety-alignment.md), [09](09-papers.md))*.
- **Context window** — max tokens a model attends over; KV cache grows with it.
- **Cross-encoder** — joint query–doc scorer; accurate reranker over top-k *(→ [04](04-applied-llm.md))*.
- **Cross-entropy** — NLL loss for classification/LMs; `perplexity = exp(CE)` *(→ [12](12-math-stats.md))*.

## D

- **DDP / Data Parallel** — replicate model, all-reduce gradients *(→ [02](02-ml-and-llm-depth.md))*.
- **Decode** — autoregressive token-by-token phase; **memory-bandwidth bound**; sets TPOT.
- **Distillation** — train a small "student" to match a large "teacher".
- **DPO** — Direct Preference Optimization; alignment from pairs, no reward model/RL *(→ [02](02-ml-and-llm-depth.md), [09](09-papers.md))*.
- **Dropout** — randomly zero activations during training to regularize.

## E

- **Elo** — rating from pairwise wins (Chatbot Arena) *(→ [04](04-applied-llm.md))*.
- **Embedding** — dense vector representation of text/tokens for similarity/retrieval.
- **Emergent abilities** — capabilities that appear at scale; often a metric artifact *(→ [09](09-papers.md))*.
- **Eval flywheel** — prod logs → error analysis → new eval cases → fix → re-eval *(→ [04](04-applied-llm.md))*.

## F

- **FSDP** — Fully Sharded Data Parallel ≈ ZeRO-3; shards params/grads/optimizer states.
- **Few-shot / in-context learning** — task adaptation from prompt examples, no weight update *(→ [09](09-papers.md))*.
- **FlashAttention** — IO-aware exact attention; tiling + online softmax, no `T×T` in HBM *(→ [09](09-papers.md), [14](14-cuda-and-kernels.md))*.
- **FLOPs** — train ≈ `6·N·D`; inference ≈ `2·N`/token *(→ [10](10-numbers-and-hardware.md))*.
- **fp8 / fp16 / fp32** — float precisions: 1 / 2 / 4 bytes per value.

## G

- **GEMM** — General Matrix Multiply; the dominant compute in transformers *(→ [14](14-cuda-and-kernels.md))*.
- **GELU / SwiGLU** — activation functions; SwiGLU (gated) is the modern FFN choice.
- **GQA** — Grouped-Query Attention; fewer KV heads → smaller KV cache *(→ [02](02-ml-and-llm-depth.md))*.
- **Gradient checkpointing** — see *activation checkpointing*.
- **Guardrails** — input/output filters, sandboxing, approvals around a model/agent *(→ [05](05-safety-alignment.md))*.

## H

- **HBM** — High-Bandwidth Memory (GPU VRAM); large but slow vs SRAM; the bottleneck *(→ [10](10-numbers-and-hardware.md))*.
- **HNSW** — graph-based ANN index; high recall, more memory *(→ [04](04-applied-llm.md))*.
- **Hybrid retrieval** — combine dense + lexical (BM25), fuse with RRF *(→ [lab06](../labs/lab06_rag/))*.

## I

- **In-context learning** — see *few-shot*.
- **Induction heads** — circuit that copies/continues patterns; mechanism behind ICL *(→ [09](09-papers.md))*.
- **Instruction tuning** — SFT on (instruction, response) pairs to follow tasks.
- **IVF / IVF-PQ** — inverted-file ANN index; `nprobe` trades recall vs speed *(→ [lab06](../labs/lab06_rag/))*.

## J–K

- **Jailbreak** — prompt that bypasses safety training *(→ [05](05-safety-alignment.md))*.
- **KL divergence** — asymmetric distance between distributions; PPO penalty, distillation *(→ [12](12-math-stats.md))*.
- **KTO** — alignment from unpaired good/bad labels (no pairs needed).
- **KV cache** — stored keys/values so decode skips recompute; bytes = `2·L·n_kv·d_head·B·T` *(→ [10](10-numbers-and-hardware.md))*.

## L

- **LayerNorm / RMSNorm** — normalization; RMSNorm drops mean/bias, cheaper *(→ [01](01-coding.md))*.
- **LoRA / QLoRA** — low-rank adapters; QLoRA adds a 4-bit frozen base *(→ [02](02-ml-and-llm-depth.md), [09](09-papers.md))*.
- **Lost-in-the-middle** — long-context models under-use middle content *(→ [04](04-applied-llm.md))*.

## M

- **MFU** — Model FLOPs Utilization; ~40–55% is good *(→ [10](10-numbers-and-hardware.md))*.
- **MCP** — Model Context Protocol; open standard to expose tools/data to models *(→ [04](04-applied-llm.md))*.
- **MoE** — Mixture of Experts; sparse routing grows params at ~constant FLOPs/token.
- **MQA / MHA** — Multi-Query / Multi-Head Attention; MQA = 1 KV head (smallest cache).
- **MLE / MAP** — maximum likelihood / with a prior (= regularization) *(→ [12](12-math-stats.md))*.

## N

- **NCCL** — NVIDIA collective comms library (all-reduce, all-gather) for multi-GPU.
- **nDCG / MRR / recall@k** — ranking/retrieval metrics *(→ [lab06](../labs/lab06_rag/), [12](12-math-stats.md))*.
- **NF4** — 4-bit NormalFloat quantization used by QLoRA.
- **NVLink** — high-bandwidth GPU-to-GPU interconnect (keep tensor-parallel within it).

## O

- **Occupancy** — active warps/SM ÷ max; helps hide latency *(→ [14](14-cuda-and-kernels.md))*.
- **Online softmax** — running max/denominator softmax; enables FlashAttention.
- **ORPO** — preference + SFT in one loss, reference-model-free.
- **Outer/inner alignment** — right objective vs the model internalizing it *(→ [05](05-safety-alignment.md))*.
- **Over-refusal** — rejecting benign requests; a measurable failure *(→ [05](05-safety-alignment.md))*.

## P

- **PagedAttention** — KV cache in fixed pages (like virtual memory); the vLLM idea *(→ [09](09-papers.md))*.
- **PEFT** — Parameter-Efficient Fine-Tuning (LoRA, adapters, prefix tuning).
- **Perplexity** — `exp(cross_entropy)`; effective branching factor.
- **PPO** — RL algorithm used in RLHF, with a KL penalty *(→ [02](02-ml-and-llm-depth.md))*.
- **Prefill** — prompt-processing phase; **compute-bound**; sets TTFT.
- **Prefix caching** — reuse KV of a shared prompt prefix → lower TTFT/cost.
- **Prompt injection** — malicious instructions via tool/retrieved content; #1 agent threat *(→ [05](05-safety-alignment.md))*.
- **PQ** — Product Quantization; compresses vectors ~4–32× for ANN.

## Q

- **Quantization** — lower-precision weights/activations (int8/int4); ↓ memory, eval quality after.
- **QPS** — queries/sec; `avg = daily/86,400`, `peak ≈ 2–5× avg` *(→ [10](10-numbers-and-hardware.md))*.

## R

- **RAG** — Retrieval-Augmented Generation; ground answers in retrieved docs *(→ [04](04-applied-llm.md), [lab06](../labs/lab06_rag/))*.
- **ReAct** — Reason+Act agent loop: Thought → Action → Observation *(→ [lab07](../labs/lab07_agent/), [09](09-papers.md))*.
- **Reward hacking** — optimizing the proxy reward, not the true goal *(→ [05](05-safety-alignment.md))*.
- **RLHF / RLAIF** — RL from human / AI feedback *(→ [02](02-ml-and-llm-depth.md))*.
- **RoPE** — Rotary Position Embedding; encodes relative position by rotating Q/K.
- **Roofline** — perf model: memory-bound vs compute-bound by arithmetic intensity *(→ [14](14-cuda-and-kernels.md))*.
- **RRF** — Reciprocal Rank Fusion; merges rankings by rank, not score *(→ [lab06](../labs/lab06_rag/))*.
- **RSP** — Responsible Scaling Policy; ties safeguards to capability (ASL) *(→ [05](05-safety-alignment.md))*.

## S

- **SAE** — Sparse Autoencoder; extracts monosemantic features (interpretability) *(→ [05](05-safety-alignment.md), [09](09-papers.md))*.
- **Scalable oversight** — supervising tasks humans can't directly evaluate *(→ [05](05-safety-alignment.md))*.
- **Scaling laws** — loss falls as a power law in N, D, C *(→ [09](09-papers.md))*.
- **SFT** — Supervised Fine-Tuning; mask the prompt, train on responses *(→ [02](02-ml-and-llm-depth.md))*.
- **Shared memory** — per-block on-chip scratchpad; key to tiling *(→ [14](14-cuda-and-kernels.md))*.
- **SM** — Streaming Multiprocessor; runs thread blocks.
- **Speculative decoding** — small draft proposes, big model verifies; lossless ~2–3× *(→ [02](02-ml-and-llm-depth.md))*.
- **Superposition** — packing more features than neurons via overlap *(→ [09](09-papers.md))*.
- **Sycophancy** — telling users what they want to hear *(→ [05](05-safety-alignment.md))*.

## T

- **Temperature** — scales **logits** before softmax to tune diversity *(→ [01](01-coding.md), [11](11-debugging.md))*.
- **Tensor cores** — units doing small MMA per instruction (bf16/fp16/fp8) *(→ [14](14-cuda-and-kernels.md))*.
- **Tensor / Pipeline parallel** — split a layer's matmul / split layers across GPUs *(→ [02](02-ml-and-llm-depth.md))*.
- **Top-k / Top-p** — sampling: keep k tokens / smallest nucleus summing to p *(→ [01](01-coding.md))*.
- **TTFT / TPOT** — Time-To-First-Token (prefill) / Time-Per-Output-Token (decode).
- **Triton** — Python DSL for GPU kernels; tile-level, auto-coalescing *(→ [14](14-cuda-and-kernels.md))*.

## U–Z

- **vLLM** — high-throughput serving engine using PagedAttention + continuous batching *(→ [03](03-system-design.md))*.
- **Warp** — 32 threads executing in lockstep (SIMT) *(→ [14](14-cuda-and-kernels.md))*.
- **Weak-to-strong generalization** — can a weak supervisor elicit a stronger model's ability *(→ [05](05-safety-alignment.md))*.
- **Weight tying** — share input/output embedding matrices to save params.
- **ZeRO (1/2/3)** — shard optimizer states / +grads / +params across GPUs *(→ [02](02-ml-and-llm-depth.md))*.

---

## Symbols & formulas (most-used)

| Symbol | Meaning |
|--------|---------|
| `N` | model parameters |
| `D` | training tokens |
| `C` | compute (FLOPs); `C ≈ 6·N·D` train, `≈ 2·N`/token infer |
| `L`, `d`, `n_kv`, `d_head` | layers, model dim, KV heads, head dim |
| `T`, `B` | sequence length, batch size |
| `SE`, `CI` | standard error `√(p(1−p)/n)`, confidence interval ≈ est ± 1.96·SE |
| `ppl` | perplexity = `exp(CE)` |

> Use this as a **warm-up and a safety net**, not a study source — if a term here is fuzzy, that's your
> signal to open the linked file. Recognition isn't mastery; aim to *define each out loud* in a sentence.
