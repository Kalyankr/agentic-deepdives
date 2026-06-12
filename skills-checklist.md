# Skills Checklist — Senior AI/ML Engineer

A self-assessment. You should be able to honestly check most of these before interviewing at a frontier lab. Use it to **test out** of modules and to find gaps. Be ruthless: "I could do it with notes" ≠ checked.

> Scoring: aim for ≥90% in your target area(s) and ≥70% across the board. The from-scratch builds and the whiteboard derivations are the highest-signal items.

---

## Foundations [Modules 00–01]
- [ ] Derive the gradient of softmax + cross-entropy by hand.
- [ ] Implement reverse-mode autograd from scratch and train an MLP with it.
- [ ] Explain AdamW, warmup + cosine schedule, weight decay, gradient clipping.
- [ ] Explain compute-bound vs. memory-bound via the roofline model.
- [ ] Write async Python; explain the GIL, threads vs. processes vs. async.
- [ ] Recall latency numbers (cache/DRAM/SSD/network) to an order of magnitude.

## Transformers [Module 02]
- [ ] Implement multi-head causal self-attention from memory.
- [ ] Implement a KV cache and explain the prefill/decode split.
- [ ] Explain RoPE, GQA/MQA, RMSNorm, SwiGLU, FlashAttention — and *why* each exists.
- [ ] Derive parameter count, FLOPs/token, and KV-cache size for a given config.
- [ ] Explain Chinchilla-optimal scaling and its implications.
- [ ] Explain BPE tokenization and the failure modes it causes.

## Training, RLHF & DPO [Module 03]
- [ ] Run SFT with correct chat templating and loss masking.
- [ ] Explain LoRA/QLoRA math and trade-offs; fine-tune with both.
- [ ] Explain the 3-stage RLHF pipeline and the role of the KL penalty.
- [ ] Derive/explain the DPO loss; implement DPO from scratch.
- [ ] Implement a basic reward model + PPO loop; explain reward hacking.
- [ ] Explain Constitutional AI / RLAIF.

## GPU & inference [Module 04]
- [ ] Explain the GPU memory hierarchy, tensor cores, and SIMT execution.
- [ ] Write a tiled matmul and a fused softmax in CUDA/Triton; place them on a roofline.
- [ ] Explain PagedAttention and continuous batching.
- [ ] Quantize a model (INT8/INT4) and reason about quality/latency/memory.
- [ ] Explain and apply speculative decoding.
- [ ] Stand up vLLM and produce/interpret a latency–throughput curve.

## Distributed systems & inference [Module 05]
- [ ] Explain DP vs. TP vs. PP vs. ZeRO/FSDP and when to use each.
- [ ] Do the training memory math (params + grads + optimizer + activations).
- [ ] Explain NCCL collectives and why interconnect topology matters.
- [ ] Train with DDP and FSDP; report scaling efficiency.
- [ ] Serve a model with tensor parallelism + multiple replicas; report tail latency.
- [ ] Explain prefill/decode disaggregation.

## RAG & vector DBs [Module 06]
- [ ] Explain HNSW and IVF-PQ; tune recall vs. latency vs. memory in FAISS.
- [ ] Build a RAG pipeline with hybrid retrieval + reranking + citations.
- [ ] Name the highest-leverage RAG quality levers and justify them.
- [ ] Build a RAG eval harness (retrieval + generation metrics).
- [ ] Plan the memory/sharding footprint of a billion-scale index.

## Agents [Module 07]
- [ ] Articulate agents vs. workflows; choose the simplest design that works.
- [ ] Implement ReAct + tool calling + memory from scratch.
- [ ] Build/expose an MCP tool.
- [ ] Instrument trajectory tracing (steps/tokens/latency/cost) and optimize a task.
- [ ] Evaluate agent success rate and defend against prompt injection.

## Prompt orchestration [Module 08]
- [ ] Design chaining/routing/parallel orchestration with validation + fallbacks.
- [ ] Enforce structured outputs via constrained decoding.
- [ ] Optimize prompts programmatically (DSPy/APE) against an eval set.
- [ ] Cut cost with prefix/response caching and model routing.

## Evaluations [Module 09]
- [ ] Design a valid eval for a fuzzy capability.
- [ ] Build and *validate* an LLM-as-judge; account for its biases.
- [ ] Wire evals into CI to block regressions.
- [ ] Design safety/red-team and prompt-injection evals.
- [ ] Reason about statistical significance and contamination.

## Infra, monitoring & safety [Module 10]
- [ ] Instrument LLM observability (TTFT/TPOT, cost, quality, drift) with alerts.
- [ ] Define SLOs and build graceful degradation/reliability mechanisms.
- [ ] Threat-model an agent; implement layered safety + sandboxing.
- [ ] Deploy on Kubernetes with autoscaling.
- [ ] Describe RSP/Preparedness-style capability gating.

## System design & cost [Module 11]
- [ ] Estimate QPS, GPU count, storage, and bandwidth for a product, out loud, with assumptions.
- [ ] Derive serving GPU count from tokens/s and HBM bandwidth.
- [ ] Run a full system-design interview with trade-offs and failure analysis.
- [ ] Drive down cost/token and prove quality held via evals.

## Interview readiness [Module 12]
- [ ] Flagship end-to-end project shipped publicly with docs + evals + design doc.
- [ ] 2–3 deep portfolio projects (ideally one paper reproduction).
- [ ] Whiteboard attention in <15 min; capacity-estimate in <10 min.
- [ ] Hold a substantive safety/alignment conversation.
- [ ] Explain every trade-off in your projects with numbers.
