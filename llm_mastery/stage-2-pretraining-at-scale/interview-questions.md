# Stage 2 — Interview Questions (full-fledged, all levels)

> **Scope:** screening through **senior / staff / principal**. Angles: conceptual, math, coding, system design, debugging. `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math · 💻 Coding · 🏗️ Design · 🐞 Debug
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What objective is a base LLM trained on?
2. What is perplexity, intuitively?
3. Name the three levers that determine pretraining outcome.
4. What is gradient accumulation and why use it?
5. Why warm up the learning rate?
6. What is gradient clipping protecting against?
7. What is mixed-precision training?
8. What does deduplication of training data buy you?
9. What is data parallelism in one sentence?
10. Why is data quality often more important than model architecture?

## 🟡 Core (L4–L5)
11. State the Chinchilla result. What was wrong with GPT-3's compute allocation?
12. Use `C ≈ 6ND` to explain the compute–params–tokens relationship.
13. Why is bf16 preferred over fp16 for training stability?
14. Break down GPU memory during training into its four components.
15. Contrast data, tensor, and pipeline parallelism by communication pattern.
16. What does gradient checkpointing trade, and when is it worth it?
17. Why is AdamW's optimizer state so memory-hungry?
18. What is a data mixture and why is its ratio a tuned hyperparameter?
19. What is benchmark contamination and why does it matter at pretraining time?
20. What's the difference between Kaplan and Chinchilla scaling laws?

## 🔴 Senior / Staff deep dives (with follow-ups)
21. You have a fixed compute budget C. Walk me through choosing N (params) and D (tokens).
    → *covers:* `C ≈ 6ND`, Chinchilla ~20 tokens/param optimum, then the **inference-cost caveat** (over-train a smaller model to cut serving cost).
22. Compute the VRAM to full-fine-tune a 7B model in bf16 + AdamW. Show every term.
    → *covers:* params 2N (14GB) + grads 2N (14GB) + optimizer 12N (fp32 master 4N + m 4N + v 4N = 84GB) ≈ **112GB before activations** → motivates ZeRO/FSDP/PEFT.
23. Explain ZeRO stages 1/2/3 and what each sustains vs shards.
    → *covers:* stage1 shards optimizer state, stage2 +gradients, stage3 +parameters; gather-on-demand; comms/memory tradeoff.
24. Design the parallelism strategy to train a 70B model on N×8×A100 nodes.
    → *covers:* TP within node (NVLink), PP across nodes, DP/ZeRO across replicas; the pipeline bubble; activation memory.
25. Your loss spikes intermittently during a long run. How do you diagnose and recover?
    → *covers:* grad clipping, LR/warmup, bad data shards, skip-and-rollback from checkpoint, bf16 vs fp16, optimizer state corruption.
26. Why might you deliberately train *past* the Chinchilla-optimal point?
    → *inference economics:* a smaller, over-trained model is cheaper to serve at scale (LLaMA philosophy).
27. How would you build a pretraining data pipeline from raw web crawl? Name the stages.
    → *covers:* extraction, language ID, quality filtering/classifiers, dedup (MinHash), decontamination, PII handling, mixture, tokenization, sharding.
28. How do you detect and prevent training-data contamination of your eval benchmarks?

## 🧮 Math & derivations
29. Derive the four memory terms for AdamW mixed-precision training in bytes/param.
30. Show why `C ≈ 6ND` (factor ~6 = 2 fwd + 4 bwd FLOPs per param per token).
31. Given 256 A100-80GB and a 1-week budget, estimate the largest Chinchilla-optimal model you can train (order-of-magnitude).
32. Relate perplexity to cross-entropy loss; what does PPL=20 mean operationally?

## 💻 Coding / implementation
33. Write a VRAM estimator: inputs (N, dtype, optimizer, batch, seq, layers) → memory breakdown.
34. Implement gradient accumulation correctly (loss scaling by accumulation steps).
35. Add bf16 autocast + grad clipping + cosine schedule with warmup to a training loop.
36. Implement a streaming, sharded data loader that never loads the whole corpus into RAM.
37. Wrap a model in FSDP / ZeRO and explain each config flag you set.

## 🏗️ System design / applied
38. Design the full training infrastructure for a 13B model: data, checkpointing, fault tolerance, monitoring, cost.
39. A 3-week run will be interrupted by spot-instance preemptions. How do you make it resilient?
40. You must cut training cost 40% with minimal quality loss. What levers do you pull and in what order?

## 🐞 Debugging / scenarios
41. Throughput (tokens/sec) is far below the GPU's roofline. Where do you look?
    → *covers:* data loader stalls, small batch, no fused kernels, comms-bound parallelism, activation recompute overhead, low MFU.
42. Training loss is good but the model memorizes/regurgitates training data. Cause and fix?
    → *insufficient dedup; reduce epochs; data filtering; DP if sensitive.*
43. Multi-node run is 3× slower than single-node-scaled expectation. Diagnose.
    → *interconnect bound, wrong parallelism placement, TP across slow links.*
44. Loss decreases then plateaus far above expected. Suspects?

## ✅ What strong candidates demonstrate
- Fluent **memory & FLOPs arithmetic** done live, with correct units.
- Treat **data work as the real lever**, not just architecture.
- Know **when scaling laws apply and when to break them** (inference cost).
- Can map a model size to a concrete **parallelism + infra** plan.

---
Related: the **🔥 Mastery checks** in [README.md](README.md) are the minimum bar.
