# Stage 5 — Interview Questions (full-fledged, all levels)

> **Scope:** screening through **senior / staff / principal** (incl. ML-systems/perf roles). Angles: conceptual, math, coding, system design, debugging. `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math · 💻 Coding · 🏗️ Design · 🐞 Debug
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What are the two phases of autoregressive generation?
2. What is the KV cache and why does it exist?
3. Greedy vs sampling vs beam search — when use which?
4. What do temperature, top-k, and top-p each control?
5. What is quantization and why does it speed up inference?
6. What is FlashAttention solving?
7. What is continuous (in-flight) batching?
8. What is speculative decoding at a high level?
9. Name three LLM serving frameworks and a use case for each.
10. Define TTFT, TPOT, and throughput.

## 🟡 Core (L4–L5)
11. Why is decode memory-bandwidth-bound while prefill is compute-bound?
12. Derive the KV-cache size for a given model, context, and batch.
13. How do MQA and GQA reduce the KV cache, and what's the tradeoff?
14. Why does FlashAttention speed things up *without changing the math*?
15. Compare GPTQ, AWQ, and GGUF.
16. How does speculative decoding preserve the exact output distribution?
17. What does PagedAttention do for memory and batch size?
18. Explain the latency ↔ throughput tradeoff and how batch size moves it.
19. Weight-only vs weight+activation quantization — why is the latter harder?
20. When would you reach for distillation or pruning instead of quantization?

## 🔴 Senior / Staff deep dives (with follow-ups)
21. Design a serving stack to hit p99 < 300ms TTFT at 1000 RPS for a 13B model. Walk the whole design.
    → *covers:* GQA model, quantization, vLLM/TensorRT-LLM, continuous batching, prefix caching, autoscaling, prefill/decode disaggregation, hardware choice.
22. Your tokens/sec is far below GPU FLOPs roofline during decode. Explain why that's *expected*.
    → *covers:* decode does tiny GEMVs; bound by HBM bandwidth moving weights + KV per token; arithmetic intensity is low → batch to raise it.
23. Walk through every lever to cut cost-per-token by 2× and the quality risk of each.
    → *covers:* quantization, smaller/distilled model, batching, speculative decoding, KV-cache compression, prompt/context reduction, caching, routing.
24. How does speculative decoding work in detail, including the accept/reject step?
    → *covers:* draft proposes k tokens, target verifies in one parallel pass, accept via the modified-rejection-sampling rule that provably matches the target distribution; expected speedup ∝ acceptance rate.
25. You serve 100 LoRA adapters on one base model. Design the inference path.
    → *covers:* shared base weights, batched multi-adapter (e.g., S-LoRA/punica) kernels, dynamic loading, memory accounting.
26. Long context (128K) is blowing up memory. Options at the serving layer?
    → *covers:* GQA/MQA, KV-cache quantization, paged KV, eviction/streaming, chunked prefill, attention sinks.
27. Explain MFU (model FLOPs utilization). Why is it low in decode and how do you raise it?
28. Disaggregated prefill/decode serving — why split them onto different pools?

## 🧮 Math & derivations
29. Compute KV-cache bytes for: 32 layers, 8 KV heads, head_dim 128, seq 8192, batch 16, fp16. Show the formula.
30. Estimate decode latency from model size and HBM bandwidth (bandwidth-bound model).
31. Derive the arithmetic intensity of a decode step and why batching helps.
32. Given int4 vs fp16 weights, estimate the memory-movement (and thus speed) ratio for decode.

## 💻 Coding / implementation
33. Implement greedy + temperature + top-k + top-p sampling from logits.
34. Implement an incremental KV cache and the single-step decode loop.
35. Implement a simple speculative-decoding accept/reject loop given draft + target logits.
36. Write a benchmark harness measuring TTFT, TPOT (p50/p99), and throughput under concurrency.
37. Implement naive int8 weight quantization + dequant for a linear layer.

## 🏗️ System design / applied
38. Design the inference platform for a multi-tenant LLM API: routing, batching, autoscaling, quotas, observability, cost controls.
39. Edge deployment: run a 7B model on a laptop/phone. What do you change?
    → *covers:* GGUF/4-bit, llama.cpp, smaller context, CPU/Metal, memory mapping.
40. Pick hardware (A100 vs H100 vs L4 vs CPU) for three workloads (chat, batch summarization, embeddings) and justify.

## 🐞 Debugging / scenarios
41. Throughput collapses when a few long-context requests arrive. Why, and fix?
    → *KV-cache pressure / fragmentation → PagedAttention, length-based scheduling, separate pools.*
42. int4 quantization tanked accuracy on one task only. Diagnose.
    → *outlier channels / activation sensitivity → AWQ, mixed precision, keep sensitive layers fp16.*
43. p50 latency is fine but p99 is terrible. Likely causes?
    → *head-of-line blocking, batch scheduling, GC/paging, cold cache, long-prompt prefill stalls.*
44. Speculative decoding gave no speedup. Why might that happen?
    → *low acceptance rate (weak/mismatched draft), verification overhead, short outputs.*

## ✅ What strong candidates demonstrate
- Identify the **correct bottleneck** (memory bandwidth vs compute) before optimizing.
- Do **KV-cache and latency math** live.
- Know the **tradeoff** behind every speedup (quality, memory, complexity).
- Think in **p50/p99 and cost-per-token**, not averages.

---
Related: the **🔥 Mastery checks** in [README.md](README.md) are the minimum bar.
