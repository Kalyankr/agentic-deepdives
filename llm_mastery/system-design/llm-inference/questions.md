# LLM Inference Service — Interview Questions (all levels)

> **Scope:** screening through **senior / staff / principal** ML-systems / Applied-Scientist interviews. The reference design is [README.md](README.md). `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math/Estimation · 🏗️ Design · 🐞 Debug/Ops
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What does an LLM inference service do, and why is it its own system?
2. Why is decode memory-bandwidth-bound while prefill is compute-bound?
3. What is arithmetic intensity, and why does it force you to batch?
4. Define TTFT and TPOT and the SLO each one drives.
5. What is continuous (in-flight) batching?
6. What is the KV cache and why does it cap concurrency?
7. What is PagedAttention and what problem does it solve?
8. Why expose an OpenAI-compatible API?
9. What is MFU and why is it the cost budget?
10. Name the components a request flows through, gateway to GPU and back.

## 🟡 Core design
11. Walk through the lifecycle of a single streaming completion request.
12. Design the scheduler — admission, priority, and batch formation.
13. How do you serve many models on one fleet (multi-model packing)?
14. How do you serve thousands of fine-tunes cheaply (multi-LoRA)?
15. Design KV-cache management under memory pressure.
16. How does the router pick a replica for a request?
17. How do you autoscale a GPU fleet for inference?
18. How do you handle model loading and cold starts?
19. How do you enforce multi-tenant fairness and quotas?
20. How do you do zero-downtime model deploys and rollbacks?

## 🔴 Senior / Staff deep dives
21. Why disaggregate prefill and decode onto separate pools?
22. Walk through speculative decoding end to end — and when does it not help?
23. How do you hit p99 TTFT under heavy load? Enumerate every lever.
24. A few long prompts are tanking everyone's TPOT. Diagnose and fix.
25. Make a streaming request survive a replica failure mid-generation.
26. Cut $/1M tokens by 2× without hurting quality — in order.
27. How does prefix-aware routing work and what does it buy?
28. Design scale-to-zero for cold models without hurting warm-model SLOs.

## 🧮 Math & estimation
29. Estimate single-stream decode latency for a 70B model on an 8×H100 node.
30. Estimate KV-cache bytes per sequence and the batch size it allows.
31. Estimate how many GPUs to serve 2M output tokens/sec.
32. Write the speculative-decoding acceptance criterion.
33. How does batch size trade off against TTFT and TPOT?
34. Estimate the throughput gain of continuous vs static batching.

## 🏗️ Design variations
35. Design serving for reasoning models with long test-time compute.
36. Design an embeddings-serving service — what changes vs chat?
37. Design a batch / offline inference tier on spot GPUs.
38. Design heterogeneous-hardware serving across mixed GPU types.

## 🐞 Debugging & ops
39. MFU is low but the GPUs look "busy." Diagnose.
40. TTFT is fine but TPOT degrades as load rises. Why?
41. OOM crashes under long-context bursts. Fix it without buying GPUs.
42. Turning on speculative decoding made throughput worse. Why?

---
Back to the [HLD](README.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
