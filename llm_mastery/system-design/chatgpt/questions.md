# System Design — Interview Questions (Design ChatGPT, all levels)

> **Scope:** screening through **senior / staff / principal** ML-systems / Applied-Scientist interviews. The reference design is [README.md](README.md) (the ChatGPT HLD). `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math/Estimation · 🏗️ Design · 🐞 Debug/Ops
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What makes serving an LLM product fundamentally different from a normal stateless web service?
2. Define TTFT and TPOT — which part of the user experience does each affect?
3. Why keep the data plane stateless while conversation state lives elsewhere?
4. Why stream tokens (SSE) instead of returning the full completion at once?
5. What is the difference between prefill and decode, and why does it matter for serving?
6. What is continuous (in-flight) batching and why is it the biggest throughput lever?
7. What is the KV cache and why does it dominate inference memory?
8. What is the role of the API gateway in this system?
9. Why route between multiple model sizes instead of always using the biggest model?
10. Name the layers a request passes through from client to GPU and back.

## 🟡 Core design
11. Walk through the end-to-end lifecycle of a single streaming chat request.
12. How do you assemble the prompt when conversation history exceeds the context window?
13. How does the model router decide which model serves a request?
14. Design the inference scheduler — how are requests admitted, prioritized, and batched?
15. How do you serve many per-tenant fine-tunes without dedicating a GPU per tenant?
16. How does PagedAttention work and what problem does it solve?
17. Design the streaming layer for millions of concurrent connections.
18. How do you integrate RAG into the chat path, and what do you evaluate separately?
19. How does tool / function calling work in the orchestration loop, and how do you bound it?
20. Where do you place safety/moderation, and what does each layer catch?
21. Design the storage layer — what data goes where, and why those stores?
22. How does the data flywheel turn production traffic into a better model?

## 🔴 Senior / Staff deep dives
23. Why disaggregate prefill and decode onto separate pools? What does it buy and cost?
24. How do you hit p99 TTFT < 2 s under heavy load? Enumerate every lever.
25. A few long-context requests are tanking p99 for everyone. Diagnose and fix.
26. How do you autoscale a GPU fleet given slow provisioning and spiky demand?
27. Walk through speculative decoding end-to-end — and when does it fail to help?
28. Design multi-region serving with data residency and graceful degradation.
29. How do you make a stateful streaming request survive a replica failure mid-generation?
30. Cut serving cost per token by 2× without hurting quality — what do you do, in order?
31. How do you defend an agentic RAG ChatGPT against indirect prompt injection?
32. How do you roll out a new model safely to 100M users?

## 🧮 Math & estimation
33. Estimate the peak request rate and output-token throughput for 100M DAU.
34. Estimate the GPU fleet size to serve a 70B-class model at that load.
35. Compute the KV-cache size for a model/context/batch, and size it for 1M concurrent sessions.
36. Break down the TTFT budget and derive the per-token decode latency lower bound.
37. Estimate conversation and vector-index storage per year.
38. Estimate cost per 1M tokens and identify what dominates it.
39. How does MoE change active params/FLOPs, and therefore the fleet size?
40. How many concurrent streams can one gateway node hold, and what's the limiting resource?

## 🏗️ Design variations
41. How does the design change for a reasoning ("thinking") model with heavy test-time compute?
42. Add multimodal (image input/output) — what changes across the stack?
43. Design the developer **API product** (rate limits, quotas, billing, batch/async).
44. Add per-user **long-term memory** across sessions — design it end to end.
45. Design an **on-device / edge** small-model tier and the routing policy to it.
46. Design **semantic caching** — when does it help and when is it dangerous?

## 🐞 Debugging & ops
47. TTFT p99 spiked 3× with no change in request rate. Where do you look?
48. GPU utilization reads high but tokens/sec/GPU is low. What's wrong?
49. Users report responses cut off mid-sentence. Enumerate suspects.
50. Cost per token crept up 30% over a month with no deploy. Investigate.
51. The fleet is throwing OOM under long-context load. Fix it.
52. Quality silently regressed after a model deploy. How do you detect it and roll back?

---
Back to the [HLD](README.md) · [Index](../../README.md)
