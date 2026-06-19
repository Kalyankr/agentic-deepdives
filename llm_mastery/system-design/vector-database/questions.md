# Vector Database Internals — Interview Questions (all levels)

> **Scope:** screening through **senior / staff / principal** ML-systems / Applied-Scientist interviews. The reference design is [README.md](README.md). `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math/Estimation · 🏗️ Design · 🐞 Debug/Ops
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What is a vector database, and what does it do that a regular (B-tree/hash) database can't?
2. Why **approximate** nearest neighbor instead of exact — what's the curse of dimensionality?
3. Cosine vs dot-product vs L2 — which metric when, and how do they relate?
4. What is HNSW at a high level?
5. What is IVF (an inverted-file index)?
6. What is product quantization (PQ), and why use it?
7. Why are vectors so memory-hungry, and what's the dominant cost driver?
8. What does recall@k mean here, and why is it the key quality metric?
9. What is filtered (metadata + vector) search, and why is it hard?
10. Why are inserts/deletes harder in a vector index than in a B-tree?

## 🟡 Core design
11. Walk through a top-k search request end to end.
12. Design the storage engine / write path for a vector DB.
13. How does HNSW search work, and what do M and ef control?
14. How does IVF-PQ search work, step by step?
15. Design filtered search so a selective filter doesn't destroy recall.
16. How do you shard and route queries across nodes?
17. How do you handle deletes and updates in a graph index?
18. How do you make freshly inserted vectors searchable within seconds?
19. How do you make the system durable and crash-safe?
20. How do you tune the recall / latency / memory tradeoff?

## 🔴 Senior / Staff deep dives
21. Recall is too low at an acceptable latency. Diagnose and fix.
22. The index doesn't fit in RAM. What are your options, in order?
23. A highly selective filter returns too few / wrong results. Fix it.
24. Designing for 100B vectors — what changes versus 1B?
25. One shard is a hot spot. Diagnose and rebalance.
26. Choose HNSW vs IVF-PQ vs DiskANN for given constraints — and justify.
27. How do you support real-time streaming upserts without rebuilding the whole index?
28. How do you keep the vector index and the metadata/payload consistent?

## 🧮 Math & estimation
29. Estimate the RAM to hold 1B 768-dim vectors at fp32, fp16, and PQ.
30. Estimate PQ compression ratio and the codebook size.
31. Estimate the distance computations for an IVF search given nlist and nprobe.
32. Estimate HNSW memory (graph links + vectors).
33. How does PQ make distance computation fast — the lookup-table (ADC) trick?
34. Estimate query latency for in-memory HNSW vs disk-based (DiskANN).

## 🏗️ Design variations
35. Design a disk-based ANN index (DiskANN-style) for billion-scale on one node.
36. Design multi-tenant collections with isolation and per-tenant scale.
37. Design the vector DB as the index behind the RAG platform — what's the contract?
38. Design snapshot / backup / restore for a large distributed vector index.

## 🐞 Debugging & ops
39. p99 latency spikes periodically while p50 is fine. Why?
40. Recall dropped after a big batch of deletes. Why, and how to recover?
41. Search misses vectors that were just inserted. Why, and how to fix?
42. Memory blows up (OOM) during an index build. Triage it.

---

> **How to practice:** start every answer with the **memory estimate** ($N \times d \times$ bytes), then reason about which **ANN family** fits and which knob (ef / nprobe / PQ bits) moves recall. Check yourself against [answers.md](answers.md) and the [one-page cheat-sheet](cheat-sheet.md).

[← Back to vector DB HLD](README.md) · [Answer key](answers.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
