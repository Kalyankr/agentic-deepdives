# RAG Platform — Interview Questions (all levels)

> **Scope:** screening through **senior / staff / principal** ML-systems / Applied-Scientist interviews. The reference design is [README.md](README.md). `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math/Estimation · 🏗️ Design · 🐞 Debug/Ops
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What is RAG and what problem does it solve over a plain LLM call?
2. Why retrieve external knowledge instead of fine-tuning it into the model?
3. What are the two planes of a RAG platform and why separate them?
4. What is chunking and why does it matter so much for quality?
5. What is an embedding, and what does "semantic search" actually mean?
6. Dense vs sparse (BM25) retrieval — what does each catch that the other misses?
7. What is reranking and why add it after the initial retrieval?
8. What does "grounding" mean, and how do citations and abstention help?
9. Why must access control be enforced at retrieval, not at generation?
10. What is the vector index for, and why not just scan every vector?

## 🟡 Core design
11. Walk through the end-to-end lifecycle of a single query in a RAG platform.
12. Design the ingestion pipeline from a raw document to an indexed chunk.
13. How do you choose a chunking strategy for a given corpus?
14. Design hybrid retrieval — how do you combine dense and sparse results?
15. How do you enforce per-document ACLs efficiently during retrieval?
16. How do you keep the index fresh as source documents change?
17. How do you handle document deletions and right-to-be-forgotten?
18. Design the multi-tenant isolation model for the platform.
19. How do you assemble context and prompt the LLM for a grounded, cited answer?
20. How do you evaluate a RAG system end to end?

## 🔴 Senior / Staff deep dives
21. The right document is in the corpus but never makes it into the answer. Diagnose.
22. How do you scale the vector index to 10B+ chunks?
23. You must upgrade the embedding model — how do you re-index with zero downtime?
24. Pre-filtering vs post-filtering for ACLs/metadata — tradeoffs and recall impact.
25. Design retrieval for a query that needs multi-hop reasoning across documents.
26. A grounded system still hallucinates. How do you drive it down?
27. How do you defend against indirect prompt injection from retrieved content?
28. HNSW vs IVF-PQ — when do you pick each at platform scale?

## 🧮 Math & estimation
29. Estimate storage for 1B chunks of 1024-dim fp16 vectors plus index overhead.
30. Estimate the embedding compute to index 1B chunks once.
31. Lay out a query-path latency budget across the retrieval funnel.
32. Write the Reciprocal Rank Fusion formula and explain how it merges lists.
33. How do you size the number of shards and replicas for the index?
34. Estimate the cost per query and per ingested document.

## 🏗️ Design variations
35. How would you design GraphRAG / structured retrieval over a knowledge graph?
36. Design multimodal RAG (images, tables, scanned docs).
37. Design agentic / iterative retrieval where the model decides what to fetch.
38. Design a RAG cache that respects freshness and per-user personalization.

## 🐞 Debugging & ops
39. Retrieval p99 latency spiked right after a large ingestion job. Diagnose.
40. Recall dropped after you added a metadata filter. Why, and how to fix?
41. A tenant reports seeing another tenant's document in results. Triage it.
42. Faithfulness scores fell after a prompt change. How do you localize the cause?

---
Back to the [HLD](README.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
