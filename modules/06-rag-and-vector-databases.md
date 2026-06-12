# Module 06 · RAG & Vector Databases

> **Goal:** Build retrieval-augmented generation systems that are accurate, fast, and grounded — and understand the vector-search internals (ANN indexes) underneath them well enough to tune them at scale.

**Duration:** ~4 weeks. **Prereqs:** [Module 02](02-transformer-internals.md).

---

## 6.1 Why RAG

- LLMs have frozen, lossy parametric memory; RAG injects fresh, verifiable, private knowledge at inference time
- Reduces hallucination, enables citations, avoids constant re-training
- RAG vs. long-context vs. fine-tuning — when to use which (cost, freshness, controllability)

## 6.2 Embeddings & semantic search

- Embedding models: bi-encoders (dense retrieval), sentence transformers, instruction-tuned embeddings
- Embedding dimensions, normalization, **cosine vs. dot vs. Euclidean**
- Matryoshka embeddings (truncatable dims), multilingual & domain embeddings
- Contrastive training of embeddings (intuition); why retrieval ≠ generation models

## 6.3 Vector databases & ANN internals

The "database" interviewers probe — know the algorithms, not just the API.

- The problem: exact nearest neighbor is $O(N)$; we need **Approximate NN (ANN)**
- **HNSW** (Hierarchical Navigable Small World) — graph-based, great recall/latency; `M`, `efConstruction`, `efSearch` knobs
- **IVF** (inverted file) — cluster then search probes (`nprobe`)
- **Product Quantization (PQ)** / OPQ — compress vectors for memory; IVF-PQ
- ScaNN, DiskANN (SSD-resident, billion-scale)
- Recall vs. latency vs. memory trade-offs; how to measure recall@k
- **Hybrid search** — combine dense + sparse (BM25/SPLADE); **Reciprocal Rank Fusion**
- Metadata filtering (pre- vs. post-filtering) and its impact on ANN
- Tools: **FAISS** (library), and managed/servers: Qdrant, Weaviate, Milvus, pgvector, Pinecone, Vespa, Elasticsearch/OpenSearch

> **Build:** Implement brute-force kNN, then use **FAISS** with `IndexFlat`, `IVF`, `IVFPQ`, and `HNSW`. Plot recall@10 vs. latency vs. memory across configs on a real dataset (e.g., 1M vectors). This is a classic, high-signal exercise.

## 6.4 The RAG pipeline

### Indexing
- **Chunking** strategies: fixed, recursive, semantic, structure-aware; overlap; chunk size trade-offs
- Metadata extraction, parent/child (small-to-big) retrieval
- Document parsing (PDF, HTML, tables) — the messy real-world part

### Retrieval
- Query construction, **query rewriting / expansion**, HyDE (hypothetical doc embeddings)
- Multi-query, RAG-fusion
- **Re-ranking** with a cross-encoder (e.g., bge-reranker, Cohere Rerank) — usually the biggest quality win
- Maximal Marginal Relevance (diversity)

### Generation
- Prompt assembly, context ordering ("lost in the middle"), citation/grounding
- Handling "no answer" / abstaining; reducing hallucination
- Context window budgeting and compression

### Advanced patterns
- **GraphRAG** (entity/relationship graphs), knowledge graphs
- Agentic / iterative RAG (retrieve → reason → retrieve again) — bridges to [Module 07](07-agentic-systems.md)
- Long-context RAG, contextual retrieval (Anthropic's contextual embeddings/BM25), caching

## 6.5 Evaluating RAG

- **Retrieval metrics:** recall@k, precision@k, MRR, nDCG, hit rate
- **Generation/end-to-end:** faithfulness/groundedness, answer relevance, context precision/recall
- Frameworks: RAGAS, TruLens, ARES; LLM-as-judge pitfalls
- Building a golden eval set; detecting regressions (ties to [Module 09](09-evaluations.md))

## 6.6 Production concerns

- Incremental indexing, freshness, deletes, re-embedding on model upgrades
- Sharding & replication of the vector index at scale; memory footprint planning
- Latency budget across embed → search → rerank → generate
- Multi-tenancy, access control on retrieved data, PII

---

## Module 06 capstone — **A production-quality RAG service**

1. The FAISS ANN benchmark (recall/latency/memory across index types).
2. A full RAG service over a real corpus (your docs/a public dataset) with chunking, hybrid retrieval, **re-ranking**, and cited answers.
3. A **RAGAS-style eval harness** with retrieval + generation metrics and a golden set; show that adding reranking/hybrid search measurably improves the numbers.
4. A latency breakdown across the pipeline and one optimization (caching, smaller reranker, etc.) with before/after.

## Exit criteria
- [ ] You can explain HNSW and IVF-PQ and tune recall vs. latency vs. memory.
- [ ] You can design a RAG pipeline and name the highest-leverage quality levers (reranking, chunking, hybrid).
- [ ] You can build a RAG eval harness and use it to drive improvements.
- [ ] You can plan the memory/sharding footprint of a billion-scale index.

## Core papers / sources
- *Retrieval-Augmented Generation* — Lewis et al., 2020
- *Dense Passage Retrieval* — Karpukhin et al., 2020
- *HNSW* — Malkov & Yashunin, 2016
- *Product Quantization for NN Search* — Jégou et al., 2011
- *Lost in the Middle* — Liu et al., 2023
- *Contextual Retrieval* — Anthropic, 2024
- FAISS, Qdrant, RAGAS docs
