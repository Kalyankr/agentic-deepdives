# 🃏 RAG Platform — One-Page Cheat-Sheet

> Last-minute recall card for the [full HLD](README.md). Drill the bold bits.

## The one idea
**Fetch relevant, access-controlled evidence at query time and ground the LLM in it** (with citations + abstention) → fixes stale knowledge, hallucination, and attribution. Two hard parts: **retrieval quality** and **keeping a huge index fresh + access-controlled**.

## Two planes (separate them)
- **Ingestion** (async, throughput): parse → **chunk** → embed → index. Scales to the *change rate*.
- **Query** (sync, latency): rewrite → embed → **retrieve → filter → rerank → ground**. Scales to *QPS*.
- They meet only at the **shared index + metadata/ACL store**.

## Numbers (state assumptions)
- 1B chunks × (1024-dim fp16 = **2 KB/vec**) → **2 TB raw**, ×~2–3 HNSW → **~5 TB**, shard ~50–100 nodes (+replicas). PQ cuts vectors ~8–16×.
- Index 1B chunks once ≈ $\frac{10^9}{3\text{K/s}} \approx$ **90 GPU-hours** (parallelizable); steady-state embed only the **delta (CDC)**.
- Query budget: embed ~5 ms → **ANN ~20–50 ms** → rerank top-100 ~50–100 ms → generate (streamed). Keep retrieval+rerank **<~200 ms**.

## Retrieval = a funnel
`query → rewrite/HyDE → [dense ANN ∥ BM25] → ACL+metadata filter → RRF fuse → cross-encoder rerank (100→8) → ground+cite`
- **Hybrid:** dense = semantics/paraphrase; **sparse/BM25** = exact terms/IDs/rare words. Fuse by **RRF** $=\sum 1/(k+\text{rank})$ (rank-based, no score calibration).
- **Bi-encoder** (fast, indexable) for **recall** → **cross-encoder** (joint, slow) for **precision**.
- **HyDE / query rewrite** bridges the question↔answer vocabulary gap.

## Chunking (highest-leverage knob)
Too small = fragmented; too big = diluted embedding + wasted tokens. **Structure-aware + ~10–20% overlap**, attach metadata, **parent-child** (embed child, return parent). Default ~256–512 tok. **Tune against the eval set.**

## Vector index
**HNSW** = great recall/latency, high memory, fragments on updates → compaction. **IVF-PQ** = compressed/scalable, needs centroid training, lower recall. Shard (scatter-gather) + replicate; **co-locate metadata for filtered search**; PQ to save memory.

## Access control (the #1 rule)
**Authorize at RETRIEVAL, never at generation** — the LLM summarizes anything in its context. **Pre-filter inside the ANN** (post-filter wrecks recall + leaks existence). Per-tenant **namespaces** for hard isolation; per-doc ACL predicates for fine-grained.

## Grounding (kill hallucination)
Answer **only from context** + **per-claim citations** + **abstain** if unsupported + **verify** cited spans. Most "hallucinations" are actually **retrieval misses** → fix retrieval first.

## Freshness & deletes
**CDC** streams create/update/delete → re-embed only changes, upsert by stable chunk id, **version embeddings**. **Deletes = tombstone now + purge everywhere** (compliance). **Embedding-model upgrade invalidates the whole index** → **dual-index re-embed + atomic swap**. Compact to fight fragmentation.

## Evaluate the two stages SEPARATELY
- **Retrieval:** recall@k, MRR, nDCG vs **golden query→doc** set.
- **Generation:** **faithfulness** (claims supported), answer relevance, citation correctness (LLM-judge + human).
- Wrong answer? **Attribution split:** gold doc retrieved? No → retrieval bug; yes but ignored/contradicted → generation bug.
- **Offline CI gate** on index/prompt/model changes + **online** (thumbs, citation-click, abstention).

## Caching
Embedding cache (text→vec) · retrieval cache (query→top-k, invalidate on doc change) · answer cache (scoped per tenant **+ ACL set**) · LLM prefix-cache the stable grounding prompt.

## Security
Per-doc ACL at retrieval · tenant isolation/encryption/residency · PII redaction + right-to-be-forgotten · **retrieved content = untrusted → indirect prompt injection** (delimit/label, least privilege, output guards).

## Cost order
dedup + cache → quantize (PQ) index memory → scale-to-zero idle tenants → rerank only when needed → **route generation to a smaller model** → spot for backfills. Track **$/query** and **$/doc** per tenant.

## Top tradeoffs / failure modes
chunking (frag↔dilute) · **filtered ANN** (pre vs post → recall/leak) · recall↔latency↔cost · freshness↔re-embed cost · **ACL-at-generation leak** · parsing quality (bad PDF→bad retrieval) · hallucination (ground+cite+abstain) · index drift (compaction) · **embedding upgrade re-index** · indirect injection.

---
[← HLD](README.md) · [Q&A](questions.md) · [Answers](answers.md) · [Index](../../README.md)
