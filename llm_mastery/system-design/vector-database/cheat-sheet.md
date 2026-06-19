# 🃏 Vector Database Internals — One-Page Cheat-Sheet

> Last-minute recall card for the [full HLD](README.md). Drill the bold bits.

## The one idea
A vector DB is **a storage engine wrapped around an approximate (ANN) index** that answers top-$k$ nearest-neighbor queries over billions of embeddings. The [RAG platform](../rag-platform/README.md) *uses* it; here we *build* it. Core tension: **recall vs. latency vs. memory vs. freshness.**

## The number that drives everything
$$\text{RAM} = N \times d \times \text{bytes}$$
| 1B × 768-dim | per vec | total |
|---|---|---|
| fp32 | 3,072 B | **~3 TB** |
| fp16 | 1,536 B | ~1.5 TB |
| int8 | 768 B | ~0.75 TB |
| **PQ m=96** | **96 B** | **~96 GB** |

→ raw vectors **won't fit one box** → **shard / quantize / tier**. Exact NN = $O(Nd)$/query → hopeless → **ANN exists**.

## Why approximate (curse of dimensionality)
In high-$d$, distances **concentrate** and trees degrade to full scans. Accept ~95–99% recall for **100–1000×** speed + memory. **Never promise exact at scale.**

## Metrics
**cosine** (angle; text default) · **dot** (angle+magnitude; MIPS/recsys) · **L2** (vision). On **normalized** vectors all three rank the same → normalize + dot.

## ANN families (know cold)
| Family | Idea | Win | Cost |
|---|---|---|---|
| **HNSW** | multi-layer graph, greedy hops | best **in-RAM** recall/latency | heavy memory, awkward deletes |
| **IVF** | k-means cells, scan `nprobe` | cheap, tunable, fast build | recall ∝ nprobe; needs training |
| **PQ** | $m$ subvector codebooks → codes | **8–64× smaller** + fast | lossy → rerank |
| **IVF-PQ** | IVF narrows, PQ stores/scores | **billion-scale in RAM** | most knobs |
| **DiskANN** | graph on SSD + PQ in RAM | billions on **1 node** | SSD latency |

**Pick:** fits RAM + recall-critical → HNSW · memory-bound 100M–10B → IVF-PQ · 1 box / cost-bound → DiskANN. **Always: compressed search + full-precision rerank.**

## HNSW
Greedy descent top→bottom, ~**$O(\log N)$**. **M** = links/node (recall↔memory) · **ef_construction** = build quality · **ef_search** = **recall↔latency dial** (ef ≥ k). Best in-RAM curve; **deletes = tombstone + rebuild**.

## IVF-PQ search
(1) query vs `nlist` centroids → pick **`nprobe`** cells. (2) build PQ **distance table** ($m\times256$). (3) each code's dist = **sum of $m$ lookups** (ADC, no full math). (4) **rerank top-N full precision.** Scan ≈ **`nprobe`·N/nlist** (vs N).
PQ: split $d$ into $m$ subvecs, 256-centroid codebook each → **1 byte/subvec** (768 fp32 → 96 B = **32×**); codebooks ~MB (free).

## Storage engine (LSM-style)
`upsert → WAL (durable) → mutable in-mem segment (instantly searchable) → flush → immutable segment + ANN index → compaction (merge + rebuild + drop tombstones)`. Search = **fan-out over all segments + merge**. Segment count ↔ latency/freshness tradeoff.

## Inserts / deletes / updates (the hard part)
ANN ❤ static data. **Insert** = append (mutable seg). **Delete** = **tombstone** (mark + filter at query, reclaim in compaction) — can't edit a graph in place. **Update** = delete + insert. Heavy churn → recall decays → **compact by tombstone ratio.**

## Filtered search (first-class hard problem)
- **post-filter** (search→drop): selective filter ⇒ **recall collapse.**
- **pre-filter** (matches→search): great if small set, else brute force.
- **in-filter** (only traverse matching neighbors) + payload indexes co-located → the scalable answer.
- **selectivity-aware planner** picks: selective → pre-filter; broad → in-filter ANN (bump ef).

## Distribution
**Shard** vectors (random = balanced fan-out · clustered = fewer shards, skew risk). **Scatter-gather** → latency = **slowest shard** (hedged reads). **Replicas** = HA + read QPS (leader writes, WAL ship). Rebalance from snapshots.

## Durability & consistency
**WAL** + **snapshots → object storage** + replicas. Recover = snapshot + **replay WAL tail**. Usually **eventual** (offer read-your-writes). **Commit vector + payload atomically** (one WAL record).

## Tuning = the recall–latency–memory surface
Sweep **M/ef · nlist/nprobe · PQ m/bits · rerank depth** vs **brute-force ground truth**. Fix recall target (≥0.95), minimize latency/memory. **Rerank to cheat the curve.** Never quote latency without recall.

## Cost order
PQ/int8 + rerank → DiskANN (SSD) over RAM → right-size ef/nprobe → scale-to-zero idle collections → share shards for small tenants → schedule compaction. Track **$/M vectors** + **$/M queries**.

## Top failure modes
low recall (raise ef/nprobe, rerank, compact) · won't fit RAM (PQ/DiskANN/shard) · **selective-filter recall collapse** · **delete-driven decay** (compact) · **p99 spikes from compaction/GC** · stale-after-insert (mutable seg) · **centroid drift** (retrain) · OOM build (per-segment) · scatter-gather tail · index↔payload mismatch.

---
[← HLD](README.md) · [Q&A](questions.md) · [Answers](answers.md) · [Index](../../README.md)
