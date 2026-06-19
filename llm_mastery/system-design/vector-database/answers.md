# Vector Database Internals — Answer Key

> Full worked answers to [questions.md](questions.md). The bar: **lead with the memory estimate** ($N\times d\times$ bytes → vectors dominate), **know the ANN families and their math** (HNSW / IVF / PQ), treat the system as a **storage engine wrapped around an approximate index**, and always pair **recall with latency/memory**. Reference design: [README.md](README.md).
>
> Notation: $N$ = #vectors, $d$ = dimensions, $k$ = neighbors returned, ANN = approximate nearest neighbor, ADC = asymmetric distance computation, ef/nprobe = the recall↔latency dials.

---

## 🟢 Fundamentals

**1. What is a vector database, and what can't a regular DB do?**
A vector DB stores high-dimensional **embeddings** and answers **nearest-neighbor** queries — "find the $k$ vectors most similar to this one" by cosine/dot/L2. A B-tree/hash index answers **exact** point and **range** queries on scalars with a useful sort order; similarity in 100s of dimensions has **no such order** and exact search degrades to a full scan. So a vector DB is built around **approximate index structures** (HNSW/IVF/PQ) plus a storage engine for durability, filtering, and updates.

**2. Why approximate, and the curse of dimensionality.**
Exact NN is $O(Nd)$ per query (brute force) because high-dimensional space defeats space-partitioning: as $d$ grows, distances **concentrate** (everything is roughly equidistant) and tree methods (kd-trees) must visit nearly all cells → no better than scanning. **ANN** accepts ~95–99% recall for **100–1000×** speedup and huge memory savings. At billion-scale, exact search per query (1B × 768 ≈ $10^{12}$ ops) is hopeless, so approximation isn't a shortcut — it's the only viable approach.

**3. Cosine vs dot vs L2.**
- **Cosine** — angle only (magnitude-invariant); the default for text embeddings where direction = meaning.
- **Dot product** — angle **and** magnitude; used when magnitude encodes importance (some recsys/MIPS models).
- **L2 (Euclidean)** — straight-line distance; common for image/vision features.
They relate: on **normalized** vectors, cosine, dot, and (monotonically) L2 give the **same ranking**, so many systems L2-normalize and use dot internally. Pick the metric the embedding model was trained for.

**4. HNSW in one breath.**
**Hierarchical Navigable Small World** = a multi-layer proximity graph. Upper layers are sparse with long-range links (express lanes); the bottom layer holds all vectors with dense short links. Search **greedily hops** toward the query from the top entry point, descending layer by layer — ~**$O(\log N)$**. It gives the **best recall/latency in RAM**, at the cost of high memory (vectors + graph links) and awkward deletes.

**5. What is IVF?**
**Inverted File index**: run **k-means** to learn `nlist` centroids, assign each vector to its nearest centroid's "cell" (posting list). At query time, find the **`nprobe`** nearest centroids and scan **only those cells** (≈ `nprobe`·$N/\text{nlist}$ vectors) instead of all $N$. `nprobe` is the recall↔latency dial. Cheap memory and fast build, but recall is sensitive to `nprobe` and centroid quality (boundary vectors get missed).

**6. What is PQ and why use it?**
**Product Quantization** compresses a vector into a tiny code: split it into **$m$ subvectors**, learn a 256-entry **codebook** per subspace (k-means), and store each subvector as a **1-byte centroid id**. A 768-dim fp32 vector (3072 B) becomes **~96 B** ($m=96$) — **~32× smaller** — so billions of vectors fit in RAM. It also makes distance **fast** via lookup tables (Q33). The cost is **lossy, approximate** distances → pair it with a full-precision rerank.

**7. Why vectors are memory-hungry.**
Raw storage is $N\times d\times$ bytes and **dominates everything**: 1B × 768-dim fp32 = **~3 TB**. The dimensionality (hundreds–thousands) times billions of items is the cost driver — far more than the metadata or the graph/centroid overhead. That's why the entire design centers on **shrinking and tiering vectors** (fp16/int8/PQ, DiskANN) and sharding them across machines.

**8. What is recall@k?**
Recall@k = (the $k$ results the ANN returned ∩ the **true** $k$ nearest by exact search) ÷ $k$ — how many of the genuine nearest neighbors you actually found. It's the key quality metric because the index is **approximate**: latency without recall is meaningless (you can always be fast at low recall). You measure it against a **brute-force ground truth on a sample** and tune knobs to hit a target (e.g. ≥0.95).

**9. Filtered search and why it's hard.**
Filtered search combines a **metadata predicate** (`category = X`) with vector similarity. It's hard because the ANN index is built on geometry, not the predicate: **post-filtering** (search then drop non-matches) collapses when the filter is selective (top-$k$ may contain zero matches); **pre-filtering** (restrict to matches first) can degrade to a brute-force scan of the subset. The good answer is **in-filter** traversal (only walk to neighbors that satisfy the predicate) with metadata **co-located** with vectors, chosen by selectivity (Q15).

**10. Why inserts/deletes are hard.**
ANN indexes are optimized for **static** data. A B-tree edits in place; an HNSW **graph** can't cleanly drop a node without breaking navigability, and an IVF cell's quantizer assumes a fixed distribution. So deletes become **tombstones** (mark + filter at query time, reclaim during compaction/rebuild), updates become **delete + re-insert**, and heavy churn **degrades recall** until you rebuild. Freshness needs a separate **mutable segment**. This insert/delete/rebuild dance is the heart of vector-DB engineering.

---

## 🟡 Core design

**11. Top-k search end to end.**
Client sends `{vector, k, ef/nprobe, filter}` → **router** resolves the collection and **scatters** to all shards → each shard searches its **ANN index across all segments** (mutable + immutable), applying the filter **during** traversal, and returns local top-$k$ → router **gathers and merges** to a global top-$k$ → optional **full-precision rerank** of candidates → attach payloads → return ids + distances. Latency is bounded by the **slowest shard** plus merge/rerank.

**12. Storage engine / write path.**
**LSM-style immutable segments.** Write → **WAL** (durable) → **mutable in-memory segment** (immediately searchable by brute force) → **flush** to an **immutable segment** with its own built ANN index + payload columns → **background compaction** merges small segments, rebuilds the index, and drops tombstones. Search fans out over all segments and merges. This gives fast durable writes, bounded read amplification, and a clean place to reclaim deletes — at the cost of compaction work and a freshness/segment-count tradeoff.

**13. HNSW search; M and ef.**
Start at the top-layer entry point, **greedily move** to the neighbor nearest the query until no neighbor is closer, **descend** a layer, repeat; at the bottom keep a candidate list of size **`ef`** and return the best $k$.
- **`M`** = links per node → higher M improves recall but costs memory (~$M$ neighbors/node) and build time.
- **`ef_construction`** = build-time breadth → better graph, slower build.
- **`ef_search`** = query-time breadth → **the recall↔latency dial** (set ef ≥ k).

**14. IVF-PQ search step by step.**
(1) Compare the query to the **`nlist` centroids**, pick the **`nprobe`** nearest cells. (2) Build the PQ **distance table**: for each of the $m$ subspaces, distances from the query subvector to all 256 sub-centroids ($m\times256$). (3) For every PQ code in the probed cells, approximate distance = **sum of $m$ table lookups** (ADC) — no full distance math. (4) Keep the top candidates, then **rerank with full-precision vectors** to recover quantization loss. Tune `nprobe` (recall) and PQ `m`/bits (memory vs accuracy).

**15. Filtered search without wrecking recall.**
Make it **selectivity-aware**. Maintain **payload indexes** and co-locate metadata with vectors. **Highly selective filter** (few matches) → **pre-filter**: gather matching ids and brute-force/HNSW-search just those. **Non-selective filter** → **in-filter ANN**: traverse the graph/scan the lists but only accept neighbors satisfying the predicate (filterable HNSW), bumping **ef** to compensate for the pruned graph. The planner estimates selectivity to choose. Never plain post-filter on a selective predicate — recall collapses.

**16. Sharding & routing.**
Partition vectors across shards so data and index memory scale horizontally. **Random/round-robin** sharding balances load but every query **scatters to all shards** and merges; **clustered** sharding (route by centroid) hits fewer shards but risks skew. The **router** scatters, collects local top-$k$, and merges to a global top-$k$ — so latency tracks the **slowest shard**. Add replicas per shard for HA/read throughput. Rebalancing splits/moves shards (rebuild from snapshots) in the background.

**17. Deletes & updates in a graph index.**
You **can't** cleanly remove a node from HNSW without corrupting navigability, so: mark it in a **tombstone bitset**, **filter it out at query time**, and **physically reclaim** it during compaction/rebuild. Updates = **delete + insert** (segment vectors are immutable); payload-only changes go in a side column. Schedule compaction by **tombstone ratio** (not just size), because churn bloats the graph with dead ends and **drops recall** until rebuilt.

**18. Freshness for new inserts.**
Land new vectors in the **mutable in-memory segment** that's searched by **brute force** (or a tiny index) and merged into every query's results — so they're visible within **seconds**, before the heavy immutable index is built. Background flush + incremental build moves them into a real ANN segment later. This mutable-segment pattern is exactly how a vector DB reconciles "ANN wants static data" with "writes stream in."

**19. Durability & crash safety.**
**WAL** (fsync) ahead of the in-memory buffer; periodic **snapshots** of segments to **object storage**; **replicas** as live copies (WAL shipping, leader for writes). Recovery = reload last snapshot + **replay the WAL tail**. Commit the **vector and its payload atomically** under one WAL record so a result can't reference a deleted payload. Snapshots double as backup/restore and rebalancing units.

**20. Tuning recall/latency/memory.**
Treat it as a surface and **sweep the knobs against a brute-force ground truth**: HNSW **`M`/`ef`**, IVF **`nlist`/`nprobe`**, PQ **`m`/bits**, rerank depth, segment count. Fix a **recall target** (e.g. 0.95) and minimize latency/memory at it (or plot the full recall-vs-QPS curve). The cheat code: **compressed search + full-precision rerank** usually gives the best recall per byte. More shards/replicas trade money for throughput/tail.

---

## 🔴 Senior / Staff deep dives

**21. Low recall at acceptable latency.**
Diagnose the knob that's starving recall: **`ef_search`/`nprobe` too low** (raise them), **too many small segments** (each searched shallowly → compact), **over-aggressive PQ** (too few bits/subvectors → add a **full-precision rerank** or reduce compression), **selective filter** post-filtering (switch to pre/in-filter), or **centroid drift** in IVF (retrain). Measure recall@k against exact ground truth first so you're tuning the real metric, then trade a little latency (higher ef) or memory (rerank) for the recall you need.

**22. Index won't fit in RAM (in order).**
(1) **Lower precision** — fp16, then **int8** scalar quantization (4×) with light loss. (2) **PQ/OPQ** codes (8–64×) + **full-precision rerank** to recover recall — the big lever. (3) **DiskANN**: graph on SSD + compressed vectors in RAM → billion-scale on one node. (4) **Shard** across more machines so each holds a slice. (5) **Tier**: keep hot collections in RAM, cold ones on disk/object storage. Usually PQ + rerank or DiskANN solves it before you pay for more RAM.

**23. Selective filter returns too few / wrong results.**
This is **post-filtering on a selective predicate** — the ANN top-$k$ barely intersects the matching set. Fix: **pre-filter** (fetch matching ids via a payload index, then search just those — cheap because the set is small) or **in-filter** traversal with a higher **ef**. Ensure metadata is **co-located/indexed** with vectors so the predicate is cheap to evaluate during search. Add a planner rule: below some selectivity threshold, always pre-filter.

**24. Designing for 100B vectors vs 1B.**
Memory goes from ~TBs to ~**100s of TB raw** → you **must** quantize hard (PQ/OPQ) and **shard widely** (1000s of shards), making **scatter-gather fan-out and tail latency** the dominant problem (routing, hedged reads, maybe **clustered sharding** to hit fewer shards). Build becomes a massive distributed job (per-segment, from snapshots); **DiskANN/SSD tiering** is likely mandatory; centroid retraining and compaction are continuous background pipelines. Operationally it shifts from "tune an index" to "run a distributed storage system."

**25. Hot-spot shard.**
A shard taking disproportionate QPS/latency — often from **clustered sharding** (a popular topic concentrated on one shard) or skewed tenant load. Diagnose via per-shard QPS/latency/memory. Fix: **add replicas** to that shard for read throughput, **split** it and redistribute its vectors, switch hot collections to **random sharding** for even fan-out, or isolate the noisy tenant onto dedicated shards. Rebalance in the background from snapshots.

**26. HNSW vs IVF-PQ vs DiskANN.**
- **HNSW** — data fits in RAM and recall/latency is paramount (≤ ~10–100M/shard): best in-memory curve, accept the memory cost.
- **IVF-PQ** — memory-bound at 100M–10B: compress to fit RAM, tune nprobe, rerank for recall.
- **DiskANN** — billion-scale on **one node** or hard cost limits: SSD-resident graph + compressed RAM vectors, accept SSD latency.
Justify by the binding constraint (RAM vs cost vs recall) and the data size per shard; mention **PQ + rerank** as the common accuracy recovery across all three.

**27. Real-time streaming upserts.**
Don't rebuild globally. New vectors go to the **mutable segment** (brute-forced, instantly searchable) → background **flush** to incremental immutable segments (HNSW supports incremental insert; IVF-PQ batches into new segments) → **compaction** periodically merges/rebuilds and reclaims tombstones. Deletes are tombstones until compaction. Retrain IVF centroids periodically as the distribution drifts. The segment architecture is precisely what lets a static-friendly index absorb a live write stream.

**28. Index ↔ payload consistency.**
A search must never return an id whose payload was already deleted, so **commit the vector and its payload atomically** — same WAL record / transaction boundary — and apply deletes to **both** (tombstone the vector, mark the payload). Replicate via the same WAL so replicas stay consistent. Offer **read-your-writes** for clients that need it. Treat the segment as the unit of atomic visibility: a vector becomes searchable and its payload readable together, or neither.

---

## 🧮 Math & estimation

**29. RAM for 1B 768-dim vectors.**
$N\times d\times$ bytes:
- **fp32:** 768×4 = 3,072 B → **~3.0 TB**
- **fp16:** 768×2 = 1,536 B → **~1.5 TB**
- **PQ (m=96, 8-bit):** 96 B → **~96 GB**
fp32 needs **~50–100 nodes** (or sharding) just for vectors; **PQ brings it to a few nodes**. Add ~+50–100% for an HNSW graph, or just centroids for IVF. This estimate is the first thing to say in the interview — it dictates shard/quantize/tier.

**30. PQ compression & codebook size.**
With $m$ subvectors and $b$ bits, each vector = **$m\cdot b/8$ bytes** (e.g. $m=96, b=8$ → 96 B vs 3072 B = **32×**). The **codebooks** are tiny and shared: $m$ subspaces × $2^b$ centroids × ($d/m$) floats × 4 B = for $m=96,b=8,d=768$ → $96\times256\times8\times4 \approx$ **0.75 MB total** — negligible next to 96 GB of codes. So PQ's overhead is essentially free; the savings are all in the per-vector codes.

**31. IVF distance computations.**
With `nlist` cells and `nprobe` probed, you scan ≈ **`nprobe` × $N/\text{nlist}$** vectors (plus `nlist` centroid comparisons to choose cells). Example: $N$=1B, nlist=65,536, nprobe=64 → centroid step ~65K, then ~$64\times(10^9/65536)\approx$ **~1M candidate scans** instead of 1B — a **~1000×** reduction. Raising nprobe scans more cells → higher recall, higher cost. (With PQ, each scan is a cheap ADC table sum, not a full distance.)

**32. HNSW memory.**
Vectors: $N\times d\times$ bytes (e.g. fp16 → 1.5 TB for 1B×768). **Graph links:** ~$M$ neighbor ids per node at L0 (plus sparse upper layers), each id ~4–8 B → ≈ $N\times M\times$ 4–8 B. For $M$=32, 1B nodes → ~**128–256 GB** of links on top of the vectors (~+10–17% here, larger for small vectors). So HNSW ≈ **vectors + ~$NM$ link bytes** — and the vectors usually dominate, which is why HNSW+PQ exists.

**33. PQ's fast-distance trick (ADC).**
**Asymmetric Distance Computation.** Precompute, once per query, a table of distances from each **query subvector** to all $2^b$ sub-centroids in its subspace → an **$m\times2^b$** table. Then any database vector's approximate distance = **$\sum_{i=1}^{m}$ table[$i$][code$_i$]** — $m$ lookups + adds, no multiplies over $d$ dims. This turns each candidate's distance into a handful of cache-friendly lookups, which is why IVF-PQ can scan millions of codes per query in milliseconds.

**34. Latency: in-memory HNSW vs DiskANN.**
**In-memory HNSW:** ~$O(\log N)$ hops × ~$M$·ef distance computes, all in RAM/cache → **sub-ms to low-ms**. **DiskANN:** the graph/full vectors live on **SSD**, so traversal incurs **SSD reads** (~50–100 µs each) — it minimizes them by keeping **compressed vectors in RAM** to guide the walk and only fetching a few full vectors from disk, landing at **a few ms**. Trade: DiskANN is ~10× slower than pure RAM but fits **billions on one node** at a fraction of the RAM cost.

---

## 🏗️ Design variations

**35. DiskANN-style disk-based ANN.**
Keep a **single navigable graph on SSD** (Vamana-style, built for short paths to bound disk reads) with **PQ-compressed vectors in RAM** to guide traversal cheaply; fetch only a handful of **full-precision vectors from SSD** to rerank the final candidates. Layout the graph so each hop is a contiguous SSD read; cache hot nodes in RAM. Result: **billion-scale on one machine** at ~a few ms, trading SSD latency for ~10× less RAM. Build is heavy (done offline, parallelized).

**36. Multi-tenant collections.**
Per-tenant **collections/namespaces** with their own schema and index params. **Big tenants → dedicated shards** (isolation, independent scale); **small tenants → shared shards** with a tenant-id **filter/namespace** (watch noisy neighbors). Enforce **per-tenant quotas** (vector count, QPS, memory) and fair scheduling so one big scan can't starve others. Encrypt at rest; authz per collection. Route via the shard map in metadata. Scale-to-zero idle tenants to object storage.

**37. Vector DB behind the RAG platform — the contract.**
The [RAG platform](../rag-platform/README.md) owns chunking, embedding, reranking, and ACLs; the vector DB owns **store + ANN search + filtered retrieval + freshness/deletes**. Contract: RAG **upserts** `{id, vector, payload(incl. ACL + tenant + source)}`, and **searches** with `{query_vector, k, filter=ACL∧tenant, ef}`; the DB returns ids+distances+payload with the **filter enforced in-engine** (never trust the caller). The DB guarantees fast filtered ANN, durable upserts/deletes (right-to-be-forgotten), and tenant isolation; RAG guarantees good embeddings and rerank.

**38. Snapshot / backup / restore.**
Because segments are **immutable**, a snapshot = a **consistent set of segment files + the WAL offset + the shard map/metadata**, shipped to **object storage**. Backup = flush mutable segments, record the manifest, copy (incremental — only new segments). Restore/rebuild a node/shard = pull its segments + metadata, **replay the WAL tail** for anything after the snapshot. Snapshots are also the unit for **rebalancing** (move a shard) and point-in-time recovery. Test restores regularly.

---

## 🐞 Debugging & ops

**39. Periodic p99 spikes, p50 fine.**
Classic **background-work interference**: **compaction / segment merges / index rebuilds / GC** competing for CPU/IO, or a **scatter-gather tail** where one slow shard sets p99. Diagnose by correlating spikes with compaction/GC timing and per-shard tail latency. Fix: **throttle/schedule compaction** off-peak, isolate it (IO/CPU limits), add **hedged requests** (ask a replica, take the first answer) to cut tail, and balance shards so no single one lags.

**40. Recall dropped after mass deletes.**
Deletes are **tombstones**, not removals — the HNSW graph still routes **through dead nodes**, wasting ef budget and breaking paths, so effective recall falls and memory stays high. Recovery: **trigger compaction/rebuild** to physically drop tombstoned vectors and re-link the graph; going forward, **schedule compaction by tombstone ratio** (e.g. rebuild a segment past ~20% deleted) rather than purely by size. Temporarily raising ef can mask it until compaction runs.

**41. Just-inserted vectors are missed.**
The vector is durable (WAL) but **not yet in the built immutable index**, and the **mutable-segment** brute-force search either isn't wired into the query path or the write is still buffered/replicating. Fix: ensure every query **also scans the mutable segment**, flush/commit promptly, and offer **read-your-writes** (wait-for-flush) for clients that need immediate visibility. On replicas, account for WAL-shipping lag. This is the freshness-vs-static-index tradeoff showing up as a bug.

**42. OOM during index build.**
Building an ANN index is **memory-heavy** (HNSW holds vectors + the growing graph; IVF-PQ holds training data + codebooks). Triage: build **per-segment** (bound each build's footprint) and stream rather than loading all vectors at once; **lower `M`/`ef_construction`**; **quantize earlier** (build on PQ/int8, rerank later); **shard more** so each node builds a smaller slice; train IVF centroids on a **sample**, not the full set. Then resume the build from the last completed segment.

---

## What strong answers share
- **Memory estimate first:** $N\times d\times$ bytes → **1B×768 fp32 ≈ 3 TB**, vectors dominate → the whole design is shard/quantize/tier.
- **ANN families cold:** **HNSW** (graph, best in-RAM, heavy memory, tombstone deletes), **IVF** (k-means cells + `nprobe`), **PQ** (codebooks + **ADC lookup tables**, ~32×), **IVF-PQ / DiskANN** for billion-scale — and *when* to pick each.
- **The math:** HNSW ~$O(\log N)$ with **ef** as the recall dial; IVF scans ~`nprobe`·$N/\text{nlist}$; PQ = $m$ subvector codes + table-lookup distances; **compressed search + full-precision rerank** recovers recall.
- **Storage-engine mindset:** **WAL + immutable segments + compaction**, **deletes = tombstone-then-rebuild**, freshness via a **mutable segment**, atomic vector+payload commits.
- **Filtered search is first-class:** post vs pre vs **in-filter**, selectivity-aware planning — don't let a selective filter collapse recall.
- **Always pair recall with latency/memory** and name the real failure modes: **p99 spikes from compaction**, **scatter-gather tail**, **delete-driven recall decay**, **centroid drift**.

---

[← Back to vector DB HLD](README.md) · [Questions](questions.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
