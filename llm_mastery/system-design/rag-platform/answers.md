# RAG Platform — Answer Key

> Full worked answers to [questions.md](questions.md). The bar: **separate the ingestion and query planes**, treat retrieval as a **funnel** (recall → filter → rerank → grounded generation), **evaluate retrieval and generation independently**, and name the RAG-specific traps — **chunking, ACL-at-retrieval, freshness/deletes, and embedding-model upgrades**. Reference design: [README.md](README.md).
>
> Notation: $N$ = #chunks, $d$ = embedding dim, $k$ = top-k returned, $N_{rr}$ = rerank candidates, RRF = reciprocal rank fusion, ANN = approximate nearest neighbor.

---

## 🟢 Fundamentals

**1. What is RAG and what problem does it solve?**
Retrieval-Augmented Generation **fetches relevant context at query time and puts it in the prompt** so the LLM answers from that evidence instead of only its frozen parametric memory. It solves three problems at once: **stale knowledge** (the corpus updates without retraining), **hallucination** (answers are grounded in retrieved text with citations), and **access/attribution** (you can scope and cite sources). The cost is a retrieval system you must build, tune, and keep fresh.

**2. Why retrieve instead of fine-tuning the knowledge in?**
Fine-tuning bakes knowledge into weights: it's expensive to update, can't cite sources, blurs facts, and can't enforce per-document access control. Retrieval keeps knowledge **external, fresh, attributable, and access-controlled** — you add/delete a document in seconds, not a training run. Rule of thumb: **fine-tune to change behavior/format/skill; retrieve to supply knowledge.** They compose (a fine-tuned model that also retrieves).

**3. The two planes, and why separate them.**
The **ingestion plane** (parse → chunk → embed → index) is **throughput-oriented and asynchronous**; the **query plane** (rewrite → embed → retrieve → filter → rerank → generate) is **latency-critical and synchronous**. They have opposite SLOs and scaling profiles, so you decouple them: ingestion scales to the document change rate, query scales to QPS, and they meet only at the **shared index + metadata store**. Coupling them would let a big import stall live queries.

**4. What is chunking and why does it matter?**
Chunking splits documents into the retrievable, embeddable units. It's the **highest-leverage knob** because the chunk is both what you match *and* what you feed the model: too small fragments context and loses meaning; too large dilutes the embedding and wastes the context budget on noise. Good chunking is structure-aware (sections, paragraphs), carries metadata, and is tuned against the eval set — bad chunking caps quality no matter how good the rest is.

**5. Embeddings and semantic search.**
An **embedding** maps text to a vector such that semantically similar texts land near each other (by cosine/dot-product distance). **Semantic search** = embed the query, find the nearest chunk vectors → it matches on *meaning*, so "how do I reset my password" finds a doc titled "credential recovery" even with no shared keywords. The tradeoff: it can miss exact tokens (IDs, rare terms), which is why you pair it with sparse search.

**6. Dense vs sparse.**
**Dense** (embeddings) captures **semantics/paraphrase** but blurs exact strings. **Sparse** (BM25/keyword) nails **exact terms, rare words, codes, names, numbers** but misses synonyms. They're complementary, so production retrieval is **hybrid** — dense recovers BM25's vocabulary-mismatch misses; BM25 recovers dense's exact-match misses. Fuse their ranked lists (RRF) rather than picking one.

**7. Reranking.**
First-stage retrieval uses a **bi-encoder** (query and doc embedded separately) — fast and indexable but approximate. A **reranker** is a **cross-encoder** that scores query+document *jointly*, capturing fine-grained relevance the bi-encoder can't, applied to the top-$N$ candidates (e.g. 100 → 8). It gives a large precision lift for the price of GPU latency on $N$ pairs, so you cap $N$ and rerank only when quality needs it.

**8. Grounding, citations, abstention.**
**Grounding** = instruct the model to answer *only* from the retrieved context. **Citations** attach each claim to its source span so users can verify and you can measure faithfulness. **Abstention** = "I don't know / not in the sources" when support is missing, instead of confabulating. Together they convert the LLM from a confident guesser into an auditable, evidence-bound answerer — the core value proposition of RAG.

**9. Why ACLs at retrieval, not generation?**
Because the LLM will faithfully summarize **anything** placed in its context — it has no notion of who's allowed to see what. If an unauthorized document reaches the prompt, it can leak through the answer even if you "tell" the model to be careful. So **authorization must filter documents before they enter the context**, during retrieval — never rely on the generation step to withhold access. This is the single most important RAG security rule.

**10. Why an index, why not scan?**
At 1B+ chunks, brute-force exact nearest-neighbor is $O(N\cdot d)$ per query — billions of dot products, far too slow. An **ANN index** (HNSW, IVF-PQ) trades a little recall for **sub-linear** search, returning approximate top-k in milliseconds. The index is what makes semantic search over a huge corpus feasible at interactive latency; the recall/latency/memory tradeoff is then a tuning problem.

---

## 🟡 Core design

**11. Query lifecycle end to end.**
Query → **rewrite/expand** (resolve context, optional HyDE) → **embed** the query → **hybrid retrieve** (dense ANN + BM25, each scatter-gathered across shards) → **filter** by ACL + metadata (ideally pre-filtered inside ANN) → **fuse** (RRF) → **cross-encoder rerank** top-$N$ → assemble top-$k$ into a grounding prompt → **LLM generates** a cited answer with abstention → **verify citations**, stream back. Async: log the query + contexts + feedback for eval.

**12. Design the ingestion pipeline.**
Connector emits a doc/change event → durable **queue** → **parse** (extract clean text + structure + metadata, OCR if needed) → **chunk** (structure-aware + overlap) → **embed** in GPU batches → **upsert** vector + BM25 postings + chunk text + ACL **atomically and versioned** → raw doc to object storage. Make it **idempotent by content hash** (dedup), with retries/backoff and a **dead-letter queue** for poison docs, and per-tenant rate isolation so one import can't starve others.

**13. Choosing a chunking strategy.**
Start from the corpus structure and the eval set, not a fixed number. **Structure-aware** (by heading/section, code by function) preserves meaning; add ~10–20% **overlap** so answers spanning boundaries survive; use **parent-child** (embed small children for precise matching, return the parent for context) when units are short; consider **sentence-window** retrieval. Then **measure** recall@k across candidate sizes (256–512 tokens is a sane default) and pick empirically — chunking is tuned, not guessed.

**14. Design hybrid retrieval.**
Run **dense ANN** and **BM25** in parallel, each returning a ranked list, then fuse with **Reciprocal Rank Fusion** (rank-based, no score calibration needed). Apply ACL/metadata filters during/after retrieval, then **rerank** the fused candidates with a cross-encoder. Tune the candidate counts per retriever and the RRF constant; expose weights so corpora that are code-heavy (favor BM25) vs prose-heavy (favor dense) can be tuned.

**15. Enforce per-document ACLs efficiently.**
Store the **authorized principals/groups per chunk** as indexed metadata; the query carries the caller's identity/groups; the ANN search **pre-filters** to authorized docs during traversal (filtered ANN) so unauthorized vectors never surface. Avoid naive post-filtering (retrieve then drop): it **wrecks recall** (you may drop all of top-k) and can **leak existence** through result counts/latency. For coarse isolation, use per-tenant namespaces; for fine-grained, indexed ACL predicates.

**16. Keep the index fresh.**
Connectors stream **CDC** (create/update/delete) events; only changed docs are re-parsed/re-embedded and **upserted by stable chunk id**, retiring stale chunks. Track an embedding-model **version** per chunk. Watch **freshness lag** (ingest → searchable) as an SLO. Periodic **compaction** keeps incremental updates from fragmenting the ANN graph and degrading latency/recall.

**17. Deletions / right-to-be-forgotten.**
Support **tombstones** for *immediate* exclusion from results (filter out at query time the instant a delete arrives), backed by a **background physical purge** across *all* stores — vector index, BM25 postings, metadata/chunk store, caches, and raw object storage. Verify deletion completes within the compliance SLA and is auditable. Soft-delete-then-purge avoids expensive synchronous index rebuilds while still honoring the guarantee quickly.

**18. Multi-tenant isolation.**
Per-tenant **namespaces/logical indexes** so a tenant's query can only ever traverse its own vectors; dedicated physical indexes for large or regulated tenants. Enforce **per-tenant quotas** on ingest and query (noisy-neighbor protection), tenant-scoped encryption and optional **data residency**, and tenant-scoped caches. The invariant: **no query path can return another tenant's chunk**, enforced structurally, not just by a filter you might forget.

**19. Context assembly and grounding prompt.**
Take the reranked top-$k$, order by relevance, and pack to fit the model's budget (drop or summarize overflow); include a stable **source id** with each passage. The prompt instructs: *answer only from the context, cite the source for each claim, and abstain if unsupported.* Keep the **system/instruction prefix stable** for prefix-cache reuse. Optionally **verify** that cited spans actually support each claim and drop unsupported ones before returning.

**20. Evaluate end to end.**
**Separate the two stages.** Retrieval: recall@k, MRR, nDCG against a **golden query→relevant-doc** set. Generation: **faithfulness** (claims supported by context), answer relevance, and citation correctness via LLM-judge plus human spot-checks. Gate index/prompt/model changes with an **offline CI eval**, and watch **online** signals (thumbs, citation-click, abstention rate) with drift alerts. Per-tenant golden sets matter because quality is corpus-specific.

---

## 🔴 Senior / Staff deep dives

**21. Right doc exists but never appears — diagnose.**
Walk the funnel and localize: (a) **Indexed?** is the doc actually chunked/embedded and in the right namespace? (b) **Retrieved?** does it appear in the raw dense/BM25 candidates before filtering — if not, it's a **chunking/embedding/recall** problem (chunk too big/diluted, vocabulary mismatch → add HyDE/hybrid, raise k). (c) **Filtered out?** an over-broad ACL/metadata predicate dropped it (post-filter recall loss). (d) **Reranked down?** the cross-encoder mis-scored it. (e) **Ignored by the LLM?** it was in context but lost — context too long / poor ordering. Fix at the stage that drops it.

**22. Scale the index to 10B+ chunks.**
**Shard** vectors across nodes (queries scatter-gather → merge top-k) and **replicate** each shard for HA + read throughput. **Compress** with PQ/scalar quantization to fit memory; **tier** cold tenants to cheaper storage and scale-to-zero idle ones. Shard by tenant for isolation or by hash for balance, with **dedicated shards for whale tenants**. Keep metadata/ACL co-located for filtered search, and run **compaction** to control fragmentation. The index won't fit one box — distribution + compression are mandatory.

**23. Embedding-model upgrade with zero downtime.**
Changing the embedding model **invalidates every vector** (old and new vectors aren't comparable). Run a **dual-index migration**: keep serving the old index while a **background job re-embeds the whole corpus** into a new index, validate quality on the eval set, then **atomically swap** (or shadow/canary the new index for a fraction of traffic first). This is why chunks carry an **embedding-version**; without versioning you can't migrate safely. It's a major, planned operational event.

**24. Pre- vs post-filtering.**
**Post-filter** (retrieve top-k, then drop unauthorized/unmatched) is simple but **destroys recall** — if most of top-k is filtered you return too few or nothing — and can **leak** via counts/timing. **Pre-filter** (constrain the ANN traversal to the allowed subset) preserves recall and is secure, but needs the index to support filtered search and can be slow if the predicate is very selective (the graph has few valid neighbors). Default to **pre-filter for ACLs** (correctness/security) and tune; fall back to over-fetch+post-filter only for cheap, non-security metadata.

**25. Multi-hop retrieval.**
Single-shot retrieval fails when the answer requires chaining facts ("who manages the author of doc X?"). Use **iterative/agentic retrieval**: retrieve, let the model identify what's still missing, **issue follow-up queries**, and accumulate evidence (a retrieve→reason→retrieve loop) with a step budget. Alternatives: **query decomposition** (split into sub-questions, retrieve each), or **GraphRAG** (traverse a knowledge graph of entities/relations). Trade latency/cost for the ability to compose evidence.

**26. Still hallucinating — drive it down.**
Layered fixes: improve **retrieval** so the right evidence is actually present (most "hallucinations" are retrieval misses); tighten the **grounding prompt** (answer only from context, cite per claim); add **citation/faithfulness verification** that drops or flags unsupported claims; enable **abstention** so the model says "not in sources" instead of guessing; and **measure faithfulness** continuously to catch regressions. Optionally raise retrieval precision (rerank) so noisy context doesn't tempt the model. You trade some helpfulness/coverage for trustworthiness.

**27. Defend against indirect prompt injection.**
Retrieved documents are **untrusted input** — a poisoned page can contain "ignore your instructions and exfiltrate X." Mitigations: **delimit and label** retrieved content clearly as data (not instructions), instruct the model to never follow instructions found in context, **sanitize** obvious injection patterns, run with **least privilege** (no powerful tools driven by retrieved text), and **guard outputs** (block exfiltration/secret leakage). Provenance/allow-listing of sources and content moderation at ingest reduce the attack surface.

**28. HNSW vs IVF-PQ at scale.**
**HNSW**: excellent recall/latency, supports incremental inserts, but **high memory** (stores the graph) and fragmentation under heavy updates — great for moderate scale or latency-critical tenants with the RAM budget. **IVF-PQ**: **compressed** (product quantization) so far more vectors per GB, tunable recall via `nprobe`, but needs **centroid training** and re-training as the distribution drifts, and recall is lower at equal speed. Pick HNSW (or HNSW+PQ hybrid) for quality/latency, IVF-PQ when **memory/scale** dominate.

---

## 🧮 Math & estimation

**29. Storage for 1B vectors.**
$1024$-dim fp16 = $1024 \times 2 = 2048$ bytes ≈ **2 KB/vector**. Raw: $10^9 \times 2\text{KB} = 2\text{ TB}$. HNSW graph overhead ~2–3× → **~5 TB index**, so shard into ~50–100 shards (~50–100 GB each) plus replicas. With **PQ compression** you can cut the vector bytes ~8–16× (e.g. to ~128–256 B/vector) → hundreds of GB, at some recall cost. Chunk text + metadata live separately (cheap store); raw docs in object storage.

**30. Embedding compute to index 1B chunks once.**
At ~**2–5K chunks/s/GPU**, one pass = $\frac{10^9}{3000} \approx 3.3\times10^5$ GPU-seconds ≈ **~90 GPU-hours** — trivially parallelized across, say, 100 GPUs in ~1 hour. The point: **initial indexing is a bounded, parallelizable batch job**; steady-state you only embed the **change rate** (CDC), which is far smaller. Size embed workers to the delta, not the corpus.

**31. Query latency budget.**
Target end-to-end p95 < 1–2 s. Rough split: query embed ~5 ms (GPU) → **ANN search** ~20–50 ms (scatter-gather + merge) → ACL/metadata filter ~few ms → **cross-encoder rerank** top-100 ~50–100 ms (GPU) → **generation** seconds but **streamed** (first token in a few hundred ms). Retrieval+rerank should fit in **~150–200 ms** so the user perceives the LLM, not the search. Cache hits collapse the front of this budget.

**32. Reciprocal Rank Fusion.**
For each document $d$ ranked by multiple retrievers, $\text{RRF}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}$, where $\text{rank}_r(d)$ is $d$'s position in retriever $r$'s list and $k$ is a constant (commonly 60). It fuses lists using **ranks, not raw scores**, so you don't have to calibrate incomparable dense-distance vs BM25 scores; documents ranked high by multiple retrievers float to the top. Simple, robust, parameter-light.

**33. Sizing shards and replicas.**
**Shards** = total index size ÷ per-node memory budget (e.g. 5 TB ÷ ~64–100 GB usable per node → ~50–80 shards), adjusted so each shard's search stays within the latency target. **Replicas** = max(HA requirement, read-QPS ÷ per-replica QPS capacity). Scatter-gather latency is bounded by the **slowest shard**, so keep shards balanced and watch tail shards. Add headroom for growth and compaction.

**34. Cost per query and per doc.**
**Per ingested doc** ≈ parse + (chunks × embed cost) + index write + storage; dominated by **embedding** and amortized index memory. **Per query** ≈ query-embed + ANN search (CPU/RAM amortized) + **rerank** (GPU on $N$ pairs) + **generation** (LLM tokens — usually the largest term). Drive both down with **dedup/caching** (skip re-embeds and repeat queries), **quantized indexes** (memory), **rerank only when needed**, and **routing generation to a smaller model**. Track $/doc and $/query per tenant.

---

## 🏗️ Design variations

**35. GraphRAG / structured retrieval.**
Build a **knowledge graph** (entities + relations) from the corpus during ingestion, alongside vectors. At query time, retrieve relevant nodes, then **traverse edges** to gather connected context (good for multi-hop and "global" questions a flat top-k misses). Combine graph traversal with vector retrieval (hybrid): vectors find entry points, the graph supplies structured neighborhood and relationships. Costs: graph construction/maintenance and more complex retrieval logic.

**36. Multimodal RAG.**
Ingest images/tables/audio by either **captioning/OCR/transcribing** them into text for a text index, or using **multimodal embeddings** (shared text-image space) to retrieve raw modalities directly. Tables need structure-preserving parsing (don't flatten); scanned docs need OCR; charts may need vision models. Generation uses a **multimodal LLM** that can read the retrieved image/table alongside text. Evaluation must cover modality-specific retrieval quality.

**37. Agentic / iterative retrieval.**
Instead of one fixed retrieval, give the model a **retrieval tool** and let it decide *what* and *when* to fetch in a **retrieve→reason→retrieve** loop, refining queries until it has enough evidence (with a **step/cost budget** to bound it). It handles multi-hop, ambiguous, and exploratory queries far better than single-shot, at the cost of **higher latency and token spend** and harder evaluation. Add caching and early-stop heuristics to control cost.

**38. RAG cache with freshness + personalization.**
Cache at multiple layers: **embedding cache** (text→vector, safe to cache long), **retrieval cache** (query→top-k, TTL'd, **invalidated on index updates** to its docs), and **answer cache** (query→cited answer, scoped per tenant and per **ACL set** so personalization/permissions aren't leaked). Use **semantic** cache keys (similar queries hit) carefully — only when freshness and personalization allow. Bypass cache for queries whose docs recently changed.

---

## 🐞 Debugging & ops

**39. Retrieval p99 spiked after a big ingestion.**
Large ingest fragments the **HNSW graph** (or invalidates IVF centroids) and inflates a few shards, so scatter-gather waits on slow tail shards; background indexing also competes for CPU/IO with queries. Fixes: **rate-limit/offpeak** heavy ingest, isolate ingest from query resources, run **compaction/rebuild** to defragment, **rebalance** hot shards, and (IVF) **re-train centroids** after big distribution shifts. Monitor per-shard latency to spot the tail.

**40. Recall dropped after adding a metadata filter.**
You're almost certainly **post-filtering**: the ANN returns top-k by similarity, *then* the filter removes most of them, leaving few/none — the filter shrank the candidate pool after the fact. Fix by **pre-filtering inside the ANN** (constrain traversal to matching docs so you still get k *valid* results), or **over-fetch** (retrieve a much larger k before filtering) as a stopgap. Verify the filter predicate/metadata indexing is correct too.

**41. Cross-tenant document leak — triage.**
Treat as a **sev-1 isolation incident**: immediately reproduce and scope blast radius, then check where isolation failed — was the query missing the tenant namespace, an ACL predicate mis-applied/post-filtered, a shared cache returning another tenant's answer, or a mis-tagged chunk at ingest? Short-term: disable the affected path/cache and add a hard tenant guard. Long-term: enforce tenant scoping **structurally** (namespace per tenant) so it can't be bypassed by a forgotten filter, add tests, and audit logs.

**42. Faithfulness dropped after a prompt change — localize.**
Because retrieval is unchanged, suspect **generation**: the new prompt may have weakened the "answer only from context / cite / abstain" instructions, changed context ordering/truncation, or over-encouraged helpfulness. Use the **attribution split** — confirm the gold doc is still retrieved (it is, retrieval didn't change), then A/B the old vs new prompt on the faithfulness eval set to isolate the regression. Roll back or fix the specific instruction; add the prompt to the **CI eval gate** so it can't regress silently.

---

## What strong answers share
- **Separate the planes and stages:** async **ingestion** vs latency-critical **query**; retrieval is a **funnel** (recall → filter → rerank → grounded generation), evaluated **stage by stage**.
- **Name the RAG-specific traps:** chunking tradeoffs, **pre-filtered ANN for ACLs (authorize at retrieval, never at generation)**, **freshness/deletes**, and **embedding-model upgrades invalidating the index**.
- **Ground hard:** hybrid retrieval + rerank + citations + abstention + **faithfulness eval**; treat retrieved content as **untrusted** (indirect injection).
- **Quantify:** vector storage, embedding compute, latency budget, shard/replica sizing, and $/query vs $/doc.
- **Treat quality, latency, cost, and freshness as explicit, tunable tradeoffs** — per tenant and per corpus.

---
Back to [questions](questions.md) · [HLD](README.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
