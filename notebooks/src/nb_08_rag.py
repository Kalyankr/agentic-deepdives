"""Build NB08 — RAG & vector databases."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 08 · RAG & Vector Databases

> Module: **06 · RAG & Vector Databases**.

**Goal:** build retrieval-augmented generation and understand the **approximate nearest
neighbor (ANN)** search underneath. We implement embeddings, cosine search, an IVF-style index,
**reranking**, and **RAG evaluation** — all self-contained (no downloads).

### Learning objectives
1. Embed text and retrieve by **cosine similarity**.
2. Understand **HNSW / IVF / PQ** trade-offs (recall vs latency vs memory).
3. Assemble a RAG pipeline: chunk → retrieve → **rerank** → generate (grounded).
4. Measure retrieval quality (**recall@k, MRR**).
"""),
    md(r"""
## 1. Why RAG

An LLM's parametric memory is frozen, lossy, and un-citable. **RAG** injects fresh, private,
verifiable knowledge at inference time: embed a query, retrieve relevant chunks, and condition
generation on them. It reduces hallucination and enables citations — without retraining.

**RAG vs long-context vs fine-tuning:** RAG for large/changing knowledge bases & citations;
long-context for whole-doc reasoning; fine-tuning for behavior/format/skills.
"""),
    md(r"""
## 2. Embeddings & cosine similarity

An embedding maps text to a vector so that *semantically similar* text is *geometrically close*.
We use a tiny deterministic **hashing bag-of-words** embedding here (real systems use trained
bi-encoders like `bge`, `e5`, `gte`). Similarity is the **cosine** of the angle between vectors.
"""),
    code(r"""
import numpy as np, re
rng = np.random.default_rng(0)

DIM = 256
def embed(text):
    v = np.zeros(DIM)
    for tok in re.findall(r"[a-z]+", text.lower()):
        h = hash(tok) % DIM
        v[h] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v          # L2-normalize so dot product == cosine

def cosine(a, b):
    return float(a @ b)

print("sim(cat,kitten docs):", round(cosine(embed("the small cat"), embed("a tiny kitten cat")), 3))
print("sim(cat, finance)   :", round(cosine(embed("the small cat"), embed("interest rate policy")), 3))
"""),
    md(r"""
## 3. Brute-force search vs ANN

Exact nearest-neighbor is $O(N)$ per query — fine for thousands, too slow for billions. **ANN**
indexes trade a little recall for big speedups:

- **HNSW** — a navigable small-world **graph**; excellent recall/latency; knobs `M`, `efSearch`.
- **IVF** — **cluster** vectors, only search the nearest `nprobe` clusters.
- **PQ (Product Quantization)** — compress vectors into codes (e.g. 32× smaller) for memory;
  often combined as **IVF-PQ**. DiskANN pushes this to SSD for billion-scale.

Let's build an exact index and an IVF-style index and measure the recall/speed trade-off.
"""),
    code(r"""
# Build a corpus of random docs + an exact (brute force) and an IVF (clustered) index.
N = 4000
DB = rng.standard_normal((N, DIM))
DB /= np.linalg.norm(DB, axis=1, keepdims=True)

def exact_search(q, k=10):
    sims = DB @ q
    return set(np.argsort(sims)[::-1][:k])

# IVF: k-means-ish clustering (one pass), then search only the nearest `nprobe` centroids.
n_clusters = 64
cent_idx = rng.choice(N, n_clusters, replace=False)
centroids = DB[cent_idx].copy()
assign = np.argmax(DB @ centroids.T, axis=1)            # nearest centroid per doc
buckets = {c: np.where(assign == c)[0] for c in range(n_clusters)}

def ivf_search(q, k=10, nprobe=4):
    near = np.argsort(centroids @ q)[::-1][:nprobe]
    cand = np.concatenate([buckets[c] for c in near]) if len(near) else np.array([], int)
    if len(cand) == 0: return set()
    sims = DB[cand] @ q
    return set(cand[np.argsort(sims)[::-1][:k]])

# recall@10 of IVF vs exact, averaged over queries, for several nprobe
queries = rng.standard_normal((200, DIM)); queries /= np.linalg.norm(queries, axis=1, keepdims=True)
for nprobe in [1, 4, 8, 16]:
    rec = np.mean([len(ivf_search(q, 10, nprobe) & exact_search(q, 10)) / 10 for q in queries])
    frac = nprobe / n_clusters
    print(f"nprobe={nprobe:2d}: recall@10={rec:.2f}, scans ~{frac:.0%} of the DB")
print("\n-> more probes = higher recall but more work. This is the ANN dial.")
"""),
    code(r"""
# Product Quantization idea: split each vector into m sub-vectors, replace each with the id of
# the nearest centroid in a small codebook -> store tiny integer codes instead of floats.
def pq_compress(vectors, m=8, ksub=256):
    d = vectors.shape[1]; sub = d // m
    codes = np.zeros((len(vectors), m), dtype=np.uint8)
    codebooks = []
    for j in range(m):
        part = vectors[:, j*sub:(j+1)*sub]
        cb = part[rng.choice(len(part), ksub, replace=False)]      # toy codebook
        codes[:, j] = np.argmin(((part[:, None, :] - cb[None])**2).sum(-1), axis=1)
        codebooks.append(cb)
    orig_bytes = vectors.nbytes
    comp_bytes = codes.nbytes
    return codes, comp_bytes, orig_bytes

_, comp, orig = pq_compress(DB[:1000])
print(f"PQ memory: {comp/1e3:.0f} KB vs {orig/1e3:.0f} KB float -> {orig/comp:.0f}x smaller")
"""),
    md(r"""
## 4. The RAG pipeline

```
INDEX:    documents -> chunk -> embed -> vector index (+ keyword/BM25 index)
QUERY:    question  -> embed -> retrieve top-N -> RERANK to top-k -> assemble prompt -> LLM (cite)
```

Highest-leverage quality levers (in rough order): **reranking** (cross-encoder), **chunking**
strategy, **hybrid** (dense + BM25) retrieval, and query rewriting. Let's build a mini pipeline.
"""),
    code(r"""
corpus = [
    "The KV cache stores past keys and values so decoding does not recompute attention.",
    "PagedAttention manages the KV cache in fixed-size pages to avoid fragmentation.",
    "RoPE encodes positions by rotating query and key vectors.",
    "Chinchilla showed compute-optimal training uses about 20 tokens per parameter.",
    "LoRA adds a low-rank update to frozen weights for cheap fine-tuning.",
    "DPO aligns models from preference pairs without a separate reward model.",
    "Bananas are a good source of potassium.",   # distractor
]
index = np.stack([embed(c) for c in corpus])

def retrieve(query, n=3):
    sims = index @ embed(query)
    order = np.argsort(sims)[::-1][:n]
    return [(corpus[i], float(sims[i])) for i in order]

def rerank(query, candidates):
    # toy cross-encoder: reward exact token overlap (a real reranker scores the pair jointly)
    qset = set(re.findall(r"[a-z]+", query.lower()))
    scored = [(c, s + 0.1*len(qset & set(re.findall(r"[a-z]+", c.lower())))) for c, s in candidates]
    return sorted(scored, key=lambda x: -x[1])

def rag_answer(query):
    cands = retrieve(query, n=4)
    ranked = rerank(query, cands)
    context = "\n".join(f"[{i+1}] {c}" for i, (c, _) in enumerate(ranked[:2]))
    # In production the LLM writes the answer from `context` and cites [1],[2].
    return f"Q: {query}\nRetrieved context:\n{context}\n\n(LLM would answer here, grounded in [1],[2].)"

print(rag_answer("how does the kv cache speed up decoding?"))
"""),
    md(r"""
## 5. Evaluating retrieval — recall@k and MRR

You can't improve what you don't measure. Build a small **golden set** of
`(query, relevant_doc_id)` and track:
- **recall@k** — was a relevant doc in the top k?
- **MRR** — 1/rank of the first relevant doc (rewards ranking it high).
End-to-end you also measure **faithfulness/groundedness** and **answer relevance** (NB11).
"""),
    code(r"""
gold = [("kv cache decoding", 0), ("paged attention fragmentation", 1),
        ("rotary position embeddings", 2), ("compute optimal tokens", 3),
        ("low rank fine tuning", 4), ("preference pairs alignment", 5)]

def recall_at_k(k=3):
    hits = 0
    for q, rel in gold:
        got = [corpus.index(c) for c, _ in retrieve(q, n=k)]
        hits += rel in got
    return hits / len(gold)

def mrr():
    total = 0.0
    for q, rel in gold:
        got = [corpus.index(c) for c, _ in retrieve(q, n=len(corpus))]
        rank = got.index(rel) + 1
        total += 1.0 / rank
    return total / len(gold)

print(f"recall@1 = {recall_at_k(1):.2f}")
print(f"recall@3 = {recall_at_k(3):.2f}")
print(f"MRR      = {mrr():.2f}")
"""),
    md(r"""
## 6. Advanced patterns & production
- **Hybrid search** (dense + BM25) fused with **Reciprocal Rank Fusion**.
- **Query rewriting / HyDE**, multi-query, RAG-fusion.
- **Contextual retrieval** (Anthropic): prepend a short doc-level context to each chunk before
  embedding — big recall gains.
- **GraphRAG** over entity/relationship graphs for multi-hop questions.
- Production: incremental indexing, freshness/deletes, re-embedding on model upgrades,
  sharding/replication, metadata filtering, "lost in the middle" context ordering.

## Exercises
1. Swap the toy embedder for `sentence-transformers`; rebuild the index with **FAISS** (`IndexHNSWFlat`, `IVFPQ`).
2. Plot recall@10 vs latency vs memory across FAISS index types on 100k+ vectors.
3. Add **hybrid** retrieval (BM25 + dense + RRF) and measure the recall lift.
4. Build a **RAGAS**-style faithfulness check with an LLM judge over your answers.

## Resources
- *RAG* (Lewis 2020); *Dense Passage Retrieval* (Karpukhin 2020).
- *HNSW* (Malkov 2016); *Product Quantization* (Jégou 2011); FAISS docs.
- *Lost in the Middle* (Liu 2023); *Contextual Retrieval* (Anthropic 2024); RAGAS docs.
"""),
]

if __name__ == "__main__":
    write(cells, "08_rag_and_vector_databases.ipynb")
