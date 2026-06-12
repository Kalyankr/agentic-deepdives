# Lab 06 — RAG & Vector Search

> Module: [06 · RAG & Vector Databases](../../modules/06-rag-and-vector-databases.md)

Build the **retrieval core** behind RAG from scratch — cosine similarity, exact kNN, an **IVF**
approximate index, and the retrieval **metrics** — then watch a full pipeline (embed → retrieve →
hybrid fuse → grounded answer) run end to end. **NumPy only**, no model or network.

## What you implement

All `TODO`s live in [retrieval.py](retrieval.py):

| Function | Concept |
|----------|---------|
| `cosine_similarity` | the similarity metric (normalize → dot) |
| `ExactIndex.search` | brute-force top-k — the recall = 1.0 baseline |
| `IVFIndex.search` | approximate ANN: probe `nprobe` nearest clusters, then exact-rank |
| `recall_at_k` | retrieval quality — did we find the relevant chunks? |
| `mrr` | mean reciprocal rank of the first hit |

The embedder, chunking, RRF fusion, lexical ranker, and prompt assembly are provided in
[pipeline.py](pipeline.py); k-means + inverted lists for the IVF index are provided too.

## Run

```bash
# spec tests (fail until you implement the TODOs)
uv run pytest -m todo tests/test_lab06_rag.py

# the end-to-end demo (works once the TODOs pass)
uv run python -m lab06_rag.demo
```

You should see exact and IVF retrieval agree (when `nprobe` covers all clusters), a hybrid
(dense + lexical) ranking via RRF, a grounded prompt with citations, and recall@k / MRR scores.

## Why these pieces matter (interview-relevant)

- **Exact vs ANN:** exact search is `O(N)`; **IVF** scans only `nprobe` clusters — the fundamental
  recall–latency dial. With `nprobe = n_clusters` you get exact results back.
- **Hybrid + RRF:** dense retrieval misses exact tokens (IDs, names); a lexical ranker catches them.
  **Reciprocal Rank Fusion** merges the two using ranks (not incomparable scores).
- **Metrics first:** most RAG failures are *retrieval* failures — `recall@k` / `MRR` tell you whether
  the right chunk was even in the candidate set before you blame the generator.

## Stretch goals

- Add **Product Quantization** (compress vectors, approximate distances) and measure the
  recall/memory trade-off vs. the uncompressed index.
- Sweep `nprobe` and `n_clusters` and plot **recall@10 vs. work** (install the `viz` extra).
- Add a **cross-encoder-style reranker** (re-score the top-N) and measure the quality lift.
- Swap `hash_embed` for a real sentence-embedding model and `stub_generate` for a real LLM call.
- Add an `IndexHNSW` (graph-based) and compare recall/latency to IVF.
