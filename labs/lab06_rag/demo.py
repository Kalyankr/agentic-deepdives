"""End-to-end RAG demo over a tiny corpus — runs once you implement `retrieval.py`.

    uv run python -m lab06_rag.demo

Shows: embed a corpus → exact vs IVF retrieval → hybrid (dense + lexical) fusion →
grounded answer → retrieval metrics (recall@k, MRR).
"""

from __future__ import annotations

from lab06_rag.pipeline import (
    embed_corpus,
    hash_embed,
    lexical_rank,
    rag_prompt,
    reciprocal_rank_fusion,
    stub_generate,
)
from lab06_rag.retrieval import ExactIndex, IVFIndex, mrr, recall_at_k

CORPUS = [
    "The KV cache stores past keys and values so decoding does not recompute attention.",
    "PagedAttention manages the KV cache in fixed-size pages to avoid fragmentation.",
    "RAG retrieves relevant documents and grounds the model's answer in them with citations.",
    "DPO aligns a model from preference pairs without training a separate reward model.",
    "Tensor parallelism splits a single matmul across GPUs and is kept within a node.",
    "HNSW is a graph-based ANN index with high recall and low latency.",
    "Product quantization compresses vectors to cut memory, trading a little accuracy.",
    "Continuous batching swaps finished sequences out and new ones in each decode step.",
]


def main() -> None:
    dim = 256
    corpus_vecs = embed_corpus(CORPUS, dim)
    question = "how does the kv cache speed up decoding?"
    q_vec = hash_embed(question, dim)

    # 1) exact retrieval (the recall=1.0 baseline)
    exact = ExactIndex(corpus_vecs)
    dense = exact.search(q_vec, k=3)
    print("dense top-3:", dense)
    for i in dense:
        print(f"   [{i}] {CORPUS[i]}")

    # 2) approximate retrieval with IVF; nprobe = n_clusters reproduces exact
    ivf = IVFIndex(corpus_vecs, n_clusters=4, seed=0)
    approx = ivf.search(q_vec, k=3, nprobe=2)
    print("\nIVF (nprobe=2) top-3:", approx)

    # 3) hybrid: fuse dense + lexical rankings with RRF
    lexical = lexical_rank(question, CORPUS)
    fused = reciprocal_rank_fusion([dense, lexical])[:3]
    print("\nlexical:", lexical)
    print("hybrid (RRF) top-3:", fused)

    # 4) assemble a grounded prompt + (stub) answer
    contexts = [CORPUS[i] for i in fused]
    print("\n--- prompt ---")
    print(rag_prompt(question, contexts))
    print("\n--- answer ---")
    print(stub_generate(question, contexts))

    # 5) retrieval metrics against a labeled relevant set (chunks 0 and 1 are about KV cache)
    relevant = {0, 1}
    print("\nrecall@3 (dense):", recall_at_k(dense, relevant, k=3))
    print("MRR (dense)     :", round(mrr(dense, relevant), 3))


if __name__ == "__main__":
    main()
