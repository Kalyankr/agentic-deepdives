"""RAG pipeline glue for Lab 06 — fully provided.

A deterministic hashing embedder (no model/network), chunking, hybrid fusion (RRF),
a tiny lexical ranker, and grounded-prompt assembly. These exercise the retrieval
core you implement in `retrieval.py`. In production you'd swap `hash_embed` for a
real embedding model and the prompt for a real LLM call.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic bag-of-words **hashing** embedder.

    Each token is hashed (crc32, so it's stable across runs/processes — unlike
    Python's salted `hash`) into one of `dim` buckets; the L2-normalized bucket
    counts are the embedding. Crude (no semantics) but reproducible and dependency
    free — enough to exercise retrieval. Replace with a real model in production.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for tok in _tokens(text):
        vec[zlib.crc32(tok.encode()) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def chunk_text(text: str, size: int = 160, overlap: int = 40) -> list[str]:
    """Split text into overlapping fixed-size windows (simple, structure-agnostic)."""
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, max(1, len(text)), step)]


def embed_corpus(chunks: list[str], dim: int = 256) -> np.ndarray:
    """Embed a list of chunks into an `(N, dim)` matrix."""
    return np.vstack([hash_embed(c, dim) for c in chunks])


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse several ranked lists (e.g. dense + lexical) into one.

    RRF score for an item = Σ 1 / (k + rank_in_list). Robust because it uses ranks,
    not raw (incomparable) scores. Returns indices sorted by fused score, best first.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return [idx for idx, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def lexical_rank(query: str, chunks: list[str]) -> list[int]:
    """A tiny BM25-lite lexical ranker (token overlap) to demonstrate hybrid search."""
    q = set(_tokens(query))
    scored = [(i, sum(t in q for t in _tokens(c))) for i, c in enumerate(chunks)]
    return [i for i, s in sorted(scored, key=lambda kv: -kv[1]) if s > 0]


def rag_prompt(question: str, contexts: list[str]) -> str:
    """Assemble a grounded prompt with numbered citations (the 'A' of RAG)."""
    ctx = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using ONLY the context below. Cite sources like [1].\n"
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def stub_generate(question: str, contexts: list[str]) -> str:
    """A stand-in for the LLM call: returns the top context as the 'grounded answer'.

    Replace with a real model in production; here it keeps the demo offline and
    deterministic while showing where generation plugs in.
    """
    if not contexts:
        return "I don't know — no relevant context was retrieved."
    return f"(grounded in [1]) {contexts[0]}"
