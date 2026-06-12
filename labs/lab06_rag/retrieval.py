"""Vector retrieval core for Lab 06 — implement the `TODO`s.

This is the conceptual heart of RAG's "R": cosine similarity, exact kNN, an **IVF**
(inverted-file) approximate index, and the retrieval **metrics**. NumPy only — no
embedding model or network needed (the embedder lives in `pipeline.py`).

The index `build` steps (k-means + inverted lists) are provided. Your job is the
search math and the metrics — look for `raise NotImplementedError`.

Run the spec:  uv run pytest -m todo tests/test_lab06_rag.py
"""

from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize along an axis (provided helper)."""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query `(d,)` and each row of `matrix` `(N, d)` → `(N,)`.

    TODO:
      1. normalize `query` to unit length (use `normalize`).
      2. normalize each row of `matrix` to unit length.
      3. cosine = dot product of unit vectors → return `matrix_unit @ query_unit`.
    """
    raise NotImplementedError("Implement cosine similarity — see the TODO above")


class ExactIndex:
    """Brute-force exact nearest-neighbor search — the recall = 1.0 baseline (`O(N)`)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float64)  # (N, d)

    def search(self, query: np.ndarray, k: int = 5) -> list[int]:
        """Return the indices of the top-`k` most similar vectors, best first.

        TODO:
          1. sims = cosine_similarity(query, self.vectors)            # (N,)
          2. take the indices of the k largest sims. For O(N) selection use
             `np.argpartition(-sims, k-1)[:k]`, then sort those k by sim descending.
          3. return them as a Python list of ints.
        """
        raise NotImplementedError("Implement exact top-k search — see the TODO above")


class IVFIndex:
    """An **IVF** (inverted-file) approximate index — the core ANN idea.

    Vectors are clustered with k-means; at query time you only scan the `nprobe`
    nearest clusters instead of the whole database. More probes → higher recall but
    more work. With `nprobe == n_clusters` this is exact (recall 1.0).

    `build` (k-means + inverted lists) is provided. YOU implement `search`.
    """

    def __init__(self, vectors: np.ndarray, n_clusters: int = 16, seed: int = 0):
        self.vectors = np.asarray(vectors, dtype=np.float64)
        self.n_clusters = min(n_clusters, len(self.vectors))
        self.centroids, assignments = self._kmeans(self.vectors, self.n_clusters, seed)
        # inverted lists: cluster id -> indices of vectors assigned to it
        self.lists: dict[int, np.ndarray] = {
            c: np.where(assignments == c)[0] for c in range(self.n_clusters)
        }

    @staticmethod
    def _kmeans(x: np.ndarray, k: int, seed: int, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
        """Tiny cosine k-means (provided — does not depend on your TODOs)."""
        rng = np.random.default_rng(seed)
        centroids = x[rng.choice(len(x), k, replace=False)].copy()
        assign = np.zeros(len(x), dtype=int)
        xn = normalize(x)
        for _ in range(iters):
            new = (xn @ normalize(centroids).T).argmax(axis=1)
            if np.array_equal(new, assign):
                break
            assign = new
            for c in range(k):
                members = x[assign == c]
                if len(members):
                    centroids[c] = members.mean(axis=0)
        return centroids, assign

    def search(self, query: np.ndarray, k: int = 5, nprobe: int = 1) -> list[int]:
        """Approximate top-`k`: probe the `nprobe` nearest clusters, then exact-rank.

        TODO:
          1. score the query against `self.centroids` with `cosine_similarity`.
          2. pick the `nprobe` best cluster ids (largest centroid similarity).
          3. gather candidate vector indices from those clusters' inverted lists
             (`self.lists[c]`); concatenate them.
          4. exact-rank the candidates: compute cosine of the query to
             `self.vectors[candidates]`, take the top-k, and map back to the
             original indices. Return a list of ints (best first).
          Edge case: if there are no candidates, return [].
        """
        raise NotImplementedError("Implement IVF search — see the TODO above")


def recall_at_k(retrieved: list[int], relevant: set[int] | list[int], k: int) -> float:
    """Fraction of the relevant items that appear in the top-`k` retrieved.

    `retrieved` is a ranked list of indices; `relevant` is the ground-truth set.

    TODO: return |{top-k retrieved} ∩ relevant| / |relevant|  (0.0 if no relevant items).
    """
    raise NotImplementedError("Implement recall@k — see the TODO above")


def mrr(retrieved: list[int], relevant: set[int] | list[int]) -> float:
    """Reciprocal rank of the first relevant hit: `1 / rank` (1-based), else 0.0.

    TODO:
      1. scan `retrieved` in order; find the first index that is in `relevant`.
      2. return 1 / (its 1-based rank). If none is relevant, return 0.0.
    """
    raise NotImplementedError("Implement MRR — see the TODO above")
