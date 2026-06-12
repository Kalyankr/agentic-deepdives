"""Spec for Lab 06 — RAG retrieval core.

These FAIL until you implement the TODOs in lab06_rag/retrieval.py
(`cosine_similarity`, `ExactIndex.search`, `IVFIndex.search`, `recall_at_k`, `mrr`).

    uv run pytest -m todo tests/test_lab06_rag.py
"""

from __future__ import annotations

import numpy as np
import pytest

from lab06_rag.retrieval import ExactIndex, IVFIndex, cosine_similarity, mrr, recall_at_k

pytestmark = pytest.mark.todo


def test_cosine_similarity_values():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sims = cosine_similarity(np.array([1.0, 0.0]), matrix)
    assert sims[0] == pytest.approx(1.0)  # identical
    assert sims[1] == pytest.approx(0.0)  # orthogonal
    assert sims[2] == pytest.approx(1 / np.sqrt(2))  # 45 degrees


def test_exact_index_finds_planted_neighbor():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((40, 16))
    query = vecs[7] + 0.001 * rng.standard_normal(16)  # closest to row 7
    assert ExactIndex(vecs).search(query, k=1)[0] == 7


def test_exact_index_returns_k_sorted():
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((30, 8))
    out = ExactIndex(vecs).search(vecs[3], k=5)
    assert len(out) == 5
    assert out[0] == 3  # a vector is most similar to itself
    sims = cosine_similarity(vecs[3], vecs)
    assert [sims[i] for i in out] == sorted((sims[i] for i in out), reverse=True)


def test_ivf_full_probe_matches_exact():
    rng = np.random.default_rng(2)
    vecs = rng.standard_normal((60, 16))
    query = vecs[11] + 0.01 * rng.standard_normal(16)
    exact = ExactIndex(vecs).search(query, k=5)
    ivf = IVFIndex(vecs, n_clusters=6, seed=0)
    # probing every cluster reduces IVF to exact search
    assert set(ivf.search(query, k=5, nprobe=6)) == set(exact)


def test_ivf_single_probe_returns_k():
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((60, 16))
    ivf = IVFIndex(vecs, n_clusters=6, seed=0)
    out = ivf.search(vecs[20], k=5, nprobe=1)
    assert len(out) <= 5
    assert all(0 <= int(i) < len(vecs) for i in out)


def test_recall_at_k():
    retrieved = [3, 1, 4, 1, 5]
    relevant = {1, 4}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)  # top3=[3,1,4]
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(0.0)  # top1=[3]


def test_mrr():
    assert mrr([3, 1, 4], {4}) == pytest.approx(1 / 3)
    assert mrr([2, 4, 1], {4}) == pytest.approx(0.5)
    assert mrr([0, 1, 2], {9}) == pytest.approx(0.0)
