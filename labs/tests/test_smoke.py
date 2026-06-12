"""Smoke tests — fast, no heavy deps, no unimplemented TODOs required.

These run in CI and should be green on a fresh clone. They check that modules import
and that forward-only paths and pure functions work, without calling any code that
depends on the learning exercises.
"""

from __future__ import annotations

import math

import pytest

from lab01_micrograd.engine import Value
from lab04_inference_bench.benchmark import RequestResult, summarize


def test_value_forward_ops():
    # forward-only: does not require the TODO gradients
    assert (Value(2.0) + Value(3.0)).data == pytest.approx(5.0)
    assert (Value(2.0) * Value(3.0)).data == pytest.approx(6.0)
    assert (Value(2.0) ** 3).data == pytest.approx(8.0)
    assert Value(-1.0).relu().data == pytest.approx(0.0)
    assert Value(2.0).relu().data == pytest.approx(2.0)
    assert Value(0.0).tanh().data == pytest.approx(0.0)
    assert Value(0.0).exp().data == pytest.approx(1.0)


def test_summarize_basic():
    results = [
        RequestResult(ok=True, e2e=0.10, ttft=0.02, tpot=0.001, output_tokens=80),
        RequestResult(ok=True, e2e=0.20, ttft=0.04, tpot=0.002, output_tokens=120),
        RequestResult(ok=False, e2e=0.05, error="boom"),
    ]
    m = summarize(results, wall_time=1.0)
    assert m["requests"] == 3
    assert m["ok"] == 2
    assert m["failed"] == 1
    assert m["output_token_throughput_tps"] == pytest.approx(200.0)
    assert m["request_throughput_rps"] == pytest.approx(2.0)
    # p50 of [20ms, 40ms] with interpolation = 30ms
    assert m["ttft_p50_ms"] == pytest.approx(30.0)
    assert not math.isnan(m["e2e_p95_ms"])


def test_gpt_constructs():
    # only runs locally where torch is installed; skipped in lightweight CI
    pytest.importorskip("torch")
    from lab02_nanogpt.config import GPTConfig
    from lab02_nanogpt.model import GPT

    model = GPT(GPTConfig(block_size=16, vocab_size=32, n_layer=2, n_head=2, n_embd=32))
    assert model.num_params() > 0


def test_lab06_provided_glue():
    # provided pipeline pieces — no retrieval TODOs required
    import numpy as np

    from lab06_rag.pipeline import hash_embed, lexical_rank, reciprocal_rank_fusion

    # hashing embedder is deterministic across calls (crc32, not salted hash())
    assert np.allclose(hash_embed("the kv cache"), hash_embed("the kv cache"))
    # RRF fuses two rankings into one covering all items
    assert set(reciprocal_rank_fusion([[0, 1, 2], [2, 1, 0]])) == {0, 1, 2}
    # lexical ranker puts the overlapping chunk first
    assert lexical_rank("kv cache", ["the kv cache stores values", "unrelated text"])[0] == 0


def test_lab07_provided_tools():
    # provided tools + brain — no agent-loop TODOs required
    from lab07_agent.brain import mock_brain
    from lab07_agent.tools import default_tools

    tools = default_tools()
    assert tools["calculator"]("12 * (3 + 4)") == "84"
    assert tools["calculator"]("rm -rf /").startswith("ERROR")
    assert tools["lookup"]("explain the kv cache").startswith("The KV cache")
    out = mock_brain([{"role": "user", "content": "what is 2 + 2?"}])
    assert "Action: calculator" in out
