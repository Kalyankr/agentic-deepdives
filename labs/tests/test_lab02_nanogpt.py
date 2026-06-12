"""Spec for Lab 02 — causal self-attention.

These FAIL until you implement `CausalSelfAttention.forward` in lab02_nanogpt/model.py.
Requires the `nn` extra (PyTorch):  uv sync --extra dev --extra nn

    uv run pytest -m todo tests/test_lab02_nanogpt.py
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.todo

torch = pytest.importorskip("torch")

from lab02_nanogpt.config import GPTConfig  # noqa: E402
from lab02_nanogpt.model import GPT, CausalSelfAttention  # noqa: E402


def _config() -> GPTConfig:
    return GPTConfig(block_size=16, vocab_size=32, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def test_attention_preserves_shape():
    cfg = _config()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 8, cfg.n_embd)
    y = attn(x)
    assert y.shape == x.shape


def test_forward_shapes_and_loss():
    cfg = _config()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_attention_is_causal():
    """Changing a future token must not affect earlier positions' outputs."""
    cfg = _config()
    model = GPT(cfg).eval()
    torch.manual_seed(0)

    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    targets = torch.zeros_like(idx)
    logits1, _ = model(idx, targets)

    idx2 = idx.clone()
    idx2[0, -1] = (idx[0, -1] + 1) % cfg.vocab_size  # perturb only the LAST token
    logits2, _ = model(idx2, targets)

    # all positions before the last must be unchanged (no peeking at the future)
    assert torch.allclose(logits1[:, :-1], logits2[:, :-1], atol=1e-5)
    # the last position should change (its own input changed)
    assert not torch.allclose(logits1[:, -1], logits2[:, -1], atol=1e-5)


def test_generate_length():
    cfg = _config()
    model = GPT(cfg).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 11)
