"""Stage 1 — Lab B: self-attention.

You implement the core attention math and the multi-head wiring. The module
shapes and projections are provided so you can focus on the mechanism.

Shapes convention:
    B  = batch
    T  = sequence length (tokens)
    C  = n_embd (model width)
    nh = n_head
    hd = head dim = C // nh
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: F401  (you'll use F.softmax)


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """Core attention. q, k, v have shape (B, nh, T, hd). Returns (B, nh, T, hd).

    TODO(stage1-labB):
      1. scores = (q @ k.transpose(-2, -1)) / sqrt(hd)        # (B, nh, T, T)
      2. if mask is not None: scores = scores.masked_fill(mask == 0, float('-inf'))
      3. attn = F.softmax(scores, dim=-1)                     # over the KEY axis
      4. if dropout is not None: attn = dropout(attn)
      5. return attn @ v                                      # (B, nh, T, hd)

    Whiteboard check: why divide by sqrt(hd)? (saturating softmax / vanishing grads)
    """
    raise NotImplementedError("Implement scaled_dot_product_attention (Lab B)")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # one projection that produces q, k, v stacked, then an output projection
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # lower-triangular causal mask, shape (1, 1, block, block)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        # TODO(stage1-labB):
        #   1. qkv = self.c_attn(x); split into q, k, v each of shape (B, T, C)
        #   2. reshape each to (B, nh, T, hd) with hd = C // self.n_head
        #        e.g. q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        #   3. y = scaled_dot_product_attention(
        #            q, k, v,
        #            mask=self.mask[:, :, :T, :T],
        #            dropout=self.attn_dropout)
        #   4. reshape y back to (B, T, C):  y.transpose(1, 2).contiguous().view(B, T, C)
        #   5. return self.resid_dropout(self.c_proj(y))
        raise NotImplementedError("Implement CausalSelfAttention.forward (Lab B)")


if __name__ == "__main__":
    # tiny shape smoke-test (works once implemented)
    from config import GPTConfig

    cfg = GPTConfig(block_size=16, n_embd=32, n_head=4)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 16, 32)  # (B, T, C)
    y = attn(x)
    assert y.shape == x.shape, y.shape
    print("attention shape OK ✓", tuple(y.shape))
