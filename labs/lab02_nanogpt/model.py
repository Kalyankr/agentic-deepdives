"""A GPT (decoder-only transformer) implemented from scratch in PyTorch.

Module 02 keystone. Everything is wired up for you **except the core attention
computation** — implement `CausalSelfAttention.forward` (look for `TODO`).

Once attention is correct:
    uv run pytest -m todo tests/test_lab02_nanogpt.py   # spec
    uv run python -m lab02_nanogpt.train                # train on TinyShakespeare
    uv run python -m lab02_nanogpt.sample --prompt "To be"

Stretch goals (see README): add a KV cache, swap LayerNorm→RMSNorm, add RoPE & GQA.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from lab02_nanogpt.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head *causal* self-attention.

    The projections, dropout, and a precomputed causal mask are provided.
    YOU implement the forward pass.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        # one fused linear that produces q, k, v together
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # causal mask: (1, 1, block_size, block_size) lower-triangular of ones
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) where C == n_embd
        #
        # TODO — implement causal self-attention:
        #   1. q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        #   2. reshape each to (B, n_head, T, head_dim)   (split C into heads, then transpose)
        #   3. attention scores:  (q @ k.transpose(-2, -1)) / sqrt(head_dim)   -> (B, nh, T, T)
        #   4. apply the causal mask: set scores to -inf where self.bias[:, :, :T, :T] == 0
        #      (use masked_fill with float('-inf'))
        #   5. softmax over the last dim, then self.attn_dropout
        #   6. y = att @ v                                -> (B, nh, T, head_dim)
        #   7. transpose/reshape back to (B, T, C)
        #   8. return self.resid_dropout(self.c_proj(y))
        #
        # Hint: once it works, try replacing steps 3–6 with a single call to
        #   F.scaled_dot_product_attention(q, k, v, is_causal=True) and compare speed.
        raise NotImplementedError("Implement causal self-attention — see the TODO above")


class MLP(nn.Module):
    """Position-wise feed-forward network with 4x expansion and GELU."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """A pre-norm transformer block: x = x + attn(ln1(x)); x = x + mlp(ln2(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: input embedding and output projection share weights
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Parameter count (excluding the position embedding, like nanoGPT)."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.transformer.wpe.weight.numel()

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"sequence length {t} > block_size"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        pos_emb = self.transformer.wpe(pos)  # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference: only compute logits for the final position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample `max_new_tokens` tokens given a context `idx`."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
