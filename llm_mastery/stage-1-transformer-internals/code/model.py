"""Stage 1 — Labs C & D: the MLP, the Block, and the full GPT.

Provided: module structure, embeddings, weight tying, init, optimizer helper.
You implement: the forward passes (the assembly) and generation.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import CausalSelfAttention


class MLP(nn.Module):
    """Per-token feed-forward: expand 4x, nonlinearity, project back."""

    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # TODO(stage1-labC): c_fc -> GELU (F.gelu) -> c_proj -> dropout
        raise NotImplementedError("Implement MLP.forward (Lab C)")


class Block(nn.Module):
    """One transformer block: PRE-NORM attention + MLP, each with a residual."""

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # TODO(stage1-labC): pre-norm residuals (note: norm is INSIDE the branch)
        #   x = x + self.attn(self.ln_1(x))
        #   x = x + self.mlp(self.ln_2(x))
        #   return x
        raise NotImplementedError("Implement Block.forward (Lab C)")


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying: share the token embedding with the output projection
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns (logits, loss).

        TODO(stage1-labD):
          1. B, T = idx.shape ; assert T <= self.cfg.block_size
          2. pos = torch.arange(T, device=idx.device)
             x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)
          3. for block in self.blocks: x = block(x)
             x = self.ln_f(x)
          4. logits = self.lm_head(x)                               # (B, T, vocab)
          5. loss = None
             if targets is not None:
                 loss = F.cross_entropy(
                     logits.view(-1, logits.size(-1)), targets.view(-1))
          6. return logits, loss
        """
        raise NotImplementedError("Implement GPT.forward (Lab D)")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend idx (B, T). Returns (B, T + max_new_tokens).

        TODO(stage1-labD) — this is the autoregressive decode loop:
          for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]      # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # last step only
            if top_k is not None:                         # optional top-k filter
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
          return idx
        """
        raise NotImplementedError("Implement GPT.generate (Lab D)")

    def configure_optimizers(self, lr, weight_decay=0.1):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
