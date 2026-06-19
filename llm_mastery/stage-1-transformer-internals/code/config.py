"""Stage 1 — config.

Small hyperparameter container for the GPT. This file is COMPLETE (no TODOs):
its job is just to hold dimensions so the other modules stay clean.
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 65      # set from your tokenizer / dataset
    block_size: int = 128     # context length (T) — max tokens the model attends over
    n_layer: int = 4          # number of transformer blocks (depth)
    n_head: int = 4           # number of attention heads
    n_embd: int = 128         # model width (d_model). Must be divisible by n_head
    dropout: float = 0.0
    bias: bool = True         # use bias in Linears / LayerNorms
