"""Model configuration for the GPT."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 128  # max context length
    vocab_size: int = 65  # set by the dataset (char-level default)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head
