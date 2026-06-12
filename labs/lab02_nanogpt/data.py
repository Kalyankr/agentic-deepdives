"""Character-level dataset utilities.

By default this trains on `data/input.txt` if present. To use TinyShakespeare:

    curl -o data/input.txt \\
      https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

If no file is found, a tiny built-in corpus is used so the pipeline runs offline.
"""

from __future__ import annotations

from pathlib import Path

import torch

DATA_PATH = Path(__file__).parent.parent / "data" / "input.txt"

_FALLBACK = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them.\n"
) * 200


def load_text() -> str:
    if DATA_PATH.exists():
        return DATA_PATH.read_text(encoding="utf-8")
    print(
        f"[data] {DATA_PATH} not found — using a tiny built-in corpus.\n"
        "       See data.py for how to download TinyShakespeare."
    )
    return _FALLBACK


class CharTokenizer:
    """Maps characters <-> integer ids."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = dict(enumerate(chars))

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (inputs, targets) where targets are inputs shifted by 1."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
