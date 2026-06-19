"""Stage 1 — Lab D: sample text from a trained checkpoint.

Run after train.py has produced checkpoints/gpt.pt:
    uv run python sample.py
"""
import os

import torch

from config import GPTConfig
from model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
here = os.path.dirname(os.path.abspath(__file__))

# rebuild the SAME char vocab used in train.py
with open(os.path.join(here, "data", "input.txt"), "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
itos = {i: c for i, c in enumerate(chars)}
decode = lambda ids: "".join(itos[i] for i in ids)  # noqa: E731


def main():
    cfg = GPTConfig(
        vocab_size=len(chars), block_size=128, n_layer=4, n_head=4, n_embd=128
    )
    model = GPT(cfg).to(device)
    ckpt = os.path.join(here, "checkpoints", "gpt.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)  # start token
    out = model.generate(idx, max_new_tokens=500, temperature=0.8, top_k=50)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
