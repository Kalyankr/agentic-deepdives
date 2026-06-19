"""Stage 1 — Lab D: train the GPT on TinyShakespeare.

This training plumbing is COMPLETE so you can focus on the model internals.
It will run once you've implemented attention.py and model.py.

Uses a simple CHAR-level vocab by default (no dependency on bpe.py) so you can
validate the model immediately. Once Lab A is done, swap in your BPETokenizer.

Run:
    uv run python prepare_data.py    # once
    uv run python train.py
"""
import os

import torch

from config import GPTConfig
from model import GPT

# ---------------- hyperparameters ----------------
batch_size = 32
block_size = 128
max_iters = 3000
eval_interval = 250
eval_iters = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

here = os.path.dirname(os.path.abspath(__file__))

# ---------------- data (char-level) ----------------
with open(os.path.join(here, "data", "input.txt"), "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]          # noqa: E731
decode = lambda ids: "".join(itos[i] for i in ids)  # noqa: E731

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )
    model = GPT(cfg).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | device={device}")

    optimizer = model.configure_optimizers(lr=learning_rate)
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    ckpt_dir = os.path.join(here, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "gpt.pt"))
    print("saved checkpoint -> checkpoints/gpt.pt")


if __name__ == "__main__":
    main()
