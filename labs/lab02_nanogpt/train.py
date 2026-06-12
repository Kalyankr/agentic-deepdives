"""Train the GPT on a character-level corpus.

    uv run python -m lab02_nanogpt.train               # small defaults (CPU-friendly)
    uv run python -m lab02_nanogpt.train --max-iters 5000 --device cuda

Saves a checkpoint to checkpoints/ckpt.pt for `sample.py` to load.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lab02_nanogpt.config import GPTConfig
from lab02_nanogpt.data import CharTokenizer, get_batch, load_text
from lab02_nanogpt.model import GPT

CKPT_PATH = Path(__file__).parent.parent / "checkpoints" / "ckpt.pt"


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, iters=20):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=tok.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(config).to(args.device)
    print(f"device={args.device}  params={model.num_params() / 1e6:.2f}M  vocab={tok.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            tr = estimate_loss(model, train_data, args.block_size, args.batch_size, args.device)
            va = estimate_loss(model, val_data, args.block_size, args.batch_size, args.device)
            print(f"iter {it:5d}  train {tr:.4f}  val {va:.4f}")

        x, y = get_batch(train_data, args.block_size, args.batch_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config, "stoi": tok.stoi, "itos": tok.itos},
        CKPT_PATH,
    )
    print(f"saved checkpoint -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
