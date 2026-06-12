"""Generate text from a trained checkpoint.

uv run python -m lab02_nanogpt.sample --prompt "To be" --max-new-tokens 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lab02_nanogpt.model import GPT

CKPT_PATH = Path(__file__).parent.parent / "checkpoints" / "ckpt.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--ckpt", default=str(CKPT_PATH))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    config = ckpt["config"]
    stoi, itos = ckpt["stoi"], ckpt["itos"]

    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    start = args.prompt if all(c in stoi for c in args.prompt) else "\n"
    idx = torch.tensor([[stoi[c] for c in start]], dtype=torch.long, device=args.device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
