# Lab 02 — nanoGPT

> Module: [02 · Transformer Internals](../../modules/02-transformer-internals.md)

Build a **GPT (decoder-only transformer)** from scratch and train it to generate text.

## Your task

Implement `CausalSelfAttention.forward` in [model.py](model.py) (look for the `TODO`).
Everything else — the block wiring, embeddings, weight tying, training loop, sampler — is done.

The steps are spelled out in the docstring:
1. project to q, k, v and split into heads
2. scaled dot-product scores `q @ kᵀ / √head_dim`
3. apply the **causal mask** (no attending to the future)
4. softmax → dropout
5. weighted sum with v, reassemble heads, output projection

## Check your work

```bash
uv sync --extra dev --extra nn          # installs torch
uv run pytest -m todo tests/test_lab02_nanogpt.py
```

The spec verifies output shape **and causality** (early positions must not change when a
future token is altered) — the property the mask guarantees.

## Train & sample

```bash
# (optional) real data:
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

uv run python -m lab02_nanogpt.train --max-iters 2000      # add --device cuda if you have a GPU
uv run python -m lab02_nanogpt.sample --prompt "To be"
```

## Stretch goals (the Module 02 capstone)

- **KV cache**: cache past keys/values in `generate` and measure the speedup vs. recompute.
- **RMSNorm** instead of LayerNorm.
- **RoPE** rotary position embeddings (drop the learned `wpe`).
- **GQA / MQA**: share K/V across heads; measure the KV-cache memory reduction.
- **SwiGLU** FFN instead of GELU-MLP.
- Derive your model's parameter count and KV-cache size; confirm against `model.num_params()`.
