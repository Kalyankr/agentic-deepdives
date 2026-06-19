# Stage 1 — Code skeleton

Runnable scaffolding for the Stage 1 labs. The **plumbing is done**; the
**conceptual core is left as guided `TODO`s** (marked `TODO(stage1-labX)`),
because implementing those parts is the whole point. Don't look up solutions —
derive them. Search the TODOs with: `grep -rn "TODO(stage1" .`

## File → lab map

| File | Lab | You implement |
|------|-----|---------------|
| [bpe.py](bpe.py) | A | `get_stats`, `merge`, `BPETokenizer.{train,encode,decode}` |
| [attention.py](attention.py) | B | `scaled_dot_product_attention`, `CausalSelfAttention.forward` |
| [model.py](model.py) | C, D | `MLP.forward`, `Block.forward`, `GPT.forward`, `GPT.generate` |
| [config.py](config.py) | — | (complete) hyperparameter container |
| [prepare_data.py](prepare_data.py) | — | (complete) downloads TinyShakespeare |
| [train.py](train.py) | D | (complete) training loop — runs once model is done |
| [sample.py](sample.py) | D | (complete) generates from a checkpoint |

> `train.py`/`sample.py` use a simple **char-level** vocab so you can train the
> model *before* finishing the BPE lab. Once Lab A works, swap your
> `BPETokenizer` in as an exercise.

## Suggested order

1. **Lab A** — implement `bpe.py`, then `uv run python bpe.py` (expect `round-trip OK`).
2. **Lab B** — implement `attention.py`, then `uv run python attention.py` (expect `shape OK`).
3. **Lab C** — implement `MLP.forward` and `Block.forward` in `model.py`.
4. **Lab D** — implement `GPT.forward` + `GPT.generate`, then:
   ```bash
   uv run python prepare_data.py
   uv run python train.py      # val loss should fall well below 2.0
   uv run python sample.py     # should emit (vaguely) Shakespeare-ish text
   ```
5. **Lab E (ablations)** — copy `attention.py`/`model.py` and try: remove the
   `1/sqrt(hd)` scaling, drop positional embeddings, switch pre-norm → post-norm.
   Record what breaks in [../experiment-log.md](../experiment-log.md).
6. **Lab F (stretch)** — replace learned `pos_emb` with RoPE.

## Setup (if not done yet)

```bash
# from the workspace root
uv add torch numpy
```

Everything runs CPU-only (TinyShakespeare + this tiny model trains fine on CPU).

## Done when

- `bpe.py` and `attention.py` self-tests pass.
- `train.py` drives val loss down and `sample.py` produces coherent-ish text.
- You can pass the Stage 1 **mastery checks** in [../README.md](../README.md) without notes.
