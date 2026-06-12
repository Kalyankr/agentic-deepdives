"""Build NB03 — Pretraining objectives & scaling laws."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 03 · Pretraining & Scaling Laws

> Module: **03 · LLM Training** (pretraining half).

**Goal:** understand what "pretraining a language model" actually optimizes, how data and
compute are budgeted, and the **scaling laws** that decide model/data size. You'll implement
the **cross-entropy / next-token** objective and compute training cost.

### Learning objectives
1. State the LM objective and connect cross-entropy ↔ **perplexity**.
2. Understand the data pipeline (dedup, filtering, packing, decontamination).
3. Apply **Chinchilla** compute-optimal scaling and the $C\approx 6ND$ rule.
4. Estimate the cost (FLOPs, GPU-hours) of a training run.
"""),
    md(r"""
## 1. The objective: next-token prediction

A base LLM is trained to predict the next token given all previous ones. For a sequence
$x_1,\dots,x_T$ the loss is the average **negative log-likelihood**:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})$$

This is exactly **cross-entropy** between the model's predicted distribution and the one-hot
true next token. Minimizing it = maximizing the likelihood of the training corpus.
"""),
    code(r"""
import numpy as np

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z); return e / e.sum(-1, keepdims=True)

def cross_entropy(logits, targets):
    # logits: (T, V), targets: (T,) int ids
    p = softmax(logits)
    T = len(targets)
    nll = -np.log(p[np.arange(T), targets] + 1e-12)
    return nll.mean()

rng = np.random.default_rng(0)
V, T = 1000, 16
logits = rng.standard_normal((T, V))
targets = rng.integers(0, V, size=T)
loss = cross_entropy(logits, targets)
print(f"random-init loss ~ {loss:.3f}  (sanity check: ln(V) = {np.log(V):.3f})")
# A freshly-initialized model should score ~ ln(V): it's guessing uniformly over the vocab.
"""),
    md(r"""
## 2. Perplexity — the loss in "effective vocabulary" units

$$\text{PPL} = e^{\mathcal{L}}$$

Perplexity is the exponentiated cross-entropy: roughly "how many tokens is the model choosing
between on average." Random init ≈ $V$; a good model drops it to a small number. It's the
classic intrinsic metric for language modeling.
"""),
    code(r"""
print(f"perplexity at init     ~ {np.exp(loss):.0f}  (~ vocab size, i.e. clueless)")
# Suppose after training the loss were 2.0:
print(f"perplexity at loss=2.0 = {np.exp(2.0):.1f}  (choosing among ~7 tokens on average)")
"""),
    md(r"""
## 3. The data pipeline (where most of the quality comes from)

Pretraining is *mostly a data problem*. The pipeline:

1. **Collect** — web (Common Crawl/FineWeb), code, books, papers, multilingual.
2. **Filter** — quality classifiers, language ID, dedup (MinHash/exact), remove boilerplate/NSFW.
3. **Decontaminate** — remove eval-benchmark text so scores aren't cheating.
4. **Mix** — choose ratios (e.g. more code improves reasoning); upsample high-quality sources.
5. **Tokenize & pack** — concatenate documents into fixed-length sequences (with separators)
   so no compute is wasted on padding.

> Rule of thumb learned the hard way: **better data beats more data**, and dedup matters a lot.
"""),
    code(r"""
# Sequence packing: glue short docs together to fill the context window (no wasted padding).
def pack(token_streams, block_size):
    flat = [t for doc in token_streams for t in (doc + [-1])]   # -1 = doc separator
    blocks = [flat[i:i+block_size] for i in range(0, len(flat) - block_size + 1, block_size)]
    return np.array(blocks)

docs = [list(range(5)), list(range(3)), list(range(8))]
blocks = pack(docs, block_size=4)
print("packed into blocks of 4:\n", blocks)
"""),
    md(r"""
## 4. Compute-optimal scaling — Chinchilla

How should you spend a fixed compute budget $C$? Kaplan (2020) found power-law scaling;
**Chinchilla** (2022) refined it: for compute-optimal training, **parameters and training
tokens should grow in roughly equal proportion** — about **20 tokens per parameter**.

The key cost identity:

$$C \approx 6\,N\,D$$

where $N$=params, $D$=training tokens, and 6 = (2 fwd + 4 bwd) FLOPs per param per token.
"""),
    code(r"""
def chinchilla(N_params, tokens_per_param=20):
    D = tokens_per_param * N_params
    C = 6 * N_params * D            # total training FLOPs
    return D, C

for N in [1e9, 7e9, 70e9]:
    D, C = chinchilla(N)
    print(f"{N/1e9:5.0f}B params -> {D/1e9:7.0f}B tokens, {C:.2e} FLOPs (compute-optimal)")
"""),
    md(r"""
## 5. From FLOPs to GPU-hours (capacity planning)

Wall-clock time depends on hardware throughput and how efficiently you use it (**MFU**,
model FLOPs utilization — real runs hit ~30–55%).

$$\text{GPU-seconds} = \frac{C}{\text{peak FLOP/s}\times \text{MFU}}$$
"""),
    code(r"""
def gpu_hours(C, peak_flops=9.89e14, mfu=0.4, n_gpus=1):
    # H100 ~ 989 TFLOP/s dense BF16 (with sparsity off). MFU ~0.4 is a decent real-world number.
    gpu_seconds = C / (peak_flops * mfu)
    return gpu_seconds / 3600 / n_gpus

_, C = chinchilla(7e9)
print(f"7B compute-optimal training:")
print(f"  total FLOPs   : {C:.2e}")
print(f"  on 1x H100    : {gpu_hours(C):,.0f} GPU-hours")
print(f"  on 256x H100  : {gpu_hours(C, n_gpus=256):,.1f} hours wall-clock (~{gpu_hours(C, n_gpus=256)/24:.1f} days)")
"""),
    md(r"""
## 6. Training stability (what goes wrong at scale)

- **Loss spikes:** sudden divergence; mitigations: LR warmup, gradient clipping, careful init,
  skipping/rewinding bad batches, **z-loss** (regularize the softmax normalizer).
- **LR schedule:** linear **warmup** (transformers are unstable early) → **cosine decay**.
- **Precision:** train in **bf16** (wide range, no loss-scaling needed) with fp32 master weights;
  watch for overflow in attention/softmax.
- **Mixed approaches:** gradient checkpointing (recompute activations) to fit memory — covered in NB07.

## 7. Emergence & inference-time scaling (frontier topics)
- Some capabilities appear abruptly with scale ("emergence") — debated, partly a metric artifact.
- **Inference-time scaling**: spend more compute *at test time* (longer chains of thought,
  best-of-N, search) to raise accuracy — the o1/R1 paradigm (more in NB05 & NB09).

## Exercises
1. Train the `nanoGPT` lab and plot **val loss vs. tokens seen**; estimate its scaling slope.
2. For a 1.4B-param budget, compute Chinchilla-optimal tokens and GPU-hours on A100 vs H100.
3. Implement **label smoothing** in the cross-entropy and observe the effect on loss/PPL.

## Resources
- *Scaling Laws for Neural LMs* (Kaplan 2020); *Training Compute-Optimal LLMs / Chinchilla* (Hoffmann 2022).
- *The Llama 3 Herd of Models* (data + training report); *FineWeb* dataset report.
- Karpathy `nanoGPT`; *The Ultra-Scale Playbook* (Hugging Face).
"""),
]

if __name__ == "__main__":
    write(cells, "03_pretraining_and_scaling_laws.ipynb")
