"""Build NB04 — Fine-tuning: SFT & PEFT/LoRA."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 04 · Fine-tuning — SFT & Parameter-Efficient Tuning (LoRA/QLoRA)

> Module: **03 · LLM Training** (fine-tuning half).

**Goal:** turn a **base** model into an **instruction-following** assistant via Supervised
Fine-Tuning (SFT), and do it cheaply with **LoRA/QLoRA**. You'll implement chat-template
**loss masking** and the **LoRA** low-rank update in NumPy.

### Learning objectives
1. Format chat data and **mask the loss to assistant tokens only**.
2. Understand the LoRA math $W' = W + \tfrac{\alpha}{r}BA$ and why it works.
3. Know when to use full FT vs LoRA vs QLoRA (memory/quality trade-offs).
"""),
    md(r"""
## 1. Base vs. instruct models

A **base** model just continues text (great at completion, bad at following instructions).
**SFT** trains it on `(prompt, ideal response)` pairs so it learns the *assistant* behavior:
answer questions, follow formats, use a system prompt. SFT is the first post-training stage
before alignment (RLHF/DPO, next notebook).
"""),
    md(r"""
## 2. Chat templates & loss masking

Chat data is rendered into a single token stream using a **chat template** with role markers
(system / user / assistant). Crucially, we compute the loss **only on the assistant's tokens** —
the model shouldn't be trained to *generate* the user's prompt, only to *respond* to it.
We implement this with a label mask (set non-assistant labels to `-100` / ignore).
"""),
    code(r"""
import numpy as np

# A toy "chat template" -> token ids, plus a mask marking which tokens are assistant tokens.
def render_and_mask(turns):
    # turns: list of (role, text). We fake tokenization as word ids for clarity.
    ids, train_mask = [], []
    vocab = {}
    def tok(s):
        out = []
        for w in s.split():
            vocab.setdefault(w, len(vocab))
            out.append(vocab[w])
        return out
    for role, text in turns:
        marker = tok(f"<|{role}|>")
        body = tok(text)
        ids += marker + body
        # train ONLY on assistant body tokens (not markers, not user/system)
        train_mask += [0]*len(marker) + ([1]*len(body) if role == "assistant" else [0]*len(body))
    return np.array(ids), np.array(train_mask)

ids, mask = render_and_mask([
    ("system", "You are helpful."),
    ("user", "what is 2 plus 2 ?"),
    ("assistant", "2 plus 2 is 4 ."),
])
print("tokens:", ids)
print("loss mask (1 = train on this token):", mask)
print("-> we backprop loss on", int(mask.sum()), "assistant tokens only")
"""),
    code(r"""
# Masked cross-entropy: ignore non-assistant positions (label = -100 convention)
def masked_ce(logits, targets, mask):
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    T = len(targets)
    nll = -logp[np.arange(T), targets]
    return (nll * mask).sum() / max(mask.sum(), 1)

rng = np.random.default_rng(0)
Vsz = 50
logits = rng.standard_normal((len(ids), Vsz))
loss = masked_ce(logits, ids, mask)
print(f"masked SFT loss = {loss:.3f}  (computed over assistant tokens only)")
"""),
    md(r"""
## 3. LoRA — Low-Rank Adaptation

Full fine-tuning updates *all* weights (huge memory: weights + grads + optimizer states).
**LoRA** freezes $W$ and learns a tiny low-rank update:

$$W' = W + \frac{\alpha}{r} B A,\quad A\in\mathbb{R}^{r\times d},\ B\in\mathbb{R}^{d\times r},\ r\ll d$$

Only $A,B$ train. For $d=4096, r=8$ that's ~**0.4%** of the parameters — yet it recovers most
of full-FT quality, because fine-tuning updates are empirically **low-rank**. Initialize $B=0$
so training starts exactly at the base model.
"""),
    code(r"""
class LoRALinear:
    def __init__(self, W, r=8, alpha=16):
        self.W = W                      # frozen base weight (d_out, d_in)
        d_out, d_in = W.shape
        self.A = np.random.randn(r, d_in) * 0.01    # down-projection
        self.B = np.zeros((d_out, r))               # up-projection (zero init => no change at start)
        self.scale = alpha / r
    def __call__(self, x):
        return x @ self.W.T + (x @ self.A.T) @ self.B.T * self.scale
    def trainable_params(self):
        return self.A.size + self.B.size
    def base_params(self):
        return self.W.size

d_out, d_in, r = 4096, 4096, 8
layer = LoRALinear(np.random.randn(d_out, d_in) * 0.02, r=r)
tp, bp = layer.trainable_params(), layer.base_params()
print(f"base params     : {bp:,}")
print(f"LoRA params     : {tp:,}")
print(f"trainable share : {100*tp/bp:.2f}%   (rank r={r})")
"""),
    md(r"""
## 4. QLoRA — LoRA on a quantized base

**QLoRA** fits big models on one GPU by:
- storing the **frozen base in 4-bit** (NF4, an information-theoretically nice format for
  normally-distributed weights),
- **double quantization** (quantize the quantization constants),
- **paged optimizers** to survive memory spikes,
- training LoRA adapters in bf16 on top.

Net effect: fine-tune a 65B model on a single 48GB GPU, near full-FT quality.

| Method | Trains | Memory | When |
|--------|--------|--------|------|
| Full FT | all weights | very high (≈16 B/param) | max quality, lots of GPUs |
| LoRA | small adapters | low | most fine-tuning |
| QLoRA | adapters + 4-bit base | lowest | big model, one GPU |
"""),
    md(r"""
## 5. Serving many adapters & practical tips

- **Multi-LoRA serving:** keep one base in memory, hot-swap many task adapters per request
  (vLLM supports this) — cheap multi-tenant specialization.
- **Data quality > quantity** (LIMA: ~1k great examples can beat 50k mediocre ones).
- **Catastrophic forgetting:** mix in some general data; keep LR modest.
- **Merge** LoRA into base for zero inference overhead, or keep separate for swapping.

## Exercises
1. Train a real LoRA SFT with `peft`+`trl` on a 0.5–1.5B model; compare base vs SFT outputs.
2. Sweep LoRA rank $r\in\{4,8,16,64\}$; plot quality vs trainable-params.
3. Implement **DoRA** (decompose weight into magnitude + direction) and compare.
4. Verify your masked-CE only updates assistant tokens by checking grads are 0 elsewhere.

## Resources
- *LoRA* (Hu 2021); *QLoRA* (Dettmers 2023); *DoRA* (2024).
- *LIMA: Less Is More for Alignment* (2023); HF `peft`, `trl`, `bitsandbytes` docs.
- Sebastian Raschka — practical LoRA write-ups.
"""),
]

if __name__ == "__main__":
    write(cells, "04_finetuning_sft_and_peft.ipynb")
