# Chapter 8 — Pretraining & Scaling Laws · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-3-llm-stack/08-pretraining.md)

---

## Interview answers

### Q: "What's the pretraining objective?"

**Autoregressive next-token prediction.** Given a sequence, the model predicts the probability distribution of token $t{+}1$ from tokens $1..t$, and we minimize the **cross-entropy / negative log-likelihood** of the true next token, averaged over a huge corpus:

$$\mathcal{L} = -\frac{1}{N}\sum_{t} \log p_\theta(x_{t+1}\mid x_{\le t}).$$

That's it — one self-supervised objective, no labels. The surprise of the last decade is that optimizing this single objective at scale induces broad capabilities (grammar, facts, translation, reasoning) as a side effect of getting good at prediction.

### Q: "Explain Chinchilla / compute-optimal scaling."

For a **fixed training compute budget**, there's an optimal split between model size $N$ and training tokens $D$. Chinchilla (DeepMind, 2022) found that to be compute-optimal you should scale them **together**, at roughly **$D \approx 20N$ tokens per parameter** — many earlier models (like GPT-3, 175B params on ~300B tokens) were **too big and undertrained**. A Chinchilla-optimal 70B-on-1.4T model beat GPT-3 with far fewer parameters. The headline: don't just grow the model — feed it proportionally more data.

### Q: "Then why do modern small models train on 15T tokens (way past 20×)?"

Because Chinchilla optimizes **training** compute, but production cares about **lifetime** cost = training **+** all future inference. If you'll serve a model billions of times, it's worth **over-training a smaller model** far past the 20× point: you spend more upfront to get a model that's permanently cheaper and faster to run. Llama-3 8B on ~15T tokens (~1900 tokens/param) is "Chinchilla-suboptimal" for training but **optimal for total economics** — a smaller, over-trained model that's cheap to serve. It's a training-vs-serving tradeoff, and serving usually dominates.

### Q: "bf16 vs fp16 in training?"

Both 16-bit. **bf16** keeps fp32's 8-bit exponent (huge range) with a 7-bit mantissa → it rarely overflows/underflows, so **no loss scaling** is needed. **fp16** has a 5-bit exponent (narrow range) with a 10-bit mantissa → more precision but it **needs loss scaling** to keep small gradients from underflowing to zero. For training, bf16 is preferred where supported because range matters more than precision and it removes the loss-scaling machinery. (Full bit-layout reasoning in the [Chapter 4 solutions](04-cs-fundamentals-solutions.md#q-bf16-vs-fp16-for-training).)

### Q: "What is gradient checkpointing?"

A memory-for-compute trade. Normally you keep **all** intermediate activations from the forward pass to use in backward — that's a lot of memory for deep models. Gradient (activation) checkpointing **discards most activations** and **recomputes** them during the backward pass from a few saved checkpoints. Cost: roughly one extra forward pass (~30% more compute); benefit: dramatically lower activation memory, which lets you fit bigger models or longer sequences. It's standard for training large models.

### Q: "Why is data quality so important?"

Because **architecture is largely commoditized** — everyone uses the same transformer recipe — so the differentiator is the **data pipeline**: aggressive **deduplication** (dups waste compute and hurt generalization), **quality filtering** (remove spam/boilerplate/low-quality text), **decontamination** (remove benchmark test sets to avoid inflated scores), and a deliberate **domain mix** (code, math, web, books). Most of the quality gap between models that "should" be equal comes from data work, not model tweaks. Garbage in, garbage out — at trillion-token scale.

### Q: "What are emergent abilities and are they real?"

"Emergent abilities" are capabilities that appear **abruptly past a scale threshold** — a task the model fails at small sizes, then suddenly does well once it's large enough. They're real as an *observation*, but **whether the jump is fundamental is debated**: Schaeffer et al. argued many "emergences" are **metric artifacts** — using a harsh all-or-nothing metric (like exact match) makes smooth underlying improvement *look* like a sudden jump; switch to a continuous metric and the curve is smooth. A strong answer acknowledges both: capabilities clearly improve with scale and some look emergent, but be skeptical of sharp-threshold claims and check whether the metric is doing the work.

---

## Exercise solutions

### Exercise 1 — Next-token cross-entropy with the correct shift

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
V, T = 32, 16
text = torch.randint(0, V, (1, T + 1))           # one short "document"

# THE SHIFT: inputs are tokens [0..T-1], targets are tokens [1..T]
x, y = text[:, :-1], text[:, 1:]

emb = nn.Embedding(V, 64); head = nn.Linear(64, V)
opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=1e-2)

for step in range(300):
    logits = head(emb(x))                        # (1, T, V)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0: print(step, round(loss.item(), 4))

print("final loss:", round(loss.item(), 4))      # -> ~0, the model memorized the sequence
print("ln(V) random baseline:", round(torch.log(torch.tensor(float(V))).item(), 4))
```

**Result:** loss starts near $\ln V \approx 3.47$ (uniform-random guessing) and drops to ~0 as the tiny model **memorizes** the short sequence — confirming the shift is correct (position $t$ predicts $t{+}1$). Getting the off-by-one shift wrong is a classic bug; this overfit-one-sequence test catches it instantly.

### Exercise 2 — A baby scaling law (loss vs model size)

```python
import torch, torch.nn as nn, torch.nn.functional as F, math

torch.manual_seed(0)
V, T, N = 64, 32, 2000
data = torch.randint(0, V, (N, T + 1))           # fixed dataset with some structure
data[:, 1:] = (data[:, :-1] + 1) % V             # next token = prev+1 (learnable pattern)

def train_model(d_model, steps=300):
    emb = nn.Embedding(V, d_model)
    net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, V))
    opt = torch.optim.Adam(list(emb.parameters()) + list(net.parameters()), lr=3e-3)
    for _ in range(steps):
        b = data[torch.randint(0, N, (64,))]
        logits = net(emb(b[:, :-1]))
        loss = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

for d in (8, 16, 32, 64, 128):
    n_params = V*d + d*d + d*V
    print(f"d_model={d:4d}  params≈{n_params:7d}  loss={train_model(d):.4f}")
```

**Result:** loss falls monotonically as model size grows, and on a log–log plot (loss vs params) the points fall roughly on a straight line — a miniature **scaling law**. That predictable power-law relationship is exactly what lets labs forecast a frontier model's loss from small-scale runs before spending millions on the full one.

### Exercise 3 — Gradient accumulation == one big batch

```python
import torch, torch.nn as nn

torch.manual_seed(0)
model = nn.Linear(10, 1)
X = torch.randn(32, 10); Y = torch.randn(32, 1)
loss_fn = nn.MSELoss()

# (A) one big batch of 32
model.zero_grad()
loss_fn(model(X), Y).backward()
big = model.weight.grad.clone()

# (B) 4 micro-batches of 8 with accumulation; divide each loss by n_accum
model.zero_grad()
n_accum = 4
for i in range(n_accum):
    xb, yb = X[i*8:(i+1)*8], Y[i*8:(i+1)*8]
    (loss_fn(model(xb), yb) / n_accum).backward()    # accumulate into .grad
acc = model.weight.grad.clone()

print("identical gradient:", torch.allclose(big, acc, atol=1e-6))   # True
```

**Result:** accumulating gradients over 4 micro-batches of 8 (dividing each loss by `n_accum`) gives the **identical** gradient to one batch of 32. This is how you simulate a large batch size that wouldn't fit in memory — train at a big *effective* batch on small hardware. The key detail is the `/ n_accum` so the mean is over the full effective batch.

### Exercise 4 — Near-dedup with MinHash

```python
import numpy as np, random

def shingles(text, k=4):
    text = text.lower()
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def minhash(s, num_perm=64, seed=1):
    rng = random.Random(seed)
    # random hash functions: (a*x + b) mod p
    p = 2**61 - 1
    fns = [(rng.randrange(1, p), rng.randrange(0, p)) for _ in range(num_perm)]
    sig = []
    for a, b in fns:
        sig.append(min(((a * (hash(sh) % p) + b) % p) for sh in s))
    return np.array(sig)

def estimated_jaccard(a, b):
    return np.mean(minhash(a) == minhash(b))

docs = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog!",   # near-dup of #0
    "machine learning models are trained on large datasets",
    "the quick brown fox leaps over the lazy dog",     # near-dup of #0
    "completely unrelated text about cooking pasta",
]

kept, removed = [], []
for i, d in enumerate(docs):
    if any(estimated_jaccard(d, docs[j]) > 0.7 for j in kept):
        removed.append(i)
    else:
        kept.append(i)
print("kept docs   :", kept)         # [0, 2, 4]
print("removed dups:", removed)      # [1, 3]  -> ~40% removed
```

**Result:** MinHash estimates Jaccard similarity from compact signatures (no $O(n^2)$ exact comparison of full shingle sets), flagging the near-duplicate variants of doc 0 and removing ~40% of this toy corpus. Real pipelines run this with LSH banding over **billions** of documents — dedup is one of the highest-leverage data-quality steps.

### Exercise 5 — Chinchilla-optimal vs over-trained token counts

```python
params = {"1B": 1e9, "7B": 7e9, "70B": 70e9}
print(f"{'model':6s} {'Chinchilla (20×)':>18s} {'over-trained (real)':>22s}")
for name, n in params.items():
    chin = 20 * n                      # compute-optimal tokens
    real = {"1B": 15e12, "7B": 15e12, "70B": 15e12}[name]   # modern over-training
    print(f"{name:6s} {chin/1e9:14.0f} B  {real/1e12:18.0f} T  ({real/n:,.0f} tok/param)")
```

**Result:**

| Model | Chinchilla-optimal (20×) | Modern over-trained |
|---|---|---|
| 1B | 20 B tokens | 15 T (~15,000×) |
| 7B | 140 B tokens | 15 T (~2,100×) |
| 70B | 1.4 T tokens | 15 T (~210×) |

**Discussion:** Chinchilla says the *training-compute-optimal* point is ~20 tokens/param. Modern models (Llama-3 etc.) train **far past** it because the goal is **lifetime cost**: a small model trained on 15T tokens is permanently cheaper to serve than a larger Chinchilla-optimal model of equal quality. You "waste" training FLOPs to save vastly more inference FLOPs — rational when you'll serve billions of requests. The tradeoff flips only if a model will be served rarely.

---

[← Chapter 7 solutions](07-llm-architecture-solutions.md) · [Solutions index](README.md) · [Next: Chapter 9 solutions →](09-alignment-solutions.md)
