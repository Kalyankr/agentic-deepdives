# Chapter 6 — The Transformer from Scratch · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-2-deep-learning/06-transformer-from-scratch.md)

---

## Interview answers

### Q: "Walk me through scaled dot-product attention."

Given queries $Q$, keys $K$, values $V$ (each a matrix of token vectors):

1. **Score**: $QK^\top$ — every query's dot product with every key measures relevance.
2. **Scale**: divide by $\sqrt{d_k}$ to keep the scores' variance ~1 (see next answer).
3. **Mask** (decoder only): set future positions to $-\infty$ so they get zero weight.
4. **Softmax** over each row → a probability distribution (the attention weights).
5. **Aggregate**: multiply weights by $V$ → each output is a weighted average of value vectors.

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

In one sentence: *each token builds its new representation by softly looking up information from the tokens most relevant to it.*

### Q: "Why divide by $\sqrt{d_k}$?"

If $Q,K$ have components with ~unit variance, the dot product $q\cdot k = \sum_{i=1}^{d_k} q_i k_i$ has variance $\propto d_k$ — it **grows with dimension**. Large scores push softmax into a saturated regime (one weight ≈ 1, rest ≈ 0), where its gradient is ~0 and learning stalls. Dividing by $\sqrt{d_k}$ rescales the variance back to ~1, keeping softmax in its responsive range with healthy gradients. Exercise 6 below makes you *feel* this by removing it.

### Q: "Why multiple heads?"

A single attention head can only learn one kind of relationship at a time. **Multi-head** attention runs $h$ attentions in parallel on lower-dimensional projections, so different heads specialize — one tracks syntax, another coreference, another induction/copy patterns, another local position. Their outputs are concatenated and projected, combining many relationship types per layer. It's cheap (the per-head dimension is $d_\text{model}/h$, so total compute is unchanged) and strictly more expressive than one big head.

### Q: "Encoder vs decoder?"

- **Encoder**: **bidirectional** attention — every token sees the whole sequence. Good for *understanding* (classification, embeddings); e.g., BERT.
- **Decoder**: **causal/masked** attention — token $i$ only sees tokens $\le i$. Required for *generation*, so the model can be trained to predict the next token without peeking at the answer; e.g., GPT.

Modern LLMs are decoder-only: the causal mask makes every position a training signal (predict the next token) while preserving autoregressive generation.

### Q: "Why residual + LayerNorm in every block?"

- **Residuals** ($x + \text{sublayer}(x)$) give a gradient highway (derivative $1 + \dots$) so very deep stacks train without vanishing gradients — same mechanism as Chapter 5.
- **LayerNorm** normalizes each token's activation vector to zero mean/unit variance, keeping scales stable across layers and steps so training doesn't blow up or stall.
- **Pre-norm** (normalize *before* the sublayer: $x + \text{sublayer}(\text{LN}(x))$) keeps the residual path clean and is much more stable than the original post-norm at depth — it's what modern LLMs use.

### Q: "Why do transformers need positional encoding?"

Attention is a weighted sum — it's **permutation-invariant**, so without position information "dog bites man" and "man bites dog" look identical. Positional encodings (sinusoidal, learned, or rotary/RoPE) inject *where* each token is so the model can use word order. RoPE (Chapter 7) is today's default because it encodes **relative** position and extrapolates to longer contexts.

---

## Exercise solutions

### Exercise 1 — Scaled dot-product attention (rows sum to 1)

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    w = softmax(scores, axis=-1)
    return w @ V, w

rng = np.random.default_rng(0)
Q, K, V = rng.standard_normal((4, 8)), rng.standard_normal((4, 8)), rng.standard_normal((4, 8))
out, w = attention(Q, K, V)
print("output:", out.shape, "weights:", w.shape)      # (4, 8) (4, 4)
print("rows sum to 1:", np.allclose(w.sum(-1), 1.0))  # True
```

**Result:** every attention row sums to 1 — it's a probability distribution over keys. That invariant is the #1 sanity check for an attention implementation.

### Exercise 2 — Causal masking (no peeking at the future)

```python
T = 4
causal = np.tril(np.ones((T, T), dtype=bool))   # True on/below diagonal
out, w = attention(Q, K, V, mask=causal)
print(np.round(w, 3))
# upper triangle is exactly 0 -> token i puts zero weight on tokens > i
print("future weights are zero:", np.allclose(np.triu(w, k=1), 0.0))   # True
```

**Result:** the weight matrix is lower-triangular — position $i$ assigns exactly 0 weight to any position $> i$. Masking with $-\infty$ *before* softmax (not zeroing after) guarantees the surviving weights still renormalize to 1. This single change turns attention into a **decoder**.

### Exercise 3 — Multi-head attention (output dim == d_model)

```python
def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    T, d_model = X.shape
    d_head = d_model // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    # split into heads: (T, n_heads, d_head) -> (n_heads, T, d_head)
    def split(M): return M.reshape(T, n_heads, d_head).transpose(1, 0, 2)
    Qh, Kh, Vh = split(Q), split(K), split(V)
    mask = np.tril(np.ones((T, T), dtype=bool)) if causal else None
    heads = [attention(Qh[h], Kh[h], Vh[h], mask)[0] for h in range(n_heads)]
    concat = np.concatenate(heads, axis=-1)        # (T, d_model)
    return concat @ Wo

d_model, n_heads, T = 32, 4, 6
X = rng.standard_normal((T, d_model))
Wq, Wk, Wv, Wo = (rng.standard_normal((d_model, d_model)) for _ in range(4))
out = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads)
print("output shape:", out.shape)        # (6, 32) == (T, d_model)
```

**Result:** output is `(T, d_model)` — multi-head attention is shape-preserving. Each of the 4 heads works in a 8-dim subspace ($d_\text{model}/n_\text{heads}$); concatenation + the output projection $W_o$ recombine them, so total compute matches a single full-width head.

### Exercise 4 — A full pre-norm block, stacked N times

```python
def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def feed_forward(x, W1, W2):
    return np.maximum(0, x @ W1) @ W2            # FFN with ReLU (4x hidden)

def block(x, params, n_heads):
    # pre-norm + residual around attention, then around FFN
    a = multi_head_attention(layer_norm(x), *params['attn'], n_heads)
    x = x + a
    f = feed_forward(layer_norm(x), params['W1'], params['W2'])
    return x + f

def make_params(d_model):
    h = 4 * d_model
    return {'attn': [rng.standard_normal((d_model, d_model)) for _ in range(4)],
            'W1': rng.standard_normal((d_model, h)) * 0.02,
            'W2': rng.standard_normal((h, d_model)) * 0.02}

x = rng.standard_normal((T, d_model))
layers = [make_params(d_model) for _ in range(6)]   # stack of 6
for p in layers:
    x = block(x, p, n_heads)
print("after 6 blocks:", x.shape)         # (6, 32) -> shape preserved, depth added
```

**Result:** stacking is trivial *because* each block is shape-preserving ($x \to x$). The residual path carries the representation forward while each sublayer adds a refinement. This is the entire transformer trunk; a GPT is just embeddings → N of these → final norm → vocab projection.

### Exercise 5 — Char-level GPT on TinyShakespeare (PyTorch)

A compact, complete nanoGPT-style script. Drop `input.txt` (TinyShakespeare) next to it, or it falls back to a tiny built-in string.

```python
import torch, torch.nn as nn, torch.nn.functional as F, os, math

torch.manual_seed(0)
text = open('input.txt').read() if os.path.exists('input.txt') else "to be or not to be, " * 200
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])
T, n_emb, n_head, n_layer, B = 64, 128, 4, 4, 32

def get_batch():
    ix = torch.randint(len(data) - T - 1, (B,))
    x = torch.stack([data[i:i+T] for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    return x, y

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_emb), nn.LayerNorm(n_emb)
        self.attn = nn.MultiheadAttention(n_emb, n_head, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(n_emb, 4*n_emb), nn.GELU(), nn.Linear(4*n_emb, n_emb))
    def forward(self, x):
        m = torch.triu(torch.ones(x.size(1), x.size(1)), 1).bool().to(x.device)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=m)  # pre-norm
        x = x + a
        return x + self.ff(self.ln2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, n_emb); self.pos = nn.Embedding(T, n_emb)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_emb); self.head = nn.Linear(n_emb, V)
    def forward(self, idx, targets=None):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.lnf(self.blocks(x)); logits = self.head(x)
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, V), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n):
        for _ in range(n):
            logits, _ = self(idx[:, -T:])
            probs = F.softmax(logits[:, -1, :], -1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], 1)
        return idx

model = GPT(); opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(2000):
    x, y = get_batch(); _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(step, loss.item())

start = torch.zeros((1, 1), dtype=torch.long)
print(''.join(itos[i] for i in model.generate(start, 300)[0].tolist()))
```

**Result:** loss drops from ~$\ln V$ (random) toward ~1.5 on real Shakespeare, and generation goes from gibberish to Shakespeare-flavored text with plausible words, line breaks, and character names. This *is* GPT — GPT-3 is the same architecture scaled ~5 orders of magnitude. (Use a GPU and the full `input.txt` for legible samples.)

### Exercise 6 — Ablate the $\sqrt{d_k}$ scaling

```python
import numpy as np

def attn_scores(Q, K, scale=True):
    d_k = Q.shape[-1]
    s = Q @ K.T
    return s / np.sqrt(d_k) if scale else s

rng = np.random.default_rng(0)
for d_k in (8, 64, 512):
    Q, K = rng.standard_normal((4, d_k)), rng.standard_normal((4, d_k))
    s_scaled, s_raw = attn_scores(Q, K, True), attn_scores(Q, K, False)
    print(f"d_k={d_k:4d}  raw score std={s_raw.std():7.2f}  scaled std={s_scaled.std():4.2f}")
```

**Result:** raw-score standard deviation grows like $\sqrt{d_k}$ (≈ 8× larger at $d_k{=}512$ than $d_k{=}8$), while the scaled scores stay ~1. In a real training run, dropping the scale makes softmax saturate at large $d_k$ → near-zero gradients → unstable/failed training. Feeling this is the point: the $\sqrt{d_k}$ isn't decoration, it's what keeps gradients alive.

---

[← Chapter 5 solutions](05-neural-networks-solutions.md) · [Solutions index](README.md) · [Next: Chapter 7 solutions →](07-llm-architecture-solutions.md)
