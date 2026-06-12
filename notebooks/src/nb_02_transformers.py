"""Build NB02 — Transformers from scratch (numpy)."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 02 · Transformers From Scratch

> Module: **02 · Transformer Internals** — the single most important module for frontier-lab interviews.

**Goal:** implement the transformer's core — **scaled dot-product attention**, **multi-head**
attention, the **causal mask**, and a full block — in plain NumPy, then understand every
modern variant (**RoPE, GQA, RMSNorm, KV cache, FlashAttention, MoE**) and be able to derive
**params / FLOPs / KV-cache** on a whiteboard.

### Learning objectives
1. Implement attention from the equation, including causal masking.
2. Build multi-head attention and a transformer block.
3. Explain *why* each modern component exists.
4. Do the cost math: parameters, FLOPs/token, KV-cache size.
"""),
    md(r"""
## 1. Tokenization (the model's alphabet)

LLMs don't see text — they see **token ids**. Modern models use **Byte-Pair Encoding (BPE)**:
start from bytes/chars and greedily merge the most frequent adjacent pair, building a vocab.
Here's a minimal char-level tokenizer to make the idea concrete.
"""),
    code(r"""
import numpy as np

text = "the cat sat on the mat. the cat ate."
vocab = sorted(set(text))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

ids = encode("the cat")
print("vocab size:", len(vocab))
print("encode('the cat') ->", ids)
print("decode back      ->", decode(ids))
# Real models use byte-level BPE (e.g. ~50k-128k merges) so any UTF-8 text is representable.
"""),
    md(r"""
## 2. Scaled dot-product attention — the core operation

Given queries $Q$, keys $K$, values $V$:

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Intuition: each position builds a **query**, compares it (dot product) to every position's
**key** to get attention weights, then takes a weighted sum of **values**.

- **Why divide by $\sqrt{d_k}$?** Dot products of $d_k$-dim vectors have variance $\propto d_k$;
  scaling keeps them ~unit variance so softmax doesn't saturate into one-hot (which kills gradients).
"""),
    code(r"""
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)   # (..., T, T)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)        # block disallowed positions
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

rng = np.random.default_rng(0)
T, d = 4, 8
Q = rng.standard_normal((T, d)); K = rng.standard_normal((T, d)); V = rng.standard_normal((T, d))
out, w = attention(Q, K, V)
print("output shape:", out.shape, " attention weights row sums:", w.sum(-1).round(3))
"""),
    md(r"""
## 3. Causal masking (so the model can't see the future)

For autoregressive generation, position $t$ may only attend to positions $\le t$. We enforce
this with a lower-triangular mask before the softmax. This is *the* property that makes a GPT
trainable on all positions in parallel while still predicting "the next token."
"""),
    code(r"""
causal = np.tril(np.ones((T, T))).astype(bool)   # True where attention is allowed
print("causal mask (lower-triangular):\n", causal.astype(int))
out_c, w_c = attention(Q, K, V, mask=causal)
print("\nattention weights (note zeros above the diagonal):\n", w_c.round(2))
assert np.allclose(np.triu(w_c, 1), 0), "future positions leaked!"
print("\ncausality holds -> OK")
"""),
    md(r"""
## 4. Multi-head attention

Instead of one attention with dimension $d$, use $h$ **heads** of dimension $d/h$ in parallel.
Each head can specialize (syntax, coreference, position, …); we concatenate their outputs and
project. This is just attention applied per-head with reshaping.
"""),
    code(r"""
def multi_head_attention(X, Wq, Wk, Wv, Wo, n_head, causal=True):
    T, d = X.shape
    hd = d // n_head
    Q = (X @ Wq).reshape(T, n_head, hd).transpose(1, 0, 2)   # (h, T, hd)
    K = (X @ Wk).reshape(T, n_head, hd).transpose(1, 0, 2)
    V = (X @ Wv).reshape(T, n_head, hd).transpose(1, 0, 2)
    mask = np.tril(np.ones((T, T))).astype(bool) if causal else None
    heads, _ = attention(Q, K, V, mask=mask)                 # (h, T, hd)
    concat = heads.transpose(1, 0, 2).reshape(T, d)          # (T, d)
    return concat @ Wo

d, n_head = 32, 4
X = rng.standard_normal((6, d))
Wq, Wk, Wv, Wo = (rng.standard_normal((d, d)) * 0.1 for _ in range(4))
y = multi_head_attention(X, Wq, Wk, Wv, Wo, n_head)
print("MHA output:", y.shape)   # (T, d), same shape in = shape out (residual-friendly)
"""),
    md(r"""
## 5. The full transformer block

A modern (pre-norm) decoder block is:

```
x = x + MHA(RMSNorm(x))      # communicate across positions
x = x + FFN(RMSNorm(x))      # think per-position (usually 4x wider, with SwiGLU/GELU)
```

The **residual stream** (the `x +`) is the highway that lets gradients flow through deep nets;
attention and the FFN *read from* and *write to* it. Stacks of these blocks = a GPT.
"""),
    code(r"""
def rms_norm(x, eps=1e-5):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def ffn(x, W1, W2):
    return gelu(x @ W1) @ W2

def block(x, params, n_head):
    a = multi_head_attention(rms_norm(x), *params["attn"], n_head)
    x = x + a
    f = ffn(rms_norm(x), params["W1"], params["W2"])
    return x + f

d = 32
params = {
    "attn": tuple(rng.standard_normal((d, d)) * 0.1 for _ in range(4)),
    "W1": rng.standard_normal((d, 4*d)) * 0.1,
    "W2": rng.standard_normal((4*d, d)) * 0.1,
}
x = rng.standard_normal((6, d))
for _ in range(3):           # stack 3 blocks
    x = block(x, params, n_head=4)
print("after 3 blocks:", x.shape, " (shape preserved by residual stream)")
"""),
    md(r"""
## 6. Positional information — RoPE

Attention is **permutation-invariant**: it has no idea of order. We must inject position.
Modern models use **RoPE (Rotary Position Embeddings)**: rotate the query/key vectors by an
angle proportional to their position. The dot product then depends only on **relative**
position $(m-n)$, which generalizes to longer contexts and underpins context-extension tricks
(position interpolation, YaRN, NTK-aware scaling).
"""),
    code(r"""
def rope(x, base=10000):
    # x: (T, d) with d even. Rotate pairs (x_2i, x_2i+1) by m * theta_i.
    T, d = x.shape
    half = d // 2
    theta = base ** (-np.arange(0, half) / half)     # frequencies per pair
    m = np.arange(T)[:, None]                         # positions
    ang = m * theta[None, :]                          # (T, half)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out

q = rng.standard_normal((5, 8))
# RoPE makes <q_m, k_n> depend on (m-n): same content at different offsets => same score pattern
print("RoPE applied, shape:", rope(q).shape)
"""),
    md(r"""
## 7. The KV cache — the #1 inference concept

Generation is autoregressive: to produce token $t{+}1$ we recompute attention over tokens
$1..t$. Without caching, that's $O(T^2)$ wasted work. The **KV cache** stores past keys/values
so each new token only computes its *own* Q, K, V and attends to the cache — turning per-step
cost from $O(T)$ recompute into $O(1)$ append + an attention read.

- **Prefill** = process the whole prompt at once (compute-bound), fills the cache.
- **Decode** = one token at a time (memory-bandwidth-bound), reads weights + cache every step.

The KV cache, not the weights, is usually the **memory bottleneck** at serving time.
"""),
    code(r"""
class KVCache:
    def __init__(self):
        self.K, self.V = None, None
    def append(self, k, v):
        self.K = k if self.K is None else np.concatenate([self.K, k], axis=0)
        self.V = v if self.V is None else np.concatenate([self.V, v], axis=0)
        return self.K, self.V

# decode loop sketch: each step appends ONE row of k,v and attends over the whole cache
cache = KVCache()
d = 8
for step in range(4):
    x_t = rng.standard_normal((1, d))     # the single new token's hidden state
    k_t, v_t = x_t * 0.5, x_t * 0.3       # (stand-in for x_t @ Wk, x_t @ Wv)
    K, V = cache.append(k_t, v_t)
    q_t = x_t
    out, _ = attention(q_t, K, V)         # attend over all cached positions
print("cache length after 4 decode steps:", cache.K.shape[0])
"""),
    md(r"""
## 8. Efficiency variants you must be able to explain

| Idea | What it does | Why |
|------|--------------|-----|
| **MQA / GQA** | share K/V across (groups of) heads | shrinks the **KV cache** → bigger batches / longer context (Llama-3, Mistral use GQA) |
| **FlashAttention** | IO-aware, tiled attention; never materializes the $T\times T$ matrix | turns attention memory from $O(T^2)$ to $O(T)$ and is much faster |
| **Sliding-window / sparse** | attend to a local window | linear cost for long sequences |
| **MoE** | many FFN "experts," route each token to top-$k$ | scale parameters without scaling FLOPs/token (Mixtral, DeepSeek) |

**Architecture families:** *encoder-only* (BERT, understanding), *decoder-only* (GPT/Llama/Claude,
generation — the dominant design), *encoder-decoder* (T5, seq2seq).
"""),
    md(r"""
## 9. The cost math (whiteboard this in interviews)

For a decoder-only model: $L$ layers, hidden size $d$, vocab $V$, sequence length $T$.

- **Parameters** $\approx 12 L d^2$ (attention $4d^2$ + FFN $8d^2$ per layer) $+\,Vd$ embeddings.
- **FLOPs/token (fwd)** $\approx 2N$ where $N$ = #params; **training** fwd+bwd $\approx 6N$ per token
  → the famous $C\approx 6ND$ (compute ≈ 6 × params × tokens).
- **KV-cache** $= 2\,(K{+}V)\times L\times T\times d_{kv}\times \text{bytes}$ per sequence.
"""),
    code(r"""
def model_stats(L, d, V, dtype_bytes=2, T=4096, n_head=None, n_kv_head=None):
    p_attn = 4 * d * d * L
    p_ffn  = 8 * d * d * L
    p_embed = V * d
    N = p_attn + p_ffn + p_embed
    d_kv = d if n_kv_head is None else d * n_kv_head // (n_head or 1)
    kv_bytes = 2 * L * T * d_kv * dtype_bytes
    return N, kv_bytes

# ~ a 7B-ish config
N, kv = model_stats(L=32, d=4096, V=32000, dtype_bytes=2, T=4096)
print(f"params       ~ {N/1e9:.2f} B")
print(f"weights mem  ~ {N*2/1e9:.1f} GB (fp16)")
print(f"fwd FLOPs/tok~ {2*N/1e9:.1f} GFLOP")
print(f"KV cache/seq ~ {kv/1e9:.2f} GB at T=4096  <-- grows linearly with batch & length")
"""),
    md(r"""
## 10. Scaling laws (why models are the size they are)

- **Kaplan et al. 2020:** loss falls as a **power law** in params, data, and compute.
- **Chinchilla (Hoffmann et al. 2022):** for a fixed compute budget, params and tokens should
  scale **together** (~20 tokens/param) — most pre-2022 models were *under-trained*. This is why
  a well-trained 7B can beat an under-trained 13B.

## Exercises
1. Add **GQA** to `multi_head_attention` (fewer K/V heads) and recompute KV-cache size.
2. Implement **SwiGLU** FFN: $(\,\text{Swish}(xW_1)\odot xW_3\,)W_2$ and swap it in.
3. Build a tiny **greedy + temperature** sampler over a random logit vector.
4. Derive params for Llama-3-8B's real config and compare to 8B.

## Resources
- *Attention Is All You Need* (Vaswani 2017); *The Illustrated/Annotated Transformer*.
- **RoPE** (Su 2021), **GQA** (Ainslie 2023), **FlashAttention** 1/2/3 (Dao), **Mixtral** (2024).
- **Chinchilla** (Hoffmann 2022); Karpathy *Let's build GPT* + `nanoGPT`.
- The lab with spec tests (you implement attention): `labs/lab02_nanogpt/`.
"""),
]

if __name__ == "__main__":
    write(cells, "02_transformers_from_scratch.ipynb")
