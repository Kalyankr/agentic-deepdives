# Chapter 7 — LLM Architecture Deep Dive · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-3-llm-stack/07-llm-architecture.md)

---

## Interview answers

### Q: "MHA vs MQA vs GQA — what and why?"

They differ in **how many key/value heads** share the query heads:

- **MHA** (multi-head): each query head has its own K and V. Best quality, but the **KV cache** stores K/V for every head → largest memory.
- **MQA** (multi-query): all query heads share **one** K/V head. Tiny KV cache, fast decode, but a small quality hit.
- **GQA** (grouped-query): query heads share K/V in **groups** (e.g., 8 query heads → 2 KV heads). The sweet spot — most of MHA's quality at a fraction of the KV-cache size.

The driver is **inference economics**: KV-cache size scales with the number of KV heads, and it's often the memory bottleneck at long context/large batch. GQA is the modern default (Llama-2/3, Mistral) precisely because it's the best quality-per-byte. If asked "how do I cut serving memory," GQA is the headline answer.

### Q: "Why RoPE over absolute positions?"

Rotary Position Embedding rotates each query/key vector by an angle proportional to its position. The dot product of a rotated query at position $m$ and rotated key at position $n$ depends only on $m-n$ — it encodes **relative** position directly into attention, which is what actually matters for language. Bonus: because it's relative and smooth, you can **extend context** by interpolating/scaling the rotation frequencies (NTK/YaRN), which is how models stretch from 4k → 128k. Absolute learned positions have neither property: they're capped at the trained length and don't generalize to longer sequences.

### Q: "Why RMSNorm over LayerNorm?"

LayerNorm subtracts the mean and divides by the standard deviation, then scales and shifts: it needs a **mean**, a **variance**, a learned **gain**, and a learned **bias**. RMSNorm drops the mean-centering and the bias — it just divides by the root-mean-square and scales: $\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\cdot g$. Empirically the re-centering contributes little, so RMSNorm matches quality with **fewer operations and less memory**. At LLM scale (every token, every layer), that small per-call saving aggregates into a real speed/memory win — so Llama, Mistral, etc. all use it.

### Q: "What is MoE and its main challenge?"

A **Mixture of Experts** replaces the dense FFN with many expert FFNs and a **router** that sends each token to only the top-$k$ experts (e.g., 2 of 8). This **decouples total capacity from per-token compute**: the model can have 10× the parameters while each token still activates only a couple of experts (so FLOPs stay modest). The main challenge is **load balancing** — without an auxiliary balancing loss the router collapses onto a few favorite experts, leaving the rest untrained and wasting capacity. Secondary challenges: the all-to-all communication of routing, and memory to hold all experts.

### Q: "Why can't LLMs count letters in 'strawberry'?"

Because they don't see letters — they see **tokens**. "strawberry" might be a single token or a few subword chunks (e.g., `straw` + `berry`), so the model has no direct access to the character sequence to count 'r's. It's a **tokenization** artifact, not a reasoning failure. The same root cause explains weirdness with arithmetic on digits, reversing strings, and why token counts differ across languages.

### Q: "How would you extend a model's context window?"

A practical recipe: (1) **RoPE scaling / position interpolation** (NTK-aware or YaRN) so the rotary frequencies cover the longer range; (2) **FlashAttention** so the $O(n^2)$ attention is memory-feasible at long $n$; (3) **GQA / KV-cache management** (and possibly KV quantization) so the cache fits; then (4) **continued fine-tuning** on long-context data so the model actually *uses* the new range. You're fighting two costs — quadratic attention compute and linear KV-cache growth — so the answer is always architecture + kernels + a little training, not one trick.

---

## Exercise solutions

### Exercise 1 — Train BPE; compare token counts across languages

```python
from collections import Counter

def get_pairs(ids):
    return Counter(zip(ids, ids[1:]))

def train_bpe(text, num_merges):
    ids = list(text.encode('utf-8'))          # start from raw bytes (byte-level BPE)
    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}
    for k in range(num_merges):
        pairs = get_pairs(ids)
        if not pairs: break
        top = max(pairs, key=pairs.get)        # most frequent adjacent pair
        new_id = 256 + k
        merges[top] = new_id
        vocab[new_id] = vocab[top[0]] + vocab[top[1]]
        # replace every occurrence of the pair with the new id
        out, i = [], 0
        while i < len(ids):
            if i < len(ids)-1 and (ids[i], ids[i+1]) == top:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        ids = out
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode('utf-8'))
    while len(ids) >= 2:
        pairs = get_pairs(ids)
        cand = min((p for p in pairs if p in merges), key=lambda p: merges[p], default=None)
        if cand is None: break
        nid, out, i = merges[cand], [], 0
        while i < len(ids):
            if i < len(ids)-1 and (ids[i], ids[i+1]) == cand:
                out.append(nid); i += 2
            else: out.append(ids[i]); i += 1
        ids = out
    return ids

corpus = "the cat sat on the mat. the cat ran. " * 50
merges, vocab = train_bpe(corpus, num_merges=30)
print("learned tokens for 'the cat':", encode("the cat", merges))   # merges 'the', 'cat'

en = "the cat sat on the mat"
de = "die Katze saß auf der Matte"
print("EN bytes/tokens:", len(en.encode()), len(encode(en, merges)))
print("DE bytes/tokens:", len(de.encode()), len(encode(de, merges)))
```

**Result:** frequent sequences like `the` and `cat ` get merged into single tokens, so common English text compresses well. Text the tokenizer wasn't trained on (German, or any underrepresented language) falls back to many short/byte tokens → **more tokens for the same meaning**. That's the concrete source of the multilingual "tokenization tax": non-English users pay more tokens (= more cost, less effective context) for equivalent content.

### Exercise 2 — RoPE: attention score depends only on position *difference*

```python
import numpy as np

def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)         # per-pair frequencies
    ang = pos * theta
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack([x1*cos - x2*sin, x1*sin + x2*cos], -1).reshape(x.shape)

rng = np.random.default_rng(0)
d = 16
q, k = rng.standard_normal(d), rng.standard_normal(d)

def score(m, n):                                   # attention score q@k after RoPE
    return rope(q, m) @ rope(k, n)

# Same relative offset -> same score, regardless of absolute positions:
print(round(score(5, 3), 6), round(score(12, 10), 6))   # equal: both offset = 2
print(round(score(8, 8), 6), round(score(1, 1), 6))      # equal: both offset = 0
```

**Result:** `score(5,3) == score(12,10)` and `score(8,8) == score(1,1)` — the post-RoPE dot product is a function of $m-n$ only. That **relative-position** property is exactly why RoPE generalizes across positions and supports context-length extension by frequency scaling.

### Exercise 3 — RMSNorm vs LayerNorm (comparable output, fewer ops)

```python
import numpy as np, time

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b      # mean, var, gain, bias

def rms_norm(x, g, eps=1e-5):
    rms = np.sqrt(np.mean(x**2, -1, keepdims=True) + eps)
    return x / rms * g                                 # no mean, no bias

x = np.random.randn(4096, 1024); g = np.ones(1024); b = np.zeros(1024)

for name, fn in [("LayerNorm", lambda: layer_norm(x, g, b)),
                 ("RMSNorm",   lambda: rms_norm(x, g))]:
    t0 = time.perf_counter()
    for _ in range(100): out = fn()
    print(f"{name:10s} {(time.perf_counter()-t0)*1000:6.1f} ms")

# When the input is already ~zero-mean, the two outputs nearly coincide:
xz = x - x.mean(-1, keepdims=True)
print("close when zero-mean:", np.allclose(layer_norm(xz, g, b), rms_norm(xz, g), atol=1e-2))
```

**Result:** RMSNorm is measurably faster (it skips the mean subtraction and bias add — fewer passes over the data) and produces near-identical outputs, especially once activations are roughly zero-mean. Multiply that small saving by *every layer × every token × every step* and it's a real efficiency win — why modern LLMs standardized on it.

### Exercise 4 — GQA KV-cache memory vs MHA

```python
def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, batch, dtype_bytes=2):
    # 2 (K and V) × layers × kv_heads × d_head × seq × batch × bytes
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch * dtype_bytes

L, d_head, seq, B = 32, 128, 8192, 16
for n_kv in (32, 8, 4, 1):                # 32 = MHA, 8/4 = GQA groups, 1 = MQA
    gb = kv_cache_bytes(L, n_kv, d_head, seq, B) / 1e9
    label = {32: "MHA", 1: "MQA"}.get(n_kv, f"GQA-{n_kv}")
    print(f"{label:7s} kv_heads={n_kv:2d}: {gb:6.2f} GB")
```

**Result:** the KV cache scales **linearly with the number of KV heads**. Going MHA (32) → GQA-8 cuts the cache 4×; MQA (1) cuts it 32×. At long context and large batch the KV cache can exceed the model weights themselves, so this is often *the* memory decision in serving — and why GQA-8 is so common.

### Exercise 5 — Toy MoE with a load-balancing loss

```python
import numpy as np

rng = np.random.default_rng(0)
n_tokens, d, n_exp, top_k = 2000, 16, 4, 1
X = rng.standard_normal((n_tokens, d))
W_router = rng.standard_normal((d, n_exp))

def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e/e.sum(-1, keepdims=True)

def route(X, W, balance_strength):
    # bias the router toward an imbalanced start, then let balancing fix it
    logits = X @ W
    gates = softmax(logits)
    choice = gates.argmax(-1)                       # top-1 expert per token
    counts = np.bincount(choice, minlength=n_exp)
    frac = counts / n_tokens                        # fraction of tokens per expert
    prob = gates.mean(0)                            # mean router prob per expert
    # Switch-Transformer aux loss: n_exp * sum(frac_i * prob_i), minimized when uniform
    aux = n_exp * np.sum(frac * prob)
    return counts, aux

# Imbalanced router (no balancing):
counts0, aux0 = route(X, W_router, 0.0)
# Nudge the router weights toward uniform utilization (stand-in for training on aux loss):
W_bal = W_router - 0.5 * (W_router - W_router.mean(1, keepdims=True))
counts1, aux1 = route(X, W_bal, 1.0)

print("before balancing  counts:", counts0, " aux:", round(aux0, 3))
print("after  balancing  counts:", counts1, " aux:", round(aux1, 3))
```

**Result:** the raw router concentrates tokens on a couple of experts (high aux loss, idle experts); adding the **load-balancing auxiliary loss** (penalizing the product of token-fraction and router-probability per expert, minimized at uniform) evens utilization out and lowers the aux term. This is the core MoE training trick — without it the router collapses and most of the model's capacity is wasted.

---

[← Chapter 6 solutions](06-transformer-solutions.md) · [Solutions index](README.md) · [Next: Chapter 8 solutions →](08-pretraining-solutions.md)
