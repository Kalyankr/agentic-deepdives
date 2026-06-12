# 01 · Coding Questions (Practical ML + DSA)

> Frontier-lab coding rounds are usually **practical and ML-flavored**: implement a core piece of
> the stack from scratch (attention, a sampler, a tokenizer, a metric) in plain Python/NumPy, with
> clean code and tests. Occasionally you'll get a DSA medium. Practice in a **plain editor** (no
> autocomplete), narrate your approach, and write at least one test.

How to use: read the prompt, implement it yourself first, *then* compare. Each problem lists the
**follow-ups** an interviewer will push on — have answers ready.

---

## 1. Numerically stable softmax

**Prompt:** Implement softmax over the last axis. Why subtract the max?

```python
import numpy as np

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)   # shift for numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
```

**Why subtract the max:** `exp` overflows for large logits (e.g. `exp(1000)=inf`). Subtracting the
max makes the largest exponent `e^0=1`, so values stay in `(0,1]`. Softmax is shift-invariant
(`softmax(x+c)=softmax(x)`), so the result is unchanged.

**Follow-ups:**
- *Temperature?* Divide logits by `T` before softmax: `softmax(x/T)`. `T→0` ⇒ argmax; `T→∞` ⇒ uniform.
- *log-softmax?* Compute `x - logsumexp(x)` directly — more stable than `log(softmax(x))` and what
  cross-entropy uses.

---

## 2. Scaled dot-product attention

**Prompt:** Implement `attention(Q, K, V)` with the `1/√d_k` scaling and an optional causal mask.

```python
def attention(Q, K, V, mask=None):
    # Q,K,V: (..., T, d_k) / (..., T, d_v)
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)   # (..., T, T)
    if mask is not None:                                # mask: True = keep, False = block
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))         # lower-triangular: pos i sees ≤ i
```

**Why `1/√d_k`:** dot products grow with dimension (variance ∝ `d_k`); without scaling the softmax
saturates into near-one-hot, killing gradients. Dividing by `√d_k` keeps the variance ~1.

**Follow-ups:**
- *Mask value −1e9 vs −inf?* `-inf` is cleaner but can produce `NaN` if a whole row is masked; a
  large finite negative avoids that.
- *Complexity?* `O(T²·d)` time and `O(T²)` memory — this is why long context is expensive and why
  FlashAttention (tiling, no materialized `T×T`) and sparse/linear attention exist.

---

## 3. Multi-head attention

**Prompt:** Extend to `h` heads. Why multiple heads instead of one big one?

```python
def multi_head_attention(X, Wq, Wk, Wv, Wo, n_head, causal=True):
    # X: (T, d_model); Wq/Wk/Wv: (d_model, d_model); Wo: (d_model, d_model)
    T, d_model = X.shape
    d_head = d_model // n_head
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                      # each (T, d_model)

    def split(M):                                         # (T, d_model) -> (h, T, d_head)
        return M.reshape(T, n_head, d_head).transpose(1, 0, 2)

    Qh, Kh, Vh = split(Q), split(K), split(V)
    mask = causal_mask(T) if causal else None
    out, _ = attention(Qh, Kh, Vh, mask)                  # (h, T, d_head)
    out = out.transpose(1, 0, 2).reshape(T, d_model)      # concat heads
    return out @ Wo
```

**Why multiple heads:** each head attends in a different subspace (e.g. syntax vs. coreference),
letting the model capture several relationships per layer. Total compute is ~the same as one head of
size `d_model` because each head is `d_model/h` wide.

**Follow-ups:** GQA/MQA (share K/V across heads to shrink the KV cache); where RoPE is applied (to Q,K
before the dot product).

---

## 4. Sampling: greedy, temperature, top-k, top-p (nucleus)

**Prompt:** Given logits, implement the common decoding strategies.

```python
def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=np.random):
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0:                       # greedy
        return int(np.argmax(logits))
    logits = logits / temperature

    if top_k is not None:                      # keep only the k largest logits
        kth = np.sort(logits)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)

    probs = softmax(logits)

    if top_p is not None:                      # nucleus: smallest set with cumulative prob >= p
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        cutoff = np.searchsorted(cum, top_p) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()

    return int(rng.choice(len(probs), p=probs))
```

**When to use which:** greedy/low-T for deterministic tasks (code, extraction); top-p ~0.9–0.95 for
open-ended generation; top-k as a simpler cap. top-p adapts the candidate set to the distribution's
shape, which is usually better than a fixed k.

**Follow-ups:** repetition/frequency penalties; why `temperature` then `top_p` order matters;
min-p sampling; how this interacts with speculative decoding (must match the target distribution).

---

## 5. KV cache

**Prompt:** Implement a KV cache for autoregressive decoding. What problem does it solve?

```python
class KVCache:
    """Stores past keys/values so each decode step is O(T) not O(T^2)."""
    def __init__(self):
        self.k = None   # (T, d)
        self.v = None

    def append(self, k_t, v_t):              # k_t,v_t: (1, d) for the new token
        self.k = k_t if self.k is None else np.concatenate([self.k, k_t], axis=0)
        self.v = v_t if self.v is None else np.concatenate([self.v, v_t], axis=0)
        return self.k, self.v

def decode_step(x_t, Wq, Wk, Wv, cache):
    q_t = x_t @ Wq                            # only the NEW token's query
    k_t, v_t = x_t @ Wk, x_t @ Wv
    K, V = cache.append(k_t, v_t)             # all keys/values so far
    scores = q_t @ K.T / np.sqrt(q_t.shape[-1])
    return softmax(scores) @ V
```

**What it solves:** during decoding, past tokens' K/V don't change, so recomputing attention over the
whole prefix every step is wasted work (`O(T²)` total). Caching them makes each step `O(T)`.

**The cost:** memory. `kv_bytes = 2 · n_layers · seq_len · n_kv_heads · d_head · bytes`. It grows
**linearly with batch × sequence length** and often dominates memory at long context — motivating
GQA/MQA, paged attention, and KV-cache quantization.

**Follow-ups:** eviction for long context (this is an **LRU**-style problem — see #11); PagedAttention
(fixed-size pages, no fragmentation); prefix caching (reuse the cache for a shared system prompt).

---

## 6. Cross-entropy loss (with label smoothing & ignore_index)

**Prompt:** Implement next-token cross-entropy from logits, supporting masked (ignored) positions.

```python
def cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.0):
    # logits: (N, V), targets: (N,) int
    N, V = logits.shape
    logp = logits - logsumexp(logits, axis=-1, keepdims=True)   # log-softmax (stable)
    mask = targets != ignore_index
    idx = np.where(mask, targets, 0)
    nll = -logp[np.arange(N), idx]                              # -log p(correct)
    if label_smoothing > 0:                                     # mix in uniform target
        smooth = -logp.mean(axis=-1)
        nll = (1 - label_smoothing) * nll + label_smoothing * smooth
    return (nll * mask).sum() / mask.sum()                      # average over real tokens only

def logsumexp(x, axis=-1, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)
```

**Why `ignore_index`/masking:** in SFT you only train on **assistant** tokens, not the prompt; mask
the prompt positions out of the loss. **Label smoothing** discourages over-confident logits and can
improve calibration.

**Follow-ups:** relation to perplexity (`ppl = exp(CE)`); why we use log-softmax not `log(softmax)`;
how this becomes the SFT loss with a chat template.

---

## 7. Byte-Pair Encoding (BPE) — core merge loop

**Prompt:** Implement BPE training (learn merges) and encoding. Why subword tokenization?

```python
from collections import Counter

def get_stats(tokens):
    return Counter(zip(tokens, tokens[1:]))           # counts of adjacent pairs

def merge(tokens, pair, new_id):
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(tokens[i]); i += 1
    return out

def train_bpe(text, num_merges):
    tokens = list(text.encode("utf-8"))               # start from raw bytes (0..255)
    merges, next_id = {}, 256
    for _ in range(num_merges):
        stats = get_stats(tokens)
        if not stats: break
        pair = max(stats, key=stats.get)              # most frequent adjacent pair
        tokens = merge(tokens, pair, next_id)
        merges[pair] = next_id; next_id += 1
    return merges

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    for pair, idx in merges.items():                  # apply merges in learned order
        tokens = merge(tokens, pair, idx)
    return tokens
```

**Why subword:** word-level vocab is huge and can't handle OOV; char/byte-level makes sequences very
long. BPE is the middle ground — frequent words become one token, rare words split into pieces, and
**every** string is representable (byte fallback). This is why token counts ≠ word counts and why
non-English / code can cost more tokens.

**Follow-ups:** byte-level BPE (GPT-2) guarantees no OOV; the cost of a poor tokenizer on math/code;
why "strawberry has 3 r's" is hard (the model sees tokens, not characters).

---

## 8. A metric: `pass@k` (and why the unbiased estimator)

**Prompt:** You sample `n` solutions per problem and `c` pass the tests. Estimate `pass@k`.

```python
from math import comb

def pass_at_k(n, c, k):
    # P(at least one of a random k samples is correct) — unbiased (Codex/HumanEval)
    if n - c < k:               # not enough wrong ones to fill k -> guaranteed a pass
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
```

**Why not just `c/n ≥ ...`:** the naive estimate `1-(1-c/n)^k` is **biased** for small `n`. The
combinatorial form is the exact probability that a random size-`k` subset contains ≥1 correct sample,
which is unbiased. Used on HumanEval / coding benchmarks.

**Follow-ups:** report a **confidence interval** (bootstrap); pass@1 vs pass@10 trade-offs; how
sampling temperature affects pass@k.

---

## 9. Top-k nearest neighbors (retrieval core)

**Prompt:** Given a query embedding and a matrix of doc embeddings, return the top-k by cosine.

```python
def top_k_cosine(query, docs, k=5):
    # query: (d,), docs: (N, d)
    q = query / (np.linalg.norm(query) + 1e-9)
    D = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9)
    sims = D @ q                                  # (N,) cosine similarities
    idx = np.argpartition(-sims, kth=k-1)[:k]     # O(N) partial selection, not full sort
    return idx[np.argsort(-sims[idx])]            # sort just the k winners
```

**Why `argpartition`:** full sort is `O(N log N)`; partial selection is `O(N)` then sort only `k`.
This is exact (brute-force) search — the baseline ANN indexes (HNSW, IVF-PQ) approximate to scale.

**Follow-ups:** cosine vs dot vs L2 (normalize ⇒ they rank the same); when to normalize; how this
becomes the retrieval step in RAG; ANN recall/latency trade-off.

---

## 10. RMSNorm / LayerNorm

**Prompt:** Implement LayerNorm and RMSNorm. Why did modern LLMs move to RMSNorm + pre-norm?

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def rms_norm(x, gamma, eps=1e-5):
    rms = np.sqrt(np.mean(x**2, -1, keepdims=True) + eps)
    return gamma * x / rms                         # no mean-subtraction, no bias
```

**Why RMSNorm:** drops the mean-centering and bias — cheaper and empirically as good (LLaMA, etc.).
**Pre-norm** (normalize *inside* the residual branch) gives a clean identity path through the network,
which stabilizes very deep training; **post-norm** is harder to train deep.

**Follow-ups:** where norms sit relative to the residual; why training stability matters at scale;
numerical precision of the norm in bf16.

---

## 11. DSA: LRU cache (maps directly to KV-cache eviction)

**Prompt:** Implement an `O(1)` LRU cache. (Framed for ML: evicting cold entries from a KV/prefix cache.)

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()

    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)        # mark most-recently-used
        return self.d[key]

    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)  # evict least-recently-used
```

**ML framing:** prefix/KV caches and embedding caches are finite; LRU (or LFU) decides what to evict.
Bring up that real serving systems (vLLM) manage the KV cache in **pages** with reference counting,
not a naive LRU.

**Follow-ups:** implement it with a hashmap + doubly-linked list (no `OrderedDict`); LFU vs LRU;
cache-hit-rate as the metric you'd track.

---

## 12. Training step from scratch (PyTorch)

**Prompt:** Write one full training step. What's the canonical order, and what bugs hide here?

```python
import torch

def train_step(model, batch, optimizer, scaler=None, grad_clip=1.0):
    model.train()
    x, y = batch
    optimizer.zero_grad(set_to_none=True)          # clear stale grads
    if scaler:                                      # mixed precision
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
    else:
        loss = torch.nn.functional.cross_entropy(
            model(x).view(-1, model.vocab), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return loss.item()
```

**Common bugs interviewers probe:** forgetting `zero_grad` (grads accumulate); clipping **before**
`unscale_` with AMP; calling `.item()` inside the loop too often (sync stalls); not using
`set_to_none=True`; wrong loss reshape; LR schedule/warmup missing.

**Follow-ups:** gradient accumulation for large effective batch; why warmup + cosine decay; where
gradient checkpointing trades compute for memory.

---

## Coding round tactics
- **Clarify first:** input shapes, dtypes, edge cases (empty, single element, all-masked row).
- **Talk before typing;** state the approach and complexity, then implement.
- **Test as you go:** a tiny example with a hand-checked expected output (e.g. attention weights sum
  to 1; causal mask zeros the upper triangle).
- **State complexity** (time/space) unprompted.
- **Leave it clean:** names, no dead code; mention what you'd add with more time (vectorization, batch dim).
