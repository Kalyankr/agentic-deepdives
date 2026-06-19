# Stage 1 — Answer Key (Transformer Internals)

> Full worked answers to [interview-questions.md](interview-questions.md). Each answer is written at the depth a strong senior/staff candidate would give out loud, then a sentence or two deeper than that for study. Read the question first, answer from memory, *then* check here.
>
> Notation: $B$ batch, $T$ sequence length, $d$ (or $C$) model width, $h$ heads, $d_k=d/h$ head dim, $V$ vocab, $L$ layers.

---

## 🟢 Fundamentals

**1. Explain self-attention in one sentence.**
Each token produces a query, key, and value; it compares its query against every token's key to get a softmax distribution of relevance, then takes that weighted average of the values — so every position's new representation is a content-based, learned mixture of all positions.

**2. Why do transformers need positional encodings at all?**
The attention operation is a sum over a *set* — it is permutation-equivariant, so without position information "dog bites man" and "man bites dog" produce identical representations. Positional encodings inject order so the model can distinguish sequences that differ only in arrangement. (FFN and LayerNorm are applied per-token and also carry no cross-position order, so the burden falls entirely on attention + positions.)

**3. Encoder-only vs decoder-only vs encoder-decoder.**
- **Encoder-only** (BERT): bidirectional attention, trained with masked-LM; great for *understanding*/classification/embeddings. Use case: retrieval embeddings, NER, sentiment.
- **Decoder-only** (GPT, Llama): causal attention, next-token prediction; great for *generation* and, at scale, in-context learning. Use case: chat, code completion.
- **Encoder-decoder** (T5, original Transformer): bidirectional encoder + causal decoder with cross-attention; great for *seq-to-seq* with a clear input→output mapping. Use case: translation, summarization.

**4. What does causal masking do, and where?**
It sets attention scores for future positions to $-\infty$ *before* the softmax, so position $t$ can only attend to $\le t$. Applied inside each self-attention layer of a decoder, on the $T\times T$ score matrix. It is what lets us train on all positions in parallel while preserving the autoregressive "no peeking" property.

**5. Why multiple heads instead of one big head?**
A single softmax can only express one attention distribution per token — it must average all relationships together. Multiple heads let the model attend to different things in different subspaces simultaneously (e.g. one head tracks syntax, another coreference, another previous-token). It's the difference between one and many relational "channels"; total compute is similar because each head is $d/h$ wide.

**6. Job of the feed-forward (MLP) sub-layer?**
Attention *moves* information between positions; the MLP *processes* it per-position. The 2-layer MLP (up-project ×4, nonlinearity, down-project) is where most of the model's "knowledge"/computation lives — it acts like a large key-value memory applied independently to each token.

**7. Why a residual connection around each sub-layer?**
It gives gradients a direct, identity path to every layer (mitigating vanishing gradients at depth) and lets each sub-layer learn a *delta* to the running representation rather than a full re-encoding. This is what makes very deep stacks trainable — it's the backbone of the "residual stream."

**8. LayerNorm vs BatchNorm — why LayerNorm?**
LayerNorm normalizes across the feature dimension *per token*, independent of other examples and other positions. That's essential because (a) sequences have variable length, (b) batch statistics are noisy/unavailable at inference for a single stream, and (c) autoregressive decoding processes one token at a time. BatchNorm's cross-batch statistics break all three.

**9. What is BPE and why byte-level?**
Byte-Pair Encoding starts from a base vocabulary and greedily merges the most frequent adjacent symbol pair, repeatedly, building subword units; common words become single tokens, rare words split into pieces. **Byte-level** means the base alphabet is the 256 bytes, so *any* Unicode string is representable with zero out-of-vocabulary tokens — no `<UNK>`, robust to emoji/code/multilingual text.

**10. Time and memory complexity of self-attention in sequence length?**
Compute is $O(T^2 d)$ (the $QK^\top$ and the score·$V$ matmuls are each $T^2 d$). Memory for the naive score matrix is $O(T^2)$ per head. Both are quadratic in $T$ — the reason long context is hard.

---

## 🟡 Core (L4–L5)

**11. Every tensor shape from token ids → logits.**
- ids: $(B,T)$ ints →
- token embedding lookup: $(B,T,d)$; add positional info (or apply RoPE inside attention) →
- per block: LayerNorm $(B,T,d)$; QKV projection $(B,T,3d)$ → split/reshape to $(B,h,T,d_k)$ each; scores $QK^\top$ $(B,h,T,T)$; softmax; ×$V$ → $(B,h,T,d_k)$; merge → $(B,T,d)$; output proj $(B,T,d)$; residual add; LayerNorm; MLP $(B,T,d)\to(B,T,4d)\to(B,T,d)$; residual add →
- final LayerNorm $(B,T,d)$ →
- LM head ($d\times V$, often tied): logits $(B,T,V)$.

**12. Why divide by $\sqrt{d_k}$? What breaks without it?**
If $q,k$ have ~zero-mean unit-variance independent entries, $q\cdot k=\sum_{i=1}^{d_k} q_i k_i$ has variance $\approx d_k$, so magnitude $\propto\sqrt{d_k}$. Feeding large-magnitude logits to softmax saturates it (one near-1, rest near-0), whose Jacobian $\approx 0$ → vanishing gradients and unstable training. Dividing by $\sqrt{d_k}$ keeps logits $O(1)$ and the softmax in a high-gradient regime.

**13. Pre-norm vs post-norm — which and why?**
Modern LLMs use **pre-norm** ($x + \text{Sublayer}(\text{LN}(x))$). Post-norm ($\text{LN}(x+\text{Sublayer}(x))$, the original) puts LN on the residual path, which attenuates the identity signal and makes deep stacks need careful warmup/learning-rate-warmup to avoid divergence. Pre-norm keeps a clean identity residual stream so gradients flow to depth; the tradeoff is a growing residual magnitude, often handled with a final LN.

**14. Absolute vs learned vs RoPE vs ALiBi — why did RoPE win?**
- **Sinusoidal absolute:** fixed, parameter-free, some extrapolation, but injects *absolute* position additively.
- **Learned absolute:** a trainable position embedding table; simple but cannot exceed trained length and is purely absolute.
- **RoPE:** rotates Q/K by an angle proportional to position, so the dot product depends only on *relative* offset; no parameters, composes with attention naturally, and interpolates/extrapolates (YaRN/NTK) to longer context.
- **ALiBi:** adds a linear distance penalty to scores; cheap, extrapolates, but coarser.
RoPE won because it encodes *relative* position directly in the attention dot product, adds no parameters, and supports principled context extension — the right set of properties for scaled decoder LLMs.

**15. Why tie input embedding and output projection?**
Both are $d\times V$ matrices mapping between token space and model space; tying them ($W_\text{out}=W_\text{emb}^\top$) saves $\sim Vd$ parameters (large for big vocab), regularizes, and reflects the symmetry that "the representation of a token" and "the direction that predicts that token" should be related. Empirically improves perplexity.

**16. The "residual stream" mental model.**
Think of a per-token vector "bus" of width $d$ running through all layers. Each sub-layer *reads* from the stream (via LN), computes something, and *writes back* by addition. Blocks communicate by writing features that later blocks read; attention writes information moved across positions, MLPs write per-token transformations. Superposition lets many features share the $d$ dimensions.

**17. GELU vs ReLU vs SwiGLU.**
ReLU is the cheap baseline (hard gate, zero for negatives). GELU is a smooth, probabilistic gate ($x\cdot\Phi(x)$) that improves gradients and is standard in GPT/BERT. **SwiGLU** is a *gated* unit: $(\text{Swish}(xW)\odot(xV))W_2$ — two projections, one gates the other. It consistently improves quality per parameter, so Llama/PaLM use it (typically scaling hidden to $\tfrac{8}{3}d$ to keep param count comparable).

**18. Where do most parameters live — attention or MLP?**
The **MLP**. Per layer, attention is $4d^2$ (Q,K,V,O) and the MLP is $2\cdot d\cdot 4d = 8d^2$ — so ~2/3 of block parameters (and most "knowledge") are in the MLP. (With SwiGLU the ratio shifts slightly but MLP still dominates.)

**19. Why is attention permutation-invariant without positions?**
Attention output for token $i$ is $\sum_j \text{softmax}_j(q_i\cdot k_j)\,v_j$ — a sum over $j$. Permuting the keys/values permutes the summands but not the sum, so the output is invariant to input order (equivariant for the query index). Position information is the only thing that breaks this symmetry.

**20. How does the KV cache change attention at inference vs training?**
Training computes attention over the *whole* sequence in parallel (all queries × all keys). At autoregressive inference you generate one token at a time; the KV cache stores past keys/values so each new step computes attention of just the *new* query against the cached $K,V$ — turning per-step cost from recomputing $O(T^2)$ into $O(T)$, at the cost of $O(T)$ memory per layer/head that grows with context.

---

## 🔴 Senior / Staff deep dives

**21. Derive the $\sqrt{d}$ scaling.**
Let $q,k\in\mathbb{R}^{d_k}$ have i.i.d. entries with mean 0, variance 1. Then $s=q\cdot k=\sum_{i=1}^{d_k} q_i k_i$. Each term has $\mathbb{E}[q_ik_i]=0$ and $\text{Var}(q_ik_i)=\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]=1$ (independence). Summing $d_k$ independent terms: $\text{Var}(s)=d_k$, so $\text{std}(s)=\sqrt{d_k}$. Large $s$ pushes softmax toward a one-hot, where its Jacobian $\text{diag}(p)-pp^\top \to 0$, killing gradients. Dividing by $\sqrt{d_k}$ rescales $\text{Var}\to 1$, keeping logits $O(1)$ and gradients healthy.

**22. A product needs 128K-token context. What breaks, options?**
*What breaks:* attention compute is $O(T^2)$ (16k× more than 1k); the **KV cache** grows linearly — $2\cdot L\cdot h\cdot d_k\cdot T$ values — and dominates memory/bandwidth, capping batch size and throughput; quality degrades ("lost in the middle"); positions exceed training range.
*Options:* **FlashAttention** (IO-aware, no $T^2$ materialization — fixes memory, not the FLOPs), **sliding-window / sparse / dilated** attention, **GQA/MQA** to shrink the cache, **RoPE interpolation / YaRN / NTK-scaling** to extend positions, **KV-cache quantization** and eviction/attention-sinks for streaming, and retrieval to avoid stuffing everything into context. State at the end: pick based on whether the bottleneck is compute, memory, or position range.

**23. Design an attention variant to shrink the KV cache; quality tradeoff.**
The cache scales with the number of *KV heads*. **MHA** has $h$ KV heads (full quality, full cache). **MQA** uses a single shared KV head ($h\times$ smaller cache, big throughput win, but a measurable quality drop and training instability). **GQA** groups query heads to share $g$ KV heads ($1<g<h$) — recovers almost all MHA quality with most of MQA's memory savings, which is why GQA is the modern default (Llama-2/3). The knob is $g$: smaller = cheaper but lossier.

**24. Why is post-norm unstable at depth; what does pre-norm do to gradients?**
In post-norm, LN sits *on* the residual path, so the identity signal is repeatedly renormalized and its variance is controlled by LN rather than passed cleanly — the effective gradient to early layers shrinks, and the output variance can grow with depth, requiring warmup to avoid early divergence. Pre-norm normalizes only the *input* to each sub-layer, leaving the residual highway untouched: $\partial x_L/\partial x_\ell$ retains an identity term, so gradients reach depth and training is stable without delicate tuning.

**25. Numerical stability of softmax.**
$\text{softmax}(x)_i = e^{x_i}/\sum_j e^{x_j}$. Large $x_i$ overflows $e^{x_i}$ (inf in fp16, whose max ≈ 65504), and the result is invariant to a constant shift, so we compute $e^{x_i-\max_k x_k}/\sum_j e^{x_j-\max_k x_k}$. Now the largest exponent is $e^0=1$ (no overflow) and at least one denominator term is 1 (no divide-by-zero). Critical in fp16/bf16 where dynamic range is small; FlashAttention carries a running max to do this in a streaming, blockwise fashion.

**26. Many "dumb" failures trace to one design choice — which?**
**Tokenization.** The model's atom is the token, not the character. It can't reliably spell or reverse a word it sees as one token, struggles with digit-by-digit arithmetic because numbers tokenize irregularly, and chokes on unusual whitespace/formatting (e.g. Python indentation, the "SolidGoldMagikarp" glitch tokens). Fixes: better number handling (digit splitting), byte-level fallback, careful vocab.

**27. Redesign attention for streaming / infinite generation.**
Naive generation grows the KV cache without bound and degrades once you exceed training length. Use a **sliding window** (attend to the last $W$ tokens) plus **attention sinks** — keep the first few tokens permanently, because the softmax dumps "excess" attention mass onto them and dropping them collapses quality (StreamingLLM). Combine with cache eviction so memory is $O(W)$, enabling effectively infinite streams at bounded cost.

**28. Inductive biases: CNN/RNN vs transformer; why transformers won at scale.**
CNNs bake in locality + translation equivariance; RNNs bake in sequential/Markov recurrence and a fixed-size state. These strong priors help with little data but *cap* what can be learned and (RNN) prevent parallelism + create long-range gradient decay. Transformers have a *weak* prior (any token can attend to any token), so they need more data/compute but scale better and parallelize fully over sequence length — given internet-scale data and GPUs, low bias + high capacity wins (the "bitter lesson").

---

## 🧮 Math & derivations

**29. Full attention equation with shapes.**
$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$
$Q,K,V\in\mathbb{R}^{T\times d_k}$ (per head); $QK^\top\in\mathbb{R}^{T\times T}$ (score per query-key pair); $M\in\mathbb{R}^{T\times T}$ causal mask (0 or $-\infty$); softmax over the last (key) axis → rows sum to 1; output $\in\mathbb{R}^{T\times d_k}$. With heads/batch: $Q,K,V\in\mathbb{R}^{B\times h\times T\times d_k}$.

**30. Show a block has ≈ $12d^2$ params; where does $8d^2$ come from?**
Attention: four $d\times d$ projections (Q, K, V, output) $=4d^2$. MLP: up-projection $d\times 4d=4d^2$ and down-projection $4d\times d=4d^2$, total $8d^2$ (the ×4 expansion is why). Sum $=12d^2$ per block (LN/bias terms are $O(d)$, negligible). So total model ≈ $12 L d^2$ parameters.

**31. Justify "forward ≈ $2N$ FLOPs per token."**
Almost all compute is matmuls, and each parameter participates in one multiply-add per token = 2 FLOPs (1 multiply + 1 add). With $N$ parameters that's $2N$ FLOPs/token for the forward pass. Backward is ~2× forward, so training ≈ $6N$ FLOPs/token — the basis of the Chinchilla compute estimate $C\approx 6ND$.

**32. Gradient of softmax-cross-entropy w.r.t. logits.**
For loss $\mathcal{L}=-\log p_y$ where $p=\text{softmax}(z)$: $\partial\mathcal{L}/\partial z_i = p_i - \mathbb{1}[i=y]$, i.e. $\nabla_z\mathcal{L}=\text{softmax}(z)-\text{onehot}(y)$. The softmax Jacobian $\text{diag}(p)-pp^\top$ telescopes against the cross-entropy's $-1/p_y$, leaving this clean "predicted minus target" form — which is also why it's numerically friendly and why the fused `cross_entropy(logits, target)` is preferred over `log(softmax)`.

**33. Derive $O(T^2 d)$ cost; which matmul dominates at long $T$?**
$QK^\top$: $(T\times d_k)(d_k\times T)=T^2 d_k$ MACs. Score·$V$: $(T\times T)(T\times d_k)=T^2 d_k$. Summed over $h$ heads → $O(T^2 d)$. The QKV/output projections are $O(Td^2)$. So cost $\approx \alpha T d^2 + \beta T^2 d$; the **$T^2 d$ attention term dominates once $T\gtrsim d$** — i.e. long sequences are attention-bound, short ones are projection/MLP-bound.

---

## 💻 Coding / implementation

**34. Scaled dot-product attention with causal mask, from scratch.**
```python
import torch, torch.nn.functional as F

def attention(q, k, v, mask=None):
    # q,k,v: (B, h, T, d_k)
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / d_k ** 0.5      # (B, h, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = scores.softmax(dim=-1)                         # over keys
    return attn @ v, attn                                 # (B, h, T, d_k)

T = q.size(-2)
causal = torch.tril(torch.ones(T, T, device=q.device)).bool()
out, _ = attention(q, k, v, mask=causal)
```

**35. Reshape `(B,T,C)` ↔ `(B,h,T,d_k)`.**
```python
B, T, C = x.shape
# split heads
x = x.view(B, T, h, C // h).transpose(1, 2)   # (B, h, T, d_k)
# ... attention ...
# merge heads
x = x.transpose(1, 2).contiguous().view(B, T, C)
```
`.transpose` then `.contiguous()` before the merge `view` is required because transpose makes the tensor non-contiguous.

**36. One BPE merge step.**
```python
from collections import Counter

def get_stats(ids):
    return Counter(zip(ids, ids[1:]))          # adjacent pair frequencies

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

# usage: pick the most frequent pair, mint a new id, replace it
pair = max(get_stats(ids), key=get_stats(ids).get)
ids = merge(ids, pair, new_id=256)
```

**37. KV cache for single-token decoding.**
```python
class KVCache:
    def __init__(self): self.k = self.v = None
    def update(self, k_new, v_new):              # each (B, h, 1, d_k)
        self.k = k_new if self.k is None else torch.cat([self.k, k_new], dim=2)
        self.v = v_new if self.v is None else torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

# decode step: project only the new token, append, attend new q over full cache
q, k_new, v_new = proj(x_new)                    # x_new: (B,1,C)
k, v = cache.update(k_new, v_new)
out, _ = attention(q, k, v, mask=None)           # new q sees all past keys
```

**38. Implement RoPE and apply to Q, K.**
```python
def rope(x, base=10000.0):
    # x: (B, h, T, d_k), d_k even
    B, h, T, d = x.shape
    theta = base ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    pos = torch.arange(T, device=x.device).float()
    freqs = torch.outer(pos, theta)              # (T, d/2)
    cos, sin = freqs.cos(), freqs.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)

q, k = rope(q), rope(k)   # apply BEFORE the QK^T dot product
```
Key point: rotating Q and K by position makes $q_m\cdot k_n$ depend only on the relative offset $m-n$.

**39. Vectorize a naive triple-nested attention loop.**
The naive `for b: for i: for j:` computing $\sum_j \text{softmax}_j(q_i\cdot k_j)v_j$ collapses to three batched ops: `scores = q @ k.transpose(-2,-1) / d_k**0.5` (replaces the $i,j$ loops), one `softmax(dim=-1)`, then `scores @ v`. Batch ($b$) and head dims are leading axes handled by broadcasting. This turns $O(T^2)$ Python iterations into 2 GPU matmuls — orders of magnitude faster and the whole point of the architecture.

---

## 🏗️ System design / applied

**40. Tokenizer for NL + code + multilingual.**
- **Byte-level BPE** base so every byte is representable (no UNK across languages/emoji/code).
- **Vocab size** ~50–130k: bigger vocab = shorter sequences (cheaper attention) but a larger embedding table and rarer tokens; pick by measuring fertility (tokens/word) across all target languages so no language is starved.
- **Whitespace/indentation:** preserve it as tokens (critical for Python); consider tokens for runs of spaces. Don't normalize away case/accents.
- **Digits:** split into individual digits (or consistent chunks) to help arithmetic.
- **Code:** ensure common identifiers/operators and newline/tab are well-represented; train on a corpus mixture that reflects the real NL:code:multilingual ratio. Validate by checking compression and that no domain has pathological fertility.

**41. Architecture for long legal documents under a latency budget.**
Long inputs, mostly *understanding* + grounded generation. Favor: **RoPE (with interpolation/YaRN)** or ALiBi for length extrapolation; **GQA** to keep the KV cache affordable at long $T$; **FlashAttention** kernels; possibly **sliding-window + sparse global** tokens so cost isn't fully $O(T^2)$. Choose **deeper-over-wider** within the latency budget for better reasoning, but cap depth so per-token latency meets SLA; pick heads so $d_k\approx 64$–128. Strongly consider **retrieval/chunking** instead of stuffing the whole document — cheaper and often more accurate than max context.

---

## 🐞 Debugging

**42. Loss → ~0 but generation is gibberish.**
Classic **causal-mask leak**: during teacher-forced training the model can see future tokens (mask missing/wrong axis/off-by-one), so predicting the next token is trivial (loss→0), but at inference there is no future to peek at → gibberish. Check the mask is applied, lower-triangular, and aligned to the right axis; verify train-time loss on shuffled future fails.

**43. Loss → NaN after a few hundred steps. Suspects.**
No/short **LR warmup**, **LR too high**, **fp16 overflow** (switch to bf16 or use loss scaling), **missing gradient clipping**, bad weight **init** (too large), exploding activations from post-norm at depth, a divide-by-zero/log(0) in a custom loss, or corrupt/duplicated data causing a huge-gradient batch. Triage by adding grad-norm logging and clip, lowering LR, and switching precision.

**44. Model insensitive to word order. Where to look?**
**Positional information is missing or wrong** — embeddings not added, added on the wrong axis, zeroed by a bug, or RoPE not applied to Q/K. Sanity check: a model with positions should change output when you shuffle tokens; if it doesn't, positions aren't reaching attention.

**45. Generations loop/repeat. Diagnose.**
*Decoding causes:* greedy/low-temperature decoding falls into high-probability loops; fix with sampling (temperature/top-p), **repetition/no-repeat-ngram penalties**, or beam diversity. *Training causes:* undertrained model, exposure bias, or degenerate data. First try decoding changes (cheap); if even sampled text degenerates, suspect the model/training.

**46. Val loss ≫ train loss within the first epoch — what's suspicious?**
Within epoch 1 the model hasn't seen data twice, so a large gap isn't classic overfitting. Suspect a **train/val distribution mismatch or leakage bug**, a **preprocessing/tokenization difference** between splits, an incorrectly built validation set (different domain/length), **dropout/eval-mode** handled wrong, or **data ordering** (curriculum/sorted data making early train batches easy). Inspect a few val examples and confirm identical preprocessing.

---

## What strong answers share
Reason from **shapes and first principles**; connect every choice to **training stability** or **inference cost**; be able to **derive** ($\sqrt{d}$, $12d^2$, $2N$ FLOPs) and **implement** (attention, BPE, KV cache, RoPE); and know the *why behind the why*.

---
Back to [questions](interview-questions.md) · [Stage README](README.md) · [Index](../README.md)
