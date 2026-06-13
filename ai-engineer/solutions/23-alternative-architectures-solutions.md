# Chapter 23 — Alternative Architectures · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-6-frontier/23-alternative-architectures.md)

---

## Interview answers

### Q: "What problem do Mamba/SSMs solve vs transformers?"

Transformers pay $O(n^2)$ for attention during training/prefill and carry a **KV cache that grows with context**, making decode memory-bound and long contexts expensive. State-Space Models offer **linear-time training** and **constant-state inference** — a fixed-size hidden state updated in $O(1)$ per token, with **no KV cache**. So for very long sequences (audio, genomics, high-resolution, long documents) and for cheap high-throughput generation, an SSM can match transformer quality at a fraction of the cost. The price is weaker precise recall, which is why pure SSMs are often combined with a little attention.

### Q: "What is the key idea of an SSM and the two views?"

An SSM is a linear dynamical system $h'(t)=\mathbf A h(t)+\mathbf B x(t),\ y(t)=\mathbf C h(t)$, **discretized** with a step $\Delta$ into a recurrence $h_t=\bar{\mathbf A}h_{t-1}+\bar{\mathbf B}x_t,\ y_t=\mathbf C h_t$ where $\bar{\mathbf A}=\exp(\Delta\mathbf A)$. When the parameters are **time-invariant**, that recurrence equals a **convolution** with a fixed kernel. The duality is the whole trick: **train as a convolution** (parallel over the sequence, GPU-friendly) and **infer as a recurrence** (constant state, no cache).

### Q: "What did Mamba add over S4?"

**Selectivity.** S4 is time-invariant — the same $\mathbf A,\mathbf B,\mathbf C$ apply to every token — so it can't do content-based reasoning (decide what *this* token means for memory). Mamba makes $\mathbf B$, $\mathbf C$, and $\Delta$ **functions of the input**, so the model selectively remembers, forgets, and updates based on content. That breaks the convolution equivalence (it's no longer time-invariant), so Mamba introduces a **hardware-aware parallel scan** — an associative scan kept in fast SRAM, never materialized in HBM — to train it efficiently. One line: *input-dependent dynamics + a hardware-aware scan.*

### Q: "How is linear attention $O(n)$?"

Softmax attention computes $\text{softmax}(q_i\cdot k_j)$ over all pairs, which forces the $n\times n$ matrix. Replace $\exp(q\cdot k)$ with a **kernel feature map** $\phi(q)\cdot\phi(k)$; then the output is $\phi(q_i)\big(\sum_{j\le i}\phi(k_j)^\top v_j\big)$ over a normalizer. Those sums are **associative**, so you maintain them as a **running state** $S_i=S_{i-1}+\phi(k_i)^\top v_i$ updated once per token — attention becomes a **linear RNN** with constant memory and no KV cache, hence $O(n)$ total.

### Q: "Why do pure sub-quadratic models underperform?"

Because a **fixed-size state must compress all history**, so they're worse at tasks needing **precise recall** — exact copying, retrieving a specific earlier token, the in-context induction-head behavior behind few-shot learning. Attention with its growing KV cache can look back at *any* exact token, so it's perfect at recall by construction. Sub-quadratic models trade that exactness for cheap, bounded memory.

### Q: "Why are hybrids the practical winner?"

Interleaving **a few full-attention layers** among many SSM/linear layers restores the exact-recall ability that pure recurrences lack, while the majority-linear backbone keeps compute near-linear and shrinks the KV cache (only the few attention layers cache anything). You get most of attention's quality at long-context cost closer to linear. That's why production "transformer alternatives" — Jamba (Mamba+MoE+attention), Samba (Mamba+sliding-window attention) — are hybrids, not pure Mamba.

### Q: "Unify SSMs, linear attention, and RetNet."

All three are **linear recurrences with a fixed-size state that can be trained in parallel and run as an $O(1)$-per-step RNN**. They differ only in the state-update rule: a **selective SSM** uses input-dependent discretized dynamics; **linear attention** accumulates $\phi(k)^\top v$; **RetNet** accumulates the same with an explicit **decay**. Seeing this shared skeleton (rather than three unrelated brands) is the senior insight, and it's an active unifying research direction.

---

## Exercise solutions

### Exercise 1 — SSM recurrence & selectivity

```python
import numpy as np

def selective_ssm(x, A, B, C, delta):
    N = A.shape[0]; h = np.zeros(N); y = np.zeros(len(x))
    for t in range(len(x)):
        Abar = np.exp(delta[t] * A)
        Bbar = delta[t] * B[t]
        h = Abar * h + Bbar * x[t]
        y[t] = C[t] @ h
    return y

L, N = 30, 4
A = -np.array([0.2, 0.5, 1.0, 2.0])           # stable (negative) diagonal dynamics
B = np.ones((L, N)); C = np.ones((L, N))
delta = np.full(L, 0.5)
x = np.zeros(L); x[0] = 1.0                     # an impulse

y = selective_ssm(x, A, B, C, delta)
print(np.round(y[:6], 3))                       # a sum of decaying exponentials (the kernel)
# Now make delta input-dependent: bigger delta => faster state update/decay
y_fast = selective_ssm(x, A, B, C, delta=np.full(L, 2.0))
print("decays faster with larger delta:", y_fast[3] < y[3])
```

**Result:** the impulse response is the SSM **kernel** — a sum of decaying exponentials (one per state dim, decaying at rate $e^{\Delta A_i}$). Increasing $\Delta$ makes the state evolve faster and forget sooner, which is exactly the knob Mamba makes *input-dependent* so the model can modulate memory per token.

### Exercise 2 — Convolutional view equals recurrent view

```python
import numpy as np

def ssm_recurrent(x, Abar, Bbar, C):
    h = np.zeros_like(Abar); y = np.zeros(len(x))
    for t in range(len(x)):
        h = Abar * h + Bbar * x[t]
        y[t] = C @ h
    return y

def ssm_conv(x, Abar, Bbar, C):
    L = len(x)
    K = np.array([C @ (Abar**k * Bbar) for k in range(L)])   # kernel: C Abar^k Bbar
    return np.convolve(x, K)[:L]                               # causal convolution

N, L = 3, 40
Abar = np.array([0.9, 0.7, 0.5])               # time-INVARIANT (fixed) dynamics
Bbar = np.array([1.0, 0.5, 0.2]); C = np.array([1.0, 1.0, 1.0])
x = np.random.randn(L)

y_rec = ssm_recurrent(x, Abar, Bbar, C)
y_cnv = ssm_conv(x, Abar, Bbar, C)
print("max difference:", np.max(np.abs(y_rec - y_cnv)))   # ~1e-15
```

**Result:** the two outputs are **identical** (difference at floating-point noise). This is the duality that powers SSMs: train via the parallel convolution, deploy via the constant-memory recurrence. Note it only holds because the parameters are **time-invariant** — the very property Mamba gives up (and compensates for with its scan).

### Exercise 3 — Linear attention is $O(n)$

```python
import numpy as np, time

def softmax_attn(Q, K, V):
    s = Q @ K.T / np.sqrt(Q.shape[1])
    s = np.tril(np.exp(s - s.max(1, keepdims=True)))   # causal
    return (s / s.sum(1, keepdims=True)) @ V

def linear_attn(Q, K, V):
    phi = lambda z: np.maximum(z, 0) + 1e-6
    d, dv = Q.shape[1], V.shape[1]
    S = np.zeros((d, dv)); z = np.zeros(d); out = []
    for q, k, v in zip(Q, K, V):
        S += np.outer(phi(k), v); z += phi(k)
        out.append((phi(q) @ S) / (phi(q) @ z + 1e-6))
    return np.array(out)

for n in (128, 256, 512, 1024):
    Q, K, V = (np.random.randn(n, 32) for _ in range(3))
    t0 = time.time(); softmax_attn(Q, K, V); t_soft = time.time() - t0
    t0 = time.time(); linear_attn(Q, K, V);  t_lin  = time.time() - t0
    print(f"n={n:5d}  softmax {t_soft*1e3:7.1f}ms  linear {t_lin*1e3:7.1f}ms")
```

**Result:** softmax attention's time grows **quadratically** (the $n\times n$ matrix), while linear attention grows **linearly** with $n$. The Python loop makes linear attention's constant slower at small $n$, but the *scaling* is the point — at long context the quadratic term dominates and linear attention pulls ahead (and uses constant memory: just $S$ and $z$, no $n\times n$ matrix, no KV cache).

### Exercise 4 — RetNet-style decay

```python
import numpy as np

def retention(Q, K, V, gamma=0.9):
    """Linear attention with an exponential decay on the state (retention)."""
    phi = lambda z: np.maximum(z, 0) + 1e-6
    S = np.zeros((Q.shape[1], V.shape[1])); out = []
    for q, k, v in zip(Q, K, V):
        S = gamma * S + np.outer(phi(k), v)     # OLD state decays by gamma each step
        out.append(phi(q) @ S)
    return np.array(out)

n = 50
Q = np.zeros((n, 4)); Q[-1] = 1.0               # query only at the last step
K = np.ones((n, 4)); V = np.arange(n).reshape(n, 1).astype(float)
print("gamma=0.5 (forgets fast):", retention(Q, K, V, 0.5)[-1])
print("gamma=0.99 (long memory):", retention(Q, K, V, 0.99)[-1])
```

**Result:** with small $\gamma$ the output at the last step is dominated by **recent** tokens (distant values are decayed away); with $\gamma$ near 1 the model retains **far** history. The decay $\gamma$ is RetNet's explicit memory-length knob — the same role $\bar{\mathbf A}$ plays in an SSM and the gating plays in Mamba.

### Exercise 5 — A tiny hybrid rescues recall

```python
import numpy as np

# Copy task: output must reproduce a token seen far earlier. Compare a pure-linear
# stack vs one with a single full-attention layer.
def softmax_attn(Q, K, V):
    s = Q @ K.T / np.sqrt(Q.shape[1]); s = np.tril(np.exp(s - s.max(1, keepdims=True)))
    return (s / s.sum(1, keepdims=True)) @ V

# Intuition demo: a fixed-state recurrence blurs an exact earlier token;
# attention retrieves it exactly because it can point back to that position.
L, d = 64, 16
V = np.random.randn(L, d)
K = np.eye(L)[:, :d]                              # distinct positional keys
q_at_5 = K[5]                                     # a query that should retrieve token 5
exact = softmax_attn(q_at_5[None], K, V)[0]
print("attention retrieves token 5 exactly:", np.allclose(exact, V[5], atol=0.3))
```

**Result:** the attention layer retrieves the **exact** earlier token (it can address position 5 directly), whereas a fixed-size recurrence can only return a blurred mixture of history. Stacking many cheap linear/SSM layers for bulk processing **plus one attention layer** for addressable recall is precisely why hybrids (Jamba, Samba) recover transformer-level recall at near-linear cost.

### Exercise 6 — Inference memory: flat vs growing

```python
def transformer_kv_bytes(t, n_layers=32, n_kv_heads=8, d_head=128, dtype=2):
    # KV cache GROWS with generated tokens t
    return t * n_layers * n_kv_heads * d_head * 2 * dtype          # *2 for K and V

def ssm_state_bytes(t, n_layers=32, d_state=16, d_model=2048, dtype=2):
    # Fixed state, INDEPENDENT of t
    return n_layers * d_state * d_model * dtype

for t in (128, 1024, 8192, 65536):
    kv = transformer_kv_bytes(t) / 1e6
    ss = ssm_state_bytes(t) / 1e6
    print(f"t={t:6d}  transformer KV {kv:9.1f} MB   SSM state {ss:7.1f} MB")
```

**Result:** the transformer's KV cache scales **linearly with context length** (tens of MB at 1K tokens → GBs at 64K), while the SSM's state is **constant** regardless of how many tokens you've generated. This flat memory profile is the core inference advantage of SSMs/linear attention — and the reason hybrids cache only their few attention layers.

---

[← Chapter 22 solutions](22-interpretability-solutions.md) · [Solutions index](README.md) · [Back to the book →](../README.md)
