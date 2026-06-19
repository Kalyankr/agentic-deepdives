# Stage 1 — Transformer Internals (the bedrock)

> **Objective:** Implement a decoder-only transformer **from scratch** and explain every tensor shape, every matmul, and every design choice on a whiteboard. This is the foundation everything else stands on. Do not rush it.

[← Back to index](../README.md) · Next: [Stage 2 — Pretraining](../stage-2-pretraining-at-scale/README.md)

📝 **Interview prep:** [interview-questions.md](interview-questions.md) · ✅ [answer key](answers.md)

---

## Why this stage matters

Every later stage — fine-tuning, alignment, quantization, KV-cache, serving — is a modification of this core machinery. If you deeply understand the transformer forward pass, 80% of "advanced" topics become obvious variations. People who are vague here stay vague forever.

**You are done with this stage when:** you can rebuild a GPT forward pass from a blank file, without looking anything up.

---

## Mental model (hold this in your head)

A decoder-only LLM is a stack of identical blocks that repeatedly do two things:
1. **Mix information across tokens** (attention) — "let each token look at previous tokens."
2. **Process each token independently** (MLP) — "think harder about each token's representation."

Everything else (norms, residuals, positional info) exists to make that stack *trainable* and *position-aware*.

```
tokens → embed → [ Attention → MLP ] × N → final norm → unembed → logits → softmax → next-token probs
```

---

## Concept-by-concept deep dive

### 1.1 Tokenization (input boundary)
- **What:** text → integer IDs the model can embed.
- **BPE (Byte-Pair Encoding):** start from characters/bytes, iteratively merge the most frequent adjacent pair into a new token. Repeat until vocab size is reached.
- **Byte-level BPE** (GPT-2): operates on raw bytes, so it can encode *any* string with no "unknown" token.
- **Why vocab size matters:** bigger vocab → shorter sequences (cheaper attention) but larger embedding matrix and softmax. Typical: 32k–128k.
- **Gotchas:** tokenization explains many "dumb" model behaviors (can't spell, struggles with arithmetic, weird whitespace handling). The token is the model's atom, not the character.

> **Lab:** implement BPE training + encode/decode on a text file. Verify `decode(encode(x)) == x`.

### 1.2 Embeddings
- Token IDs index into an embedding matrix `E ∈ ℝ^{vocab × d_model}`.
- Output: a vector per token. This is the only place raw token identity enters.
- **Weight tying:** reuse `E` (transposed) as the output unembedding. Saves params, often improves quality.

### 1.3 Self-attention (the heart)
The core equation:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- **Q, K, V:** three linear projections of the input. Intuition: each token emits a **query** ("what am I looking for?"), a **key** ("what do I offer?"), and a **value** ("what I'll pass on if matched").
- **`QKᵀ`:** similarity of every query to every key → an `(seq × seq)` score matrix.
- **`/√d_k`:** scaling. Without it, dot products grow with dimension, pushing softmax into saturated regions where gradients vanish. This is *the* reason for the scale factor.
- **softmax:** turns scores into a probability distribution over tokens to attend to.
- **`·V`:** weighted sum of values = the new representation.

**Shapes (single head):** input `(B, T, d)` → Q,K,V each `(B, T, d_k)` → scores `(B, T, T)` → output `(B, T, d_k)`.

### 1.4 Multi-head attention
- Split `d_model` into `h` heads of size `d_k = d_model/h`. Run attention in parallel per head, concat, then a final output projection `W_O`.
- **Why:** different heads learn different relationships (syntax, coreference, position). One big head can't specialize.
- **Shapes:** `(B, T, d)` → reshape to `(B, h, T, d_k)` → attention per head → concat back to `(B, T, d)` → `W_O`.

### 1.5 Causal masking
- For autoregressive LMs, token *t* must **not** see tokens > *t* (that would be cheating — leaking the answer).
- Implement by adding `-∞` to the upper triangle of the score matrix before softmax → those positions get probability 0.
- This single trick is what lets you train on **all positions at once** (next-token prediction in parallel) while preserving causality.

### 1.6 Feed-forward / MLP block
- Two linear layers with a nonlinearity: `W_2 · act(W_1 · x)`. Inner dimension is usually `4 × d_model`.
- **Activations:** ReLU → GELU → **SwiGLU** (gated, used in LLaMA). Know that SwiGLU uses a gating branch and typically a `~8/3 × d_model` inner dim to keep params comparable.
- Applied **per-token, identically** — this is where most parameters live.

### 1.7 Residuals + Normalization
- **Residual stream:** `x = x + Attention(norm(x))` then `x = x + MLP(norm(x))`. The residual stream is the "highway" information flows along; blocks read from and write to it.
- **LayerNorm vs RMSNorm:** RMSNorm (LLaMA) drops the mean-centering, cheaper, works as well.
- **Pre-norm vs post-norm:** *pre-norm* (norm inside the residual branch) trains far more stably for deep stacks — this is why modern LLMs use it. Post-norm (original transformer) needs careful warmup.

### 1.8 Positional information
Attention is **permutation-invariant** — without position signals it can't tell word order. Options:
- **Sinusoidal** (original): fixed sin/cos patterns added to embeddings.
- **Learned absolute** (GPT-2): a learned vector per position. Simple, but doesn't extrapolate past trained length.
- **RoPE (Rotary)** (LLaMA, most modern): rotates Q and K by position-dependent angles, encoding *relative* position directly in the attention dot product. Extrapolates better; the default today.
- **ALiBi:** adds a linear distance penalty to attention scores. Strong length extrapolation.

> Know **why RoPE won**: relative position, no added parameters, good extrapolation, plays well with KV-cache.

### 1.9 Output: logits → loss
- Final norm → multiply by unembedding `(d_model × vocab)` → **logits**.
- Training loss = **cross-entropy** between predicted next-token distribution and the true next token, averaged over all positions.

---

## Ordered learning path (do in this sequence)

1. Read *The Illustrated Transformer* (Alammar) for visual intuition.
2. Watch Karpathy's *Let's build GPT* end to end — **type the code yourself**, don't copy-paste.
3. Read *Attention Is All You Need* — now it'll make sense.
4. Read the RoPE section of the RoFormer paper.
5. Re-implement nanoGPT from a blank file (see labs).

---

## 🛠️ Hands-on labs (progressive)

- [ ] **Lab A — BPE tokenizer:** train BPE on a corpus; implement encode/decode; confirm round-trip.
- [ ] **Lab B — Single attention head:** implement scaled dot-product attention with a causal mask on toy tensors; print every shape.
- [ ] **Lab C — Full block:** multi-head attention + MLP + pre-norm residuals.
- [ ] **Lab D — Full GPT:** stack N blocks, add embeddings + positional encoding + LM head; train on TinyShakespeare; sample text.
- [ ] **Lab E — Ablations:** remove the `√d` scaling, remove positional encoding, switch pre→post norm. **Observe and explain** what breaks. This builds real intuition.
- [ ] **Lab F (stretch):** swap learned positions for RoPE; confirm training still works.

---

## ⚠️ Common pitfalls & gotchas

- Forgetting the causal mask → model "cheats," train loss looks great, generation is garbage.
- Applying softmax over the wrong dimension (must be over keys/last dim).
- Off-by-one in shapes when reshaping for heads — print shapes obsessively.
- Confusing `d_model`, `d_k`, and inner MLP dim.
- Thinking in characters instead of tokens when reasoning about model behavior.
- Adding positional encoding after attention instead of to the embeddings (for absolute schemes).

---

## 🔥 Mastery checks (answer without notes)

- [ ] Derive the output shape of multi-head attention from input `(B, T, d_model)` with `h` heads.
- [ ] Explain precisely why we divide by `√d_k`. What fails without it?
- [ ] Why is attention O(T²) in time **and** memory? Where exactly does the cost live?
- [ ] What does causal masking enable during training, and how is it implemented?
- [ ] Explain the residual stream as a communication channel between blocks.
- [ ] Why did the field move from post-norm → pre-norm, and from learned positions → RoPE?
- [ ] Given a model with `d_model=4096, n_layers=32, vocab=128k`, estimate the parameter count and say where most params live (MLP vs attention vs embeddings).
- [ ] Explain two real model failures that are actually **tokenization** artifacts.

---

## ✅ Stage 1 checklist

- [ ] Read the 4 core resources
- [ ] Labs A–E complete
- [ ] Can rebuild nanoGPT forward pass from memory
- [ ] All mastery checks passable without notes
- [ ] Notes written in your own words (`notes/`)

**When all boxes are checked → proceed to [Stage 2](../stage-2-pretraining-at-scale/README.md).**
