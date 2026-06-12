# Module 02 · Transformer Internals

> **Goal:** Know the transformer so well you can implement GPT from scratch, derive its memory/FLOP costs, and explain every modern architectural choice (RoPE, GQA, RMSNorm, MoE, FlashAttention). This is **the** module for frontier-lab interviews.

**Duration:** ~6 weeks. **Prereqs:** [Module 01](01-deep-learning-foundations.md).

---

## 2.1 Attention, derived

The core operation. Given queries $Q$, keys $K$, values $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- **Why scale by $\sqrt{d_k}$?** To keep dot-product variance ~1 so softmax doesn't saturate.
- **Self-attention vs. cross-attention.**
- **Causal masking** for autoregressive LMs (no peeking at the future).
- **Multi-head attention (MHA):** $h$ parallel attention "views," concatenated and projected.

> **Build:** Implement single-head, then multi-head attention in pure PyTorch with `einsum`. Verify against `nn.MultiheadAttention`. Write the causal mask yourself.

## 2.2 The full transformer block

- Token + positional embeddings
- Pre-norm vs. post-norm (modern LLMs use **pre-norm** for stability)
- Multi-head self-attention sublayer + residual
- Position-wise FFN (MLP): `Linear → activation → Linear`, expansion ratio ~4×
- LayerNorm / **RMSNorm**
- Residual stream as the "highway" — a key mental model
- Weight tying (input embedding ↔ output projection)

## 2.3 Positional information

- Why transformers need it (attention is permutation-invariant)
- Absolute (learned, sinusoidal)
- **RoPE (Rotary Position Embeddings)** — the modern default; rotation in complex space, enables context extension
- ALiBi (linear biases)
- Long-context tricks: position interpolation, YaRN, NTK-aware scaling

## 2.4 Modern attention & efficiency

- **KV cache** — the single most important inference concept (cache past keys/values to avoid recompute). You'll build this in [Module 04](04-gpu-architecture-and-inference.md).
- **MQA (Multi-Query)** and **GQA (Grouped-Query)** attention — shrink the KV cache by sharing K/V heads (Llama 2/3, Mistral use GQA)
- **FlashAttention (v1/v2/v3)** — IO-aware, tiled attention that avoids materializing the $N\times N$ matrix; memory goes from $O(N^2)$ to $O(N)$
- Sliding-window & sparse attention (Longformer, Mistral)
- **MoE (Mixture of Experts)** — sparse FFNs, top-k routing, load balancing; how Mixtral/DeepSeek scale params without scaling FLOPs per token

## 2.5 Tokenization

- Byte-Pair Encoding (BPE), WordPiece, **SentencePiece**, byte-level BPE (GPT-2+)
- Vocabulary size trade-offs, special tokens, chat templates
- Why tokenization causes weird failures (arithmetic, spelling, non-English)

> **Build:** Train a BPE tokenizer on a corpus (use `tokenizers` lib, then understand the merge algorithm). Compare vocab sizes and compression ratios.

## 2.6 Architecture families

- **Encoder-only** (BERT) — bidirectional, for understanding/embeddings
- **Decoder-only** (GPT, Llama, Claude, Mistral) — autoregressive generation; the dominant LLM design
- **Encoder–decoder** (T5, original Transformer) — translation, seq2seq
- Know *why* decoder-only won for general-purpose LLMs

## 2.7 The math you must be able to do on a whiteboard

For a decoder-only model with $L$ layers, hidden size $d$, $n_h$ heads, vocab $V$, sequence length $N$:

- **Parameter count** ≈ $12 L d^2$ (dominant term: attention + FFN), plus $V d$ for embeddings. Be able to derive each piece.
- **FLOPs per token (forward)** ≈ $2 \times \text{params}$. Training forward+backward ≈ $6 \times \text{params} \times \text{tokens}$ (the famous $C \approx 6ND$).
- **KV cache size** = $2 \times L \times N \times d_{kv} \times \text{bytes}$ (the 2 is K and V). This is what limits batch size at inference.
- **Activation memory** during training and why gradient checkpointing helps.

> **Drill:** Given "7B params, 32 layers, d=4096, FP16, 4k context," compute params, KV-cache per sequence, and minimum serving memory. You'll do this live in interviews.

## 2.8 Scaling laws

- Kaplan et al. (2020) — power laws relating loss to params/data/compute
- **Chinchilla** (Hoffmann et al., 2022) — compute-optimal scaling (~20 tokens/param); why it changed everything
- Emergent abilities and the debate around them
- Inference-time scaling (more on this in Modules 03 & 07)

---

## Module 02 capstone — **Build nanoGPT yourself**

1. **GPT from scratch**: implement embeddings, multi-head causal attention, FFN, pre-norm blocks, weight tying. Train on TinyShakespeare and then a larger corpus (e.g., a subset of OpenWebText/FineWeb). Generate coherent text.
2. **Modernize it**: swap in **RMSNorm**, **RoPE**, **SwiGLU** FFN, and **GQA**. Benchmark loss/throughput vs. the vanilla version.
3. **Implement a KV cache** for generation and measure the speedup vs. recompute.
4. **Write-up**: derive your model's parameter count and KV-cache size and confirm against the actual tensors.

## Exit criteria
- [ ] You can implement multi-head causal attention from memory.
- [ ] You can explain RoPE, GQA, RMSNorm, SwiGLU, and FlashAttention and *why* each exists.
- [ ] You can derive params, FLOPs, and KV-cache size on a whiteboard.
- [ ] You can explain Chinchilla-optimal scaling and its implications.

## Core papers (read these directly)
- *Attention Is All You Need* — Vaswani et al., 2017
- *Language Models are Few-Shot Learners* (GPT-3) — Brown et al., 2020
- *RoFormer* (RoPE) — Su et al., 2021
- *GQA* — Ainslie et al., 2023
- *FlashAttention* (v1 & v2) — Dao et al., 2022/2023
- *Mixtral of Experts* — Jiang et al., 2024
- *Training Compute-Optimal LLMs* (Chinchilla) — Hoffmann et al., 2022
- *Llama 2 / Llama 3* technical reports

## Supplementary
- "The Illustrated Transformer" — Jay Alammar
- "The Annotated Transformer" — Harvard NLP
- Karpathy — *Let's build GPT* + *nanoGPT* repo
