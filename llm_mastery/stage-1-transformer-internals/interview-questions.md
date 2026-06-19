# Stage 1 — Interview Questions (full-fledged, all levels)

> **Scope:** screening (AS-II / L4) through **senior / staff / principal** (L5–L7) applied-scientist & ML-engineer depth. Angles: conceptual, mathematical, coding, system design, debugging.
>
> **How to use:** answer out loud in 60–90s, then take the follow-up. Saying "I'm not sure, but here's how I'd reason about it" + a correct derivation beats a memorized definition. The `→` notes are *what a strong answer covers* — don't peek until you've tried.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math · 💻 Coding · 🏗️ Design · 🐞 Debug
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals (rapid-fire)
1. Explain self-attention in one sentence.
2. Why do transformers need positional encodings at all?
3. Encoder-only vs decoder-only vs encoder-decoder — give a model + use case for each.
4. What does causal masking do, and where is it applied?
5. Why multiple attention heads instead of one big head?
6. What is the job of the feed-forward (MLP) sub-layer?
7. Why a residual connection around each sub-layer?
8. LayerNorm vs BatchNorm — why LayerNorm in transformers?
9. What is BPE and why byte-level?
10. What is the time and memory complexity of self-attention in sequence length?

## 🟡 Core (L4–L5)
11. Walk through every tensor shape from input token ids → logits.
12. Why divide attention scores by √d_k? What breaks without it?
13. Pre-norm vs post-norm — which do modern LLMs use and why?
14. Compare absolute, learned, RoPE, and ALiBi positional schemes. Why did RoPE win?
15. Why tie the input embedding and output projection weights?
16. Explain the "residual stream" mental model — how do blocks communicate?
17. GELU vs ReLU vs SwiGLU — what changed and why?
18. Where do most of a transformer's parameters live: attention or MLP? Justify.
19. Why is attention permutation-invariant without positional info?
20. How does the KV cache change what attention computes at inference vs training?

## 🔴 Senior / Staff deep dives (with follow-ups)
21. Derive why dot-product magnitude grows with dimension and how that motivates the √d scaling.
    → *strong answer:* model q,k entries as ~unit-variance independent; dot product of d terms has variance ∝ d, so std ∝ √d; dividing by √d keeps logits O(1) and softmax out of the saturated/low-gradient regime.
22. A product needs 128K-token context. What breaks, and what are your options?
    → *covers:* O(T²) attention compute + linear KV-cache growth; FlashAttention (IO), sparse/sliding-window, GQA/MQA, RoPE interpolation/YaRN, "lost in the middle."
23. Design an attention variant to shrink the KV cache. Explain the quality tradeoff.
    → *covers:* MQA (1 KV head) vs GQA (grouped) vs MHA; memory/throughput win vs small quality loss; why GQA is the modern default.
24. Why is post-norm unstable at depth, and what does pre-norm do to gradient flow?
25. Walk through numerical stability of softmax — the max-subtraction trick and why it matters in fp16/bf16.
26. Many "dumb" model failures (can't spell, bad arithmetic, weird whitespace) trace to one design choice. Which, and why?
    → *tokenization* — the token, not the character, is the atom.
27. How would you redesign attention for streaming / infinite generation?
    → *covers:* attention sinks, StreamingLLM, sliding window + cache eviction.
28. Compare the inductive biases of CNNs/RNNs vs transformers. Why did transformers win at scale?

## 🧮 Math & derivations
29. Write the full attention equation and annotate the shape of every term.
30. Show that a transformer block has ≈ 12·d² parameters (attention 4d² for Q,K,V,O + MLP 8d²). Where does the 8d² come from?
31. Justify the rule "forward pass ≈ 2N FLOPs per token" for an N-parameter model.
32. Sketch the gradient of softmax-cross-entropy w.r.t. logits. Why is it just `softmax − onehot`?
33. Derive the O(T²·d) cost of attention and identify which matmul dominates at long T.

## 💻 Coding / implementation
34. Implement scaled dot-product attention with a causal mask from scratch (no `nn.Transformer`).
35. Implement the reshape from `(B,T,C)` to `(B, n_head, T, head_dim)` and back.
36. Implement one BPE merge step (`get_stats` + `merge`).
37. Implement a KV cache for single-token incremental decoding.
38. Implement RoPE and apply it to Q and K.
39. (Pairing) Given a naive triple-nested attention loop, vectorize it.
> Reference scaffolding: [code/](code/README.md).

## 🏗️ System design / applied
40. Design a tokenizer for a mixed natural-language + source-code + multilingual corpus. What tradeoffs (vocab size, whitespace, digits, code)?
41. You're choosing the architecture for long legal documents. Pick depth/width/heads/positional scheme and justify under a latency budget.

## 🐞 Debugging / scenarios
42. Training loss drops near zero but generation is gibberish. Most likely cause?
    → *causal-mask leak* (model sees future tokens during training).
43. Loss becomes NaN after a few hundred steps. Enumerate suspects.
    → *no LR warmup, LR too high, fp16 overflow (use bf16), bad init, missing grad clip.*
44. The model is insensitive to word order. Where do you look?
    → *positional encoding not added / wrong axis.*
45. Generations loop and repeat the same phrase. Diagnose (training vs decoding causes).
46. Validation loss is much worse than training loss within the first epoch. What's suspicious?

## ✅ What strong candidates demonstrate
- Reason from **shapes and first principles**, not memorized lines.
- Connect each design choice to **training stability** or **inference cost**.
- Can **derive** (√d, param counts, FLOPs) and **implement** (attention, BPE, KV cache).
- Know the *why behind the why* (e.g., not just "RoPE is better" but the relative-position + extrapolation + no-params argument).

---
Related: the shorter **🔥 Mastery checks** in [README.md](README.md) are the minimum bar; this file is the full interview surface.
