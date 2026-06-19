# Week 1-2: Transformer Internals

This folder contains the Week 1-2 materials for building transformer intuition from first principles, then tying everything together with a tiny GPT trained on real text.

## Recommended Order

| Step | Notebook / Guide | Focus |
|------|------------------|-------|
| 1 | [Transformer Concepts Guide](01_transformer_concepts_guide.md) | High-level reference for the full stack |
| 2 | [Self Attention](02_self_attention.ipynb) | Scaled dot-product attention, Q/K/V, masks |
| 3 | [Multi-Head Attention](03_multi_head_attention.ipynb) | Parallel heads, reshaping, output projection |
| 4 | [Positional Encoding](04_positional_encoding.ipynb) | Sinusoidal, learned, RoPE, ALiBi |
| 5 | [Layer Normalization](05_layer_normalization.ipynb) | LayerNorm, RMSNorm, pre-norm vs post-norm |
| 6 | [Feed-Forward Networks](06_feed_forward_networks.ipynb) | MLP, GELU, SwiGLU, GeGLU |
| 7 | [GPT Decoder](07_gpt_decoder.ipynb) | Decoder-only architecture and KV cache |
| 8 | [Tiny GPT End-to-End with BPE](08_tiny_gpt_end_to_end_bpe.ipynb) | Tokenizers, BPE, real-text training loop |

## Completion Checklist

- [ ] Understand Q/K/V projections and attention scores.
- [ ] Implement causal masking without future-token leakage.
- [ ] Explain why multi-head attention reshapes tensors.
- [ ] Compare sinusoidal, learned, RoPE, and ALiBi position methods.
- [ ] Compare LayerNorm, RMSNorm, pre-norm, and post-norm.
- [ ] Implement classic FFN and modern gated FFN variants.
- [ ] Train the tiny GPT notebook end to end and inspect samples.
