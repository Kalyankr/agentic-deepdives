# Transformer Concepts Guide
## Complete Reference for Advanced Deep Learning

---

## Table of Contents
1. [Self-Attention](#1-self-attention)
2. [Multi-Head Attention](#2-multi-head-attention)
3. [Positional Encoding](#3-positional-encoding)
4. [Layer Normalization](#4-layer-normalization)
5. [Feed-Forward Networks](#5-feed-forward-networks)
6. [GPT Architecture](#6-gpt-architecture)

---

## 1. Self-Attention

### Intuition: Library Analogy
- **Query (Q)**: Your question - "What information do I need?"
- **Key (K)**: Labels/tags - "What information does each token have?"
- **Value (V)**: Content - "What is the actual information?"

### The Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Why divide by √d_k?**
- Dot products get very large when d_k is big
- Large values push softmax into regions with tiny gradients
- Scaling keeps gradients healthy for training

### Code Example

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Args:
        query: (batch, seq_len, d_k)
        key:   (batch, seq_len, d_k)
        value: (batch, seq_len, d_v)
        mask:  optional mask
    """
    d_k = query.size(-1)
    
    # Step 1: Compute scores (Q @ K^T)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Step 2: Scale
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask (for causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### Causal Masking (GPT-style)
Each token can only see **previous tokens**, not future ones:

```
Causal Mask:
          Tok0  Tok1  Tok2  Tok3
Tok0    [  1     0     0     0  ]  ← Only sees itself
Tok1    [  1     1     0     0  ]  ← Sees Tok0, Tok1
Tok2    [  1     1     1     0  ]  ← Sees Tok0, Tok1, Tok2
Tok3    [  1     1     1     1  ]  ← Sees all previous
```

```python
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
```

### Cross-Attention
Query comes from **target** sequence, Key/Value from **source** sequence.

Used in: Translation (decoder attends to encoder), Vision-Language models.

```python
# French (target) queries English (source)
Q = W_Q(french)   # What French words need
K = W_K(english)  # What English words have
V = W_V(english)  # English content
```

---

## 2. Multi-Head Attention

### Intuition
Instead of one attention, use **multiple parallel attention heads**. Each head can learn different patterns:
- Head 1: Subject-verb relationships
- Head 2: Adjective-noun relationships  
- Head 3: Positional patterns
- etc.

### The Formula

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### Key Insight: The Reshape Trick

**Question**: "If we split 512 dims into 8 heads of 64, doesn't each head lose information?"

**Answer**: No splitting happens! Each head's W_Q is a **512×64 matrix** that sees ALL 512 dimensions.

```
x (batch, seq, 512)
    ↓
W_Q projection: 512 → 512 (each slice can see ALL 512 dims)
    ↓
reshape: (batch, seq, 512) → (batch, seq, 8, 64)
    ↓
transpose: (batch, 8, seq, 64)  ← 8 heads, each with 64 dims
```

### Code Example

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 64
        
        # Single projection: 512 → 512 (covers all heads)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, S, D = x.shape
        
        # Project once, split into Q, K, V
        qkv = self.qkv_proj(x)  # (B, S, 3*512)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to separate heads
        q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: (B, 8, S, 64)
        
        # Attention per head (parallel!)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, 8, S, 64)
        
        # Concatenate heads
        out = out.transpose(1, 2).reshape(B, S, D)  # (B, S, 512)
        return self.out_proj(out)
```

### Do Heads Learn in Parallel?
**Yes!**
- Forward pass: All heads compute simultaneously (single matrix multiply)
- Backward pass: Gradients flow to all heads in parallel
- Each head naturally specializes through training

---

## 3. Positional Encoding

### Why Needed?
Attention is **permutation invariant** - it doesn't know token order. We need to inject position information.

### Method 1: Sinusoidal (Original Transformer)

**Intuition**: Like a clock with multiple hands
- Each dimension is a "hand" rotating at different speeds
- Position 0: All hands at starting position
- Position 1: Fast hands moved a lot, slow hands barely moved
- Any position → unique combination of hand positions

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        
        # Different frequency for each dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dims
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # Add to embeddings
```

**Pros**: Extrapolates to longer sequences, no learned params
**Cons**: Fixed, doesn't adapt to data

### Method 2: Learned Position Embedding

**Intuition**: A lookup table. Position 0 → learned vector, Position 1 → different learned vector, etc.

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Lookup table: each position gets a learnable vector
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pos_embedding(positions)
```

**Pros**: Learns optimal positions for the task
**Cons**: Can't extrapolate beyond max_len, needs initialization

### Method 3: RoPE (Rotary Position Embedding)

**Intuition**: Instead of **adding** position, **rotate** the vector!

Imagine a 2D vector. To encode position:
- Position 0: No rotation
- Position 1: Rotate by θ degrees
- Position 2: Rotate by 2θ degrees

**Why rotation is brilliant**:
- Dot product of Q at pos m and K at pos n depends on (m-n)
- Relative position is **built into the math**!
- No explicit subtraction needed

```python
# Simplified RoPE intuition
def rotate_2d(x, angle):
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack([
        x1 * cos_a - x2 * sin_a,
        x1 * sin_a + x2 * cos_a
    ], dim=-1)

# In practice: pair up dimensions and rotate each pair
def apply_rope(x, positions):
    # x: (batch, heads, seq, head_dim)
    # Rotate pairs of dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
    angles = positions.unsqueeze(-1) * freqs
    
    cos, sin = angles.cos(), angles.sin()
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
```

**Used in**: LLaMA, Mistral, GPT-NeoX, most modern LLMs

### Method 4: ALiBi (Attention with Linear Biases)

**Intuition**: No position embedding! Just add a bias to attention scores:
- Nearby tokens → small penalty
- Far tokens → large penalty

```python
# bias[i,j] = -m * |i - j|
# where m is a head-specific slope

def alibi_bias(seq_len, n_heads):
    # Each head has different slope
    slopes = 2 ** (-8 / n_heads * torch.arange(1, n_heads + 1))
    
    positions = torch.arange(seq_len)
    distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    
    # (n_heads, seq_len, seq_len)
    return -slopes.view(-1, 1, 1) * distances.unsqueeze(0)
```

**Used in**: BLOOM, some efficient transformers

### Comparison Table

| Method | Type | Extrapolates? | Params | Used In |
|--------|------|---------------|--------|---------|
| Sinusoidal | Absolute | ✓ | 0 | Original Transformer |
| Learned | Absolute | ✗ | d_model × max_len | BERT, GPT-2 |
| RoPE | Relative | ✓ | 0 | LLaMA, Mistral |
| ALiBi | Relative | ✓ | 0 | BLOOM |

---

## 4. Layer Normalization

### Why Normalize?
Deep networks suffer from **internal covariate shift**:
```
Layer 1 output: values in range [-100, 100]
Layer 2 output: values in range [-10000, 10000]  ← Exploding!
Layer 3 output: values in range [-0.0001, 0.0001] ← Vanishing!
```

Normalization keeps values in a reasonable range for stable training.

### LayerNorm vs BatchNorm

```
Input shape: (Batch, Sequence, Features)

BatchNorm: Normalize across BATCH (each feature independently)
           → Depends on batch size, bad for variable sequences

LayerNorm: Normalize across FEATURES (each sample independently)
           → Works with batch_size=1, perfect for transformers
```

### LayerNorm Formula

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- μ = mean across features
- σ² = variance across features
- γ = learnable scale (init to 1)
- β = learnable shift (init to 0)

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # Scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # Shift
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### RMSNorm (Modern Alternative)

Used in **LLaMA, Mistral** - simpler and faster!

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x)}$$

**Key difference**: No mean subtraction, no β parameter.

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

**~15% faster than LayerNorm, similar quality!**

### Pre-Norm vs Post-Norm

**Post-Norm (Original Transformer)**:
```
x → Attention → Add(x) → LayerNorm → FFN → Add → LayerNorm → output
```

**Pre-Norm (Modern - LLaMA, GPT-3)**:
```
x → LayerNorm → Attention → Add(x) → LayerNorm → FFN → Add(x) → output
```

**Why Pre-Norm is better**:
- Residual path stays "clean" - gradients flow directly
- More stable for very deep networks (100+ layers)
- Easier to train without careful learning rate tuning

---

## 5. Feed-Forward Networks

### Role in Transformer
After attention mixes information **between tokens**, FFN processes each token **independently**:
- Attention: "Which other tokens should I pay attention to?"
- FFN: "Now that I have that context, what does it mean?"

**FFN contains ~2/3 of all transformer parameters!**

### Classic FFN (Original Transformer)

```
Input (d_model) → Linear → ReLU → Linear → Output (d_model)
      512       →  2048  →      →  512
```

The 4x expansion provides "thinking space" with more capacity.

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
```

### Activation Evolution

| Era | Model | Activation |
|-----|-------|------------|
| 2017 | Original Transformer | ReLU |
| 2018 | BERT, GPT | GELU |
| 2023 | LLaMA, Mistral | SwiGLU |

### Gated Linear Units (GLU)

**Key insight**: Use one pathway to **gate** the other.

```
Traditional:  value → activation(value) → output
GLU:          value × gate → output
                    ↑
              learned separately!
```

### SwiGLU (State-of-the-Art)

Used in **LLaMA, Mistral, PaLM** - most modern LLMs.

$$\text{SwiGLU}(x) = (xW_{up}) \odot \text{SiLU}(xW_{gate})$$

where SiLU(x) = x × σ(x)

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        # Reduced d_ff to compensate for 3 matrices
        d_ff = d_ff or int(4 * d_model * 2 / 3)
        
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # SiLU activation
        up = self.w_up(x)
        return self.w_down(gate * up)
```

**Why 3 layers but similar params?**
- Classic: 2 layers, d_ff = 4 × d_model
- SwiGLU: 3 layers, d_ff = 4 × d_model × (2/3)

### FFN as Knowledge Storage

Research shows FFN layers act like **key-value memory**:
- W_up (first layer): Keys - pattern matching
- W_down (second layer): Values - what to output when pattern matches

---

## 6. GPT Architecture

### Overview

```
Input Tokens
     ↓
Token Embedding (vocab_size → d_model)
     ↓
[Optional: Position Embedding or RoPE]
     ↓
┌─────────────────┐
│   GPT Block 1   │ ← Each: Attention + FFN
├─────────────────┤
│   GPT Block 2   │
├─────────────────┤
│      ...        │ (N blocks)
├─────────────────┤
│   GPT Block N   │
└─────────────────┘
     ↓
RMSNorm (final)
     ↓
LM Head (d_model → vocab_size)
     ↓
Output Logits
```

### GPT Block (Pre-Norm Style)

```
x ─┬─► RMSNorm ─► Attention ─► + ─┬─► RMSNorm ─► FFN ─► + ─► output
   │                          │   │                    │
   └──────────────────────────┘   └────────────────────┘
        (residual)                    (residual)
```

### Complete Code

```python
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = None  # Auto: 4 * d_model * 2/3
    dropout: float = 0.0
    bias: bool = False


class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn = SwiGLU(config)
        
    def forward(self, x, kv_cache=None):
        # Pre-Norm: normalize THEN apply
        x = x + self.attention(self.ln1(x), kv_cache)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.tok_emb.weight = self.lm_head.weight
        
    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
```

### KV Cache for Fast Inference

**Without cache**: Recompute K, V for ALL tokens every step → O(n²)
**With cache**: Only compute K, V for NEW token → O(n)

```python
# Inference with KV cache
for i in range(max_new_tokens):
    if kv_cache is not None:
        # Only process last token
        curr_input = generated[:, -1:]
    else:
        curr_input = generated
    
    logits, kv_cache = model(curr_input, kv_cache)
    next_token = sample(logits[:, -1])
    generated = torch.cat([generated, next_token], dim=1)
```

### Model Sizes

| Model | d_model | n_layers | n_heads | Params |
|-------|---------|----------|---------|--------|
| GPT-2 Small | 768 | 12 | 12 | 124M |
| GPT-2 XL | 1600 | 48 | 25 | 1.5B |
| LLaMA 7B | 4096 | 32 | 32 | 7B |
| LLaMA 70B | 8192 | 80 | 64 | 70B |

### Modern vs Original

| Original (2017) | Modern (LLaMA style) |
|-----------------|----------------------|
| Post-Norm | Pre-Norm (more stable) |
| LayerNorm | RMSNorm (faster) |
| Sinusoidal PE | RoPE (relative position) |
| ReLU FFN | SwiGLU (better quality) |
| With bias | No bias (fewer params) |

---

## Quick Reference: Modern Transformer Recipe

```
✅ Pre-Norm with RMSNorm
✅ RoPE (Rotary Position Embedding)
✅ SwiGLU Feed-Forward Network
✅ No bias in Linear layers
✅ KV cache for inference
✅ Weight tying (embedding = LM head)
```

---

*Generated from Advanced Deep Learning course notebooks*
