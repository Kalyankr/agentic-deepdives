# Chapter 20 — Diffusion & Multimodal Models · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-6-frontier/20-diffusion-multimodal.md)

---

## Interview answers

### Q: "Explain diffusion models from scratch."

There's a **fixed forward process** that gradually adds Gaussian noise to data over $T$ steps until it's pure noise, and a **learned reverse process** that removes noise one step at a time. The forward process has a closed form — you can jump straight to any noise level: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, where $\bar\alpha_t$ is the cumulative product of $(1-\beta_t)$. Training is just **MSE regression**: noise an example to a random timestep $t$, ask a network $\epsilon_\theta(x_t,t)$ to predict the noise that was added, minimize $\|\epsilon-\epsilon_\theta\|^2$. To generate, start from $\mathcal N(0,I)$ and iteratively subtract the predicted noise. The key intuition: generation is hard but **denoising is easy**, so we decompose an impossible task into many trivial ones.

### Q: "Why does diffusion train more stably than GANs?"

A GAN is an **adversarial minimax game** between generator and discriminator — there's no fixed loss surface, training chases a moving Nash equilibrium, and it's prone to **mode collapse** (the generator finds one output that fools the discriminator and stops exploring). Diffusion replaces the adversary with a **stationary supervised target** — predict the noise — so it's an ordinary regression with a stable loss, no second network to balance. That stability is *why* diffusion scaled to billion-parameter text-to-image systems where GANs plateaued.

### Q: "What is classifier-free guidance (CFG)?"

A way to make conditional generation follow the condition more strongly **without a separate classifier**. During training you randomly drop the condition a fraction of the time, so the same network learns both a conditional $\epsilon_\theta(x_t,c)$ and unconditional $\epsilon_\theta(x_t,\varnothing)$ prediction. At sampling you extrapolate: $\tilde\epsilon = \epsilon_\varnothing + s\,(\epsilon_c - \epsilon_\varnothing)$. The guidance scale $s>1$ amplifies the direction the condition points, trading diversity for prompt fidelity — it's the "prompt strength" knob users feel.

### Q: "Why latent diffusion instead of pixel diffusion?"

Pixels are expensive — a $512^2\times3$ image is ~790k values. Latent diffusion first trains a **VAE** to compress the image into a small latent (e.g. $64^2\times4$, ~50× smaller), runs the *entire* diffusion process in that latent space, and decodes back to pixels at the end. The denoiser (U-Net or DiT) conditions on text via **cross-attention**. This ~50× compute reduction is what made high-res text-to-image runnable on one consumer GPU — and is exactly what Stable Diffusion is.

### Q: "How does an LLM process images (VLM architecture)?"

A **vision encoder** (typically a CLIP-pretrained ViT) turns the image into a grid of patch embeddings. A small **projection** (MLP or a Perceiver-style resampler) maps those into the LLM's token-embedding space, producing a sequence of "soft visual tokens." You **concatenate** them with the text tokens and feed the combined sequence to the transformer, which attends across both — attention doesn't care that some vectors came from pixels. Training is two-stage: **alignment** (freeze LLM + encoder, train only the projector on captions) then **instruction tuning** (multimodal Q&A). That's the LLaVA recipe; native-multimodal models like Gemini/GPT-4o instead train fused from the start, often with a unified tokenizer that can also *generate* images.

### Q: "DDPM vs DDIM?"

Same trained network, different sampler. **DDPM** ancestral sampling is stochastic (adds fresh noise each step) and needs ~1000 steps. **DDIM** defines a non-Markovian, **deterministic** reverse process that produces comparable quality in 20–50 steps — a huge speedup for free. Distillation methods (consistency/LCM) push it to 1–4 steps. So "speed" in diffusion is mostly a *sampler* choice, not a retraining.

### Q: "ε-prediction vs x₀- vs v-prediction?"

Three algebraically equivalent ways to parameterize the same target. Given $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$, knowing any one of $\{\epsilon, x_0, v\}$ determines the others. **ε-prediction** (predict the noise) is the DDPM default and is well-conditioned at high noise; **v-prediction** ($v=\sqrt{\bar\alpha_t}\epsilon-\sqrt{1-\bar\alpha_t}x_0$) is better-behaved across *all* noise levels and is preferred for distillation and high-resolution. They're interchangeable targets for the same model.

---

## Exercise solutions

### Exercise 1 — Forward process noises data to a Gaussian

```python
import numpy as np

T = 200
betas = np.linspace(1e-4, 0.02, T)
alpha_bar = np.cumprod(1.0 - betas)

def make_spiral(n=1000):
    t = np.linspace(0, 3 * np.pi, n)
    r = t / (3 * np.pi)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)   # (n, 2)

x0 = make_spiral()
for t in (0, 50, 100, 199):
    eps = np.random.randn(*x0.shape)
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    print(f"t={t:3d}  mean={xt.mean():+.3f}  std={xt.std():.3f}")
```

**Result:** as $t$ grows, the structured spiral's signal coefficient $\sqrt{\bar\alpha_t}\to0$ and the noise coefficient $\sqrt{1-\bar\alpha_t}\to1$, so `std → 1` and `mean → 0`: the data converges to an isotropic Gaussian. This is *why* sampling can start from $\mathcal N(0,I)$ — the forward process provably ends there.

### Exercise 2 — Train a tiny denoiser and sample the spiral

```python
import torch, torch.nn as nn, numpy as np

T = 200
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)

def time_embed(t, dim=16):                       # sinusoidal timestep embedding
    freqs = torch.exp(torch.linspace(0, -4, dim // 2))
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)

class Denoiser(nn.Module):
    def __init__(self, d=2, h=128, tdim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + tdim, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
    def forward(self, x, t):
        return self.net(torch.cat([x, time_embed(t)], -1))

spiral = make_spiral(2000)
X = torch.tensor(spiral, dtype=torch.float32)
model = Denoiser(); opt = torch.optim.Adam(model.parameters(), 2e-3)

for step in range(4000):
    idx = torch.randint(0, len(X), (256,))
    x0 = X[idx]
    t = torch.randint(0, T, (256,))
    eps = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps
    loss = ((eps - model(xt, t)) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def sample(n=2000):
    x = torch.randn(n, 2)
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps = model(x, tt)
        mean = (x - (1 - alphas[t]) / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()
        x = mean + (betas[t].sqrt() * torch.randn(n, 2) if t > 0 else 0)
    return x.numpy()

print("final loss:", round(loss.item(), 4))     # ~0.3–0.5; samples trace the spiral
```

**Result:** after training, `sample()` returns points that visibly reconstruct the spiral from pure noise. The loss won't reach 0 (predicting noise at high $t$ is inherently uncertain), but the *sampler* still recovers the data manifold — the whole point of diffusion.

### Exercise 3 — Classifier-free guidance on two classes

```python
# Train conditionally with label dropout, then guide at sampling.
# Add a class embedding to the input; drop it (-> unconditional) 10% of the time.
class CondDenoiser(nn.Module):
    def __init__(self, d=2, h=128, tdim=16, n_cls=2, cdim=16):
        super().__init__()
        self.cls = nn.Embedding(n_cls + 1, cdim)         # index n_cls = "null"
        self.net = nn.Sequential(nn.Linear(d + tdim + cdim, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
    def forward(self, x, t, c):
        return self.net(torch.cat([x, time_embed(t), self.cls(c)], -1))

# during training: c_in = torch.where(torch.rand(B) < 0.1, NULL, labels)
# during sampling:  eps = eps_uncond + s * (eps_cond - eps_uncond)
def guided_eps(model, x, t, c, s, NULL):
    e_c = model(x, t, c)
    e_u = model(x, t, torch.full_like(c, NULL))
    return e_u + s * (e_c - e_u)
```

**Result:** with guidance scale `s=1` you recover ordinary conditional sampling; raising `s` to 3–7 makes samples cluster **tighter** on the chosen class's region (higher fidelity) at the cost of variety — the fidelity/diversity tradeoff CFG exposes.

### Exercise 4 — DDIM vs DDPM step count

```python
@torch.no_grad()
def ddim_sample(model, n=2000, steps=50, eta=0.0):
    ts = torch.linspace(T - 1, 0, steps).long()
    x = torch.randn(n, 2)
    for i, t in enumerate(ts):
        tt = torch.full((n,), int(t))
        eps = model(x, tt)
        a_t = abar[t]
        x0_pred = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()    # predict x0
        if i < len(ts) - 1:
            a_prev = abar[ts[i + 1]]
            x = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * eps   # deterministic (eta=0)
        else:
            x = x0_pred
    return x.numpy()
```

**Result:** DDIM at **50 steps** gives spiral samples close to DDPM's 1000-step quality; at **20 steps** they're slightly looser but recognizable; DDPM at 50 steps is visibly worse (its stochastic step needs many more iterations). Conclusion: DDIM buys ~20–50× fewer steps at similar quality — same model, smarter sampler.

### Exercise 5 — VLM projector (conceptual + code shape)

```python
import torch.nn as nn
# CLIP ViT-B/16 gives 768-d patch features; suppose the LLM's d_model = 4096.
class VisionProjector(nn.Module):           # the only trainable part in stage 1
    def __init__(self, d_vis=768, d_llm=4096):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_vis, d_llm), nn.GELU(),
                                  nn.Linear(d_llm, d_llm))
    def forward(self, patch_feats):          # (B, num_patches, 768)
        return self.proj(patch_feats)        # (B, num_patches, 4096) -> soft visual tokens
```

**Explanation.** **Stage 1 (alignment):** freeze the CLIP encoder *and* the LLM; train **only** this projector on image→caption pairs so the visual tokens land where the LLM expects token embeddings — the projector learns to "translate" vision into the LLM's language. **Stage 2 (instruction tuning):** unfreeze the LLM (often with LoRA) and train on multimodal instructions (VQA, OCR, charts) so the model learns to *reason* over the now-aligned visual tokens. The encoder usually stays frozen throughout. This is the minimal, compute-cheap recipe (LLaVA) that turns a text LLM into a capable VLM.

---

[← Career solutions](18-19-career-solutions.md) · [Solutions index](README.md) · [Next: Chapter 21 solutions →](21-deep-rl-solutions.md)
