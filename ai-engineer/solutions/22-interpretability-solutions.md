# Chapter 22 — Mechanistic Interpretability · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-6-frontier/22-interpretability.md)

---

## Interview answers

### Q: "Probing vs causal intervention — what's the difference and why does it matter?"

**Probing** trains a small classifier on a layer's activations to test whether a concept is *decodable* there. If it succeeds, the concept is linearly available — but that's **correlational**: the model might encode the concept without *using* it for its output. **Causal intervention** (activation patching, ablation) actually *changes* an activation and measures the effect on behavior; if patching a component restores or destroys the correct output, that component is **causally responsible**. The distinction matters because interpretability makes *mechanism* claims, and only intervention supports them — a probe alone can fool you into a confident-but-wrong story.

### Q: "What is superposition?"

The phenomenon where a network represents **more features than it has dimensions** by storing them as **overlapping, nearly-orthogonal directions** rather than one-feature-per-neuron. It works because real features are **sparse** (rarely active simultaneously), so the occasional interference between overlapping directions is tolerable. The consequence is **polysemantic neurons** — a single neuron fires for several unrelated concepts — which is why you can't interpret a model by reading neurons one at a time. Superposition is the core obstacle that sparse autoencoders are built to overcome.

### Q: "What's a sparse autoencoder and what problem does it solve?"

An **over-complete autoencoder** (hidden layer much wider than the input, e.g. 8–32×) trained on a model's activations with an **L1 sparsity penalty** plus reconstruction loss: $\mathcal L=\|x-\hat x\|^2+\lambda\|f\|_1$. The width gives room to *unfold* superposed features into separate dimensions; the sparsity forces only a few to activate per input, which makes each learned feature tend to be **monosemantic** (one clean concept). It solves the superposition problem: instead of polysemantic neurons, you get thousands of nameable features — Anthropic's "Scaling Monosemanticity" extracted millions from Claude, including the steerable Golden Gate Bridge feature.

### Q: "Explain induction heads."

A two-attention-head **circuit** that implements in-context copying: given a sequence `…[A][B]…[A]`, it predicts `[B]`. A **previous-token head** writes "the token before me was A" into each position's residual stream; a later **induction head** uses that to attend back to the earlier spot that was preceded by `A` and copies the token that followed (`B`). It's significant because it's a **discovered algorithm** — not designed by humans — and is believed to be a primary mechanism behind **in-context learning**, the ability of LLMs to learn from examples in the prompt.

### Q: "What is the residual-stream view of a transformer?"

Because of residual connections, each attention head and MLP **reads** from a shared "residual stream" and **adds** its output back into it; the final logits are the **sum** of contributions from every component across all layers. This additive, linear structure reframes "what does this layer do?" as "what does it *write* to the stream and which later component *reads* it?", which is what makes circuits traceable. The **logit lens** exploits it: apply the final unembedding to the residual stream at intermediate layers to watch the prediction form layer by layer.

### Q: "What is activation steering?"

Controlling behavior by **adding a concept direction directly to activations at inference**: $h'=h+\alpha v$, where $v$ is a probe weight vector or an SAE decoder direction. Amplify a "Golden Gate" feature and the model fixates on the bridge; **subtract a "refusal direction"** and the model stops refusing; add a "truthfulness" direction to curb hallucination. It's cheap, reversible, needs no retraining, and is a *mechanistic* complement to RLHF — though the fact that one subtractable direction can disable safety refusals is itself an important security finding.

### Q: "Why does interpretability matter for safety specifically?"

Behavioral evaluation only tells you what a model *does* on the cases you tested; it can't rule out **deception** or hidden goals — a model can output aligned text while internally computing something else, and can behave differently on inputs you didn't try. Interpretability inspects the *computation*, offering the possibility of **detecting** misaligned cognition, **auditing** what concepts drive a decision, and **steering** behavior at the representation level. The principle: you can't trust what you can't inspect — which is why frontier safety teams treat interpretability as foundational.

---

## Exercise solutions

### Exercise 1 — Linear probe across layers

```python
# Using TransformerLens for a small GPT-2.
import torch, torch.nn as nn
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
# Two concept classes of prompts (e.g., sports vs cooking sentences).
sports = ["The striker scored a goal", "She served an ace at match point", ...]
cooking = ["Whisk the eggs and sugar", "Simmer the sauce for an hour", ...]

def resid_acts(prompts, layer):
    acts = []
    for p in prompts:
        _, cache = model.run_with_cache(p)
        acts.append(cache["resid_post", layer][0, -1])   # last-token residual
    return torch.stack(acts)

for layer in range(model.cfg.n_layers):
    X = torch.cat([resid_acts(sports, layer), resid_acts(cooking, layer)])
    y = torch.tensor([0] * len(sports) + [1] * len(cooking))
    probe = nn.Linear(X.size(1), 2)
    opt = torch.optim.Adam(probe.parameters(), 1e-2)
    for _ in range(300):
        loss = nn.functional.cross_entropy(probe(X.detach()), y)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = (probe(X).argmax(-1) == y).float().mean().item()
    print(f"layer {layer:2d}  probe acc {acc:.2f}")
```

**Result:** probe accuracy is near chance in the **earliest** layers (still mostly token-level), rises through the **middle** layers as the topic becomes linearly represented, and stays high later. The layer where accuracy jumps is where the concept "emerges." Remember the caveat: high accuracy proves the topic is **decodable**, not that the model *uses* it — that needs intervention.

### Exercise 2 — Logit lens

```python
def logit_lens(model, prompt):
    _, cache = model.run_with_cache(prompt)
    for layer in range(model.cfg.n_layers):
        resid = cache["resid_post", layer][0, -1]            # last position
        resid = model.ln_final(resid)                         # apply final norm
        logits = model.unembed(resid[None, None])[0, 0]       # to vocab
        top = model.to_string(logits.argmax())
        print(f"layer {layer:2d} -> {top!r}")
```

**Result:** for a prompt like "The Eiffel Tower is in the city of", early layers' top tokens are generic/unrelated, and the correct answer ("Paris") **sharpens into the top slot in the later layers** — a vivid demonstration that predictions are *built up gradually* through the residual stream rather than computed in one place.

### Exercise 3 — Tiny SAE and naming features

```python
import torch, torch.nn as nn

class SAE(nn.Module):
    def __init__(self, d, d_hidden):
        super().__init__()
        self.enc = nn.Linear(d, d_hidden); self.dec = nn.Linear(d_hidden, d, bias=False)
    def forward(self, x):
        f = torch.relu(self.enc(x)); return self.dec(f), f

# acts: (N, d) MLP activations collected from a layer over a text corpus
sae = SAE(d=acts.size(1), d_hidden=8 * acts.size(1))           # 8x over-complete
opt = torch.optim.Adam(sae.parameters(), 1e-3)
for step in range(2000):
    idx = torch.randint(0, len(acts), (512,))
    x = acts[idx]
    x_hat, f = sae(x)
    loss = ((x - x_hat) ** 2).mean() + 1e-3 * f.abs().mean()    # reconstruct + L1
    opt.zero_grad(); loss.backward(); opt.step()

# Name a feature: find the inputs (tokens/snippets) that maximally activate it.
_, F = sae(acts)
for feat in [0, 1, 2]:
    top = F[:, feat].topk(5).indices
    print(f"feature {feat} top contexts:", [corpus_snippets[i] for i in top])
```

**Result:** inspecting each feature's top-activating contexts reveals **monosemantic** patterns — one feature lights up on, say, code/parentheses, another on a named entity, another on a grammatical role. They're far cleaner than the raw neurons (which are polysemantic), which is the whole point: the SAE **un-mixed superposition** into nameable concepts.

### Exercise 4 — Reproduce induction heads

```python
import torch
# Feed a repeated random-token sequence; an induction head attends from the second
# occurrence of a token to the position AFTER its first occurrence.
seq = torch.randint(0, 1000, (1, 25))
rep = torch.cat([seq, seq], dim=1)                  # 50 tokens: pattern repeats
_, cache = model.run_with_cache(rep)
for layer in range(model.cfg.n_layers):
    patt = cache["pattern", layer][0]               # (n_heads, q, k)
    for head in range(patt.size(0)):
        # induction stripe: attention to position (i - seq_len + 1)
        diag = patt[head].diagonal(offset=-(seq.size(1) - 1)).mean()
        if diag > 0.3:
            print(f"induction head found at L{layer} H{head} (score {diag:.2f})")
```

**Result:** in GPT-2-small you find a small number of heads (in middle layers) whose attention forms the tell-tale **induction stripe** — high attention exactly one position after the previous occurrence of the current token. These are the induction heads; their presence coincides with the model's ability to continue the repeated pattern.

### Exercise 5 — Activation patching to localize a circuit

```python
# IOI-style task: "When John and Mary went to the store, John gave a drink to ___"
clean = "When John and Mary went to the store, John gave a drink to"   # answer: Mary
corrupt = "When John and Mary went to the store, Mary gave a drink to" # answer: John
_, clean_cache = model.run_with_cache(clean)

def patch_head(layer, head):
    def hook(z, hook):                              # z: (batch, pos, n_heads, d_head)
        z[:, :, head] = clean_cache["z", layer][:, :, head]
        return z
    logits = model.run_with_hooks(corrupt, fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook)])
    return logits[0, -1, mary_id] - logits[0, -1, john_id]   # recovery toward clean answer

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        effect = patch_head(layer, head)
        if effect > 1.0:
            print(f"L{layer} H{head} causally moves the answer (Δlogit {effect:.2f})")
```

**Result:** patching most heads does nothing, but a **small set** (the "name mover" heads and their feeders, in the IOI circuit) sharply shift the logit toward the clean answer. Those are the **causally responsible** components — a controlled experiment that localizes the circuit, which probing alone could never establish.

### Exercise 6 — Steering with a concept direction

```python
# Use a probe weight (or SAE decoder column) as the steering vector v.
v = probe.weight[1] - probe.weight[0]               # direction toward class 1
v = v / v.norm()

def steer_hook(resid, hook, alpha=8.0):
    return resid + alpha * v.to(resid.dtype)

out = model.run_with_hooks("I think the weather today is",
        fwd_hooks=[(f"blocks.{6}.hook_resid_post", steer_hook)])
print(model.to_string(out[0, -1].argmax()))
```

**Result:** adding $\alpha v$ to the residual stream **shifts generations toward the concept** encoded by $v$ (e.g., the steered topic or sentiment), and larger $\alpha$ steers harder until coherence breaks. This is the mechanism behind Golden Gate Claude and refusal-direction ablation — interpretable, training-free behavioral control, and a striking demonstration that high-level behavior often lives along **low-dimensional directions**.

---

[← Chapter 21 solutions](21-deep-rl-solutions.md) · [Solutions index](README.md) · [Next: Chapter 23 solutions →](23-alternative-architectures-solutions.md)
