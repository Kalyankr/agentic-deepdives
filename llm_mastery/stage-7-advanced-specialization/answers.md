# Stage 7 — Answer Key (Advanced Specialization)

> Full worked answers to [interview-questions.md](interview-questions.md), organized by track. The bar: explain your track **from first principles**, **implement** its core, connect it back to earlier stages, and know the **frontier + open problems**.

---

## 🟢 Cross-track fundamentals

**1. Mixture-of-Experts and why it helps.**
An MoE replaces the dense MLP with many **expert** MLPs plus a **router** that sends each token to only the top-$k$ experts. This **decouples total parameters from per-token compute**: you get the capacity (knowledge) of a huge model while only activating a small fraction per token, so quality scales with params without compute scaling proportionally.

**2. Why long context is hard (two reasons).**
(1) **Compute/memory:** attention is $O(T^2)$ and the **KV cache grows linearly** with $T$, so long sequences blow up FLOPs and memory/bandwidth. (2) **Quality:** models trained on short sequences **don't extrapolate** positionally, and even when they fit the context they suffer "**lost in the middle**" — poor use of information far from the ends.

**3. Test-time compute and why it improves reasoning.**
Spending **more compute at inference** — longer chains of thought, sampling many solutions, search/verification — instead of (or in addition to) more parameters. It helps because hard problems benefit from **exploration and self-correction**: generating, checking, and revising multiple reasoning paths finds correct answers a single greedy pass misses (the o1/R1 paradigm).

**4. How multimodal models get images into an LLM.**
A **vision encoder** (e.g. ViT/CLIP) turns the image into patch embeddings; a **projector** (MLP or cross-attention) maps those into the LLM's token-embedding space; the projected visual tokens are **prepended/interleaved** with text tokens and the LLM attends over both — images become "soft tokens."

**5. What mechanistic interpretability tries to do.**
Reverse-engineer the **internal algorithms/circuits** a network has learned — mapping specific weights/neurons/attention heads to human-understandable computations (features, circuits) — so we can *explain, predict, and audit* behavior rather than treat the model as a black box.

**6. Synthetic data + model collapse.**
**Synthetic data** is model-generated training data (from a stronger teacher or self-generation). **Model collapse** is the degradation that happens when models are trained recursively on their **own outputs**: rare events/tails vanish, diversity shrinks, and errors compound — the distribution narrows toward the model's biases. Avoided with verifiable filtering, diversity controls, and mixing in real data.

---

## Track A — Mixture of Experts (MoE)

**7. Top-k routing; why only k experts.**
A small gate network scores all experts per token; the token is sent to its **top-$k$** (usually 1–2) experts, whose outputs are combined weighted by the gate. Only $k$ experts run because that's the whole point — **sparse activation** keeps per-token FLOPs low (you pay for $k$, not all $E$ experts) while the model still *has* all experts' capacity.

**8. Huge total params, low per-token FLOPs.**
Total params = all $E$ experts (all stored in memory). But each token only routes to $k\ll E$ experts, so the **FLOPs per token** scale with $k$, not $E$. E.g. 8 experts, top-2 → 4× the params of one expert but only 2 experts' worth of compute per token. Capacity is in memory; compute is sparse.

**9. What the load-balancing loss prevents.**
**Router collapse** — without it the gate learns to send most tokens to a few favored experts, leaving others untrained (wasted capacity) and overloading some (dropped tokens). An **auxiliary balance loss** encourages uniform expert utilization so all experts get used and trained.

**10. Design an MoE layer.**
- **Routing:** gate (linear → softmax) selects **top-2** experts per token; combine outputs by gate weights.
- **Capacity factor:** each expert has a buffer = `capacity_factor × tokens/expert`; tokens beyond it **overflow/drop** (or go to the residual). Higher capacity = fewer drops but more compute/memory waste.
- **Aux loss:** load-balancing loss to equalize utilization; optional router z-loss for stability.
- **Expert parallelism:** experts sharded across devices → routing requires **all-to-all** communication (dispatch tokens, combine results).
- **Tradeoffs:** memory holds **all experts resident** (big), comms-heavy (all-to-all), training instability (routing) — in exchange for large quality gains at fixed FLOPs.

**11. Why MoEs are harder to serve.**
All experts must be **resident in memory** (high memory cost despite low active FLOPs), routing adds **all-to-all communication** and load imbalance, **batching is harder** (tokens in a batch route to different experts → irregular, variable per-expert load), and a dense model of equal quality is simpler/cheaper to deploy. So MoEs win on training-compute efficiency but pay in serving complexity and memory.

**12. Dense vs 8×7B MoE at matched active params (FLOPs & memory).**
A Mixtral-style 8×7B with top-2 has **~47B total params** but **~13B active** per token (shared attention + 2 experts). vs a dense 13B:
- **Per-token FLOPs:** roughly **equal** (~matched active params) → similar inference compute.
- **Memory:** MoE needs **all ~47B params resident** (~3.6× the dense 13B) → much larger VRAM/host footprint.
So the MoE gets quality closer to a ~47B-capacity model at ~13B compute, but pays ~47B memory.

**13. Top-2 gating + dispatch with balance loss (toy).**
```python
import torch, torch.nn as nn, torch.nn.functional as F
class MoE(nn.Module):
    def __init__(self, d, n_exp=8, k=2):
        super().__init__()
        self.k, self.gate = k, nn.Linear(d, n_exp)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, 4*d), nn.GELU(),
                                                   nn.Linear(4*d, d)) for _ in range(n_exp))
    def forward(self, x):                      # x: (T, d)
        logits = self.gate(x)                  # (T, n_exp)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.k, dim=-1)        # (T, k)
        topv = topv / topv.sum(-1, keepdim=True)        # renormalize
        out = torch.zeros_like(x)
        for slot in range(self.k):
            for e in range(len(self.experts)):
                mask = topi[:, slot] == e
                if mask.any():
                    out[mask] += topv[mask, slot, None] * self.experts[e](x[mask])
        # load-balancing aux loss (Switch-style)
        importance = probs.mean(0)                       # fraction of prob per expert
        load = F.one_hot(topi[:, 0], len(self.experts)).float().mean(0)
        aux = len(self.experts) * (importance * load).sum()
        return out, aux
```

---

## Track B — Long Context

**14. How RoPE enables extrapolation; where it breaks.**
RoPE rotates Q/K by an angle proportional to position, so the attention dot product depends only on the **relative** offset — a property that, in principle, generalizes beyond trained lengths. It **breaks** at positions far past training because the high-frequency rotation components encounter **angles never seen in training**, producing out-of-distribution attention patterns and degraded quality — which is why naive extrapolation fails and you need interpolation/YaRN.

**15. "Lost in the middle."**
Empirically, models use information best when it's at the **beginning or end** of a long context and **worst when it's in the middle** — accuracy on a fact dips sharply if it's buried mid-context. A U-shaped performance curve; implications: put critical context at the edges, rerank so the best chunk is near the top, don't assume long context = used context.

**16. Sliding-window attention; what it sacrifices.**
Each token attends only to the last $W$ tokens instead of all $T$, making attention $O(T\cdot W)$ and bounding the KV cache. It sacrifices **direct long-range attention** — information beyond the window can only propagate **indirectly** through stacked layers (effective receptive field ≈ $W\times$ layers), so very long dependencies weaken. Often combined with a few global/sink tokens.

**17. Extend 4K → 128K without full retraining.**
1. **Position Interpolation:** scale position indices down so 128K maps into the trained 4K range (compress angles into the seen distribution).
2. **NTK-aware / YaRN scaling:** adjust RoPE base frequencies so high-frequency components aren't crushed — better than naive linear interpolation.
3. **Light continued pretraining** on a small amount of long-context data to adapt.
4. **Evaluate** with **needle-in-a-haystack** and **RULER** across the full length, not just perplexity. This achieves long context cheaply vs training from scratch.

**18. KV-cache strategies for very long context.**
(Links to Stage 5.) **GQA/MQA** to shrink KV per token, **KV-cache quantization** (int8/int4), **PagedAttention** to avoid fragmentation, **eviction / sliding window + attention sinks** for streaming, **chunked prefill** to bound peak memory, and **prefix caching** for shared context. Often combine with retrieval to avoid holding everything in KV.

**19. Attention compute/memory growth; why it dominates.**
$QK^\top$ and score·$V$ each cost $O(T^2 d)$ and the score matrix is $O(T^2)$ memory (naively). The projections/MLP are $O(Td^2)$. The ratio attention/MLP ≈ $T/d$, so once **$T \gtrsim d$** the $T^2$ attention term **dominates** both compute and memory — the fundamental reason long context is expensive. FlashAttention removes the $O(T^2)$ *memory* but not the $O(T^2)$ FLOPs.

**20. RoPE position interpolation + needle-in-haystack.**
```python
def rope_interpolated(x, orig_len, target_len, base=10000.0):
    B, h, T, d = x.shape
    scale = orig_len / target_len                       # compress positions
    theta = base ** (-torch.arange(0, d, 2, device=x.device) / d)
    pos = torch.arange(T, device=x.device) * scale       # interpolate
    freqs = torch.outer(pos, theta)
    cos, sin = freqs.cos(), freqs.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], -1).flatten(-2)

# needle-in-a-haystack: insert a fact at varying depths, ask model to recall it
def niah(model, fact, question, filler, depths=(0.0,0.25,0.5,0.75,1.0), length=128000):
    results = {}
    for d in depths:
        ctx = filler[:int(length*d)] + fact + filler[int(length*d):length]
        results[d] = (model(ctx + question).strip() == fact_answer)
    return results            # exposes the U-shaped "lost in the middle" curve
```

---

## Track C — Reasoning & Test-Time Compute

**21. Outcome RM vs process RM.**
An **ORM** scores only the **final answer** (right/wrong). A **PRM** scores **each reasoning step**, rewarding a correct *process*. PRMs help because they give **dense, per-step feedback** — they catch flawed reasoning that happens to reach the right answer (and vice versa), enable step-level search/verification, and reduce reward hacking on the final token. They're costlier to label (need step annotations).

**22. Best-of-n vs self-consistency.**
Both spend more inference compute for accuracy. **Best-of-n:** sample $n$ solutions, pick the best via a **verifier/reward model** (needs a scorer). **Self-consistency:** sample $n$ CoT paths and take the **majority-vote answer** (no external verifier, exploits that correct reasoning converges). Both trade roughly **linear compute for log-ish accuracy gains** that plateau as $n$ grows.

**23. What changed with o1/R1-style models.**
They're explicitly trained (via RL on **verifiable rewards**) to **generate long internal chains of thought** and self-correct *before* answering — shifting capability gains from pretraining scale to **test-time compute**. The model learns to "think longer" on hard problems, dramatically improving math/code/reasoning, and exposing a new scaling axis (inference compute) distinct from parameters/data.

**24. RL pipeline for a reasoning model on verifiable math/code.**
- **Verifiable rewards:** use ground-truth checkers (test suites for code, exact-answer for math) — reward = did it pass.
- **Sampling:** generate **long CoT** rollouts per problem.
- **Algorithm:** **GRPO** (group-relative, no value model — cheaper) or PPO; advantage from comparing samples per prompt.
- **Reward shaping:** mostly outcome-based on verifiable signal; optionally PRM/format rewards; penalize length/gibberish to avoid hacking.
- **Anti-hacking:** KL to reference, dedup, verify with held-out checkers.
- **Eval:** held-out problems (no contamination), pass@1 and pass@k, generalization to unseen difficulty.

**25. Test-time scaling curve + limits.**
Accuracy rises with inference compute (more samples / longer CoT / search), often **log-linear**, then **plateaus**. Practical limits: **cost and latency** scale with compute (n× samples = n× cost, long CoT = slow), the **verifier ceiling** (best-of-n is capped by verifier quality), and diminishing returns once the model's reachable solution set is exhausted. You trade $/latency for accuracy — viable for high-value queries, not cheap high-QPS ones.

**26. Expected best-of-n with a perfect verifier.**
With per-sample success probability $p$ and a **perfect** verifier (always picks a correct solution if one exists), best-of-$n$ succeeds iff **at least one** of $n$ samples is correct:
$$\text{acc}(n) = 1 - (1-p)^n.$$
E.g. $p=0.3$, $n=10$: $1-0.7^{10}\approx 0.97$. Concave, approaching 1 — this is the upper bound; a real verifier does worse.

**27. Best-of-n with a verifier on GSM8K.**
```python
def best_of_n(model, verifier, problem, n=16, temperature=0.8):
    samples = [model(problem, temperature=temperature) for _ in range(n)]
    scored = [(verifier(problem, s), s) for s in samples]   # verifier: reward/correctness
    return max(scored, key=lambda t: t[0])[1]

def accuracy_vs_n(model, verifier, dataset, ns=(1,2,4,8,16,32)):
    curve = {}
    for n in ns:
        correct = sum(is_correct(best_of_n(model, verifier, ex.q, n), ex.answer)
                      for ex in dataset)
        curve[n] = correct / len(dataset)
    return curve            # plot: accuracy rises then plateaus with n
```

---

## Track D — Multimodal

**28. What CLIP's contrastive objective learns.**
CLIP trains image and text encoders jointly so that **matching (image, caption) pairs have high cosine similarity** and mismatched pairs low, via an InfoNCE contrastive loss over a batch. It learns a **shared embedding space** where semantically related images and texts align — enabling zero-shot classification, retrieval, and serving as the vision backbone for VLMs.

**29. Vision-encoder → projector → LLM (LLaVA-style).**
A **frozen vision encoder** (CLIP ViT) produces patch embeddings; a **projector** (a small MLP) maps them into the LLM's token-embedding dimension; these become **visual tokens** prepended to the text tokens, and the (often frozen-then-tuned) **LLM** generates conditioned on both. Simple, data-efficient, and modular — swap encoder or LLM independently.

**30. How images are tokenized/patchified.**
The image is split into fixed-size **patches** (e.g. 14×14 px); each patch is linearly embedded into a vector (plus positional embedding) — exactly ViT's patch embedding. An $H\times W$ image yields $\frac{H}{p}\cdot\frac{W}{p}$ patch "tokens." High-resolution variants tile the image into multiple crops to get more tokens.

**31. Design a VLM + two-stage training.**
- **Architecture:** frozen CLIP ViT → projector (MLP or cross-attn / Q-Former) → LLM.
- **Stage 1 — alignment (pretraining):** freeze encoder and LLM, train **only the projector** on large image–caption pairs so visual features map into the LLM's space.
- **Stage 2 — visual instruction tuning:** unfreeze the LLM (and/or projector), train on **multimodal instruction data** (visual QA, reasoning, OCR) for instruction-following.
- Evaluate on VQA, captioning, and hallucination benchmarks; balance data to avoid degrading text-only ability.

**32. Evaluate multimodal models + detect visual hallucination.**
Use task benchmarks (**VQAv2, TextVQA, MMMU, MathVista**) and captioning metrics, but specifically test **hallucination**: e.g. **POPE** (ask about objects *not* in the image — does it falsely affirm?), object/attribute grounding checks, and human/judge verification that claims are **visually supported**. Visual hallucination = describing things absent from the image (often language-prior-driven); measure with adversarial presence/absence questions.

**33. Wire frozen vision encoder + projector to a small LLM for VQA.**
```python
import torch, torch.nn as nn
class MiniVLM(nn.Module):
    def __init__(self, vision, llm, v_dim, d_model):
        super().__init__()
        self.vision = vision.eval().requires_grad_(False)   # frozen ViT/CLIP
        self.projector = nn.Sequential(nn.Linear(v_dim, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        self.llm = llm
    def forward(self, image, input_ids):
        with torch.no_grad():
            patches = self.vision(image)                     # (B, n_patch, v_dim)
        vis_tokens = self.projector(patches)                 # (B, n_patch, d_model)
        txt = self.llm.embed(input_ids)                      # (B, T, d_model)
        x = torch.cat([vis_tokens, txt], dim=1)              # prepend visual tokens
        return self.llm(inputs_embeds=x)
# Stage 1: train only self.projector; Stage 2: also unfreeze self.llm
```

---

## Track E — Interpretability

**34. Logit lens / activation patching.**
**Logit lens:** project a model's **intermediate residual stream** through the final unembedding to see what token it "predicts" at each layer — reveals how predictions form across depth. **Activation patching** (causal tracing): run a clean and a corrupted prompt, then **copy a specific activation** from one run into the other and measure the effect on the output — *causally* localizing which components carry the relevant information.

**35. Induction head.**
A pair of attention heads implementing **in-context copying**: given a pattern `…[A][B]…[A]`, the induction head attends from the second `[A]` back to the token that followed the first `[A]` and predicts `[B]`. It's the core circuit behind in-context learning and pattern completion, formed via a "previous-token head" feeding a "copy" head.

**36. Superposition; why SAEs help.**
**Superposition:** networks represent **more features than they have neurons** by encoding features as overlapping directions in activation space, so individual neurons are **polysemantic** (respond to many unrelated concepts). **Sparse Autoencoders** learn an **overcomplete, sparse** dictionary that decomposes activations into many **monosemantic** features — disentangling superposition into human-interpretable units you can name and manipulate.

**37. Locate and validate a circuit (e.g. induction).**
Form a **falsifiable hypothesis** ("heads X,Y implement induction"). Use a **controlled dataset** (repeated random token sequences where induction is the only way to predict). **Activation-patch / ablate** the candidate heads and measure the drop in the induction behavior; **patch them into** a corrupted run to show sufficiency. Add **controls** (ablate unrelated heads → no effect) and check the **attention pattern** matches the induction mechanism (second-A attends to post-first-A token). Causality + controls = validation, not just correlation.

**38. How interpretability improves safety/debugging.**
Concretely: **detect deception/unsafe activations** before they surface (probes for "model knows it's lying"), **find and edit** the circuits behind a bug or bias (knowledge editing, feature steering), **audit** what a model uses to make a decision (trust/compliance), **predict** failure modes from internal features, and **monitor** activations at inference for dangerous concepts. It turns black-box behavior into something you can inspect and intervene on.

**39. Find induction heads via activation patching.**
```python
import torch
def find_induction_heads(model, seq_len=50, vocab=1000):
    # repeated random sequence: prediction only solvable by induction
    half = torch.randint(0, vocab, (seq_len,))
    tokens = torch.cat([half, half]).unsqueeze(0)
    scores = {}
    base_loss = model.loss(tokens)
    for layer in range(model.n_layers):
        for head in range(model.n_heads):
            # ablate (zero) this head's attention, measure loss increase on 2nd half
            with model.ablate_head(layer, head):
                scores[(layer, head)] = (model.loss(tokens) - base_loss).item()
    # heads whose ablation most hurts the repeated half = induction heads
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
```

---

## Track F — Synthetic Data & Self-Improvement

**40. Self-instruct and RFT.**
**Self-instruct:** bootstrap an instruction dataset by prompting a strong model to **generate new (instruction, response) examples** from a few seeds, then filter — cheap alignment data without human writing. **Rejection-sampling fine-tuning (RFT):** sample many solutions from the model, **keep only the verified-correct ones** (e.g. pass tests / correct answer), and fine-tune on those — the model improves by training on its own *best, filtered* outputs.

**41. Model collapse + causes.**
Training models recursively on **model-generated data** causes progressive quality/diversity loss — the model forgets the **tails** of the distribution, outputs converge to a narrow mode, and errors/biases amplify each generation. Caused by: sampling/approximation errors compounding, loss of rare events, and no fresh real-data signal. The data distribution "collapses" toward the model's own biases.

**42. Filter synthetic data for quality + diversity.**
- **Quality:** verifiable checks (unit tests, math verifiers), reward-model/LLM-judge scoring, consistency/self-consistency, and rejecting low-confidence or malformed outputs.
- **Diversity:** **dedup** (exact + near-dup/MinHash + embedding clustering), enforce topic/length/difficulty coverage, sample at temperature for variety.
- **Hygiene:** **decontaminate** against eval sets, remove unsafe/PII content.
Then **mix with real data** rather than training purely synthetic.

**43. Synthetic-data flywheel without collapse.**
1. **Generate** from a **stronger teacher** (or current model + search) — not just the same model echoing itself.
2. **Filter with verifiable/objective signals** (tests, checkers, RM) — keep only high-quality.
3. **Diversity + dedup** to preserve the tails; cover the input distribution.
4. **Decontaminate** against evals.
5. **Mix with real data** (anchor to the true distribution).
6. **Eval-gate** each round on held-out real benchmarks; stop if quality/diversity regress.
The keys to avoiding collapse: external/verifiable signal, diversity preservation, and real-data anchoring.

**44. When distillation from a stronger teacher beats human data.**
When a **capable teacher exists**, the task has **checkable/structured** outputs, and human labeling is **slow/expensive/inconsistent** (e.g. generating millions of reasoning traces, code with tests, or covering rare cases). Teacher distillation is far cheaper, more scalable, and often higher/ more consistent quality than crowd labels. Human data still wins for **truly novel capabilities** beyond any teacher, subjective/cultural judgment, and ground-truth preferences.

**45. Self-instruct: generate, filter, fine-tune, compare.**
```python
def self_instruct(teacher, seeds, n=10000):
    data = []
    while len(data) < n:
        prompt = make_prompt(sample(seeds, 3))           # few-shot from seeds
        ex = teacher(prompt)                              # generate (instruction, response)
        if quality_ok(ex) and not near_duplicate(ex, data):   # filter + dedup
            data.append(ex)
    return decontaminate(data, eval_sets)

# experiment:
synthetic = self_instruct(teacher, seeds)
model_syn = finetune(base, synthetic)
model_hum = finetune(base, human_dataset)               # matched size baseline
compare(model_syn, model_hum, on=held_out_eval)         # win-rate, capability suite
```

---

## What strong answers share
Explaining the track **from first principles** and being able to **implement its core**; **connecting tracks to earlier stages** (MoE↔serving memory, long-context↔KV cache, reasoning↔eval/verifiers); and knowing the **current frontier and open problems** (routing instability, lost-in-the-middle, reward hacking, model collapse, superposition).

---
Back to [questions](interview-questions.md) · [Stage README](README.md) · [Index](../README.md)
