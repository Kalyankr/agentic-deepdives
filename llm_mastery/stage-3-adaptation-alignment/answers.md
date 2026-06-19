# Stage 3 — Answer Key (Adaptation & Alignment)

> Full worked answers to [interview-questions.md](interview-questions.md). The bar here is picking the **right technique for the constraints** and justifying with memory/cost math, plus knowing the alignment *objectives* and failure modes. Notation: $\pi_\theta$ policy, $\pi_\text{ref}$ frozen reference, $r$ reward, $\beta$ KL/temperature coefficient.

---

## 🟢 Fundamentals

**1. Base vs instruct model.**
A **base model** is pretrained only on next-token prediction — it *completes* text and doesn't reliably follow instructions. An **instruct model** has been further trained (SFT + alignment) to follow instructions, answer in a helpful assistant style, and respect a chat format and safety norms.

**2. What is SFT?**
**Supervised fine-tuning**: continue training the base model on curated (prompt → ideal response) pairs with the standard next-token cross-entropy loss, computed only on the response. It teaches format and instruction-following by imitation.

**3. Why mask prompt tokens in the SFT loss?**
You want the model to learn to *generate the response*, not to *predict the user's prompt*. Including prompt tokens in the loss wastes capacity modeling inputs you'll always be given and can make the model parrot/continue prompts. Mask them so only response tokens contribute gradients.

**4. LoRA in one sentence.**
Freeze the pretrained weights and learn a **low-rank update** $\Delta W = BA$ (with $B,A$ tiny) added to chosen weight matrices, training <1% of parameters while matching full-fine-tune quality.

**5. What does QLoRA solve?**
It lets you fine-tune very large models on a **single GPU** by quantizing the *frozen* base to 4-bit (slashing the dominant weight-memory term) while training small bf16 LoRA adapters on top — recovering full-fine-tune quality at a fraction of the memory.

**6. RLHF at a high level.**
Align the model to human preferences in three steps: (1) **SFT** a base model on demonstrations; (2) train a **reward model** from human pairwise preference rankings; (3) **optimize the policy** (e.g. PPO) to maximize reward while staying close (KL) to the SFT model.

**7. What is a reward model?**
A model (usually the LLM with a scalar head) that maps a (prompt, response) to a **scalar quality score**, trained on human preference pairs to score preferred responses higher. It's the learned proxy for human judgment that RL optimizes against.

**8. What is DPO and why popular?**
**Direct Preference Optimization** trains the policy *directly* on preference pairs with a simple classification-style loss, skipping the separate reward model and RL loop. Popular because it's far simpler, more stable, and cheaper than PPO while reaching comparable quality.

**9. Catastrophic forgetting.**
When fine-tuning on a narrow task overwrites the broad capabilities learned in pretraining — the model gets better at the new task but loses general knowledge/skills. Mitigated by PEFT (small updates), low LR, fewer epochs, and replaying general data.

**10. Chat template, why it matters.**
A chat template is the exact string format with special tokens/role markers (e.g. `<|user|> … <|assistant|> …`) that delimits turns. The model was trained on one specific template; using a different one at inference (wrong tokens/spacing) silently degrades or garbles outputs. Train and serve must match exactly.

---

## 🟡 Core (L4–L5)

**11. LoRA decomposition `ΔW = BA`.**
For a frozen weight $W_0\in\mathbb{R}^{d\times k}$, LoRA learns $\Delta W = BA$ where $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r\ll\min(d,k)$. The forward becomes $h = W_0 x + \tfrac{\alpha}{r}BA x$. **$A$ and $B$ train; $W_0$ stays frozen.** $A$ is usually random-init, $B$ zero-init so the adapter starts as a no-op.

**12. Three-stage RLHF pipeline.**
1. **SFT:** fine-tune base on demonstration data → a decent instruction follower $\pi_\text{SFT}$.
2. **Reward modeling:** collect human pairwise preferences, train an RM with the Bradley–Terry loss to score responses.
3. **RL (PPO):** optimize $\pi_\theta$ to maximize RM reward minus a **KL penalty** to $\pi_\text{ref}=\pi_\text{SFT}$, keeping the policy from drifting/hacking.

**13. Role of the KL penalty.**
It anchors the policy to the reference (SFT) model: the objective is $\mathbb{E}[r(x,y)] - \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_\text{ref})$. Without it the policy drifts off-distribution to **reward-hack** the imperfect RM, producing high-reward gibberish and losing fluency/diversity. $\beta$ trades reward maximization against staying close to a sane distribution.

**14. QLoRA's three tricks.**
- **NF4 (4-bit NormalFloat):** an information-theoretically optimal 4-bit datatype for normally-distributed weights — quantizes the frozen base with minimal error.
- **Double quantization:** also quantize the *quantization constants*, saving extra memory (~0.4 bits/param).
- **Paged optimizers:** use NVIDIA unified memory to page optimizer state to CPU on spikes, avoiding OOM during long-sequence steps.

**15. How DPO avoids RM + RL.**
DPO uses the closed-form solution of the KL-constrained RLHF objective, which expresses the optimal policy's **implicit reward** as $r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$. Substituting into the Bradley–Terry preference likelihood yields a loss directly on the policy's logprobs of chosen vs rejected — so one supervised-style loss replaces "train RM, then PPO."

**16. When full FT over PEFT?**
When you have **lots of high-quality data and compute**, need to **change the model deeply** (new language/modality/domain far from pretraining, or large behavior shifts), or want the absolute best quality and can afford it. PEFT can underfit very large adaptations; full FT updates everything but costs ~16 bytes/param and risks more forgetting.

**17. Reward hacking and guards.**
The policy exploits flaws in the *proxy* reward model to get high scores without genuine quality (verbosity, sycophancy, gibberish the RM mis-scores). Guards: a **KL penalty** to the reference, **early stopping**, a **stronger/ensembled RM**, fresh on-policy preference data, length normalization, and human/eval spot-checks.

**18. DPO vs IPO vs KTO vs ORPO.**
- **DPO:** needs paired (chosen, rejected); strong default. Can overfit/over-optimize on easy pairs.
- **IPO:** adds regularization to fix DPO's tendency to push margins to extremes when preferences are near-deterministic.
- **KTO:** needs only **per-sample binary** good/bad labels (no pairs) — great when you can't collect pairwise data.
- **ORPO:** combines SFT + preference optimization in **one stage** (no separate reference model), simplest pipeline.
Choose by data shape (paired vs unpaired) and pipeline simplicity.

**19. Picking LoRA `r` and `alpha`.**
`r` sets adapter capacity: small (4–8) for light tasks, larger (16–64+) for bigger domain shifts; raise it if you underfit. `alpha` scales the update ($\tfrac{\alpha}{r}$); a common heuristic is `alpha = 2r` (or keep `alpha/r` constant when sweeping `r`). Tune on validation; also choose *which* matrices (q,k,v,o and MLP) to adapt.

**20. Why a few thousand clean SFT examples beat millions of noisy.**
The base model already *knows* language and facts; SFT mainly teaches **format and behavior**, which a small, diverse, high-quality set conveys cleanly. Noisy data injects contradictory targets that the model imitates faithfully (it can't tell good from bad), degrading quality (the LIMA "less is more" result). Quality and diversity > quantity for alignment.

---

## 🔴 Senior / Staff deep dives

**21. "1 GPU, preference data, safer assistant" — method, end to end.**
Plan: **QLoRA SFT → DPO**, not PPO.
- **Memory:** 4-bit frozen 7B ≈ 3.5 GB + small bf16 adapters + paged optimizer → fits a single 24–48 GB GPU. Full FT would need ~112 GB.
- **Why not PPO:** needs an RM, an extra reference/value model, on-policy rollouts — heavy infra and unstable on one GPU. DPO consumes the preference pairs directly, stable and cheap.
- **Steps:** SFT on curated demos for format/safety → DPO on the preference pairs to push preferred over rejected (anchored to the SFT reference) → **eval**: win-rate vs baseline (LLM-judge + human), safety/red-team set, and a general-capability set to measure alignment tax. Iterate on data.

**22. DPO loss intuition and link to RLHF.**
The KL-constrained RLHF optimum is $\pi^*(y|x)\propto\pi_\text{ref}(y|x)\exp(r(x,y)/\beta)$. Invert it: $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}+\text{const}$. Plug this **implicit reward** into the Bradley–Terry model $P(y_w\succ y_l)=\sigma(r_w-r_l)$:
$$\mathcal{L}_\text{DPO}=-\mathbb{E}\Big[\log\sigma\big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\big)\Big].$$
It raises the policy's logprob of the **preferred** response relative to the **rejected**, scaled by $\beta$ and anchored to the reference — achieving RLHF's goal without an explicit RM or RL.

**23. Why LoRA works (low rank).**
Fine-tuning has a **low intrinsic dimension**: adapting a huge pretrained model to a downstream task requires moving in only a small subspace, so the weight update $\Delta W$ is approximately low-rank. A rank-$r$ factorization captures most of that update with tiny parameters. Intuition: pretraining already learned general features; adaptation re-weights/combines them, a low-rank operation.

**24. Why QLoRA fits a 7B (or 65B) on one GPU — memory math.**
The dominant cost is normally the frozen base in optimizer-bearing precision. QLoRA:
- Base weights in **4-bit NF4**: $7\text{B}\times0.5\text{B} \approx$ **3.5 GB** (frozen, no gradients/optimizer state).
- **LoRA adapters** (bf16) + their optimizer state: a few hundred MB (only ~0.1–1% of params train).
- **Paged optimizer** absorbs activation/optimizer spikes via CPU paging.
Total a few GB vs **~112 GB** for full bf16+AdamW FT. A 65B follows the same logic (~33 GB 4-bit base) — fits on a 48–80 GB card.

**25. RLHF policy → degenerate high-reward gibberish.**
This is **reward hacking**: the policy left the RM's training distribution and found inputs the RM mis-scores highly. Diagnose by inspecting samples (often repetitive/verbose/off-topic) and watching KL blow up while RM reward rises but human quality falls. Fix: **raise the KL coefficient $\beta$** (or clip), **early stop** at the reward/KL knee, **improve/retrain the RM** with on-policy data covering the exploited region, ensemble RMs, add length normalization, and validate with humans not just RM score.

**26. Alignment pipeline, domain assistant, no RL infra.**
1. **SFT (QLoRA)** on curated domain demonstrations for format + base behavior.
2. **Preference optimization without RL:** DPO or **ORPO** on (chosen, rejected) pairs; use **KTO** if you only have binary good/bad labels.
3. **Scale feedback with RLAIF / Constitutional AI:** generate and critique with an LLM against written principles to cheaply expand preference data.
4. **Red-team + safety eval** gating; **iterate** the data flywheel.
This achieves alignment with only supervised-style training — no PPO/value model.

**27. Alignment tax — measure & minimize.**
The **alignment tax** is the capability regression on general benchmarks caused by alignment training (the model becomes safer/more helpful but slightly worse at raw tasks). **Measure** by evaluating the aligned vs pre-alignment model on a fixed capability suite (MMLU, code, math) alongside helpfulness/safety. **Minimize** with PEFT/low LR, **replaying pretraining/general data** during alignment, smaller KL drift, model averaging/merging, and tuning the helpfulness–safety tradeoff rather than over-aligning.

**28. Building a high-quality preference dataset; biases.**
Collect **pairwise human rankings** of two responses to the same prompt, with clear annotator guidelines and quality control (gold checks, inter-annotator agreement). Bias sources to control: **length bias** (raters favor longer answers — normalize/penalize length), **position bias** (order of A/B — randomize), **sycophancy/style bias**, annotator demographics, and prompt-distribution skew. Augment with **RLAIF** (AI feedback) for scale, but calibrate against human labels to avoid baking in the judge model's biases.

---

## 🧮 Math & derivations

**29. LoRA trainable params for a `d×d` matrix at rank `r`.**
Full update: $d^2$ parameters. LoRA: $B\in\mathbb{R}^{d\times r}$ and $A\in\mathbb{R}^{r\times d}$ → $2dr$ parameters. Ratio $=\frac{2dr}{d^2}=\frac{2r}{d}$. E.g. $d=4096$, $r=8$ → $2\cdot4096\cdot8=65{,}536$ vs $1.67\text{M}$ → **~0.4%** of the parameters.

**30. Reward-model ranking loss (Bradley–Terry).**
Given preferred $y_w$ and rejected $y_l$ with scalar rewards $r_\phi$:
$$\mathcal{L}_\text{RM}=-\mathbb{E}_{(x,y_w,y_l)}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big].$$
It maximizes the probability (under a logistic/Bradley–Terry model) that the preferred response scores higher than the rejected.

**31. DPO loss; role of β and π_ref.**
$$\mathcal{L}_\text{DPO}=-\mathbb{E}\Big[\log\sigma\big(\beta(\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)})\big)\Big].$$
**$\pi_\text{ref}$** (frozen SFT model) anchors the policy so the update is a *relative* shift, preventing drift/over-optimization. **$\beta$** controls how hard preferences are enforced vs how close to stay to the reference: large $\beta$ = strong preference fitting but more drift; small $\beta$ = conservative.

**32. Memory: full FT vs LoRA vs QLoRA for 7B.**
- **Full FT (bf16+AdamW):** 16 B/param → params 14 + grads 14 + optim 84 = **~112 GB** (+activations).
- **LoRA (bf16 base):** frozen base in bf16 ≈ 14 GB (no grad/optim) + adapter params/grads/optim ~ hundreds of MB → **~15–16 GB**.
- **QLoRA (4-bit base):** base ≈ **3.5 GB** + adapters/optim ~1 GB → **~5 GB**.
Gap: 112 → 16 → 5 GB; QLoRA is ~20× smaller than full FT, which is why it fits on a single consumer GPU.

---

## 💻 Coding / implementation

**33. SFT loss masking (response-only).**
```python
# labels = input_ids, but prompt positions set to -100 (ignored by cross_entropy)
labels = input_ids.clone()
labels[:, :prompt_len] = -100              # mask the prompt
# (also mask padding)
labels[attention_mask == 0] = -100
loss = F.cross_entropy(logits[:, :-1].reshape(-1, V),
                       labels[:, 1:].reshape(-1), ignore_index=-100)
```
Only response tokens (label ≠ -100) contribute gradient.

**34. LoRA layer wrapping a frozen `nn.Linear`.**
```python
import torch, torch.nn as nn
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False   # freeze
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * (1 / r ** 0.5))
        self.B = nn.Parameter(torch.zeros(d_out, r))               # start as no-op
        self.scale, self.drop = alpha / r, nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.A.T) @ self.B.T
```

**35. DPO loss from logprobs.**
```python
import torch.nn.functional as F
def dpo_loss(pol_chosen, pol_rej, ref_chosen, ref_rej, beta=0.1):
    # each arg = summed logprob of that sequence under the model
    pi_logratios  = pol_chosen - pol_rej
    ref_logratios = ref_chosen - ref_rej
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()
```

**36. Chat template + multi-turn role masking.**
```python
msgs = [{"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"}]
ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt")
# mask everything except assistant spans
labels = ids.clone(); labels[:] = -100
for start, end in assistant_token_spans(msgs, tok):   # compute per-turn offsets
    labels[0, start:end] = ids[0, start:end]
```
Only assistant tokens are supervised; user/system tokens are ignored (-100). Must use the *model's own* template.

**37. Merge LoRA back into base at correct scale.**
```python
# W_merged = W0 + (alpha/r) * B @ A
with torch.no_grad():
    delta = (alpha / r) * (lora.B @ lora.A)         # (d_out, d_in)
    lora.base.weight.add_(delta.to(lora.base.weight.dtype))
# now serve plain nn.Linear with zero adapter overhead
```
The `alpha/r` scale must match training, else behavior shifts.

---

## 🏗️ System design / applied

**38. Fine-tuning service (upload data → custom adapter).**
- **Data validation:** schema/format checks, dedup, PII/safety scan, min/max examples, train/val split, leakage check; reject or warn.
- **Training:** managed **QLoRA** jobs (queued, resource-capped, reproducible configs); per-job isolation; sensible default hyperparameters with overrides.
- **Eval gating:** automatic eval on a held-out slice + safety suite; **block promotion** if quality/safety regresses vs base; show metrics to the user.
- **Serving:** store adapters; **hot-swap LoRA** onto a shared frozen base (load adapter per request/tenant), so 1000s of customer models share one base. Versioning + rollback.

**39. 50 customer-specific models cheaply.**
**One shared frozen base + 50 small LoRA adapters.** Keep the base resident in GPU memory once; **dynamically load the per-customer adapter** (a few MB) at request time, or batch requests by adapter. This avoids 50 full model copies (memory and cost ÷50), enables instant onboarding of new customers, and supports multi-adapter serving frameworks (e.g. punica/S-LoRA style) for high throughput.

**40. Human-feedback data flywheel from a deployed product.**
**Log:** prompts, responses, model/version, decoding params, and **implicit signals** (thumbs up/down, edits, regenerations, copy/accept, conversation abandonment), plus explicit ratings — with consent/privacy controls. **Close the loop:** mine logs for high-signal pairs (accepted vs regenerated → preference pairs), curate/clean, feed into the next **SFT/DPO** round, eval-gate, ship, and measure online win-rate. Continuous: production usage → preferences → training → better model → more usage.

---

## 🐞 Debugging

**41. After SFT the model parrots the question.**
**Prompt tokens weren't masked in the loss** (labels included the prompt), so the model learned to reproduce/continue the user's input. Fix: set prompt (and padding) label positions to -100 so only response tokens are supervised.

**42. After FT, lost general knowledge / repetitive.**
**Catastrophic forgetting** from too-aggressive adaptation: LR too high, too many epochs, or **full FT on narrow data** overwrote pretrained capabilities. Fix: use **PEFT/LoRA**, lower LR, fewer epochs, **replay** general/pretraining data in the mix, and early-stop on a general-capability eval.

**43. Garbled only in production.**
**Chat-template / special-token mismatch** between training and serving — different role markers, missing BOS/EOS, or extra/!missing whitespace. The model sees an out-of-distribution prompt format. Fix: use the exact same template + tokenizer special tokens in inference as in training; diff the rendered strings byte-for-byte.

**44. DPO not improving win-rate — knobs/data.**
Check **$\beta$** (too high = over-constrained, too low = drift/instability), **reference model correctness** (must be the SFT model, logprobs computed consistently), **preference data quality** (noisy or low-margin pairs where chosen≈rejected give no signal), **too few steps / LR**, and possible **length bias** dominating. Also verify chosen/rejected aren't swapped and the eval judge isn't biased.

---

## What strong answers share
Choosing the **right method for the constraints** with explicit memory/cost math (QLoRA vs full FT, DPO vs PPO); understanding alignment **objectives** (KL-constrained RLHF and DPO's implicit reward), not just APIs; knowing failure modes (**forgetting, reward hacking, template mismatch**) and their fixes; and designing the **data flywheel + eval gating**, not a single run.

---
Back to [questions](interview-questions.md) · [Stage README](README.md) · [Index](../README.md)
