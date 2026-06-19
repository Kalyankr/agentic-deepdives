# Stage 3 — Adaptation & Alignment

> **Objective:** Take a raw base model and turn it into a helpful, instruction-following, aligned assistant — and know *which* technique to reach for under given constraints (compute, data, goals).

[← Stage 2](../stage-2-pretraining-at-scale/README.md) · [Index](../README.md) · Next: [Stage 4 — Evaluation](../stage-4-evaluation/README.md)

📝 **Interview prep:** [interview-questions.md](interview-questions.md) · ✅ [answer key](answers.md)

---

## Why this stage matters

A base model completes text; it doesn't *follow instructions* or behave safely. Adaptation (SFT + PEFT) and alignment (RLHF/DPO) are what made ChatGPT-style assistants possible. This is also the stage you'll most often *do yourself* in industry — full pretraining is rare, fine-tuning is everywhere.

---

## Mental model

```
Base model (completes text)
   │  Supervised Fine-Tuning (SFT) on instruction/response pairs
   ▼
Instruct model (follows instructions)
   │  Preference optimization (RLHF / DPO) on "A is better than B" data
   ▼
Aligned model (helpful, honest, harmless)
```

- **SFT** teaches *format and task* ("when asked X, respond like Y").
- **Alignment** teaches *preferences* ("of two valid answers, humans prefer this one") and suppresses bad behavior.

---

## Concept-by-concept deep dive

### 3.1 Supervised Fine-Tuning (SFT) / instruction tuning
- **Data:** `(instruction, response)` pairs, often multi-turn, formatted with a **chat template** (special tokens marking roles: system/user/assistant).
- **Loss masking:** compute loss **only on the response tokens**, not the prompt — you want the model to learn to *produce* answers, not to model the user's text.
- **Quality > quantity:** a few thousand high-quality, diverse examples often beat millions of noisy ones (the "LIMA" lesson).
- **Catastrophic forgetting:** aggressive fine-tuning erodes pretrained knowledge. Mitigate with low LR, fewer epochs, PEFT, or data replay.

### 3.2 Parameter-Efficient Fine-Tuning (PEFT)
Full fine-tuning updates all weights (huge memory — recall the 12 bytes/param optimizer cost from Stage 2). PEFT freezes the base and trains a tiny add-on.

**LoRA (Low-Rank Adaptation):**
- Insight: the *update* `ΔW` to a weight matrix is low-rank. So represent it as `ΔW = B·A` where `A ∈ ℝ^{r×d}`, `B ∈ ℝ^{d×r}`, with rank `r ≪ d`.
- Forward: `h = Wx + (BA)x · (α/r)`. Only `A` and `B` train; `W` is frozen.
- **Params trained:** `r·(d_in + d_out)` per adapted matrix — often <1% of the model.
- **Hyperparameters:** rank `r` (8–64 typical), `alpha` (scaling), which modules to adapt (attention q/v at minimum; often all linear layers).
- **Benefits:** tiny checkpoints, swappable adapters, low memory, no inference latency if merged (`W ← W + BA`).

**QLoRA:**
- Load the **frozen base in 4-bit** (NF4 quantization) → slashes base memory.
- Train LoRA adapters in bf16 on top.
- Tricks: **double quantization** (quantize the quantization constants) + **paged optimizers** (spill optimizer state to CPU on memory spikes).
- **Result:** fine-tune a 7B on a single 24GB GPU; even 65B on a single 48GB card. This democratized fine-tuning.

**Other PEFT:** adapters (bottleneck layers), prefix/prompt tuning (learn virtual tokens). LoRA usually wins on simplicity + quality, so it's the default.

### 3.3 Why align at all? (RLHF motivation)
- SFT imitates demonstrations but can't easily say "answer A is *better* than answer B."
- Humans can **compare** outputs more reliably than they can **write** ideal ones.
- So: collect preference data (pairs ranked by humans), and optimize the model toward preferred responses.

### 3.4 RLHF (the classic 3-step pipeline)
1. **SFT** (above) → a starting policy.
2. **Reward model (RM):** train a model to predict human preference. Given two responses, it outputs a scalar; trained with a ranking loss so `reward(preferred) > reward(rejected)`.
3. **RL optimization (PPO):** fine-tune the policy to maximize RM reward, with a **KL-divergence penalty** to a frozen reference model so it doesn't drift into gibberish that hacks the reward.
- **Reward hacking:** policy finds adversarial outputs that score high but are bad → the KL term and RM quality are the guardrails.
- **Alignment tax:** alignment can slightly reduce raw capability — a known tradeoff.

### 3.5 DPO (Direct Preference Optimization) — the modern simplification
- **Key idea:** you can optimize the *same* preference objective **without** training a separate reward model or running RL. DPO derives a loss directly on the policy using the preference pairs and a reference model.
- **Loss (intuition):** increase the policy's relative log-prob of the preferred response over the rejected one, scaled by a temperature `β`, anchored to the reference model.
- **Why it took over:** far simpler, more stable, no RM, no PPO infrastructure — comparable or better results for most use cases.
- **Variants:** **IPO** (fixes DPO overfitting), **KTO** (uses unpaired good/bad labels), **ORPO** (combines SFT + preference in one step, no reference model).

### 3.6 Choosing the right technique (decision framework)

| Situation | Use |
|-----------|-----|
| Need new format/task behavior, have demonstrations | **SFT** |
| Limited GPU memory, single task | **LoRA / QLoRA** |
| Have preference (A>B) data, want better quality/safety | **DPO** (start here) |
| Large team, infra, need fine reward control | **RLHF/PPO** |
| Want one-step SFT+preference, no reference model | **ORPO** |
| Only have thumbs-up/down (unpaired) signals | **KTO** |

---

## Ordered learning path

1. Read **InstructGPT** — the canonical RLHF pipeline + motivation.
2. Read **LoRA**, then **QLoRA**.
3. Read **DPO** (and skim ORPO).
4. Work through **HuggingFace TRL** examples (SFTTrainer, DPOTrainer).
5. Do the labs end to end.

---

## 🛠️ Hands-on labs

- [ ] **Lab A — SFT:** fine-tune a small base model (e.g., a 1–7B Llama/Mistral/Qwen) on an instruction dataset using a proper chat template + loss masking.
- [ ] **Lab B — QLoRA:** redo SFT with 4-bit base + LoRA adapters on a single consumer GPU; compare memory vs full FT.
- [ ] **Lab C — DPO:** take your SFT model + a preference dataset; run DPO; produce an aligned model.
- [ ] **Lab D — Compare:** fixed prompt set → generate from base vs SFT vs DPO; write up the qualitative differences.
- [ ] **Lab E — Adapter swap:** train two LoRA adapters for two tasks; hot-swap them on the same frozen base.
- [ ] **Lab F (analysis):** sweep LoRA rank `r ∈ {4,8,16,64}`; plot quality vs trainable params.

---

## ⚠️ Common pitfalls & gotchas

- **Not masking the prompt** in SFT loss → model learns to parrot user text.
- Wrong/mismatched **chat template** between training and inference → garbage outputs.
- Over-training SFT → catastrophic forgetting, repetitive/robotic outputs.
- DPO `β` too high/low → no learning or reward-hacking-like drift.
- Forgetting the **reference model** in DPO/PPO → instability.
- Merging LoRA at the wrong scale (`α/r`) → silent quality loss.
- Evaluating only on training-like prompts → overestimating generalization.

---

## 🔥 Mastery checks (answer without notes)

- [ ] Explain the LoRA decomposition `ΔW = BA` and compute trainable params for given `r, d`.
- [ ] Why does QLoRA let you fine-tune a 7B on 24GB? Name the three tricks.
- [ ] Write the RLHF 3-step pipeline and explain the role of the **KL penalty**.
- [ ] Explain the DPO loss intuitively and why it removes the reward model + RL loop.
- [ ] What is reward hacking, and what prevents it in RLHF?
- [ ] Given a constraint (e.g., "1 GPU, preference data, need safety"), pick a method and justify it.
- [ ] Why mask the prompt tokens in SFT loss?
- [ ] When would you choose ORPO or KTO over DPO?

---

## ✅ Stage 3 checklist

- [ ] Read InstructGPT, LoRA, QLoRA, DPO
- [ ] Labs A–D complete (E/F for depth)
- [ ] Have base vs SFT vs DPO comparison written up
- [ ] All mastery checks passable
- [ ] Notes in your own words

**When complete → proceed to [Stage 4](../stage-4-evaluation/README.md).**
