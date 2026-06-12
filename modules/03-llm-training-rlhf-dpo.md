# Module 03 · LLM Training, RLHF & DPO

> **Goal:** Understand the full post-training stack that turns a base model into an aligned assistant like ChatGPT or Claude: pretraining → SFT → preference optimization (RLHF/PPO, DPO) → alignment techniques. Implement the alignment loops yourself on a small model.

**Duration:** ~6 weeks. **Prereqs:** [Module 02](02-transformer-internals.md).

---

## 3.1 Pretraining

- The objective: next-token prediction (causal LM) = cross-entropy / minimizing perplexity
- **Data is everything:** sources (web, code, books), dedup, quality filtering, decontamination, mixing ratios
- Tokenization at scale, packing sequences, document boundaries
- Curriculum & data ordering (intro)
- Compute budgeting with Chinchilla; the data-vs-params trade-off
- Stability: loss spikes, LR warmup, z-loss, careful init
- Distributed training preview (full treatment in [Module 05](05-distributed-systems-and-inference.md))

> **Note:** You won't pretrain a frontier model, but you'll continue-pretrain / train a small one to internalize the mechanics.

## 3.2 Supervised Fine-Tuning (SFT) / Instruction Tuning

- From base model → instruction-following model
- Instruction datasets (FLAN, Alpaca, Dolly, OpenAssistant, ShareGPT-style)
- **Chat templates** and role tokens (system/user/assistant); masking the loss to assistant tokens only
- Multi-turn formatting, packing, truncation strategies
- Data quality > quantity (LIMA: "less is more for alignment")

> **Build:** SFT a small base model (e.g., a 0.5–1.5B model) on an instruction dataset. Compare base vs. SFT outputs on held-out prompts.

## 3.3 Parameter-Efficient Fine-Tuning (PEFT)

- Full fine-tuning vs. PEFT trade-offs (memory, catastrophic forgetting)
- **LoRA** — low-rank adapters; the math: $W + \frac{\alpha}{r} BA$, why it works, rank/alpha choices
- **QLoRA** — 4-bit base (NF4) + LoRA; double quantization, paged optimizers — fine-tune big models on one GPU
- Adapters, prefix tuning, IA³, DoRA (survey level)
- Serving multiple LoRAs (multi-tenant adapters) — ties to [Module 04](04-gpu-architecture-and-inference.md)

> **Build:** Fine-tune the same model with LoRA and QLoRA. Compare quality, GPU memory, and training time vs. full fine-tuning.

## 3.4 Why alignment? Preference optimization

Base/SFT models are capable but not reliably **helpful, harmless, and honest**. Preference learning aligns outputs to human (or AI) preferences.

### The classic RLHF pipeline (InstructGPT / ChatGPT)
1. **SFT** an initial policy.
2. Train a **reward model (RM)** on human preference pairs (chosen vs. rejected) — Bradley-Terry loss.
3. Optimize the policy with **PPO** against the RM, with a **KL penalty** to the SFT reference to prevent reward hacking / drift.

Key concepts:
- Reward modeling, preference data collection, annotator agreement
- PPO essentials: policy/value networks, advantage estimation (GAE), clipped surrogate objective
- The **KL-to-reference** term and why it's the safety valve
- **Reward hacking** / over-optimization (Goodhart's law in RL)

### DPO and the "RL-free" family
- **DPO (Direct Preference Optimization)** — reframes RLHF as a classification loss on preference pairs; no separate reward model or RL loop. Derive the DPO loss from the RLHF objective.
- Variants: IPO, KTO, ORPO, SimPO — what problem each fixes
- DPO vs. PPO trade-offs (simplicity/stability vs. ceiling performance)

### AI feedback & constitutional methods
- **RLAIF** — AI-generated preference labels
- **Constitutional AI (Anthropic)** — self-critique + revision against a set of principles; RL from AI feedback. *Read this carefully — it's core to Anthropic's approach.*
- Rejection sampling / Best-of-N fine-tuning

> **Build (centerpiece):** Implement **DPO** from scratch on your SFT model using a small preference dataset (e.g., a subset of UltraFeedback/HH-RLHF). Then train a simple **reward model** and run a minimal **PPO** loop (TRL makes this approachable). Compare DPO vs. PPO outputs and KL drift.

## 3.5 Reasoning & inference-time training (frontier topic)

- Chain-of-thought as a training target
- **RL from verifiable rewards** (math/code with checkable answers) — the o1/R1 paradigm
- Process vs. outcome reward models (PRM vs. ORM)
- Distillation of reasoning traces
- Test-time compute scaling (preview; applied in [Module 07](07-agentic-systems.md))

## 3.6 Evaluation of alignment

- Win-rate vs. reference (LLM-as-judge, e.g., AlpacaEval, MT-Bench, Arena-style)
- Reward over-optimization curves
- Safety/refusal evals, helpfulness/harmlessness trade-off
- Full treatment in [Module 09](09-evaluations.md)

---

## Module 03 capstone — **Align a small model end-to-end**

1. SFT a small base model into an instruction follower (with proper loss masking + chat template).
2. LoRA/QLoRA variant; compare cost/quality.
3. **DPO** fine-tune on preference data; measure win-rate vs. the SFT model with an LLM judge.
4. A minimal **reward model + PPO** loop; document KL drift and any reward hacking you observe.
5. Write-up comparing SFT vs. DPO vs. PPO on helpfulness and a small safety probe set.

## Exit criteria
- [ ] You can explain the 3-stage RLHF pipeline and the role of the KL penalty.
- [ ] You can derive/explain the DPO loss and when to prefer DPO vs. PPO.
- [ ] You implemented DPO and a basic PPO loop yourself.
- [ ] You can explain Constitutional AI / RLAIF and reward hacking.

## Core papers
- *InstructGPT* — Ouyang et al., 2022
- *DPO: Your Language Model is Secretly a Reward Model* — Rafailov et al., 2023
- *Constitutional AI* — Bai et al. (Anthropic), 2022
- *Llama 2* (post-training section), *Llama 3* report
- *Proximal Policy Optimization* — Schulman et al., 2017
- *LoRA* — Hu et al., 2021; *QLoRA* — Dettmers et al., 2023
- *Training Verifiers / Let's Verify Step by Step* (process supervision) — OpenAI
- *DeepSeek-R1* — 2025 (RL for reasoning)

## Tooling
- Hugging Face `transformers`, `peft`, `trl`, `datasets`, `accelerate`
- `bitsandbytes` (quantized training)
- Weights & Biases for tracking
