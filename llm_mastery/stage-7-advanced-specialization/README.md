# Stage 7 — Advanced Specialization

> **Objective:** Go deep on 1–2 frontier tracks that match your interests/career goals. Breadth got you here (Stages 1–6); **depth in a track** is what makes you genuinely "cracked" and hard to replace.

[← Stage 6](../stage-6-production-llmops/README.md) · [Index](../README.md) · Next: [Stage 8 — Safety](../stage-8-safety-security/README.md)

📝 **Interview prep:** [interview-questions.md](interview-questions.md) · ✅ [answer key](answers.md)

---

## How to use this stage

Don't do all six tracks. **Pick 1–2**, go deep enough to implement something and explain it from first principles, then optionally sample the rest. Each track lists: the core idea, what to learn, key papers, and a project.

---

## Track A — Mixture of Experts (MoE)

**Core idea:** Replace the dense MLP with many "expert" MLPs + a **router** that sends each token to only a few experts. You get a model with huge total parameters but **constant per-token compute** (sparse activation).

**Learn:**
- Routing (top-k gating), why only k experts fire per token
- **Load balancing** loss (prevent expert collapse / hot experts)
- Capacity factor, token dropping, expert parallelism
- Tradeoffs: more VRAM (all experts resident) but cheaper FLOPs/token
- Real models: **Mixtral 8×7B**, DeepSeek-MoE, Switch Transformer

**Papers:** Switch Transformer; Mixtral; GShard.

**Project:** implement a small MoE layer with top-2 routing + a load-balancing loss; visualize which experts fire for which tokens.

---

## Track B — Long Context

**Core idea:** Extend usable context from a few K to 100K–1M tokens without retraining from scratch, and understand why long context is hard (O(n²) attention + KV-cache growth).

**Learn:**
- **RoPE scaling**: position interpolation, **NTK-aware** scaling, **YaRN**
- Efficient attention: sliding-window (Mistral), **ring attention**, streaming/StreamingLLM, attention sinks
- KV-cache pressure at long context (ties to Stage 5) + cache compression
- "Lost in the middle": models underuse mid-context information
- Long-context **evaluation**: needle-in-a-haystack, RULER

**Papers:** YaRN; StreamingLLM; Ring Attention; Lost in the Middle.

**Project:** take a model, extend its context via RoPE interpolation/YaRN, and run a needle-in-a-haystack eval before/after.

---

## Track C — Reasoning & Test-Time Compute

**Core idea:** Spend more **compute at inference** to think harder — the paradigm behind o1/o3/R1-style models. Reasoning is trained (RL on verifiable rewards) and elicited (long CoT, search).

**Learn:**
- CoT scaling, self-consistency, best-of-n, tree/graph-of-thought
- **Process reward models (PRMs)** vs outcome reward models
- RL for reasoning on **verifiable** tasks (math/code) — e.g., DeepSeek-R1's approach (GRPO)
- Inference-time search (MCTS-style), verifier-guided decoding
- The compute-vs-accuracy test-time scaling curve

**Papers:** "Let's Verify Step by Step" (PRMs); STaR; DeepSeek-R1; Tree of Thoughts.

**Project:** implement best-of-n + a verifier on GSM8K; plot accuracy vs number of samples (test-time scaling curve).

---

## Track D — Multimodal

**Core idea:** Extend LLMs to images/audio/video by aligning other modalities into the language model's embedding space.

**Learn:**
- **CLIP**-style contrastive image-text alignment
- Vision-language architectures: vision encoder → projector → LLM (**LLaVA**), cross-attention (Flamingo)
- Image tokenization / patchification; any-to-any models
- Training stages: alignment pretraining → visual instruction tuning
- Multimodal evaluation + hallucination

**Papers:** CLIP; LLaVA; Flamingo; Qwen-VL.

**Project:** wire a frozen vision encoder + projector to a small LLM; do simple visual Q&A.

---

## Track E — Interpretability

**Core idea:** Reverse-engineer *what* and *how* models compute internally — for science, debugging, and safety.

**Learn:**
- Logit lens, activation/attention analysis, probing classifiers
- **Mechanistic interpretability**: circuits, induction heads, superposition
- **Sparse Autoencoders (SAEs)** for feature disentanglement
- Activation patching / causal tracing; steering vectors

**Papers:** "A Mathematical Framework for Transformer Circuits"; "Toy Models of Superposition"; "Towards Monosemanticity" (Anthropic).

**Project:** find and validate **induction heads** in a small model via activation patching.

---

## Track F — Synthetic Data & Self-Improvement

**Core idea:** Use models to generate, filter, and curate their own training data — increasingly central to frontier progress.

**Learn:**
- Generation + **filtering** (quality, diversity, dedup, decontamination)
- Distillation from a stronger teacher; **self-instruct**
- Rejection sampling fine-tuning (RFT), RLAIF data
- Failure modes: **model collapse**, amplified bias, contamination

**Papers:** Self-Instruct; Alpaca; WizardLM (Evol-Instruct); "The Curse of Recursion" (model collapse).

**Project:** generate a synthetic instruction dataset with self-instruct, filter it, and fine-tune (reuse Stage 3); compare to a human-data baseline.

---

## 🔥 Mastery checks (for whichever track[s] you pick)

- [ ] **MoE:** why does an MoE have many params but low per-token FLOPs? What does the load-balancing loss prevent?
- [ ] **Long context:** why is attention O(n²) and how does RoPE interpolation extend context? What is "lost in the middle"?
- [ ] **Reasoning:** what is a process reward model, and why does test-time compute improve accuracy?
- [ ] **Multimodal:** how does a vision encoder's output get into the LLM? What does CLIP's contrastive objective do?
- [ ] **Interpretability:** what is an induction head and how would you locate one?
- [ ] **Synthetic data:** what is model collapse and how do you guard against it?

---

## ✅ Stage 7 checklist

- [ ] Picked 1–2 tracks
- [ ] Read the core papers for the chosen track(s)
- [ ] Completed the track project(s)
- [ ] Can explain the track from first principles
- [ ] Mastery checks for the track passable
- [ ] Notes in your own words

**When complete → proceed to [Stage 8](../stage-8-safety-security/README.md).**
