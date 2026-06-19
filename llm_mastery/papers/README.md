# 📄 Paper Reading List

> The core canon for LLM mastery, ordered by stage. **Don't just read — interrogate.** For each paper write: (1) the problem, (2) the key idea, (3) why it worked, (4) one limitation/critique. That four-line summary is the deliverable.

[← Index](../README.md)

> Note: titles + authors + year are given so you can find the authoritative source (arXiv / publisher) yourself. Prefer the latest revision on arXiv.

---

## How to read a paper (the elite method)

1. **Pass 1 (5 min):** title, abstract, figures, conclusion. What problem? What result?
2. **Pass 2 (30 min):** intro, method, key equations, main experiments. Can you explain the core idea?
3. **Pass 3 (deep):** re-derive the key math, question the experiments, find the limitation. Ideally **implement the core mechanism**.
4. Write the **4-line summary** (problem / idea / why it works / critique).

> Reading-to-understand is implementing. The papers marked **[implement]** are worth coding the core of.

---

## Stage 1 — Architecture & foundations
- [ ] **Attention Is All You Need** — Vaswani et al., 2017. *The transformer.* **[implement]** Focus: attention math, multi-head, positional encoding.
- [ ] **RoFormer: Enhanced Transformer with Rotary Position Embedding** — Su et al., 2021. Focus: RoPE and why relative position helps.
- [ ] *(Resource)* **The Illustrated Transformer** — Jay Alammar. Visual intuition.
- [ ] *(Resource)* **Let's build GPT** — Andrej Karpathy (nanoGPT). **[implement]**

## Stage 2 — Pretraining & scaling
- [ ] **Language Models are Few-Shot Learners (GPT-3)** — Brown et al., 2020. Focus: scale + in-context learning.
- [ ] **Training Compute-Optimal LLMs (Chinchilla)** — Hoffmann et al., 2022. Focus: tokens-per-param, compute-optimal scaling.
- [ ] **LLaMA** & **Llama 2** — Touvron et al., 2023. Focus: practical data + architecture choices (RMSNorm, SwiGLU, RoPE, GQA).
- [ ] **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — Rajbhandari et al., 2019. Focus: sharding optimizer/grads/params.

## Stage 3 — Adaptation & alignment
- [ ] **Training LMs to Follow Instructions with Human Feedback (InstructGPT)** — Ouyang et al., 2022. Focus: the RLHF pipeline.
- [ ] **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., 2021. **[implement]** Focus: low-rank update math.
- [ ] **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al., 2023. Focus: NF4, double quant, paged optimizers.
- [ ] **Direct Preference Optimization (DPO)** — Rafailov et al., 2023. Focus: the loss derivation, why no RM/RL needed.

## Stage 4 — Evaluation
- [ ] **Holistic Evaluation of Language Models (HELM)** — Liang et al., 2022. Focus: multi-metric, multi-scenario mindset.
- [ ] **Judging LLM-as-a-Judge (MT-Bench / Chatbot Arena)** — Zheng et al., 2023. Focus: judge biases + Elo methodology.
- [ ] *(Resource)* **lm-evaluation-harness** (EleutherAI). **[implement]** Run a real benchmark.

## Stage 5 — Inference & efficiency
- [ ] **FlashAttention** (v1, 2022) & **FlashAttention-2** (2023) — Dao et al. Focus: IO-aware exact attention.
- [ ] **Efficient Memory Management for LLM Serving with PagedAttention (vLLM)** — Kwon et al., 2023. Focus: KV-cache paging, continuous batching.
- [ ] **GPTQ** — Frantar et al., 2022, and **AWQ** — Lin et al., 2023. Focus: 4-bit post-training quantization.
- [ ] **Fast Inference from Transformers via Speculative Decoding** — Leviathan et al., 2023. Focus: draft+verify, distribution preservation.

## Stage 6 — Applications & reasoning
- [ ] **Retrieval-Augmented Generation (RAG)** — Lewis et al., 2020. Focus: retrieval + generation coupling.
- [ ] **Chain-of-Thought Prompting** — Wei et al., 2022. Focus: eliciting reasoning.
- [ ] **Self-Consistency Improves CoT** — Wang et al., 2022.
- [ ] **ReAct: Synergizing Reasoning and Acting** — Yao et al., 2022. Focus: the agent loop.
- [ ] **Toolformer** — Schick et al., 2023. Focus: learned tool use.

## Stage 7 — Advanced (read for your chosen track)
- [ ] **MoE:** Switch Transformer (Fedus et al., 2021); **Mixtral of Experts** (2024).
- [ ] **Long context:** YaRN (Peng et al., 2023); StreamingLLM (Xiao et al., 2023); Lost in the Middle (Liu et al., 2023).
- [ ] **Reasoning:** Let's Verify Step by Step (Lightman et al., 2023); Tree of Thoughts (Yao et al., 2023); DeepSeek-R1 (2025).
- [ ] **Multimodal:** CLIP (Radford et al., 2021); LLaVA (Liu et al., 2023); Flamingo (Alayrac et al., 2022).
- [ ] **Interpretability:** A Mathematical Framework for Transformer Circuits (Elhage et al., 2021); Toy Models of Superposition (2022); Towards Monosemanticity (2023).
- [ ] **Synthetic data:** Self-Instruct (Wang et al., 2022); The Curse of Recursion / model collapse (Shumailov et al., 2023).

## Stage 8 — Safety & security
- [ ] **OWASP Top 10 for LLM Applications** — the industry reference. Read fully.
- [ ] **Universal and Transferable Adversarial Attacks on Aligned LMs** — Zou et al., 2023. Focus: adversarial-suffix jailbreaks.
- [ ] **Extracting Training Data from Large Language Models** — Carlini et al., 2021. Focus: memorization/privacy.
- [ ] *(Resource)* Simon Willison's writing on **prompt injection** (direct + indirect).

---

## ✅ Reading tracker

| Stage | Papers read | 4-line summaries written | Core implemented |
|-------|-------------|--------------------------|------------------|
| 1 | [ ] | [ ] | [ ] |
| 2 | [ ] | [ ] | [ ] |
| 3 | [ ] | [ ] | [ ] |
| 4 | [ ] | [ ] | [ ] |
| 5 | [ ] | [ ] | [ ] |
| 6 | [ ] | [ ] | [ ] |
| 7 | [ ] | [ ] | [ ] |
| 8 | [ ] | [ ] | [ ] |

> **Cadence:** 1 paper/week, deeply, beats 5 papers/week skimmed. Consistency compounds.
