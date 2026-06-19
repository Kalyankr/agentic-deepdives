# Stage 7 — Interview Questions (full-fledged, all levels)

> **Scope:** screening through **senior / staff / principal / research**. Organized by track — focus on the 1–2 you specialized in, but the 🟢 fundamentals across tracks are fair game in any LLM interview. `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff/Research · 🧮 Math · 💻 Coding
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Cross-track fundamentals (everyone should know)
1. What is a Mixture-of-Experts model and why does it help?
2. Why is long context hard (two reasons)?
3. What is "test-time compute" and why does it improve reasoning?
4. How do multimodal models get images into an LLM?
5. What is mechanistic interpretability trying to do?
6. What is synthetic data and what is model collapse?

---

## Track A — Mixture of Experts (MoE)
**🟡 Core**
7. How does top-k routing work? Why only k experts per token?
8. Why does an MoE have huge total params but low per-token FLOPs?
9. What does the load-balancing loss prevent?

**🔴 Senior/Research**
10. Design an MoE layer: routing, capacity factor, token dropping, expert parallelism. Tradeoffs?
    → *covers:* gate network, top-2 routing, aux balance loss, capacity/overflow, all-to-all comms, memory (all experts resident) vs compute savings.
11. Why are MoEs harder to serve than dense models of equal quality?
12. 🧮 Compare FLOPs and memory of a dense vs 8×7B MoE at matched active params.
13. 💻 Implement a top-2 gating + dispatch for a toy MoE with a balance loss.

---

## Track B — Long Context
**🟡 Core**
14. How does RoPE enable context-length extrapolation, and where does it break?
15. What is "lost in the middle"?
16. What is sliding-window attention and what does it sacrifice?

**🔴 Senior/Research**
17. You must extend a 4K model to 128K without full retraining. Plan?
    → *covers:* position interpolation, NTK-aware scaling, YaRN, light continued pretraining, eval with needle-in-haystack/RULER.
18. Walk through KV-cache strategies for very long context (Stage 5 link).
19. 🧮 Show attention compute/memory growth with sequence length and why it dominates.
20. 💻 Implement RoPE position interpolation and run a needle-in-a-haystack eval.

---

## Track C — Reasoning & Test-Time Compute
**🟡 Core**
21. Outcome reward model vs process reward model — difference and why PRMs help.
22. How do best-of-n and self-consistency trade compute for accuracy?
23. What changed conceptually with o1/R1-style models?

**🔴 Senior/Research**
24. Design an RL pipeline to train a reasoning model on verifiable math/code.
    → *covers:* verifiable rewards, GRPO/PPO, sampling long CoT, reward shaping, avoiding reward hacking, eval on held-out problems.
25. Explain the test-time scaling curve and its practical limits (cost/latency).
26. 🧮 Given a per-sample accuracy p, derive expected best-of-n accuracy with a perfect verifier.
27. 💻 Implement best-of-n with a verifier on GSM8K; plot accuracy vs n.

---

## Track D — Multimodal
**🟡 Core**
28. What does CLIP's contrastive objective learn?
29. Describe the vision-encoder → projector → LLM architecture (LLaVA-style).
30. How are images tokenized/patchified?

**🔴 Senior/Research**
31. Design a vision-language model and its two-stage training (alignment → visual instruction tuning).
32. How do you evaluate multimodal models and detect visual hallucination?
33. 💻 Wire a frozen vision encoder + projector to a small LLM for visual QA.

---

## Track E — Interpretability
**🟡 Core**
34. What is the logit lens / activation patching?
35. What is an induction head?
36. What is superposition and why do SAEs help?

**🔴 Senior/Research**
37. Design an experiment to locate and validate a specific circuit (e.g., induction).
    → *covers:* activation patching / causal tracing, ablations, controls, falsifiable hypothesis.
38. How could interpretability improve safety or debugging concretely?
39. 💻 Find induction heads in a small model via activation patching.

---

## Track F — Synthetic Data & Self-Improvement
**🟡 Core**
40. What is self-instruct? What is rejection-sampling fine-tuning (RFT)?
41. What is model collapse and what causes it?
42. How do you filter synthetic data for quality and diversity?

**🔴 Senior/Research**
43. Design a synthetic-data flywheel that improves a model without collapse.
    → *covers:* teacher/stronger-model generation, verifiable filtering, diversity/dedup, decontamination, mixing with real data, eval gating.
44. When does distillation from a stronger teacher beat collecting human data?
45. 💻 Generate + filter a self-instruct dataset, fine-tune, and compare to a human-data baseline.

## ✅ What strong candidates demonstrate
- Explain their track **from first principles** and **implement** its core.
- Connect the track back to **earlier stages** (MoE↔serving, long-context↔KV cache, reasoning↔eval).
- Know the **current frontier** and the **open problems / failure modes**.

---
Related: the **🔥 Mastery checks** in [README.md](README.md) are the minimum bar.
