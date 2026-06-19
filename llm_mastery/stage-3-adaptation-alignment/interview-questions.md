# Stage 3 — Interview Questions (full-fledged, all levels)

> **Scope:** screening through **senior / staff / principal**. Angles: conceptual, math, coding, system design, debugging. `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math · 💻 Coding · 🏗️ Design · 🐞 Debug
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What's the difference between a base model and an instruct model?
2. What is supervised fine-tuning (SFT)?
3. Why do we mask the prompt tokens in the SFT loss?
4. What is LoRA in one sentence?
5. What problem does QLoRA solve?
6. What is RLHF at a high level?
7. What is a reward model?
8. What is DPO and why is it popular?
9. What is catastrophic forgetting?
10. What is a chat template and why does it matter?

## 🟡 Core (L4–L5)
11. Explain the LoRA decomposition `ΔW = BA`. Which matrices train, which freeze?
12. Walk through the three-stage RLHF pipeline.
13. What role does the KL-divergence penalty play in RLHF/PPO?
14. Name QLoRA's three tricks (NF4, double quant, paged optimizers) and what each does.
15. How does DPO avoid training a separate reward model and running RL?
16. When would you choose full fine-tuning over PEFT?
17. What is reward hacking and what guards against it?
18. Compare DPO, IPO, KTO, ORPO — when use which?
19. How do you pick LoRA rank `r` and `alpha`?
20. Why can a few thousand high-quality SFT examples beat millions of noisy ones?

## 🔴 Senior / Staff deep dives (with follow-ups)
21. Given "1 GPU, preference data, need a safer assistant" — choose a method and justify end to end.
    → *covers:* QLoRA SFT → DPO; memory math; why not PPO (infra/instability); eval plan.
22. Derive/justify the DPO loss intuition and its connection to the RLHF objective.
    → *covers:* implicit reward = β·log(π/π_ref); preference loss pushes preferred above rejected while anchored to reference; closed-form replaces RM+PPO.
23. Why does LoRA work — what's the empirical/theoretical justification for low rank?
    → *covers:* fine-tuning updates have low "intrinsic dimension"; ΔW is approximately low-rank; large pretrained models adapt in a small subspace.
24. Explain why QLoRA can fine-tune a 7B (even 65B) on a single GPU. Do the memory math.
    → *covers:* 4-bit frozen base (~3.5GB for 7B) + small bf16 adapters + paged optimizer; contrast with 112GB full FT.
25. Your RLHF policy starts producing degenerate high-reward gibberish. Diagnose and fix.
    → *reward hacking:* RM out-of-distribution; raise KL coefficient, improve RM, early stop, better preference data.
26. Design an alignment pipeline for a domain assistant with safety constraints and no RL infra.
    → *covers:* SFT on curated demos → DPO/ORPO on preferences → red-team eval → iterate; constitutional/RLAIF for scale.
27. What is the "alignment tax" and how do you measure/minimize it?
28. How do you build a high-quality preference dataset? Sources of bias?
    → *covers:* pairwise human ranking, annotator guidelines, length/position bias, inter-annotator agreement, AI feedback (RLAIF).

## 🧮 Math & derivations
29. Compute LoRA trainable parameters for adapting a `d×d` matrix at rank `r`. Compare to full.
30. Write the reward-model ranking loss (Bradley–Terry / logistic on reward difference).
31. Write the DPO loss and identify the role of β and π_ref.
32. For a 7B model, compute memory: full FT vs LoRA vs QLoRA. Show the gap.

## 💻 Coding / implementation
33. Implement SFT loss masking so only response tokens contribute.
34. Implement a LoRA layer (`forward = Wx + (α/r)·B(Ax)`) wrapping a frozen `nn.Linear`.
35. Implement the DPO loss given policy/reference logprobs for chosen/rejected.
36. Apply a chat template and tokenize a multi-turn conversation with correct role masking.
37. Merge LoRA adapters back into base weights at the correct scale.

## 🏗️ System design / applied
38. Design a fine-tuning service that lets users upload data and get a custom adapter. Cover data validation, training, eval gating, serving (adapter hot-swap).
39. You must support 50 customer-specific models cheaply. Architecture?
    → *covers:* shared frozen base + per-customer LoRA adapters, dynamic adapter loading.
40. Build the human-feedback data flywheel from a deployed product. What do you log and how do you close the loop?

## 🐞 Debugging / scenarios
41. After SFT the model parrots the user's question back. Likely bug?
    → *prompt tokens not masked in loss.*
42. After fine-tuning, the model lost general knowledge / got repetitive. Cause?
    → *catastrophic forgetting:* LR too high, too many epochs, full FT on narrow data → use PEFT, lower LR, replay.
43. Outputs are garbled only in production, fine in training. Suspect?
    → *chat-template / special-token mismatch between train and inference.*
44. DPO isn't improving win-rate. What knobs and data issues do you check?
    → *β tuning, reference model correctness, noisy/low-margin preference pairs, too few steps.*

## ✅ What strong candidates demonstrate
- Pick the **right technique for the constraints** and justify with memory/cost math.
- Understand alignment **objectives**, not just library calls.
- Know the **failure modes** (forgetting, reward hacking, template mismatch) and fixes.
- Think about the **data flywheel** and eval gating, not just one training run.

---
Related: the **🔥 Mastery checks** in [README.md](README.md) are the minimum bar.
