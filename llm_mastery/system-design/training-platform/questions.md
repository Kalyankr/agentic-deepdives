# LLM Training & Fine-Tuning Platform — Interview Questions (all levels)

> **Scope:** screening through **senior / staff / principal** ML-systems / Applied-Scientist interviews. The reference design is [README.md](README.md). `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math/Estimation · 🏗️ Design · 🐞 Debug/Ops
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What does a training/fine-tuning platform do, and why is it its own system separate from serving?
2. Write the training-compute rule of thumb and explain each term.
3. Why can't a large model train on a single GPU — what actually runs out of memory?
4. Data, tensor, and pipeline parallelism — what does each one split?
5. What is ZeRO / FSDP, and what does it shard?
6. What is activation (gradient) checkpointing, and what does it trade?
7. Why is checkpointing essential for large training jobs?
8. What is MFU, and why is it the headline training metric?
9. SFT vs LoRA vs RLHF vs DPO — what's the difference at a glance?
10. What is gang scheduling, and why do synchronous training jobs need it?

## 🟡 Core design
11. Walk through the lifecycle of a training job from submission to a registered model.
12. Design the data pipeline from a raw corpus to tokenized shards.
13. How do you combine data, tensor, and pipeline parallelism (3D) on a real cluster?
14. Design checkpointing for a multi-day job on thousands of GPUs.
15. How do you handle a hardware failure mid-run without losing days of work?
16. Design the scheduler for a shared, multi-tenant GPU cluster.
17. How does RLHF (PPO) differ from supervised training in resource needs?
18. How do you keep thousands of GPUs fed with data so the loader never starves them?
19. How do you make a training run reproducible?
20. How do you integrate evaluation into the training loop and the registry handoff?

## 🔴 Senior / Staff deep dives
21. The loss spikes / diverges mid-training. Diagnose and respond.
22. MFU is only 25%. Walk through how you'd raise it.
23. A few stragglers are slowing every step. Diagnose and fix.
24. Pick a parallelism plan for a 70B model vs a 7B model — and justify the difference.
25. Design elastic / fault-tolerant training that survives node loss and resizes the world.
26. Checkpoint writes stall training every N steps. Fix it **without** checkpointing less often.
27. How do you schedule both huge pretraining jobs and many small fine-tunes fairly on one cluster?
28. Spot/preemptible GPUs are half price — how do you use them safely for training?

## 🧮 Math & estimation
29. Estimate the GPU-days to pretrain a 70B model on 15T tokens.
30. Estimate the memory for AdamW mixed-precision training of a 7B model.
31. How big is a full resumable checkpoint for a 70B model, and why that size?
32. Chinchilla: given a fixed compute budget, how do you split params vs tokens?
33. Estimate the all-reduce communication volume per step for data parallelism.
34. How much memory does activation checkpointing save, and at what compute cost?

## 🏗️ Design variations
35. Design a LoRA/QLoRA fine-tuning service for many tenants on shared GPUs.
36. Design an RLHF pipeline (reward model + PPO rollouts) end to end.
37. Design a hyperparameter-sweep / experimentation system on the platform.
38. Design continued-pretraining / domain adaptation on top of an existing base model.

## 🐞 Debugging & ops
39. Training throughput dropped 30% after scaling from 512 to 2048 GPUs. Why?
40. You hit OOM only at long sequence lengths. Fix it.
41. Two runs with the same config give different final losses. Why, and how do you fix it?
42. A NaN appears in the loss after hours of training. Triage it.

---

> **How to practice:** state assumptions, do the **two estimates first** ($C\approx6ND$ and ~16 bytes/param), then reason from them. Check yourself against [answers.md](answers.md) and the [one-page cheat-sheet](cheat-sheet.md).

[← Back to training platform HLD](README.md) · [Answer key](answers.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
