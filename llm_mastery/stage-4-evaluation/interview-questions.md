# Stage 4 — Interview Questions (full-fledged, all levels)

> **Scope:** screening through **senior / staff / principal**. Angles: conceptual, math/stats, coding, system design, debugging. `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Stats · 💻 Coding · 🏗️ Design · 🐞 Debug
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What is the difference between intrinsic and extrinsic evaluation?
2. What does perplexity measure and what are its limits?
3. What does BLEU/ROUGE measure, and when does it mislead?
4. What is pass@k and why is it trustworthy?
5. Name three popular LLM benchmarks and what each tests.
6. What is benchmark contamination?
7. What is LLM-as-a-judge?
8. Why is human evaluation still needed?
9. What is a hallucination, and how is it an eval problem?
10. In RAG, what two things must you evaluate separately?

## 🟡 Core (L4–L5)
11. Why can a model top MMLU yet fail in production? Give concrete reasons.
12. List the major biases of LLM judges and a mitigation for each.
13. How would you detect contamination between a training set and a benchmark?
14. What is HELM's philosophy and why does it matter?
15. Why prefer pairwise comparison over absolute scoring for humans and judges?
16. How do you measure retrieval quality vs generation quality in RAG?
17. What is calibration, and why does it matter for an LLM?
18. What is Goodhart's law in the context of eval metrics?
19. How do you report results honestly (uncertainty, seeds, significance)?
20. What's the difference between offline benchmarks and online/production evals?

## 🔴 Senior / Staff deep dives (with follow-ups)
21. Design an evaluation for a task with **no ground-truth labels** (e.g., open-ended summarization quality).
    → *covers:* rubric-based LLM-judge + human spot-checks + pairwise preference + reference-free metrics; measure judge–human agreement; report CIs.
22. Build an end-to-end eval strategy for a customer-facing assistant before launch.
    → *covers:* capability benchmarks, safety/red-team, hallucination/faithfulness, latency/cost, A/B + online metrics, regression suite.
23. Your offline scores improved but users are unhappy. What's likely wrong with your eval?
    → *covers:* contamination, metric–goal mismatch (Goodhart), distribution shift, prompt-format sensitivity, tiny/old test set, judge bias.
24. You must trust an LLM-judge for a leaderboard. How do you validate and de-bias it?
    → *covers:* swap-order to cancel position bias, control length, multiple judges/ensembling, calibrate vs human labels, report agreement (kappa).
25. Design a contamination-resistant evaluation program for a fast-moving model team.
    → *covers:* private/held-out sets, canaries, n-gram overlap scans, time-split data, rotating fresh evals.
26. How do you evaluate a RAG system so that you can localize failures?
    → *covers:* retrieval recall@k/MRR/nDCG independently; generation faithfulness/answer-relevance; error attribution matrix.
27. How would you statistically decide whether model A beats model B?
    → *covers:* paired tests, bootstrap CIs, sample-size/power, multiple-comparison correction, effect size not just p-value.
28. Critique a popular benchmark of your choice — what does it fail to capture?

## 🧮 Stats & metrics
29. Define precision@k, recall@k, MRR, and nDCG; when use which?
30. How do you compute a bootstrap confidence interval for an accuracy metric?
31. Given two models at 71% vs 73% on 500 items, is the difference significant? How do you check?
32. Why can pass@k be estimated with an unbiased estimator from n>k samples? (sketch)

## 💻 Coding / implementation
33. Implement a pairwise LLM-judge harness that controls for **position bias** (order swap + average).
34. Implement pass@k estimation from sampled generations + unit tests.
35. Implement an n-gram overlap contamination scanner between two datasets.
36. Implement recall@k and MRR for a retriever given gold doc ids.
37. Implement bootstrap resampling to put a CI on any metric.

## 🏗️ System design / applied
38. Design a continuous evaluation system that runs on live production traffic samples and alerts on quality drift.
39. Design an eval harness that a 50-person model team can extend without stepping on each other.
40. You're asked to produce a trustworthy internal leaderboard. Design it (datasets, judges, contamination policy, refresh cadence).

## 🐞 Debugging / scenarios
41. The same model scores very differently on two "equivalent" prompt formats. What's happening and what do you do?
    → *prompt-format sensitivity:* standardize prompts, report across formats, few-shot stability.
42. Your LLM-judge always prefers the longer answer. Fix?
    → *verbosity bias:* length-control, instruct judge to ignore length, normalize.
43. Eval numbers are suspiciously high right after a data refresh. First hypothesis?
    → *contamination.*
44. A/B test shows no difference but you "know" the new model is better. What's wrong?
    → *underpowered test, wrong metric, segment effects, novelty/latency confounds.*

## ✅ What strong candidates demonstrate
- Treat evaluation as **experiment design** with uncertainty, not a single number.
- **Distrust benchmarks** intelligently (contamination, Goodhart, format sensitivity).
- Can **separate sub-system failures** (retrieval vs generation).
- Know the **statistics** (CIs, significance, power), not just the metric names.

---
Related: the **🔥 Mastery checks** in [README.md](README.md) are the minimum bar.
