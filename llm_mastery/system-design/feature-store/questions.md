# Feature & Embedding Store — Interview Questions (all levels)

> **Scope:** screening through **senior / staff / principal** ML-systems / Applied-Scientist interviews. The reference design is [README.md](README.md). `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math/Estimation · 🏗️ Design · 🐞 Debug/Ops
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What is a feature store, and what problems does it solve?
2. What is train/serve skew, and why is it so damaging?
3. What is point-in-time correctness (the as-of join), and why does it matter?
4. What is label leakage, and how does a feature store prevent it?
5. Why two stores (offline + online), and what lives in each?
6. How is a feature store different from a regular database or cache?
7. What is materialization?
8. Online vs offline feature serving — what does each power?
9. What is a feature view / feature service?
10. How does a feature store differ from a vector database (both store embeddings)?

## 🟡 Core design
11. Walk through serving features for one online prediction request.
12. Walk through generating a point-in-time-correct training dataset.
13. Design the online store for low-latency, high-fan-out lookups.
14. Design the offline store and historical (point-in-time) retrieval.
15. Design the materialization pipeline (batch + streaming).
16. How do you guarantee online/offline consistency (no skew)?
17. How do you serve embedding features, and how does that relate to the vector DB?
18. Design the feature registry and discovery/reuse.
19. How do you handle on-demand / request-time features?
20. How do you keep streaming features fresh?

## 🔴 Senior / Staff deep dives
21. A model looks great offline but degrades online. Diagnose.
22. Point-in-time joins are too slow on billions of rows. Optimize.
23. Online p99 latency is too high under heavy feature fan-out. Fix.
24. How do you backfill a new feature for historical training without leakage?
25. How do you version features and evolve schemas without breaking models?
26. Designing for 1B entities × 1000s of features — what changes?
27. How do you detect and alert on feature drift / pipeline breakage?
28. A feature is computed in both batch and streaming — how do you reconcile (lambda/kappa)?

## 🧮 Math & estimation
29. Estimate the online store size for 100M users × 1,000 features.
30. Estimate the storage for user + item embedding features.
31. Estimate the online QPS / read fan-out for a recommendation request.
32. Estimate the freshness lag budget for a streaming feature.
33. Estimate the cost/shape of a point-in-time join for a training set.
34. How big is the online store vs the offline store, and why the difference?

## 🏗️ Design variations
35. Design an embedding store for user/item embeddings in a recommender.
36. Design a feature store feeding both a recommender and a fraud model (shared features).
37. Design real-time streaming features from a clickstream.
38. Design the feature layer behind a retrieval + ranking pipeline (with the vector DB).

## 🐞 Debugging & ops
39. Online and offline values for the same feature disagree. Triage it.
40. Training data has a feature that's always null in production. Why?
41. A streaming feature is stale / lagging. Diagnose.
42. Model performance silently dropped and you suspect a feature pipeline. Investigate.

---

> **How to practice:** anchor every answer on the **two guarantees** — point-in-time correctness and train/serve consistency — and the **dual-store** architecture. Check yourself against [answers.md](answers.md) and the [one-page cheat-sheet](cheat-sheet.md).

[← Back to feature store HLD](README.md) · [Answer key](answers.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
