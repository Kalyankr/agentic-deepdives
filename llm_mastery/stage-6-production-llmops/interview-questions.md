# Stage 6 — Interview Questions (full-fledged, all levels)

> **Scope:** screening through **senior / staff / principal** (incl. applied-ML / LLM-product / platform roles). Angles: conceptual, coding, system design, debugging, product judgment. `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 💻 Coding · 🏗️ Design · 🐞 Debug
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What is prompt engineering? Give three techniques.
2. What is chain-of-thought prompting?
3. What is RAG and why use it?
4. What is an embedding and a vector database?
5. What is reranking in a retrieval pipeline?
6. What is function/tool calling?
7. What is the ReAct pattern?
8. Why use structured output (JSON mode)?
9. What are guardrails in an LLM app?
10. When would you fine-tune instead of using RAG?

## 🟡 Core (L4–L5)
11. Walk through a full RAG pipeline end to end.
12. Explain chunking tradeoffs (size vs overlap vs precision).
13. What is hybrid search and why combine dense + sparse?
14. How does reranking improve quality, and what does it cost?
15. Give the decision framework: prompt vs RAG vs fine-tune.
16. How do agents fail, and how do you contain it?
17. What is semantic caching and when does it help?
18. What is HyDE / query rewriting and why use it?
19. How do you ground a model and force citations?
20. What should you log/monitor in a production LLM app?

## 🔴 Senior / Staff deep dives (with follow-ups)
21. Design a production RAG system over 10M enterprise documents with citations and access control.
    → *covers:* ingestion/chunking, embeddings, ANN index (HNSW/IVF), hybrid + rerank, metadata/ACL filtering, grounding prompt, eval (retrieval + faithfulness), caching, observability, security (Stage 8).
22. A RAG answer is wrong. Walk me through localizing whether it's retrieval or generation.
    → *covers:* check if gold chunk was retrieved (recall@k); if yes → generation/grounding issue; if no → embedding/chunking/query issue; build an attribution harness.
23. Decide prompt vs RAG vs fine-tune for: (a) up-to-date company policy QA, (b) consistent JSON extraction, (c) a brand voice. Justify each.
    → *(a) RAG (changing knowledge); (b) fine-tune or constrained decoding (consistent behavior/format); (c) fine-tune (style) + prompt.*
24. Design an agent that books travel via tools. Cover planning, reliability, cost, and safety.
    → *covers:* tool schema, ReAct loop, step/time limits, validation/retries, idempotency, confirmation for irreversible actions, least privilege, injection defense.
25. Your RAG works in demos but hallucinates in prod. Systematic plan to fix?
    → *covers:* grounding/refusal instructions, better retrieval + rerank, chunking, faithfulness eval, context-sufficiency check, citations, guardrails.
26. How do you build the eval + observability layer so you can ship RAG changes safely?
    → *covers:* offline retrieval + generation evals, golden sets, online tracing, A/B, drift alerts, feedback capture → dataset.
27. Cut LLM API cost 60% for a high-traffic feature without hurting quality. Plan?
    → *covers:* semantic + exact caching, model routing/cascade, prompt compression, smaller/fine-tuned model, batching, context trimming, retrieval instead of long context.
28. When is an agent the wrong choice and a fixed pipeline better?
    → *covers:* determinism, latency, cost, debuggability; agents add nondeterminism + compounding errors.

## 💻 Coding / implementation
29. Implement a minimal RAG loop from scratch: chunk → embed → cosine top-k → prompt → answer.
30. Implement a reranking step with a cross-encoder over top-N candidates.
31. Implement self-consistency (sample N CoT paths, majority vote).
32. Implement a tool-calling loop (parse call → execute → feed result back) with a step limit.
33. Implement schema-validated JSON output with a retry-on-failure wrapper.
34. Implement hybrid search (combine BM25 + dense scores).

## 🏗️ System design / applied
35. Design an LLM gateway: multi-provider routing, caching, rate limiting, fallback, cost tracking, observability.
36. Design the feedback flywheel: capture thumbs up/down → datasets → eval → model/prompt improvement.
37. Design a multi-tenant knowledge assistant with per-tenant data isolation and citations.
38. Design CI/CD for prompts and RAG indexes (versioning, eval gates, rollback).

## 🐞 Debugging / scenarios
39. Retrieval recall@k is high but answers still miss the point. Where's the problem?
    → *generation/grounding:* model ignores context; fix prompt, reduce distractors, rerank, instruct to cite.
40. Latency spikes whenever context is long. Causes and fixes?
    → *prefill cost + large prompts; trim context, summarize, cache, smaller top-k, streaming.*
41. The agent occasionally loops forever or calls the wrong tool. Mitigations?
    → *step/time limits, tool schemas + validation, better planning prompt, observation grounding.*
42. Quality silently degraded over a month with no code change. Hypotheses?
    → *data/index drift, provider model update, embedding staleness, traffic shift; need monitoring + online eval.*

## ✅ What strong candidates demonstrate
- Apply the **prompt vs RAG vs fine-tune** framework with crisp justification.
- **Localize failures** across retrieval vs generation; eval each separately.
- Treat the app as a **system**: caching, routing, observability, cost, safety.
- Show **product judgment** (when agents are wrong, when determinism matters).

---
Related: the **🔥 Mastery checks** in [README.md](README.md) are the minimum bar.
