# Stage 6 — Production Systems / LLMOps

> **Objective:** Build reliable, grounded, observable LLM applications — and confidently choose between **prompting, RAG, and fine-tuning**. This is the stage most applied work actually lives in. Start it **early, in parallel** with Stages 1–5 (it needs no model training).

[← Stage 5](../stage-5-inference-optimization/README.md) · [Index](../README.md) · Next: [Stage 7 — Advanced](../stage-7-advanced-specialization/README.md)

📝 **Interview prep:** [interview-questions.md](interview-questions.md) · ✅ [answer key](answers.md)

---

## Why this stage matters

A raw model is not a product. The gap between "cool demo" and "reliable system" is engineering: grounding, structure, guardrails, observability, and cost control. Mastering this makes you the person who can actually **ship**.

---

## Mental model

> An LLM app is a **system** where the model is one component. Your job is to feed it the right context, constrain its output, verify it, observe it, and control cost.

The three ways to give a model new behavior/knowledge:

| Lever | Adds | Best for | Cost/latency |
|------|------|----------|--------------|
| **Prompting** | behavior, format | quick iteration, general tasks | cheapest |
| **RAG** | *knowledge* (dynamic, current) | factual grounding, private/changing data | medium |
| **Fine-tuning** | *behavior/style/skill* (baked in) | consistent format, narrow tasks, latency | high upfront |

Often the answer is **a combination**. Knowledge that changes → RAG. Behavior that's consistent → fine-tune. Everything starts with prompting.

---

## Concept-by-concept deep dive

### 6.1 Prompt engineering (the cheapest lever — master it first)
- **Zero-/few-shot:** show 0..k examples in the prompt.
- **Chain-of-Thought (CoT):** "think step by step" → elicits intermediate reasoning, big gains on math/logic.
- **Self-consistency:** sample multiple CoT paths, take the majority answer.
- **ReAct:** interleave **Reasoning** + **Acting** (tool calls) — the backbone of agents.
- **Structured prompting:** system/role separation, delimiters, explicit output schemas.
- **Prompt as interface:** treat prompts as versioned code (test them, track changes).

### 6.2 RAG — Retrieval-Augmented Generation (deep dive)
The dominant pattern for grounding models in your data.

**Pipeline:**
```
docs → chunk → embed → store in vector DB
query → embed → retrieve top-k → (rerank) → stuff into prompt → generate (with citations)
```

- **Chunking:** size/overlap tradeoff. Too big → diluted relevance + wasted context; too small → lost meaning. Try semantic/structure-aware chunking.
- **Embeddings:** map text to vectors; choose a good embedding model (domain matters). Normalize; pick cosine/dot.
- **Vector DBs:** FAISS, Chroma, Qdrant, pgvector, Milvus. ANN search (HNSW/IVF) for speed.
- **Retrieval:** top-k by similarity. **Hybrid search** = dense (semantic) + sparse (BM25/keyword) → best of both.
- **Reranking:** a cross-encoder re-scores the top-N for precision before stuffing context. Big quality lever.
- **Query transformation:** rewriting, **HyDE** (hypothetical answer embedding), multi-query, decomposition.
- **Advanced:** parent-document retrieval, contextual compression, metadata filtering, citations/attribution.
- **Grounding:** instruct the model to answer **only** from context and cite sources; refuse if context is insufficient.

### 6.3 Agents & tool use
- **Function/tool calling:** model emits a structured call; your code executes it; result goes back in. The model orchestrates, your code acts.
- **ReAct loop:** reason → choose tool → observe result → repeat → answer.
- **Planning:** decompose complex tasks into steps/subgoals.
- **Risks:** error compounding over steps, infinite loops, latency/cost blow-ups, tool misuse, **prompt injection via tool outputs** (see Stage 8). Add step limits, validation, and guardrails.

### 6.4 Structured output & reliability
- **JSON mode / schema-constrained / grammar-constrained decoding:** force valid, parseable output. Essential for pipelines.
- **Validation + retry:** validate against a schema; reprompt on failure.
- **Determinism:** lower temperature + fixed params for reproducible pipeline steps.

### 6.5 Guardrails & safety in production
- **Input guardrails:** filter injection, off-topic, PII, abuse.
- **Output guardrails:** toxicity/PII filters, policy checks, grounded-ness checks, schema validation.
- **Fallbacks:** graceful degradation, human handoff, cached/default responses.

### 6.6 Cost & latency engineering
- **Caching:** exact-match and **semantic caching** (cache by embedding similarity).
- **Model routing / cascades:** cheap model first, escalate to expensive only when needed.
- **Prompt compression**, shorter context, smaller models where adequate.
- **Batching / streaming** responses for UX (ties to Stage 5).

### 6.7 Observability & LLMOps
- **Tracing:** capture full request → retrieval → prompt → response → tool calls (LangSmith, Phoenix, Langfuse, OpenTelemetry).
- **Production evals:** run Stage-4 evals continuously on live traffic samples; track quality drift.
- **A/B testing:** compare prompts/models/configs on real users with real metrics.
- **Monitoring:** latency, cost, error rates, refusal rates, user feedback, hallucination flags.
- **Versioning & CI/CD:** version prompts, models, datasets, and indexes; test before deploy; enable rollback.
- **Feedback loops:** capture thumbs up/down → build datasets → improve (back to Stages 3/4).

### 6.8 The decision framework (internalize this)
1. **Start with prompting.** Cheapest, fastest to iterate.
2. **Need current/private/large knowledge?** Add **RAG**.
3. **Need consistent format/style/behavior, or lower latency/cost at scale?** **Fine-tune** (Stage 3).
4. **Hard problem?** Combine: fine-tuned model + RAG + good prompts + guardrails.

---

## Ordered learning path

1. Read the original **RAG** paper + a modern RAG survey.
2. Read **ReAct** and **Toolformer**.
3. Build with a framework (LlamaIndex/LangChain) **but** re-implement the core RAG loop yourself once, so you understand the primitives — not just the API.
4. Add observability + evals to your app.
5. Do the labs.

---

## 🛠️ Hands-on labs

- [ ] **Lab A — Prompting toolkit:** implement zero-shot, few-shot, CoT, and self-consistency on a reasoning task; compare accuracy.
- [ ] **Lab B — RAG from scratch:** chunk → embed → vector store → retrieve → generate with citations, **without** a framework, over your own docs.
- [ ] **Lab C — Add reranking + hybrid search;** measure the quality lift vs Lab B.
- [ ] **Lab D — Separate RAG evals:** retrieval recall@k *and* generation faithfulness (reuse Stage 4).
- [ ] **Lab E — Agent:** build a 2–3 tool agent with function calling, step limits, and output validation.
- [ ] **Lab F — Observability:** add tracing + logging + a small online eval to one of the above; build a tiny dashboard.
- [ ] **Lab G — Decision writeup:** for a chosen use case, argue prompt vs RAG vs fine-tune with tradeoffs.

---

## ⚠️ Common pitfalls & gotchas

- Reaching for fine-tuning when the real need is **knowledge** (use RAG) — or vice versa.
- Naive fixed-size chunking that splits mid-idea.
- Skipping reranking and blaming the LLM for retrieval misses.
- No grounding instruction → confident hallucinations despite RAG.
- Unvalidated structured output breaking downstream code.
- No observability → can't debug or prove quality in prod.
- Agents without step limits → runaway loops and cost.
- Treating prompts as throwaway instead of versioned, tested artifacts.
- **Trusting tool/retrieved content as instructions** → prompt injection (Stage 8).

---

## 🔥 Mastery checks (answer without notes)

- [ ] Given a use case, justify prompt vs RAG vs fine-tune (and combinations) with tradeoffs.
- [ ] Walk through a full RAG pipeline and name a failure mode at each step.
- [ ] Diagnose a bad RAG answer: retrieval miss or generation miss? How do you prove it?
- [ ] Explain chunking tradeoffs and when hybrid search + reranking help.
- [ ] How does ReAct turn an LLM into an agent? What are the main failure modes?
- [ ] What would you log/monitor for a production LLM app, and why?
- [ ] How do semantic caching and model routing cut cost?
- [ ] Where can prompt injection enter an agentic/RAG system?

---

## ✅ Stage 6 checklist

- [ ] Read RAG, ReAct, a RAG survey
- [ ] Built RAG **from scratch** (Lab B) + reranking (Lab C)
- [ ] Separated retrieval vs generation eval (Lab D)
- [ ] Built one agent with guardrails (Lab E)
- [ ] Added observability to one app (Lab F)
- [ ] All mastery checks passable
- [ ] Notes in your own words

**When complete → proceed to [Stage 7](../stage-7-advanced-specialization/README.md).**
