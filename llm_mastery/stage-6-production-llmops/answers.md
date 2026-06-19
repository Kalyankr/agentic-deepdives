# Stage 6 — Answer Key (Production LLMOps)

> Full worked answers to [interview-questions.md](interview-questions.md). The bar: apply the **prompt vs RAG vs fine-tune** framework with crisp justification, **localize failures** across retrieval vs generation, treat the app as a **system** (caching, routing, observability, cost, safety), and show **product judgment**.

---

## 🟢 Fundamentals

**1. Prompt engineering + three techniques.**
Designing the input to steer model behavior without changing weights. Techniques: **few-shot examples** (demonstrate the task), **chain-of-thought** ("think step by step"), and **role/structure prompting** (system role + explicit output format/constraints). Also: decomposition, delimiters, and giving the model an "out" (allow "I don't know").

**2. Chain-of-thought prompting.**
Prompting the model to produce intermediate reasoning steps before the final answer. It improves multi-step reasoning (math, logic) by letting the model allocate compute across steps and condition on its own intermediate results, rather than jumping to an answer.

**3. RAG and why.**
**Retrieval-Augmented Generation**: fetch relevant documents from a knowledge store and put them in the prompt so the model answers *grounded* in them. Use it for **up-to-date, proprietary, or large** knowledge you can't/won't bake into weights — reduces hallucination, enables citations, and updates instantly when data changes.

**4. Embedding + vector database.**
An **embedding** is a dense vector representation of text where semantic similarity ≈ vector proximity. A **vector database** stores these and does fast **approximate nearest-neighbor (ANN)** search to retrieve the most similar items to a query embedding — the retrieval backbone of RAG/semantic search.

**5. Reranking.**
A second-stage model (usually a **cross-encoder**) that re-scores the top-N candidates from cheap first-stage retrieval by jointly reading (query, document). It sharpens precision at the top (puts the truly relevant doc first) at the cost of extra latency — first stage maximizes recall, reranker maximizes precision.

**6. Function/tool calling.**
The model outputs a **structured call** (function name + JSON arguments) matching a provided schema; the app executes the real function and feeds the result back. It lets the LLM act on the world (search, DB, APIs, calculators) and ground answers in live data.

**7. ReAct pattern.**
**Reason + Act**: the model interleaves *thoughts* (reasoning), *actions* (tool calls), and *observations* (tool results) in a loop until it can answer. Reasoning guides which tool to call; observations ground the next step — the standard agent loop.

**8. Why structured output (JSON mode).**
Downstream systems need machine-parseable, schema-conformant output. JSON mode / constrained decoding guarantees valid, typed structure so you can reliably extract fields, avoid parse errors, and integrate the LLM into pipelines.

**9. Guardrails.**
Input/output checks around the model: **input** (prompt-injection/jailbreak filters, PII redaction, topic limits) and **output** (toxicity/safety classifiers, schema validation, grounding/citation checks, refusal of disallowed content). They enforce safety/policy and format independent of the model's own behavior.

**10. Fine-tune instead of RAG when…**
You need to change **behavior/format/style/skill** rather than inject **knowledge**: consistent JSON extraction, a brand voice, domain tone, latency-sensitive tasks (no retrieval round-trip), or to internalize a *stable* skill. RAG is for **changing facts**; fine-tuning is for **changing behavior**. They're often combined.

---

## 🟡 Core (L4–L5)

**11. Full RAG pipeline end to end.**
1. **Ingest** documents → clean/parse.
2. **Chunk** into passages (with overlap, metadata).
3. **Embed** chunks → store in a **vector index** (+ keyword index for hybrid).
4. **Query time:** (optional) **rewrite/expand** the query → embed → **ANN retrieve** top-N (hybrid dense+sparse).
5. **Rerank** top-N with a cross-encoder → top-k.
6. **Assemble prompt**: grounding instructions + retrieved context + question.
7. **Generate** with citation requirements.
8. **Post-process / guardrails** (faithfulness check, citations) → answer.
9. **Log** retrieval + generation for eval and the feedback flywheel.

**12. Chunking tradeoffs.**
**Large chunks** preserve context but dilute relevance (more distractors, retrieval matches on partially-relevant text, more tokens/cost). **Small chunks** are precise but may **split the answer** across boundaries, losing context. **Overlap** mitigates boundary-splitting at the cost of redundancy/storage. Tune chunk size + overlap to your content (structure-aware chunking — by section/paragraph — usually beats fixed size).

**13. Hybrid search; why dense + sparse.**
**Dense** (embeddings) captures **semantic** similarity (paraphrases, synonyms) but can miss exact terms; **sparse** (BM25) nails **exact keywords**, rare tokens, IDs, and acronyms but misses semantics. Combining them (score fusion / RRF) covers both failure modes — robust across query types, especially for enterprise/technical corpora.

**14. How reranking improves quality + cost.**
First-stage ANN is recall-oriented and approximate; a **cross-encoder reranker** reads query+doc together and scores true relevance far more accurately, pushing the best chunk to the top so the generator sees it first (and you can pass fewer chunks). Cost: an extra model pass over N candidates → added latency; mitigate by reranking only top-N and using a small reranker.

**15. Prompt vs RAG vs fine-tune framework.**
- **Prompt** when the base model already can do it and you just need steering — fastest, no infra.
- **RAG** when the bottleneck is **knowledge** that's large/changing/proprietary and you need grounding/citations.
- **Fine-tune** when the bottleneck is **behavior/format/style/skill** that's stable, or you need lower latency/cost than long prompts.
Decide by asking: *Is the gap knowledge or behavior? Does it change often? Latency/cost budget?* Often combine (fine-tune for format + RAG for facts).

**16. How agents fail; containment.**
Failure modes: **compounding errors** over steps, **infinite loops**, **wrong tool / bad arguments**, **prompt injection** via tool outputs, and **irreversible actions**. Contain with: **step/time/cost limits**, strict **tool schemas + argument validation**, **idempotency** and **human confirmation** for destructive actions, **least-privilege** tools, observation grounding, and tracing/observability to debug.

**17. Semantic caching; when it helps.**
Cache answers keyed by **embedding similarity** of the query (not exact string), so paraphrases hit the cache. Helps when traffic has many **semantically repetitive** queries (FAQs, popular questions) — big cost/latency wins. Risks: returning a cached answer for a *subtly different* query, or stale data — needs a similarity threshold and TTL/invalidation.

**18. HyDE / query rewriting.**
**Query rewriting** cleans/expands/decomposes the user query for better retrieval. **HyDE** (Hypothetical Document Embeddings) asks the LLM to *generate a hypothetical answer*, then embeds **that** to retrieve — because a full answer is semantically closer to the target documents than a short question. Both fix the **query–document mismatch** that hurts dense retrieval.

**19. Ground the model & force citations.**
Instruct it to answer **only** from the provided context, to say "I don't know" if the context is insufficient, and to **cite the chunk/source ID** for each claim. Reinforce with: passing well-reranked context, a **context-sufficiency check**, and a post-hoc **faithfulness verifier** that flags unsupported claims. Citations make grounding auditable and reduce hallucination.

**20. What to log/monitor in a production LLM app.**
Inputs/outputs (with privacy controls), prompt + model + version, **retrieved chunks + scores**, tool calls/results, tokens & **cost**, **latency (TTFT/TPOT, p50/p99)**, cache hit-rate, **user feedback** (thumbs/edits/regenerations), refusal/guardrail triggers, and quality/faithfulness eval scores on sampled traffic — plus drift alerts. Enables debugging, cost control, and the feedback flywheel.

---

## 🔴 Senior / Staff deep dives

**21. Production RAG over 10M docs with citations + access control.**
- **Ingestion:** parsers per format, structure-aware **chunking**, attach **metadata + ACL tags** (owner, group, sensitivity).
- **Index:** embeddings in an ANN index (**HNSW** for low-latency recall, or **IVF-PQ** for scale/memory) + a **BM25** index for hybrid; shard by tenant/domain.
- **Query:** rewrite → **hybrid retrieve** with **metadata/ACL filtering applied at query time** (never retrieve docs the user can't see) → **cross-encoder rerank** → top-k.
- **Generate:** grounding prompt requiring **citations**; refuse if context insufficient.
- **Eval:** retrieval (recall@k/MRR) + generation (faithfulness/answer-relevance) with golden sets.
- **Ops:** semantic + exact **caching**, **observability** (traces, scores, cost), drift alerts.
- **Security (Stage 8):** prompt-injection defenses on retrieved content, PII handling, audit logs. ACLs enforced in retrieval *and* re-checked before display.

**22. Localize a wrong RAG answer.**
Build an **attribution harness**: (1) Did retrieval fetch the **gold chunk**? Compute recall@k for that query. (2) **If yes** → it's a **generation/grounding** failure: model ignored or misused context (fix prompt, reduce distractors, rerank, force citations, maybe a stronger model). (3) **If no** → it's a **retrieval** failure: bad **chunking** (answer split), weak **embeddings/query** (semantic mismatch — try HyDE/hybrid), wrong **k** or ACL filter. Always check retrieval before blaming the model.

**23. Prompt vs RAG vs fine-tune for three cases.**
- **(a) Up-to-date company policy QA → RAG.** Knowledge that **changes**; RAG updates instantly and gives citations. Fine-tuning would bake in stale policy.
- **(b) Consistent JSON extraction → fine-tune and/or constrained decoding.** It's a **behavior/format** problem; fine-tune (or JSON-mode/grammar-constrained decoding) gives reliable structure that prompting alone won't guarantee at scale.
- **(c) Brand voice → fine-tune (style) + prompt.** Style is a **stable behavior** best internalized by fine-tuning; a system prompt reinforces tone. RAG is irrelevant — no knowledge gap.

**24. Agent that books travel — planning, reliability, cost, safety.**
- **Tools:** typed schemas for search/price/book/cancel; **least privilege** (no access beyond travel).
- **Loop:** ReAct planning with a bounded step/time/cost budget; validate every tool argument; **retry** transient failures with backoff; ensure **idempotency** (don't double-book on retry, use idempotency keys).
- **Reliability:** verify observations, handle tool errors gracefully, checkpoint state.
- **Safety:** **human confirmation before irreversible/charged actions** (booking, payment); guard against **prompt injection** from web/tool content (don't let retrieved text issue commands); audit log every action.
- **Cost:** cap tokens/steps, cache searches, use a smaller model for routing/parsing and a larger one only for planning.

**25. RAG hallucinates in prod (fine in demos) — systematic fix.**
1. **Measure** on real traffic: separate retrieval recall vs generation faithfulness.
2. **Retrieval fixes:** better chunking, **hybrid + rerank**, query rewriting/HyDE, tune k.
3. **Grounding fixes:** stronger "answer only from context / say I don't know / cite sources" prompt; reduce distractor chunks.
4. **Context-sufficiency check**: detect when retrieval is weak and **refuse/escalate** instead of guessing.
5. **Faithfulness guardrail** that verifies claims against context and blocks/regenerates unsupported ones.
6. **Eval + monitor** continuously; capture feedback to improve. Demos pass because they're in-distribution; prod needs these systematic guards.

**26. Eval + observability to ship RAG changes safely.**
- **Offline:** golden sets for **retrieval** (recall@k/MRR/nDCG) and **generation** (faithfulness/answer-relevance/LLM-judge); regression suite gated in CI.
- **Online:** **tracing** of each request (query → chunks+scores → prompt → answer → feedback), **A/B tests**, drift alerts on retrieval scores/quality.
- **Flywheel:** capture thumbs/edits → curate into eval and training sets.
Version prompts + indexes; gate promotion on eval; enable quick rollback.

**27. Cut LLM API cost 60% without hurting quality.**
- **Caching:** exact + **semantic** caching for repetitive queries (often the biggest win).
- **Model routing/cascade:** cheap/small model for easy queries, escalate hard ones; classify first.
- **Prompt/context reduction:** trim system prompts, compress/summarize context, retrieve fewer/better chunks (rerank), avoid stuffing long context.
- **Smaller/fine-tuned model** for the specific task.
- **Batching** for offline/async paths.
Order: caching + routing + context trimming first (no quality cost), then model size. Measure quality on a golden set at each step.

**28. When an agent is the wrong choice.**
When you need **determinism, low latency, predictable cost, and debuggability**. Agents add nondeterminism, **compounding errors**, variable latency/cost, and are hard to test. If the task is a **known, fixed workflow** (e.g. extract → validate → store), a **hard-coded pipeline** with targeted LLM calls is more reliable, cheaper, and auditable. Use agents only when the path genuinely can't be predetermined.

---

## 💻 Coding / implementation

**29. Minimal RAG loop.**
```python
import numpy as np
def chunk(text, size=500, overlap=50):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

def build(docs, embed):
    chunks = [c for d in docs for c in chunk(d)]
    vecs = np.array(embed(chunks))                      # (N, dim)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return chunks, vecs

def answer(q, chunks, vecs, embed, llm, k=4):
    qv = np.array(embed([q])[0]); qv /= np.linalg.norm(qv)
    top = (vecs @ qv).argsort()[-k:][::-1]               # cosine top-k
    ctx = "\n\n".join(chunks[i] for i in top)
    prompt = (f"Answer ONLY from context; cite chunk #. Say 'I don't know' if absent.\n"
              f"Context:\n{ctx}\n\nQ: {q}")
    return llm(prompt)
```

**30. Cross-encoder reranking.**
```python
def rerank(query, candidates, cross_encoder, top_k=4):
    # cross_encoder scores (query, doc) jointly
    scores = cross_encoder.predict([(query, c) for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in order[:top_k]]
```

**31. Self-consistency (majority vote over CoT).**
```python
from collections import Counter
def self_consistency(llm, prompt, n=5, temperature=0.7):
    answers = [extract_final(llm(prompt + "\nLet's think step by step.",
                                 temperature=temperature)) for _ in range(n)]
    return Counter(answers).most_common(1)[0][0]
```

**32. Tool-calling loop with step limit.**
```python
def run_agent(llm, tools, user_msg, max_steps=6):
    msgs = [{"role": "user", "content": user_msg}]
    for _ in range(max_steps):
        out = llm(msgs, tools=tools)
        if out.tool_call:
            name, args = out.tool_call.name, out.tool_call.args
            if name not in tools: 
                msgs.append(observation("error: unknown tool")); continue
            result = tools[name](**validate(args, tools[name].schema))  # validate!
            msgs.append(assistant(out)); msgs.append(observation(result))
        else:
            return out.content                       # final answer
    return "Stopped: step limit reached."            # guard against loops
```

**33. Schema-validated JSON with retry.**
```python
import json, jsonschema
def json_call(llm, prompt, schema, retries=2):
    for attempt in range(retries + 1):
        raw = llm(prompt, response_format="json")
        try:
            obj = json.loads(raw)
            jsonschema.validate(obj, schema)
            return obj
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            prompt += f"\nYour previous output was invalid: {e}. Return valid JSON only."
    raise ValueError("Failed schema validation after retries")
```

**34. Hybrid search (BM25 + dense).**
```python
import numpy as np
def hybrid(query, bm25, dense_vecs, embed, alpha=0.5, k=10):
    bm = np.array(bm25.get_scores(query.split()))
    qv = np.array(embed([query])[0]); qv /= np.linalg.norm(qv)
    dn = dense_vecs @ qv
    norm = lambda x: (x - x.min()) / (x.ptp() + 1e-9)    # min-max normalize
    score = alpha * norm(dn) + (1 - alpha) * norm(bm)    # or use RRF
    return score.argsort()[-k:][::-1]
```
(Reciprocal Rank Fusion is a robust alternative to score normalization.)

---

## 🏗️ System design / applied

**35. LLM gateway.**
A single entry point that provides: **multi-provider routing** (by model, cost, latency, availability) with **fallback** on errors/timeouts; **caching** (exact + semantic); **rate limiting / quotas** per tenant; **cost tracking** per request/tenant/model; **observability** (latency, tokens, errors, traces); **guardrails** (PII, safety) and **retries with backoff**. Decouples app code from providers and centralizes control/cost/security.

**36. Feedback flywheel.**
Capture **thumbs up/down, edits, regenerations, acceptance** alongside the full trace (consented/privacy-safe). Aggregate into datasets: positives → SFT examples, accepted-vs-rejected → **preference pairs** (DPO). Curate/clean, run through **eval gates**, then improve **prompts/RAG/model**; A/B the change; measure online lift; repeat. Production usage continuously feeds improvement.

**37. Multi-tenant knowledge assistant with isolation + citations.**
**Per-tenant data isolation:** separate indexes/namespaces (or strict metadata + row-level filters enforced at query time); never let one tenant's retrieval touch another's data. **AuthN/Z** on every request; ACL filters in retrieval. **Citations** to tenant's own sources. Per-tenant configs (prompts, models), quotas, and cost accounting. Encrypt at rest/in transit; audit logs. Test isolation explicitly (no cross-tenant leakage).

**38. CI/CD for prompts and RAG indexes.**
**Version** prompts and index builds (content hash + embedding-model version). On change: run the **offline eval suite** (retrieval + generation) as a **gate**; block on regression. **Canary/A-B** in production before full rollout; keep previous prompt/index for instant **rollback**. Re-embed when the embedding model changes; track index freshness. Treat prompts/indexes as deployable artifacts with the same rigor as code.

---

## 🐞 Debugging

**39. High recall@k but answers miss the point.**
The right context is retrieved but the **generation/grounding** is failing — the model ignores context, gets distracted by irrelevant chunks, or under-uses the answer chunk. Fix: tighten the grounding prompt (answer only from context, cite), **rerank** to put the best chunk first, **reduce distractor chunks** (smaller k after rerank), force citations, or use a stronger model. It's a generation problem, not retrieval.

**40. Latency spikes with long context.**
Long prompts mean expensive **prefill** (compute-bound, scales with prompt length) and more KV memory. Fixes: **trim/compress context** (rerank to fewer chunks, summarize), **reduce top-k**, **cache** (prefix + semantic), **stream** the response to improve perceived latency, chunked prefill, and avoid stuffing full documents — retrieve precisely instead.

**41. Agent loops or calls the wrong tool.**
Mitigations: **step/time/cost limits** to break loops; strict **tool schemas + argument validation** (reject malformed calls); a better **planning prompt** with clear tool descriptions and examples; **ground on observations** (force it to read tool results); detect repeated identical actions and abort; and add a fallback/human handoff. Trace every step to debug.

**42. Quality silently degraded over a month, no code change.**
Hypotheses: **data/index drift** (corpus changed, index stale), a **provider model update** (the API model changed underneath you), **embedding staleness** (new content embedded with an old/model-mismatched embedder), **traffic distribution shift** (new query types), or growing cache staleness. Need **monitoring + online eval** to catch this: track retrieval scores, faithfulness, and feedback over time; pin model versions; re-index on a schedule.

---

## What strong answers share
Applying the **prompt vs RAG vs fine-tune** framework with crisp justification; **localizing failures** between retrieval and generation and evaluating each separately; treating the app as a **system** (caching, routing, observability, cost, safety); and showing **product judgment** about when determinism beats agentic flexibility.

---
Back to [questions](interview-questions.md) · [Stage README](README.md) · [Index](../README.md)
