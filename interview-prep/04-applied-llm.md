# 04 · Applied LLM (RAG · Agents · Prompting · Evals)

> The core of the **Applied AI / Forward-Deployed** loop: can you build a reliable, evaluated,
> cost-aware product on top of a frontier model? These questions are about **judgment** — when to use
> what, and how you'd know it works.

Sections: [Prompting & context](#prompting--context-engineering) · [Structured output](#structured-output--tool-calling) ·
[RAG](#rag-in-practice) · [Agents](#agents-in-practice) · [Evals](#evaluation-in-practice) ·
[Cost & latency](#cost--latency-optimization) · [Build scenarios](#realistic-build-scenarios)

---

## Prompting & context engineering

**Q: Walk through your prompting toolkit, simplest first.**
Zero-shot → few-shot (2–5 examples to pin format/style) → **chain-of-thought** for reasoning →
**self-consistency** (sample N, majority vote) for hard reasoning → **decomposition** (split into
sub-prompts / a chain). Start simple; add complexity only when an eval shows you need it.

**Q: How do you write a production-grade prompt?**
Clear role + task + constraints; specify the **output contract** (format, schema); give few-shot
examples for tricky formatting; put long reference material in a delimited block and tell the model to
ground in it; state what to do on uncertainty ("say you don't know"). Then **version it and test it**
with an eval set — prompts are tested artifacts, not vibes.

**Q: What is "context engineering" and why is it the successor to prompt engineering?**
It's deliberately managing the **whole token budget** across a task/trajectory: what goes in (system
rules, few-shot, retrieved docs, tool results, history), in what order, and **what to remove** (summarize
old turns, drop stale tool output). Agents live or die on this — "context rot" (irrelevant/contradictory
context) degrades quality, so you curate, compact, and prioritize.

**Q: Few-shot vs fine-tuning — when each?**
Few-shot: fast, no training, great for format/style and low volume; costs tokens every call. Fine-tune:
when you have many examples, need consistent behavior, want to cut prompt length/latency, or need a
capability prompting can't reach. Rule of thumb: **prompt → few-shot → RAG → fine-tune**, escalating
only when evals justify it.

**Q: Chain-of-thought with reasoning models — still needed?**
Modern reasoning models do much CoT internally, so heavy "think step by step" scaffolding matters less
— but clear structure, examples, and decomposition still help, and you pay for hidden reasoning tokens,
so manage the budget.

---

## Structured output & tool calling

**Q: How do you get reliable JSON every time?**
Three levels: (1) **ask** for JSON with the schema in the prompt; (2) **constrained decoding / JSON
mode** so tokens are grammar-restricted and output *must* parse; (3) **validate + repair** — parse
against a schema (e.g. Pydantic), and on failure send the error back for a fix. In production use (2)
or (3); never trust (1) alone.

**Q: Function/tool calling — how does it work and where does it fail?**
You pass typed tool schemas; the model emits a structured **tool call** (name + JSON args); you execute
it and feed the result back. Failures: hallucinated/invalid args (validate + repair), calling the wrong
tool (clear descriptions, fewer/cleaner tools), and not knowing when to stop (max-steps). Great tool
**descriptions and error messages** matter as much as the prompt.

**Q: The model returns a confident wrong answer. How do you reduce hallucination?**
Ground it (RAG with citations), instruct it to abstain when unsure, lower temperature for factual
tasks, ask for sources and verify them, decompose-and-check, and **measure** hallucination with an eval
(faithfulness to provided context). For high-stakes, add a verification pass or human review.

---

## RAG in practice

**Q: When RAG vs long-context vs fine-tuning?**
**RAG** for large, changing, or proprietary knowledge where you need freshness + **citations** +
access control — and it's cheaper than stuffing everything in context. **Long-context** for a small,
self-contained set of docs per request (simpler, no retrieval infra, but costs tokens and can "lose the
middle"). **Fine-tuning** for *behavior/format/skill*, not for injecting fresh facts. Often combine
RAG + a fine-tuned model.

**Q: Your RAG gives bad answers. How do you debug?**
**Isolate the stage.** Measure **retrieval** first (is the right chunk in the top-k? recall@k/MRR). If
retrieval is bad → fix chunking, add hybrid (BM25+dense), add a reranker, rewrite the query. If
retrieval is good but the answer is wrong → it's a **generation** problem (prompt grounding, context
ordering, model). Most RAG failures are retrieval, and most retrieval wins come from **chunking +
reranking**.

**Q: How do you chunk?**
Structure/semantic-aware chunks (by heading/paragraph) with some **overlap**, sized to the embedding
model and the answer needs (often a few hundred tokens). Too big ⇒ diluted embeddings + wasted context;
too small ⇒ lost context. Keep metadata (source, section) for citations and filtering.

**Q: Why hybrid retrieval and reranking?**
Dense (semantic) retrieval misses exact tokens (error codes, names, IDs); **BM25** nails lexical
matches. Fuse them (Reciprocal Rank Fusion). Then a **cross-encoder reranker** rescoring the top ~100
→ top ~8 is usually the single biggest quality boost, because bi-encoder embeddings are coarse.

**Q: How do you evaluate RAG?**
Two layers: **retrieval** (recall@k, MRR, nDCG against labeled relevant chunks) and **generation**
(faithfulness/groundedness, answer relevance, citation accuracy — e.g. RAGAS/LLM-judge). Build a small
gold Q→answer+source set early; it pays for itself.

---

## Agents in practice

**Q: When should something be an agent vs a fixed workflow?**
Default to the **least agency** that works. A **workflow** (you orchestrate fixed steps) is
predictable, cheap, and testable — use it when the steps are known. An **agent** (the LLM decides the
control flow) is for open-ended tasks needing dynamic tool use — accept the higher cost/latency and
reliability burden only when the task truly needs it.

**Q: Your agent works in demos but fails in production. Why and what do you do?**
Compounding errors (`pⁿ`), context bloat, brittle tool I/O, and prompt injection. Fixes: cap steps +
budget, add verification/reflection and retries, **compact context**, robust typed tools with forgiving
error messages, and a real **eval suite** + trajectory logging so you can see *where* it fails. Shorten
the horizon; prefer workflows for the deterministic parts.

**Q: How do you design a good tool?**
Single clear purpose; typed schema; an excellent **description** (the model's only instruction manual);
**forgiving error messages** the model can recover from; idempotent where possible; least-privilege
auth; sandboxed side effects. Test the "agent–computer interface" like a UX.

**Q: Memory for agents?**
Short-term = the scratchpad/history in context (compact it). Long-term = persist facts to a store
(often a vector DB) and **retrieve** relevant memories on demand. Decide *what* is worth remembering and
*when* to recall it — unbounded memory becomes noise.

---

## Evaluation in practice

**Q: A PM says "the new prompt feels better." How do you respond?**
Turn the vibe into a number: build an **eval set** (representative inputs + graders), run both prompts,
compare with a **confidence interval**, and check for regressions on edge cases. "Feels better" isn't
shippable; an eval with a CI is. This is the most important habit at a frontier lab.

**Q: How do you build an eval from scratch for a new feature?**
Start from **real/representative inputs** (prod logs, expected queries). Define success per task
(exact-match, rubric, or pairwise). Add graders: programmatic where possible, **LLM-as-judge** for
open-ended (validated against human labels). Cover **edge/failure cases** and safety. Wire it into
**CI** so every prompt/model change is gated. Grow it from production errors (the flywheel).

**Q: LLM-as-judge — how, and what are the pitfalls?**
Use a strong model to grade pairwise or by rubric. Pitfalls: **position bias** (favors the first
option — average over both orders), **verbosity bias** (prefers longer), **self-preference**.
**Validate the judge** against human labels (measure agreement) before trusting it. Cheap, scalable,
but not ground truth.

**Q: What metrics for a chat/agent feature?**
Offline: task success rate (+CI), faithfulness, win-rate vs baseline, refusal correctness. Online:
user thumbs, retry/regeneration rate, task completion, latency (TTFT/TPOT), **cost/request**, and
safety incident rate. Tie a release to **no regression** on the gated suite.

---

## Cost & latency optimization

**Q: A feature is too expensive/slow. Levers?**
- **Prompt caching / prefix caching:** cache the large stable system prompt + tools/examples; pay only
  for the changing suffix — big savings for chat/agents.
- **Model cascade / routing:** cheap model first, escalate hard cases — most traffic stays cheap.
- **Shorten context:** retrieve+rerank fewer, better chunks; compact history; trim few-shot once
  fine-tuned.
- **Cap output tokens**, **stream** for perceived latency, **batch** independent calls.
- **Right-size the model** and quantize; **semantic cache** repeat queries.
Always quantify: tokens × price, and what the eval says about the quality you traded.

**Q: Streaming — why and any gotchas?**
Stream tokens (SSE) so the user sees output at **TTFT**, not after the full generation — large
perceived-latency win. Gotchas: you must run **output safety/moderation** on the stream (or buffer
enough to filter), and structured-output validation happens after the stream completes.

---

## Realistic build scenarios

**Q: "Build a support bot over our docs that cites sources and escalates when unsure."**
RAG with hybrid retrieval + rerank + citations; abstain/escalate when top retrieval scores are low or
the judge flags low confidence; eval on a gold set (retrieval recall + faithfulness + citation
accuracy); cache common questions; safety filter; log everything for the improvement flywheel.

**Q: "Build an agent that resolves data tasks against our internal APIs."**
Typed tools per API with authz + sandboxing; ReAct/orchestrator loop with **max-steps + spend cap**;
human approval for writes/irreversible actions; treat API output as data (injection defense);
trajectory logging + a ≥20-task eval suite (success rate, steps, cost, p95). Start with a constrained
workflow for the common path, add agency only where needed.

**Q: "Make our classification feature 10× cheaper without hurting quality."**
Measure current cost + quality on an eval set. Try: route most inputs to a small/cheap model with a
confidence-based **cascade** to the big model; fine-tune a small model on labeled data to drop few-shot
tokens; cache; constrained decoding to cut output tokens. Ship only if the gated eval shows no
meaningful quality drop, and report the cost/quality curve.

---

### How to perform in this round
Lead with the decision and the **trade-off**, name **what you'd measure**, and show a **safety reflex**
(injection, abstention, eval-before-ship). Concrete > comprehensive: a working, evaluated, cheap v1
beats an over-engineered design.
