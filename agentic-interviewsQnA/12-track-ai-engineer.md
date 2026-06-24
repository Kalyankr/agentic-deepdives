# 12 — Track: AI Engineer

> **Role in one line:** you *build* agentic products — wiring LLMs, tools, RAG, and orchestration into reliable, low-latency, cost-controlled features that ship. Less "train the model," more "make the model useful and dependable in production."

> Use this as a focused lens over the core kit. It tells you **what to emphasize**, adds **AI-Engineer-specific Q&A**, and gives a **prep plan**.

---

## What the interview actually tests

| They probe | Because the job is | Where in the kit |
|------------|--------------------|------------------|
| Application architecture (RAG, agents, orchestration) | You assemble systems from LLM building blocks | [05](05-multi-agent-systems.md), [07](07-system-design.md) |
| Tool/function-calling & MCP integration | Tools are how your product *does* things | [04](04-tools-and-function-calling.md) |
| Context & prompt engineering | The model's behavior is mostly your context | [03](03-memory-and-context.md) |
| Framework pragmatism | You pick/skip frameworks under deadlines | [06](06-frameworks-and-protocols.md) |
| Eval, observability, guardrails | You're on the hook when it breaks at 2am | [08](08-production-evaluation-security.md) |
| Latency & cost engineering | Real users + real budgets | [08](08-production-evaluation-security.md), [09c](09c-followup-questions.md) |

**They usually do NOT** require you to derive backprop, train from scratch, or prove convergence — that's the Applied Scientist track. Know the concepts, but lead with **shipping judgment**.

---

## Priority reading (in order)
1. [04 — Tools & Function Calling](04-tools-and-function-calling.md) — your bread and butter.
2. [03 — Memory & Context](03-memory-and-context.md) — RAG + context engineering.
3. [07 — System Design](07-system-design.md) — run A-G-E-N-T-S cold.
4. [08 — Production, Eval & Security](08-production-evaluation-security.md) — what separates seniors.
5. [06 — Frameworks](06-frameworks-and-protocols.md) — opinions with trade-offs.
6. [09c — Follow-Ups](09c-followup-questions.md) §C, D, G, H — the mechanism-level probes.

---

## AI-Engineer-specific Q&A (new)

### Building & architecture

**Q. Design a production "chat with your docs" feature. What are the failure points?**
"Pipeline: ingest (chunk → embed → index), query (embed → hybrid retrieve → rerank → generate with citations). Failure points and fixes: bad chunking → semantic/structure-aware chunks + small-to-big; poor recall → hybrid (BM25 + vectors) + reranker; hallucinated answers → answer-only-from-context prompt + a faithfulness/citation guardrail; stale index → scheduled/event re-indexing; cost/latency → cache embeddings and prompt prefixes, cap context. I'd separately eval retrieval (precision/recall) and generation (faithfulness) so I know which half to fix."

**Q. How do you decide single-prompt vs. RAG vs. agent for a feature?**
"Cheapest thing that hits the bar. Single prompt if the knowledge is general and static. RAG if it needs fresh/private facts with citations. An agent loop only if the task needs dynamic, multi-step tool use whose path I can't hardcode. I prototype the simplest tier, measure against a success metric, and escalate only on misses — because each tier adds latency, cost, and eval surface."

**Q. A PM wants an agent feature in two weeks. How do you scope it?**
"Clarify the one job and its success metric, map the happy path, and ship a thin vertical slice — usually a router or RAG with a human fallback — behind a flag. I'd stand up tracing and a tiny eval set on day one so I can measure, then add tools/steps only where the metric demands. I'd push back on open-ended autonomy for v1 and gate any irreversible action behind confirmation."

**Q. How do you integrate a third-party tool/API as an agent tool?**
"Wrap it as a narrow, typed tool: clear name/description (when to use and not), enum/required params, and a return that's distilled to what the model needs — not the raw payload. Add timeouts, retries with backoff, idempotency keys for writes, and actionable error messages so the model can recover. If it's reusable across apps, expose it as an MCP server. And I treat its *output* as untrusted content for injection purposes."

### Context, prompts, UX

**Q. Your prompts work in dev but degrade as the conversation grows. Fix?**
"Classic context bloat. Rolling summarization of older turns + a sliding window of raw recent turns, retrieve only relevant context per step instead of carrying everything, offload big artifacts to memory and pass handles, and pin the system/policy at the edges to dodge lost-in-the-middle. I'd add token-budget accounting so eviction is deliberate, not accidental truncation."

**Q. How do you manage prompts like code?**
"Version them in the repo, review changes in PRs, and run an offline eval set in CI that blocks regressions. I template them (no string-concatenation of untrusted input), keep variants for A/B, and tie each prompt version to traces so I can attribute a quality change to a specific edit. A prompt change is a deploy, not a casual tweak."

**Q. How do you make an agent feel fast even when it's slow?**
"Stream tokens immediately, show intermediate progress ('searching…', tool steps), render partial results, and do independent tool calls in parallel. Route trivial steps to a fast model, cache prompt prefixes and frequent tool results, and set timeouts with graceful fallbacks so one slow dependency doesn't stall the turn. Perceived latency is a UX problem as much as a systems one."

### Reliability, eval, cost

**Q. How do you build an eval harness for an agent feature with no labels yet?**
"Seed a small gold set by hand plus synthetic edge cases, wire programmatic checks where possible (schema valid? action succeeded? citation present?), and add LLM-as-judge with a clear rubric for nuance — validated against a handful of human labels. Run it in CI on every prompt/tool/model change, then grow it continuously from real production failures. The eval set becomes the feature's safety net."

**Q. The feature's LLM bill is 5× the forecast. Walk me through cutting it.**
"Trace first to find the waste — usually loops, chatty multi-agent, or bloated context. Then: model routing (cheap model for easy/routing steps), prompt + semantic + tool-result caching, cap steps/tokens, trim and summarize tool outputs, and compact context. I track cost/task as a first-class metric and alert on it so it can't silently creep again. Most of these don't touch quality."

**Q. How do you ship guardrails without killing UX?**
"Layer them: input checks (injection/PII) and output checks (schema, safety, grounding) run fast and mostly invisibly; behavioral limits (step/token budgets, allow-listed tools) bound blast radius; and HITL only on genuinely irreversible actions so I'm not nagging users. Enforce in code, not by asking the model nicely, and fail toward safe defaults or human handoff rather than a hard error."

**Q. Which framework would you actually use, and when none?**
"For a simple tool-calling assistant, often none — a ~50-line function-calling loop is transparent and lock-in-free. LangGraph when I need durable state, cycles, or HITL; an Agents-SDK/handoff style for lightweight routing; CrewAI for a quick role-based team. I keep tools/prompts/eval framework-agnostic and depend on MCP so I can swap the orchestration layer."

---

## Coding round expectations (AI Engineer)
- **Whiteboard a ReAct loop** from scratch (see [code/01_react_from_scratch.py](code/01_react_from_scratch.py)) — tools, loop, stop conditions, loop detection.
- **Implement function-calling** round trip and a tool with validation + error handling.
- **Build a tiny RAG**: chunk, embed (mock), retrieve top-k, stuff, answer with citation.
- **Add a guardrail / structured output** and a retry-on-invalid-JSON path.
- Less emphasis on hard algorithms; more on **clean, correct integration code** and edge cases (timeouts, bad tool output, empty retrieval).

## System-design round
Drive **A-G-E-N-T-S** ([07](07-system-design.md)). Favorite prompts: support agent, chat-with-data over millions of docs, email triage/draft, internal "agent that does X across our APIs." Always volunteer eval, observability, guardrails, cost/latency, and an MVP.

## Signals they grade
✅ Ships the simplest thing that works · names concrete components · obsesses over eval/observability · handles tool failure & injection · quantifies cost/latency · pragmatic about frameworks.
🚩 Over-engineers to multi-agent · no eval plan · trusts model output blindly · ignores cost · can't wire a tool call end-to-end.

## 1-week plan
- **D1–2:** [04](04-tools-and-function-calling.md) + [03](03-memory-and-context.md); run [code/](code/) examples 01–02, build a mini-RAG.
- **D3:** [07](07-system-design.md); do 2 designs on paper.
- **D4:** [08](08-production-evaluation-security.md); add tracing + an eval set to your mini project.
- **D5:** [06](06-frameworks-and-protocols.md) + this file's Q&A out loud.
- **D6:** [09c](09c-followup-questions.md) §C/D/G/H; [11 mock](11-mock-interview.md) rounds 2–4.
- **D7:** Flashcards + behavioral stories ([09b](09b-interview-answers-full.md) §K).
