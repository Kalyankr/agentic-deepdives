# 09c — Follow-Up Questions (the deeper probes)

> These are the **second- and third-level follow-ups** interviewers ask *after* your first answer, to find the edge of your knowledge. All new — they extend, not repeat, [09-interview-questions.md](09-interview-questions.md). Format: the **trigger** (what you said) → the **↳ follow-up** → a tight answer.

> 💡 Follow-ups going deeper is a *good* sign. When you hit your limit, say *"I'm not certain, but here's how I'd reason about it"* — structured reasoning beats bluffing.

---

## A. Fundamentals — deeper

**You said "an agent loops over tools."**
- ↳ **Who owns the loop — the model or your code?** Your code owns the loop and stop conditions; the model only decides the next action each turn. This is why "agent reliability" is mostly an engineering property (the harness), not just a model property.
- ↳ **So is the loop part of the model or the system?** The system. The same model is "agentic" or not depending on the scaffolding around it.

**You said "use the lowest agency that works."**
- ↳ **How do you actually measure 'works'?** Define a success metric and a target (e.g., 90% task success at <$0.05 and <3s p95), build a small eval set, and escalate agency only when the simpler tier misses the bar.
- ↳ **Give a case where you escalated.** "Routing worked for 80% of tickets; the long-tail needed dynamic multi-step lookups, so I added a ReAct loop only for the fallback path, keeping the cheap router for the head."

**You said "agents adapt to feedback."**
- ↳ **What's the difference between adaptation and just retrying?** Adaptation means the model changes its *strategy* based on the observation (different tool/args/approach); blind retry repeats the same action. Loop detection + surfacing errors into context is what enables real adaptation.

---

## B. Reasoning & planning — deeper

**You explained ReAct.**
- ↳ **Where does the 'Thought' actually help — isn't it just tokens?** It conditions the next action on explicit intermediate reasoning, which empirically improves tool selection and reduces premature answers. With reasoning models, that thinking moves *inside* the model, so you write less explicit CoT.
- ↳ **When is ReAct the wrong choice?** Long, well-structured tasks where re-reasoning every step wastes tokens — use Plan-and-Execute. Or trivial single-tool tasks where a plain function-calling step suffices.

**You mentioned reflection improves quality.**
- ↳ **How do you stop reflection from making it *worse*?** Cap rounds, require an *objective* signal (tests, validator, rubric) not just self-opinion, and keep the best-so-far so a bad revision can't regress you below it.
- ↳ **What if there's no objective signal?** Then reflection mostly amplifies the model's bias — prefer a separate critic model, ensemble/voting, or human review instead.

**You said reasoning models "think before answering."**
- ↳ **What's 'test-time compute' and why does it matter?** Spending more inference compute (longer internal chains, search, sampling) to raise accuracy without retraining. It's a knob: trade latency/cost for quality per request, which reshapes how you budget agents.
- ↳ **Downside of reasoning models in agents?** Higher latency and cost per call, hidden reasoning tokens you pay for, and they can over-think simple steps — so route simple steps to a fast model.

---

## C. Memory & context — deeper

**You described RAG.**
- ↳ **How does the vector index actually find neighbors fast?** Approximate nearest neighbor structures — **HNSW** (navigable small-world graphs, the common default) or **IVF** (clustered inverted lists), often with **product quantization** to compress vectors. You trade a little recall for big latency wins.
- ↳ **Dense vs. sparse vs. hybrid — when does sparse still win?** Sparse/BM25 wins on exact terms, rare tokens, codes, and IDs where embeddings blur meaning. Hybrid (fuse both, then rerank) is the robust default.
- ↳ **How do you pick chunk size?** Match it to the retrieval unit and embedding model's sweet spot; too large dilutes the vector and adds noise, too small loses context. Test empirically; consider semantic or structure-aware chunking and small-to-big (retrieve small, expand to parent) strategies.
- ↳ **What does a reranker add over the vector score?** A cross-encoder reads query+chunk *together* (not separately), so it scores relevance far more precisely than cosine similarity — applied to the top-k to boost precision before generation.

**You mentioned summarizing old turns.**
- ↳ **What do you lose, and how do you bound it?** You lose detail and risk context poisoning if the summary encodes an error. Bound it by keeping raw recent turns, summarizing only older history, and storing key facts/decisions in structured memory rather than prose.
- ↳ **Have you seen memory systems beyond a vector store?** Yes — hierarchical/virtual-context approaches (MemGPT/Letta-style paging between a small context and external memory), and managed memory layers that extract and update durable facts. The idea is treating context like an OS treats RAM vs. disk.

**You said "prompt caching cuts cost."**
- ↳ **How does it work mechanically?** Providers cache the model's computed state for a *static prefix* (system prompt, tools, long context); reused prefixes skip recompute, cutting latency and cost on the cached portion. So you order prompts static-first, dynamic-last to maximize hits.

---

## D. Tools & function calling — deeper

**You explained function calling.**
- ↳ **How are structured outputs guaranteed, not just requested?** Constrained/grammar-based decoding masks the token distribution at each step to only schema-valid tokens (JSON-schema or a GBNF grammar), so the output *cannot* violate the schema — stronger than "JSON mode" prompting.
- ↳ **What happens when the model hallucinates an argument that's valid JSON but wrong?** Schema validation won't catch it — you need semantic validation in the tool (range checks, existence checks) and return an actionable error so the model can correct.
- ↳ **Parallel tool calls — what breaks?** Hidden dependencies (one call should use another's result), shared-state races, and partial failures. Only parallelize provably independent calls and handle per-call errors.

**You mentioned MCP.**
- ↳ **What's the security model of MCP?** The host mediates; servers should run least-privilege with explicit user consent for sensitive tools. Risks: a malicious/compromised server, or tool *descriptions* and *returned content* carrying injection — so treat both as untrusted.
- ↳ **MCP resources vs. tools — when use which?** Resources = read-only context the host can pull in (files, rows); tools = actions the model invokes. Use resources for grounding data, tools for effects.

**You said "wrap an agent as a tool."**
- ↳ **Tools vs. handoff vs. A2A — pick one for sub-tasks.** Agent-as-tool when the caller wants a *result* and keeps control; handoff when the sub-agent should *own* the conversation; A2A when the other agent is a separate, independently-deployed service across a trust/vendor boundary.

---

## E. Multi-agent — deeper

**You chose orchestrator-worker.**
- ↳ **How do workers share findings without blowing tokens?** Write structured artifacts to shared state and pass *references/summaries*, not full transcripts; the orchestrator reads the distilled artifacts, not every worker's raw history.
- ↳ **How does the orchestrator decide when to stop spawning workers?** Marginal-value stopping: a budget (max workers/depth/tokens) plus a check that new work adds information; the orchestrator (or a critic) judges coverage.
- ↳ **How do you reach consensus when agents disagree?** Options: a judge/aggregator agent, majority voting, debate-to-convergence, or weighting by confidence/evidence. Pick by cost — voting is cheap, debate is expensive.

**You said multi-agent isolates context.**
- ↳ **But then how do you avoid contradictory decisions (fragmentation)?** Single source of truth in shared state, the orchestrator owning the plan, and structured handoffs that carry the *decisions and constraints* a sibling needs — not just free text.
- ↳ **How do you trace a failure to a specific agent?** Per-agent spans with attribution (which agent, which prompt version, which tools), correlated by a trajectory/trace ID, so you can replay the exact path.

**You mentioned A2A.**
- ↳ **What's in an Agent Card?** A machine-readable description of an agent's identity, capabilities/skills, endpoint, and auth — so other agents can discover it and delegate tasks without bespoke integration.
- ↳ **MCP and A2A together — concrete example?** A planner agent (LangGraph) calls MCP tools for data, and delegates a specialized sub-task to a separate vendor's agent over A2A — protocols compose vertically (tools) and horizontally (agents).

---

## F. Frameworks — deeper

**You'd use LangGraph.**
- ↳ **What does its checkpointer actually buy you?** Durable state per thread → pause/resume, human-in-the-loop interrupts, retry from a step, and time-travel debugging. That's the main reason to choose it over a raw loop.
- ↳ **When would LangGraph be overkill?** A single tool-calling assistant with no durable state or HITL — a plain function-calling loop is simpler and more transparent.

**You compared frameworks.**
- ↳ **How do you avoid framework lock-in?** Keep your domain logic (tools, prompts, eval) framework-agnostic, depend on standard interfaces (MCP for tools), and treat the framework as orchestration glue you can swap.
- ↳ **AutoGen v0.2 vs v0.4 — what changed?** v0.4 moved to an async, event-driven, actor-style core (autogen-core/agentchat) for scalability and observability; concepts (conversable agents, group chat) carry over but the API differs. Verify exact specifics before interview day.

---

## G. System design — deeper extensions

**You designed a support agent.**
- ↳ **How do you handle a brand-new intent you didn't build a specialist for?** A fallback general agent + retrieval, low-confidence → human escalation, and log it to expand coverage (data flywheel). Don't fail closed silently.
- ↳ **How do you prevent the agent from contradicting policy across a long chat?** Pin policy in a system layer that retrieved/user content can't override, validate actions against policy in code, and keep a structured order/case state as source of truth.
- ↳ **Multi-tenant: how do you isolate customers?** Per-tenant credentials/scopes, metadata-filtered retrieval, row-level access on memory, and tenant-tagged tracing — never rely on the prompt for isolation.

**You designed deep research.**
- ↳ **How do you guarantee citations are real, not hallucinated?** Generate claims *from* retrieved passages, attach source IDs at retrieval time, and run a verification pass that checks each claim against its cited source; drop/flag unsupported ones.
- ↳ **How do you bound cost when the question is open-ended?** Hard budgets (workers/depth/tokens), early-stop on diminishing new info, and breadth-then-depth control by the orchestrator.

**General.**
- ↳ **Sync vs. streaming vs. async — pick for a 20s task.** Stream partial output for perceived latency if interactive; if it's truly long or batchable, go async with a job ID + status. Never block a request thread for tens of seconds.
- ↳ **How do you A/B test an agent safely?** Shadow first (log, don't act), then canary a small %, compare task success / cost / latency / guardrail-trigger rates, with instant rollback.

---

## H. Production, eval & cost — deeper

**You said "build an eval set."**
- ↳ **Where do the labels come from at the start?** Seed with hand-written gold cases + synthetic edge cases, then grow from real production failures; use programmatic checks where possible and human spot-checks to calibrate LLM-judges.
- ↳ **How do you evaluate a *trajectory*, not just the answer?** Compare against a reference trajectory or score with a rubric: correct tool selection, valid args, sensible order, no redundant/looping steps, recovery on error.
- ↳ **How do you keep LLM-as-judge honest?** Clear rubric, pairwise comparisons, randomized position, a held-out human-labeled set to measure judge accuracy, and ideally a different model family than the one being judged.
- ↳ **What's your single most important online metric?** Task success rate (goal achieved), then cost/task and p95 latency; guardrail-trigger and escalation rates as health signals.

**You mentioned observability.**
- ↳ **What standard would you build on?** OpenTelemetry GenAI semantic conventions for spans (LLM calls, tool calls, token/cost attributes) so traces are vendor-neutral, plus an LLM-observability tool (LangSmith/Langfuse/Phoenix) for prompt/cost analytics.
- ↳ **What exactly is a 'span' here?** One unit of work — an LLM call or tool call — with inputs, outputs, timing, tokens, and cost, nested under a trajectory trace so you can see the whole tree.

**You said "route to a cheaper model."**
- ↳ **How do you decide per-request which model?** A lightweight classifier or rules on difficulty/length/tool-need; cheap model handles the head, escalates to the frontier model on low confidence or failure.
- ↳ **What's the token economics intuition?** Cost ≈ (input + output tokens) × price; agents multiply this by steps and (in multi-agent) by agents and repeated context. Levers: fewer/shorter steps, caching, trimming tool outputs, routing.
- ↳ **What is speculative decoding and does it help agents?** A small draft model proposes tokens that the large model verifies in parallel, cutting latency with identical output. It helps the per-call latency of long generations; orthogonal to step-count.

**Reliability.**
- ↳ **How do you make a tool call idempotent?** Idempotency keys so a retry doesn't double-execute side effects (e.g., one refund per key), plus dedupe on the server.
- ↳ **How do you reproduce a non-deterministic failure?** Full trace capture (inputs, seed/temperature, model+prompt versions, tool responses) so you can replay; evaluate over many runs and report rates, not single runs.

---

## I. Security — deeper

**You raised prompt injection.**
- ↳ **Filters get bypassed. What's the structural defense?** Don't rely on detection alone: least-privilege tools, isolate untrusted content from instructions, require validation/HITL before destructive or exfiltrating actions, and break the lethal trifecta (private data + untrusted content + outbound channel).
- ↳ **What's a dual-LLM or quarantine pattern?** A privileged LLM never sees raw untrusted content directly; a separate quarantined LLM processes untrusted data and returns only structured, validated results — limiting injection blast radius.
- ↳ **Indirect injection via a tool result — concrete mitigation?** Treat tool/web/email output as data, never as instructions; sanitize and constrain it; and scope what any single action can do so a hijack can't escalate.

**You mentioned excessive agency.**
- ↳ **How do you scope an agent's permissions in practice?** Per-tool allow-lists, short-lived scoped credentials, environment sandboxing (no network/FS unless needed), and confirmation gates on high-impact actions.
- ↳ **What do you log for an audit trail?** Every action with actor (which agent), inputs, authorization decision, and outcome — so destructive actions are attributable and reviewable.

---

## J. Emerging / cutting-edge (2025–2026)

**Agentic RAG.**
- ↳ **How is it different from 'RAG with a tool'?** The agent decides *whether*, *what*, *which source*, and *how many times* to retrieve, can decompose into sub-queries, and reasons between retrievals — multi-hop and self-correcting, not one-shot.

**Computer-use / GUI agents.**
- ↳ **Why are they still unreliable, and what improves them?** Visual grounding errors, brittleness to UI change, and latency. Improvements: accessibility-tree + pixel hybrid grounding, set-of-marks prompting, verification steps, and narrow task scoping. Always sandbox + HITL.

**Voice / realtime agents.**
- ↳ **What changes architecturally for voice?** Streaming speech-to-text and text-to-speech, low-latency turn-taking and barge-in handling, partial/interruptible generation, and structured task state so errors don't accumulate in free-form audio. Confirm before irreversible actions.

**Multimodal agents.**
- ↳ **What's hard about tools that return images/PDFs?** Token cost of vision, grounding claims to specific regions/pages, and keeping large artifacts out of context (store + reference). Summarize visual observations into text where possible.

**Self-improving / learning agents.**
- ↳ **How can an agent 'learn' without fine-tuning?** In-context/episodic learning: store successful trajectories, reflections, and reusable skills (procedural memory) and retrieve them next time — Reflexion and skill-library (Voyager-style) approaches.
- ↳ **Where does RL fit for agents?** RL on agentic trajectories (tool use, multi-step) optimizes a policy against task reward; reasoning models are RL-trained for long CoT. It's powerful but needs reward signals and infra — most teams start with prompting + evals.

**Long-horizon agents.**
- ↳ **What dominates failure over hundreds of steps?** Compounding error and context management. Mitigate with decomposition + checkpoints, external memory/state as source of truth, verification between phases, and short, recoverable sub-trajectories.

**Protocols maturing.**
- ↳ **Why do MCP and A2A matter strategically?** They turn agents and tools into interoperable, composable services (N+M, not N×M integrations), enabling cross-vendor agent ecosystems — the "networking + USB-C" layer of agentic systems.

**Evals as the moat.**
- ↳ **Why is eval increasingly the differentiator?** Models commoditize; the team that can *measure* agent quality (domain eval sets, trajectory metrics, online experiments) ships reliable agents faster. "Your eval set is your moat."

---

## K. Curveballs & meta

- ↳ **"What would you push back on if I asked for a fully autonomous agent that moves money?"** I'd require HITL on transfers, least-privilege scoped tools, hard limits, strong eval + audit logs, and a staged rollout (shadow → approval → canary). Autonomy on irreversible financial actions without these is a non-starter.
- ↳ **"Your agent is 95% accurate per step over 12 steps. Is that good?"** No — 0.95^12 ≈ 54% end-to-end. Long agents need either fewer steps, per-step verification/recovery, or decomposition with checkpoints.
- ↳ **"Cheapest way to improve a flaky agent today?"** Usually better tools and tool descriptions + an eval set to localize failures — before touching the model. Most "model problems" are scaffolding problems.
- ↳ **"When is the LLM the wrong tool entirely?"** When deterministic code, a rules engine, or classic ML solves it more reliably and cheaply. Use the model only where you need language understanding or open-ended reasoning.
- ↳ **"How do you stay current?"** Foundational papers for the *why*, vendor engineering blogs for production lessons, benchmarks for what 'good' means, and hands-on builds so it sticks — optimize for fundamentals over chasing SDKs.

---

## How to use this file
- Pair each trigger with your main answer from [09b](09b-interview-answers-full.md); rehearse the **follow-up chain** so a deeper probe never surprises you.
- For 🔥 topics, be ready to go *two* levels down and name a mechanism (HNSW, constrained decoding, prompt caching, OTel spans, dual-LLM pattern).
- If you don't know: *"I haven't used that directly — here's how I'd reason about it"* + a structured guess. That scores.
