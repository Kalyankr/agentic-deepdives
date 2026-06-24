# 09 — Interview Question Bank

> 120+ questions across conceptual, design, coding, scenario, and behavioral. Practice **out loud**. Short model answers included; deeper detail is in chapters 01–08.
>
> 🗣️ **Want the full, spoken-style answer to every question below?** See **[09b — Full Spoken Answers](09b-interview-answers-full.md)** — each one written exactly how you'd say it in the interview.
>
> 🔬 **Want the deeper probes interviewers ask *after* your answer?** See **[09c — Follow-Up Questions](09c-followup-questions.md)** — second/third-level digs plus emerging 2025–26 topics.

Legend: ⭐ = very common · 🔥 = hard/senior · 💬 = behavioral

---

## A. Fundamentals & concepts

1. ⭐ **What is an AI agent?** — LLM that dynamically decides its own control flow to reach a goal, acting via tools, observing results, looping until done. = LLM + loop + tools + memory + goal + stop condition.
2. ⭐ **Agent vs. workflow vs. single LLM call?** — Single call: no control flow. Workflow: predefined code paths. Agent: LLM directs its own process/tools at runtime.
3. **Name the five components of an agent.** — Model/reasoning, tools/actions, memory, planning/orchestration, control loop/policy.
4. ⭐ **Draw the agent loop.** — perceive → reason → act → observe → check → loop/stop.
5. **What are mandatory stop conditions?** — max steps, token/cost budget, timeout, repeat-action detection, explicit terminate tool.
6. **Levels of agency?** — pure LLM → tool router → fixed workflow → single ReAct agent → multi-agent → autonomous self-directed.
7. ⭐ 🔥 **When would you NOT use an agent?** — when a prompt/workflow suffices; tight latency/cost; high-cost irreversible errors without guardrails; can't evaluate/observe it.
8. **What changed to make agents viable now?** — reliable tool calling, long context, better reasoning (incl. reasoning models), protocols (MCP/A2A), cheaper inference.
9. **What is scaffolding?** — the harness around the LLM: loop, parsing, tools, memory, guardrails.
10. **Define grounding.** — connecting the model to real data/state so outputs are factual.
11. 🔥 **Is more autonomy always better?** — No; it trades predictability, cost, and evaluability for capability. Use the least agency that solves the task.
12. **What's context engineering?** — curating exactly what enters the window (instructions, memory, tools, retrieved data, state) within a token budget.

---

## B. Reasoning & planning

13. ⭐ **Explain Chain of Thought.** — prompt intermediate reasoning steps; helps multi-step problems; can hallucinate (no feedback).
14. **Self-consistency?** — sample N chains, majority-vote the answer; cheap reliability boost.
15. ⭐ 🔥 **Explain ReAct and why it beats CoT for agents.** — interleaves Thought→Action→Observation; reasoning grounded in real observations → less hallucination, can recover from errors, interpretable.
16. **Reflexion / reflection?** — agent critiques its own output/failure, writes a verbal reflection to episodic memory, retries better — "verbal RL" within a session.
17. ⭐ **ReAct vs. Plan-and-Execute?** — ReAct re-reasons every step (adaptive, costlier); Plan-and-Execute plans once then executes (efficient, less adaptive, add replanning).
18. **Tree of Thoughts?** — explore a tree of reasoning branches with evaluation + backtracking (BFS/DFS/beam); powerful but expensive; usually overkill in prod.
19. **ReWOO?** — decouple reasoning from observation (plan all tool calls, then execute) to save tokens vs. ReAct.
20. 🔥 **How does prompting change with reasoning models (o-series/R1)?** — give the goal/constraints, not step-by-step CoT; minimal scaffolding; let the model plan internally.
21. **Evaluator-optimizer pattern?** — generator + critic loop; use when there's a clear eval signal and iteration helps.
22. **When is reflection NOT worth it?** — no reliable success signal (you'd reinforce noise); tight latency/cost; cap rounds.
23. 🔥 **Decompose a reasoning-strategy decision.** — single task→CoT; needs actions→ReAct; long decomposable→plan-execute; clear pass/fail + iteration→add reflection; exploration→ToT.

---

## C. Memory & context

24. ⭐ **Short-term vs. long-term memory?** — short = context window (recent turns, scratchpad); long = external stores (vector/SQL/KV) across sessions.
25. ⭐ **Episodic vs. semantic vs. procedural memory?** — episodic = past experiences; semantic = facts/profile; procedural = skills/how-to (prompts/code).
26. **Manage an overflowing context window?** — summarize/compact old turns, sliding window, retrieve-don't-stuff, offload to external memory, split across sub-agents, token-budget eviction.
27. ⭐ **Walk through a RAG pipeline.** — ingest(chunk→embed→index) + query(embed→retrieve top-k→rerank→augment→generate with citations).
28. **RAG quality knobs?** — chunking, embeddings, hybrid (dense+sparse) retrieval, reranking, query rewriting/HyDE, top-k, metadata filters.
29. ⭐ 🔥 **Agentic RAG vs. naive RAG?** — agent decides whether/what/where to retrieve and can iterate (retrieve→reason→retrieve); retrieval is a tool, not a fixed pre-step.
30. **"Lost in the middle"?** — models attend less to mid-context info; put critical content at start/end.
31. **Context failure modes?** — poisoning, distraction, confusion, clash.
32. 🔥 **How do you evaluate RAG?** — separately eval retrieval (context precision/recall) and generation (faithfulness, answer relevance), e.g. RAGAS.
33. **When use a knowledge graph vs. vector DB?** — graph for entities/relationships/multi-hop reasoning (GraphRAG); vectors for semantic similarity.

---

## D. Tools & function calling

34. ⭐ **How does function calling work? Who executes?** — model emits structured tool_call (name+JSON args); **your code executes** and returns result; model continues. Model never runs code itself.
35. ⭐ **Principles of good tool design.** — clear name/description, narrow scope, strong typed schema (enums), useful returns, idempotency/safety, few non-overlapping tools, errors as actionable feedback, token-aware outputs.
36. **Handle tool errors for recovery?** — validate args, return actionable errors, retries/backoff, idempotency keys, timeouts, fallbacks/HITL, circuit breakers.
37. ⭐ 🔥 **What is MCP and what problem does it solve?** — open standard to connect tools/data/prompts to models (USB-C for AI); turns N×M integrations into N+M; primitives = tools, resources, prompts.
38. **MCP architecture?** — host (app) ↔ client (1:1) ↔ server (exposes capabilities); stdio/HTTP transports.
39. **Computer-use agents — when and risks?** — when no API exists; screenshot→action loop; slow/brittle/expensive + big security risk → sandbox + HITL.
40. **Code-as-action (CodeAct)?** — agent emits code that calls tools (composes multiple per step) vs. single JSON tool calls; run in sandbox.
41. 🔥 **Agent keeps picking the wrong tool — fix?** — tighten descriptions (when to use/avoid), remove overlaps, add enums/required params, few-shot examples, inspect traces; consolidate confusable tools.
42. **Why not dump big tool outputs into context?** — token cost, distraction, "lost in middle"; summarize/paginate/return handles.
43. **Parallel tool calls — why useful?** — independent reads/actions run concurrently → lower latency.

---

## E. Multi-agent systems

44. ⭐ 🔥 **When multi-agent vs. single agent?** — single when tightly coupled (avoid context fragmentation, Cognition view); multi when subtasks independent/parallel or need isolated context/specialization (Anthropic research win). Mind the token-cost multiplier.
45. ⭐ **Compare orchestrator-worker, hierarchical, network topologies.** — central decompose+delegate; nested supervisors; many-to-many peer. (See Ch 05 table.)
46. **Handoff/swarm pattern — where ideal?** — control passes between agents; one active at a time; great for routing/customer support (triage→specialist).
47. **Group chat / debate?** — agents share a conversation, a manager/round-robin picks speakers; debate improves reasoning/factuality at high token cost.
48. **Blackboard pattern?** — agents read/write shared workspace; controller selects who acts; opportunistic loosely-coupled collaboration.
49. ⭐ **How do agents communicate?** — shared history, direct handoffs, shared state/blackboard, structured artifacts. Biggest mistake: passing full transcripts everywhere (token blowup) — pass distilled artifacts.
50. ⭐ 🔥 **MCP vs. A2A?** — MCP = agent↔tools/data; A2A = agent↔agent task delegation across vendors (Agent Cards, tasks over HTTP).
51. **Failure modes of MAS?** — cascading errors, deadlock, context fragmentation, cost explosion, duplicate/conflicting work, lost-in-translation, hard eval.
52. **Roles you'd define?** — orchestrator/planner, researcher, worker/specialist, critic/reviewer, synthesizer/writer, router/triage, guardian.
53. 🔥 **Why can multi-agent use 15× more tokens?** — parallel agents + repeated context + inter-agent chatter; justify with value, set budgets.
54. **How to localize a failure in MAS?** — per-agent tracing/attribution; structured artifacts; checkpoints.

---

## F. Frameworks & protocols

55. ⭐ **LangGraph vs. AutoGen?** — LangGraph = explicit stateful graph (control, cycles, HITL, durable, prod); AutoGen = conversational agents/group chat (collaborative, code-exec, exploratory).
56. **CrewAI model?** — role/goal/backstory agents + tasks + process (sequential/hierarchical); fast, readable, opinionated.
57. **OpenAI Agents SDK (ex-Swarm)?** — lightweight agents + handoffs + guardrails + sessions + tracing; simple multi-agent routing.
58. **Semantic Kernel?** — enterprise SDK (C#/Java/Python), plugins/skills + planners + memory; governance/integration.
59. **LlamaIndex?** — data/RAG-first agents + workflows; best when knowledge/retrieval is the core.
60. ⭐ 🔥 **When would you use NO framework?** — simple tool-calling agent = ~50-line while-loop over the function-calling API; full transparency, no lock-in. Add a framework for state/HITL/orchestration needs.
61. **How do MCP + A2A let you mix frameworks?** — standard tool & agent interfaces; a CrewAI crew can call MCP tools and delegate to a LangGraph agent via A2A.

---

## G. System design (see Ch 07 for full walk-throughs)

62. ⭐ 🔥 **Design a customer-support agent.** — router+handoff specialists, RAG KB, per-user memory, confirmation+authz on destructive tools, HITL escalation, eval on resolution/escalation, tracing, MVP=FAQ RAG + human handoff.
63. ⭐ 🔥 **Design a deep-research assistant.** — orchestrator-worker, parallel isolated researchers, verify/citation agent, writer synthesis, budgets, factuality eval, treat web content as untrusted.
64. 🔥 **Design a coding agent.** — single-threaded ReAct + reflection on test failures, sandboxed tools, tests as success signal, reviewer on diff, HITL on PR, SWE-bench-style eval.
65. **Design a transactional/voice agent.** — strict schemas, structured order state, mandatory confirmation before charge (irreversible), idempotent payments, low-latency streaming.
66. **What's your design framework?** — A-G-E-N-T-S: Align, Ground rules/risk, Establish approach, Nodes/components, Tools/memory/context, Safeguards/eval/scale.
67. **First questions you ask?** — goal/scope, users/volume, success metrics, latency/cost budget, error cost/reversibility, data/privacy, autonomy/HITL, integrations.
68. 🔥 **Sync request blocks for a 3-min agent task — fix?** — async: queue + worker + status/streaming; checkpoint for resume.
69. **Model strategy for cost?** — route cheap model for easy/routing steps, frontier for hard reasoning; cache; consider distillation.
70. **Where put determinism?** — non-LLM logic (validation, routing rules, math) in code, not the model.

---

## H. Production, eval & security

71. ⭐ 🔥 **How do you evaluate an agent?** — outcome (task success) + trajectory (tool/step correctness); LLM-as-judge + programmatic checks + human calibration; offline regression set + online A/B.
72. **Outcome vs. trajectory eval?** — did it succeed vs. how it got there (right tools/order, no waste/loops).
73. ⭐ **LLM-as-judge pros/cons?** — scalable/cheap; biases (position, verbosity, self-preference); validate vs. humans, use rubrics + pairwise.
74. **Name agent benchmarks.** — τ-bench/τ²-bench, SWE-bench, WebArena, GAIA, AgentBench, BFCL, RAGAS.
75. ⭐ **What do you trace/observe?** — every LLM call (prompt/completion/tokens/cost/latency), every tool call (args/result/error), full path, per-agent attribution; OpenTelemetry + LangSmith/Langfuse/Phoenix.
76. ⭐ 🔥 **Prompt injection — direct vs. indirect + defenses?** — malicious instructions in user input vs. retrieved/tool content; treat external content as untrusted, separate instructions/data, least privilege, validate/HITL before destructive/exfil actions.
77. 🔥 **Excessive agency — meaning + mitigation?** — too many permissions/autonomy; least-privilege tools, scoped creds, allow-lists, confirmation/HITL.
78. 🔥 **The "lethal trifecta"?** — private data + untrusted content + outbound communication = exfiltration risk; break one leg.
79. **Insecure output handling?** — don't trust LLM output executed downstream (SQL/shell/code); validate/parameterize, never eval raw.
80. ⭐ **Stop an agent looping forever?** — max steps/budget/timeout, repeat-action detection, surface tool errors, fallback/terminate tool, reflection then graceful degrade.
81. **Cut cost without hurting quality?** — model routing, caching (prompt/semantic/tool), limit steps/tokens, compact context, parallelize, track $/task.
82. **Guardrail types?** — input (injection/PII/policy), output (schema/safety/grounding/action-authz), behavioral (budgets/limits/HITL).
83. ⭐ **Where do humans go in the loop?** — approve/reject before irreversible actions, edit plan/output, escalate on low confidence, capture feedback.
84. **CI/CD for a prompt change?** — version prompts/tools/models, run offline eval set in CI, block regressions, canary/A-B, fast rollback, mine failures back into eval set.
85. 🔥 **Why are agents non-reproducible and how to cope?** — stochastic + multi-step; fix tem/seed where possible, full tracing, deterministic code paths, eval distributions not single runs.
86. **RAG hallucinates despite retrieval — debug?** — check retrieval quality (precision/recall), reranking, chunking; enforce grounded/cited generation; add faithfulness guardrail; lower temperature.

---

## I. Scenario / troubleshooting (think-aloud)

87. 🔥 *Your multi-agent system gives inconsistent final answers.* — likely context fragmentation/conflicting sub-decisions; centralize plan in orchestrator, pass structured artifacts not full chat, single source of truth in shared state, add synthesizer + critic, per-agent tracing.
88. 🔥 *Costs 10×'d after launch.* — runaway loops / chatty agents / huge contexts; add budgets+step caps, trim tool outputs, cache, model routing, alert on $/task, inspect traces for waste.
89. *Agent works in demo, fails on real users.* — distribution shift; build eval set from real failures, add guardrails for edge inputs, HITL fallback, expand tool error handling.
90. *Latency p95 is terrible.* — tail from long trajectories/slow tools; cap steps, parallelize, stream, smaller routing model, timeouts+fallbacks, cache.
91. *Agent took a destructive action it shouldn't have.* — excessive agency; remove/scope tool, add authz + confirmation/HITL, allow-lists, validate actions, audit logs.
92. *Retrieved web page hijacked the agent.* — indirect prompt injection; isolate untrusted content, don't let it alter system prompt, least privilege, break lethal trifecta, sanitize.
93. *How to roll out a risky agent safely?* — shadow mode → HITL-gated → canary % → full; metrics + rollback at each gate.
94. *Eval set is small and stale.* — grow from production failures (data flywheel), add adversarial/edge cases, mix human + LLM-judge, track per-category.

---

## J. ML/LLM foundations they may probe

95. **Temperature/top-p effect on agents?** — higher = more exploration/variance (risk of bad tool args); often lower temp for tool-use/determinism.
96. **Structured outputs / JSON mode?** — constrained decoding guarantees schema-valid output → reliable tool args/parsing.
97. **Context window limits & cost scaling?** — attention cost grows with length; long context is pricey + "lost in middle" → curate, don't stuff.
98. **Fine-tuning vs. RAG vs. prompting for domain knowledge?** — prompting (fast), RAG (fresh/private facts, citations), fine-tuning (behavior/format/latency, not fresh facts); often combine.
99. **What's distillation and why for agents?** — train a smaller/cheaper model on a stronger model's traces to cut cost/latency for routine steps.
100. **Embeddings — what/why?** — vector representations for semantic similarity; backbone of retrieval/episodic memory.
101. 🔥 **Why do errors compound in agents?** — each step conditions on prior (possibly wrong) state; p(success)≈per-step accuracy^steps → reliability needs short paths, checks, recovery.

---

## K. Behavioral / experience 💬

102. 💬 **Tell me about an agent/LLM system you built.** — STAR: problem, why agentic, architecture choices, eval, what broke, impact/metrics.
103. 💬 **A time your agent failed in prod — what did you learn?** — show observability-driven debugging + a process fix (eval set, guardrail).
104. 💬 **How do you decide build vs. buy a framework?** — needs-driven; start simple; adopt for specific capabilities; avoid lock-in.
105. 💬 **How do you keep up with this fast-moving field?** — papers (ReAct, Reflexion, ToT), Anthropic/OpenAI/Google eng blogs, benchmarks, hands-on builds.
106. 💬 **Disagreement on going multi-agent — how resolve?** — frame trade-offs (coupling, context, cost), prototype both, let metrics decide.
107. 💬 **How do you balance autonomy vs. safety for a business?** — tie autonomy level to error cost/reversibility; HITL on high-risk; measure & expand gradually.
108. 💬 **Most interesting agentic problem you've thought about?** — be ready with a concrete, opinionated answer.

---

## L. Rapid-fire one-liners (drill these)

- **ReAct** = reason+act loop with observations.
- **Reflexion** = self-critique stored in episodic memory.
- **MCP** = USB-C for tools/data to models.
- **A2A** = agent-to-agent task delegation standard.
- **Orchestrator-worker** = lead delegates to specialists, synthesizes.
- **Excessive agency** = too much permission/autonomy.
- **Lethal trifecta** = private data + untrusted content + outbound comms.
- **Context engineering** = right info+tools in the window at the right time.
- **LLM-as-judge** = model scores outputs vs. rubric (validate it).
- **HITL** = human approves/edits before risky actions.
- **Trajectory eval** = grade the path, not just the answer.
- **Lowest agency that works** = the senior default.

---

### How to practice
- Cover the answers; say each aloud in ≤60s.
- For ⭐, have a crisp 2-sentence answer + one concrete example.
- For 🔥, be ready to go two levels deeper and name a paper/system.
- For 💬, prepare 3 STAR stories (a build, a failure, a trade-off decision).
