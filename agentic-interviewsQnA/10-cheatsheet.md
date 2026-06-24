# 10 — Cheat Sheet (one-page recall)

> Print this. If you can reconstruct it from memory, you're ready.

---

## The one-liner
**Agent = LLM + loop + tools + memory + goal + stop condition.** It *dynamically decides its own control flow*; a workflow has *predefined* paths.

## Agent loop
`perceive → reason → act → observe → check (done/error/budget) → loop or stop`

## Five components
Model (brain) · Tools (hands) · Memory (state) · Planning/orchestration · Control loop/policy

## Levels of agency
pure LLM → tool router → fixed workflow → single ReAct agent → multi-agent → autonomous. **Use the lowest that works.**

---

## Workflow patterns (use before "agents")
Prompt chaining · Routing · Parallelization (sectioning/voting) · Orchestrator-workers · Evaluator-optimizer

## Reasoning patterns
| Pattern | Idea | Use when |
|---|---|---|
| CoT | step-by-step thinking | single reasoning task |
| Self-consistency | sample N, vote | cheap accuracy boost |
| **ReAct** | thought→action→observation | needs tools, path unknown |
| Reflexion | self-critique → episodic memory → retry | clear pass/fail + iteration |
| Plan-and-Execute | plan once, execute (replan) | long, decomposable |
| Tree of Thoughts | search branches + backtrack | exploration, cost OK |
| Reasoning models | give goal, minimal scaffolding | o-series / R1 |

**ReAct > CoT** for agents because reasoning is grounded in real observations.

---

## Memory
- **Short-term** = context window (window/summary/scratchpad). **Long-term** = vector/SQL/KV/graph.
- **Episodic** (experiences) · **Semantic** (facts/profile) · **Procedural** (skills/prompts).
- **RAG:** ingest(chunk→embed→index) + query(embed→retrieve→**rerank**→augment→generate+cite). Hybrid (dense+sparse). **Agentic RAG** = retrieval as a tool the agent decides to use/iterate.
- **Context engineering** > prompt engineering. Watch "lost in the middle" → put critical info at start/end.

---

## Tools & function calling
- Model emits structured tool_call → **your code executes** → return result → model continues. Model never runs code itself.
- **Good tools:** clear name/desc, narrow scope, typed schema (enums), useful returns, idempotent/safe, few non-overlapping, errors as feedback, token-aware.
- **MCP** = standard for tools/data/prompts → model (USB-C; N+M not N×M). Primitives: **tools, resources, prompts**.
- Computer-use = screenshot→action loop; powerful, brittle, risky → sandbox + HITL.

---

## Multi-agent topologies
| Topology | Control | Parallel | Best for |
|---|---|---|---|
| Sequential | fixed | no | staged pipelines |
| **Orchestrator-worker** | central | yes | dynamic decomposition, research |
| Hierarchical | nested | yes | large complex tasks |
| Network/P2P | distributed | yes | flexible/emergent |
| Group chat/debate | manager/round-robin | limited | brainstorm, reasoning |
| Blackboard | shared state+controller | yes | opportunistic |
| Handoff/swarm | passed | 1 active | routing, support |

- **Single vs. multi:** single for tightly-coupled (avoid context fragmentation); multi for independent/parallel + isolated context + specialization. **Multi can cost 15× tokens.**
- Pass **structured artifacts**, not full transcripts. Orchestrator owns the plan.
- **MCP = agent↔tools; A2A = agent↔agent** (Agent Cards, tasks over HTTP).

---

## Frameworks (pick by need)
| Need | Pick |
|---|---|
| Control, cycles, durable state, HITL, prod | **LangGraph** |
| Conversational multi-agent + code-exec | **AutoGen** |
| Fast role/task team | **CrewAI** |
| Lightweight handoff routing (OpenAI) | **OpenAI Agents SDK** |
| Enterprise .NET/Java, governance | **Semantic Kernel** |
| RAG/knowledge core | **LlamaIndex** |
| Simple tool agent | **no framework** (50-line loop) |

---

## System design framework — A-G-E-N-T-S
**A**lign on requirements · **G**round rules/risk · **E**stablish approach (simplest!) · **N**odes/components · **T**ools/memory/context · **S**afeguards/eval/scale → wrap with trade-offs + MVP.

Ask first: goal/scope · users/volume · success metrics · latency/cost budget · error cost/reversibility · data/privacy · autonomy/HITL · integrations.

Reference components: gateway/auth · orchestrator/router · model router · tool layer (MCP) · memory (vector+SQL+cache) · guardrails · tracing · eval harness · HITL · queue/worker.

---

## Production
- **Eval:** outcome (task success) + **trajectory** (tool/step correctness). LLM-as-judge (validate vs human) + programmatic checks + offline regression set + online A/B. Benchmarks: τ-bench, SWE-bench, WebArena, GAIA, BFCL, RAGAS.
- **Observe:** trace every LLM/tool call (tokens, cost, latency, errors), per-agent attribution. OpenTelemetry + LangSmith/Langfuse/Phoenix.
- **Guardrails:** input (injection/PII/policy) · output (schema/safety/grounding/authz) · behavioral (budgets/limits/HITL).
- **Cost/latency:** model routing · caching · cap steps/tokens · compact context · parallelize · stream · track $/task.
- **Reliability:** retries/backoff · idempotency · checkpoints · graceful degradation · determinism in code.

## Security (OWASP LLM)
- **Prompt injection** (direct & **indirect** via tool/web content) — treat external content as untrusted; separate instructions/data; validate/HITL before destructive actions.
- **Excessive agency** — least privilege, scoped creds, allow-lists, confirmation.
- **Lethal trifecta** = private data + untrusted content + outbound comms → break one leg.
- Insecure output handling, supply chain (MCP servers), data poisoning, unbounded consumption.

---

## Senior signals (say these)
1. "Use the **lowest agency** that solves it."
2. "I'd **prototype framework-free**, add a framework for state/HITL/orchestration."
3. "Pass **structured artifacts**, not full chat, between agents."
4. "**Trajectory + outcome** eval, built from real failures."
5. "Treat all external content as **untrusted**; **HITL** before irreversible actions."
6. "Track **$/task** and **p95 latency** as first-class metrics."

## Top traps to avoid
Defaulting to multi-agent · no stop conditions · stuffing context · no eval/observability · destructive tools without guardrails · ignoring prompt injection · hand-waving "the LLM handles it."

---

## Must-know names/papers
ReAct · Reflexion · Tree of Thoughts · Toolformer · Voyager · "Building Effective Agents" (Anthropic) · MCP · A2A · RAG · RAGAS · τ-bench · SWE-bench · OWASP Top 10 for LLM Apps · "lethal trifecta" (Willison).
