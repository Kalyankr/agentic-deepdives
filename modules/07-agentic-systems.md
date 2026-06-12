# Module 07 · Agentic Systems

> **Goal:** Design, build, and serve LLM **agents** — systems where the model plans, uses tools, maintains memory, and acts in loops to accomplish goals. Cover single-agent and multi-agent patterns, tool/function calling, the Model Context Protocol, agent runtime/inference, and reliability. This is central to where Anthropic and OpenAI are heading.

**Duration:** ~5 weeks. **Prereqs:** [Module 03](03-llm-training-rlhf-dpo.md), [Module 06](06-rag-and-vector-databases.md).

---

## 7.1 What an agent is

- Definition: an LLM in a **loop** that observes, reasons, decides on actions (tools), executes them, and incorporates results until a goal is met
- **Agents vs. workflows** (Anthropic's framing): workflows = predefined code paths orchestrating LLM calls; agents = the LLM dynamically directs its own process. *Start simple — use the least agency that works.*
- The augmented LLM: model + retrieval + tools + memory
- When NOT to build an agent (latency, cost, reliability vs. a fixed pipeline)

> **Read first:** Anthropic's *Building Effective Agents* and OpenAI's *A Practical Guide to Building Agents*. These define the vocabulary used in interviews.

## 7.2 Core agent patterns

- **ReAct** (Reason + Act) — interleave thoughts, actions, observations
- **Tool use / function calling** — schemas, argument generation, parallel tool calls, result handling
- **Planning** — plan-and-execute, decomposition, task graphs; planner/executor split
- **Reflection / self-critique** — Reflexion, self-refine, critique loops
- **Routing** — classify then dispatch to specialized handlers
- **Prompt chaining**, **orchestrator–workers**, **evaluator–optimizer** (workflow patterns)
- Tree/graph search over actions (Tree of Thoughts, LATS) — when extra test-time compute pays off

## 7.3 Tools & the environment

- Designing good tools: clear names, typed schemas, great descriptions, error messages the model can recover from ("agent-computer interface" design)
- Function/tool calling formats; structured outputs / JSON mode / constrained decoding
- Code execution as a tool (sandboxes), computer use / browser use, retrieval as a tool
- **Model Context Protocol (MCP)** — the open standard for connecting models to tools/data; servers, resources, tools; why it matters for interoperability
- Safety of tool use: sandboxing, allow-lists, human-in-the-loop approvals, least privilege

## 7.4 Memory

- Short-term: context window, scratchpads, message history management
- **Context engineering** — what to put in context, compaction/summarization, "context rot" with long histories
- Long-term memory: vector stores (Module 06), episodic/semantic memory, memory writing/retrieval policies
- State management across turns and sessions; serialization, resumability

## 7.5 Multi-agent systems

- When multi-agent helps (parallelizable subtasks, separation of concerns) vs. hurts (coordination overhead, error compounding)
- **Orchestrator–worker** / supervisor patterns; specialist agents
- Communication protocols, shared state/blackboard, message passing
- Handoffs, delegation, agent-to-agent protocols
- Failure compounding and why error rates multiply across steps — design for it

## 7.6 Agent inference & runtime (the "inference of agentic systems")

Agents stress serving infrastructure differently from chat:
- **Multi-step, stateful** sessions — long horizons, many model calls per task
- **KV-cache & prefix caching** are huge for agents (repeated system prompts/tool defs) — ties to [Module 04](04-gpu-architecture-and-inference.md)
- Latency budgeting across a trajectory; streaming partial results; parallel tool execution
- Concurrency: many in-flight tool calls; async orchestration; timeouts, retries, idempotency
- Cost control: step caps, token budgets, model routing (cheap model for easy steps), caching
- Determinism & reproducibility for debugging; trajectory logging/replay
- Scaling agent runtimes: queues, workers, durable execution (e.g., Temporal-style), checkpoint/resume of long tasks

> **Build:** Instrument your agent with full **trajectory tracing** (every prompt, tool call, latency, token count, cost) and enable prefix caching. Show the latency/cost breakdown of a multi-step task and optimize it (caching + model routing).

## 7.7 Reliability, evaluation & safety

- Why agents are hard: compounding errors, loops, getting stuck, hallucinated tool args
- Guardrails: input/output validation, action approval, spend limits, max steps
- **Evaluation of agents** — task success rate, trajectory quality, tool-call accuracy, cost/steps to completion; benchmarks like SWE-bench, GAIA, τ-bench, WebArena (full treatment in [Module 09](09-evaluations.md))
- Security: **prompt injection** via tools/retrieved content (the top agent threat), data exfiltration, confused-deputy; sandboxing & least privilege (more in [Module 10](10-ai-infrastructure-and-production.md))

---

## Module 07 capstone — **Build a real agent**

Pick one and build it end-to-end:
- A **coding agent** that reads a repo, edits files, runs tests, and iterates to fix a bug.
- A **research agent** that plans, searches/retrieves, and produces a cited report.
- A **deep-research / multi-agent** orchestrator with a planner + worker agents.

Requirements:
1. Tool calling with typed schemas + robust error handling/recovery.
2. Memory (short-term context management + long-term store).
3. An MCP server exposing at least one tool/data source.
4. Full trajectory tracing: steps, tokens, latency, cost; prefix caching enabled.
5. An **eval suite**: ≥20 tasks with automated success checks; report success rate, avg steps, avg cost, p95 latency.
6. Guardrails: max steps, spend cap, action approval for dangerous tools, prompt-injection test cases.

## Exit criteria
- [ ] You can articulate agents vs. workflows and pick the simplest design that works.
- [ ] You can implement ReAct + tool calling + memory from scratch (not just via a framework).
- [ ] You understand how agents stress inference (prefix caching, concurrency, cost) and can optimize a trajectory.
- [ ] You can evaluate an agent's success rate and defend against prompt injection.

## Core sources
- *Building Effective Agents* — Anthropic, 2024
- *A Practical Guide to Building Agents* — OpenAI, 2025
- *ReAct* — Yao et al., 2022
- *Reflexion* — Shinn et al., 2023
- *Toolformer* — Schick et al., 2023
- *Tree of Thoughts* — Yao et al., 2023
- Model Context Protocol (MCP) spec & docs
- *τ-bench*, *SWE-bench*, *GAIA*, *WebArena* papers
- Frameworks to study (then go lower-level): LangGraph, OpenAI Agents SDK, Claude Agent SDK, AutoGen, CrewAI, DSPy
