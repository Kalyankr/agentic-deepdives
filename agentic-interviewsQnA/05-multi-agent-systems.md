# 05 — Multi-Agent Systems & Flows

> Goal: choose the right topology, orchestration, and communication pattern for multiple cooperating agents — and justify the added complexity.

---

## 5.1 Why (and why not) multi-agent

A **multi-agent system (MAS)** splits work across multiple specialized agents that coordinate to solve a task no single agent handles well.

**Reasons to go multi-agent:**
- **Separation of concerns** — each agent has a focused role, tools, and prompt → better quality and easier maintenance.
- **Context isolation** — each agent gets its own clean window, avoiding the "one giant prompt" overload. (A major real motivation.)
- **Parallelism** — independent subtasks run concurrently → lower latency for wide tasks.
- **Specialized tools/models** — route to cheap/fast or strong/expensive models per role.
- **Scalability of capability** — add agents/roles without rewriting one monolith.

**Costs / reasons NOT to:**
- **Coordination overhead**, more failure modes, harder to debug.
- **Token/cost blow-up** — multi-agent systems can use far more tokens than a single agent.
- **Error propagation** — one agent's mistake cascades.
- **Consistency** — agents can disagree or duplicate work.

> 💡 Senior framing: *"Multi-agent is a tool for managing complexity and context, not a default. Use it when a single agent's context or responsibilities become unmanageable, or when subtasks are genuinely parallel."* Anthropic found multi-agent shines for **broad, parallelizable research**; Cognition (Devin) argued for **single-threaded** agents when tasks are tightly coupled. Know both sides.

---

## 5.2 The core topologies (know these cold)

### 1) Single agent (baseline)
One agent, many tools. Start here.

### 2) Sequential / Pipeline
Agents in a fixed chain; output of one feeds the next.
```
Researcher → Writer → Editor → Publisher
```
Use for staged processes with clear handoffs.

### 3) Orchestrator–Worker (Manager / Supervisor)
A lead agent decomposes the task and delegates to workers, then synthesizes results.
```
              ┌─────────────┐
              │ Orchestrator│
              └──┬───┬───┬──┘
        delegate │   │   │  delegate
            ┌─────▼┐ ┌▼───┐ ┌▼─────┐
            │Worker│ │Wkr │ │Worker│
            └──────┘ └────┘ └──────┘
              (results flow back up, orchestrator synthesizes)
```
💡 The most common production pattern. Workers can run in parallel. Orchestrator owns planning + synthesis.

### 4) Hierarchical (supervisor of supervisors)
Orchestrator-workers nested into trees for complex orgs of agents. Mid-level managers coordinate sub-teams.

### 5) Network / Peer-to-peer
Agents talk many-to-many; any agent can hand off to any other. Flexible but harder to control; risk of chaos. Often constrained by a routing policy.

### 6) Group chat / Debate / Collaboration
Agents share a conversation; a manager (or round-robin/speaker-selection policy) decides who speaks next. **Debate** (agents argue opposing sides) and **society of minds** improve reasoning/factuality at high token cost.

### 7) Blackboard
Agents read/write a shared workspace ("blackboard"); a controller decides which agent acts on the current state. Good for opportunistic, loosely-coupled collaboration.

### 8) Handoff / Swarm
Agents transfer control to one another by "handing off" the conversation (OpenAI Swarm/Agents SDK). The active agent fully owns the turn until it hands off. Great for customer-support-style routing (triage → billing → tech).

Topology cheat table:

| Topology | Control | Parallel? | Best for |
|----------|---------|-----------|----------|
| Sequential | Fixed | No | Staged pipelines |
| Orchestrator-worker | Central | Yes | Dynamic decomposition, research |
| Hierarchical | Central, nested | Yes | Large complex tasks |
| Network/P2P | Distributed | Yes | Flexible, emergent collaboration |
| Group chat/debate | Manager/round-robin | Limited | Brainstorm, reasoning, review |
| Blackboard | Shared state + controller | Yes | Opportunistic problem solving |
| Handoff/swarm | Passed between agents | No (1 active) | Routing, customer support |

---

## 5.3 Roles & specialization patterns

Common agent roles to name:
- **Planner / Orchestrator** — decomposes and delegates.
- **Researcher / Retriever** — gathers info.
- **Worker / Specialist** — does a focused subtask (coder, analyst, SQL).
- **Critic / Reviewer / Evaluator** — checks quality (evaluator-optimizer).
- **Synthesizer / Writer** — combines results into the final output.
- **Router / Triage** — classifies and dispatches.
- **Guardian / Safety** — enforces policy/guardrails.

Pattern: **Generator + Critic** (a.k.a. actor-critic / evaluator-optimizer) — one produces, one reviews, loop until the critic approves or budget hits.

---

## 5.4 Communication & coordination

**How agents pass information:**
- **Shared message history** (group chat) — everyone sees everything; simple but token-heavy and noisy.
- **Direct messages / handoffs** — targeted transfer of control + context.
- **Shared state / blackboard** — agents read/write a common store (e.g., LangGraph state object).
- **Structured artifacts** — agents exchange typed objects (a plan, a doc, a JSON result) rather than chat.

**Coordination decisions:**
- **Who speaks next?** round-robin, manager-selects, rules, or model-decided.
- **How is work split?** static roles vs. dynamic decomposition by an orchestrator.
- **How are results merged?** orchestrator synthesis, voting, or a dedicated synthesizer.
- **Termination?** manager declares done, consensus, max rounds, or a goal check.

💡 **Context-passing is the crux.** Decide what each agent *needs* vs. the full history. Passing full transcripts everywhere is the classic token-cost mistake; pass distilled artifacts instead.

---

## 5.5 Agent communication protocols (the "flows" layer)

Emerging standards you should name:

- **MCP (Model Context Protocol)** — agent ↔ *tools/data* (vertical integration). (Ch 04.)
- **A2A (Agent2Agent, Google, 2025)** — agent ↔ *agent* interoperability across vendors/frameworks. Agents publish an **Agent Card** (capabilities, endpoint, auth); others discover and delegate **tasks** to them over HTTP/JSON-RPC/SSE. Treats other agents as opaque peers (no shared memory/tools required).
- **ACP / ANP / others** — additional proposals in the agent-interop space.

> Mental model: **MCP = how an agent uses tools; A2A = how agents talk to each other.** Together they're the "USB-C + networking" of agent ecosystems.

---

## 5.6 Failure modes of multi-agent systems

⚠️ Be ready to list these:
- **Cascading errors** — early mistake poisons everything downstream.
- **Coordination failure / deadlock** — agents wait on each other or loop.
- **Context fragmentation** — agents lack info siblings hold → contradictory actions (Cognition's core critique of naive multi-agent).
- **Cost explosion** — chatty agents burn tokens (15× single-agent is real).
- **Duplicate / conflicting work** — no clear ownership.
- **Lost-in-translation** — info degrades across handoffs.
- **Evaluation difficulty** — which agent caused the failure? (need per-agent tracing).

Mitigations: clear role boundaries, structured (not chat) handoffs, an orchestrator that owns the plan, shared state with single source of truth, budgets per agent, and end-to-end tracing.

---

## 5.7 Designing a multi-agent flow (recipe)

```
1. Decompose the task → identify distinct skills/contexts needed.
2. If subtasks are tightly coupled → prefer a single agent (avoid fragmentation).
   If subtasks are independent/parallel or need isolated context → multi-agent.
3. Pick a topology (default: orchestrator-worker).
4. Define each agent: role, system prompt, tools, model, I/O schema.
5. Define communication: shared state vs. handoff; what context each gets.
6. Define termination + synthesis.
7. Add guardrails, budgets, tracing, and HITL checkpoints.
8. Evaluate end-to-end AND per agent.
```

---

## 5.8 Worked example — research assistant (orchestrator-worker)

```
User: "Compare the EV strategies of 3 automakers and recommend an investment."

Orchestrator (Lead):
  - Plans subtasks, spawns 3 Research workers (one per company) IN PARALLEL.
Research Worker (×3):  [isolated context each]
  - web_search + retrieve + summarize → returns a structured company brief.
Analyst Worker:
  - takes 3 briefs → builds comparison table, computes metrics.
Critic:
  - checks claims have citations; flags gaps → orchestrator may re-dispatch.
Writer:
  - synthesizes final memo with citations.
Orchestrator:
  - returns memo; HITL approval before any "action" (e.g., trade).
```
Why multi-agent here: research subtasks are **independent and parallel**, each needs an **isolated context**, and a **critic/writer** separation improves quality — exactly the conditions that justify MAS.

---

## Interview questions for this chapter

1. When is multi-agent justified vs. a single agent? Give the argument on both sides. *(5.1)*
2. Compare orchestrator-worker, hierarchical, and network topologies. *(5.2)*
3. What's a handoff/swarm pattern and where is it ideal? *(5.2)*
4. How do agents communicate, and what's the most common cost mistake? *(5.4)*
5. MCP vs. A2A — what does each standardize? *(5.5)*
6. List four failure modes of MAS and how you'd mitigate them. *(5.6)*
7. Design a multi-agent system for automated customer support. *(5.7, see Ch 07)*

**Model answer to #1:** Single agents win when work is tightly coupled — shared context prevents the fragmentation and conflicting-decision problems that plague naive multi-agent setups (Cognition's view). Multi-agent wins when subtasks are independent and parallelizable, each needs an isolated context to avoid overloading one window, or distinct specialization/tools/models help — Anthropic's research system saw big gains here. Decide by coupling and context pressure, not by hype, and always weigh the token-cost multiplier.
