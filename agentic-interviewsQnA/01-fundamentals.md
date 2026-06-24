# 01 — Fundamentals of Agentic AI

> Goal: define agents precisely, explain the agent loop, classify autonomy, and know when *not* to build an agent.

---

## 1.1 What is an AI agent?

💡 **An AI agent is a system that uses an LLM to dynamically decide its own control flow to accomplish a goal**, taking actions in an environment through tools, observing the results, and iterating until the goal is met or a stop condition triggers.

Contrast three things people conflate:

| Term | Control flow | Example |
|------|--------------|---------|
| **LLM call** | None — single input → output | "Summarize this text." |
| **Workflow** | Predefined by code; LLM fills steps | RAG pipeline: retrieve → stuff → generate |
| **Agent** | Decided at runtime by the LLM | "Research X and write a report" — model picks tools, order, and when to stop |

Anthropic's widely-cited distinction: **workflows** orchestrate LLMs through *predefined code paths*; **agents** are systems where *LLMs dynamically direct their own processes and tool usage*.

### The minimal definition (memorize)
> **Agent = LLM (the brain) + a loop + tools (hands) + memory (state) + a goal/policy + a stop condition.**

---

## 1.2 The anatomy of an agent

```
        ┌─────────────────────────────────────────────┐
        │                   AGENT                      │
        │                                              │
  Goal ─┼──▶ ┌──────────┐   plan/think   ┌──────────┐  │
        │    │ Reasoning │ ─────────────▶ │ Planning │  │
        │    │  (LLM)    │ ◀───────────── │          │  │
        │    └────┬─────┘   reflect       └────┬─────┘  │
        │         │ choose action              │        │
        │         ▼                             ▼        │
        │    ┌──────────┐               ┌──────────────┐ │
        │    │  Tools   │               │    Memory    │ │
        │    │(actions) │               │ short/long   │ │
        │    └────┬─────┘               └──────────────┘ │
        └─────────┼────────────────────────────────────┘
                  ▼
            ┌───────────┐  observation/feedback
            │Environment│ ───────────────────────┐
            └───────────┘                         │
                  ▲                               │
                  └───────────────────────────────┘
```

**Five components interviewers expect you to name:**

1. **Model / reasoning core** — the LLM that interprets state and decides the next action.
2. **Tools / actions** — functions, APIs, code execution, retrieval, other agents. The agent's "hands."
3. **Memory** — context window (short-term) + external stores (long-term) to persist state across steps/sessions.
4. **Planning / orchestration** — how goals are decomposed and steps sequenced (can be implicit in the LLM or explicit in code).
5. **Policy / control loop** — the loop logic, stop conditions, guardrails, and human-in-the-loop checkpoints.

---

## 1.3 The agent loop

The canonical loop (and the heart of ReAct):

```
1. PERCEIVE   — read goal + current state + latest observation
2. REASON     — think about what to do next (often a hidden "thought")
3. ACT        — call a tool / produce an action
4. OBSERVE    — get the tool result / environment feedback
5. CHECK      — done? error? budget exceeded? → loop or stop
```

Pseudo-code you should be able to whiteboard:

```python
def run_agent(goal, tools, llm, max_steps=10):
    memory = [system_prompt(tools), user(goal)]
    for step in range(max_steps):
        thought_action = llm(memory)            # REASON
        if thought_action.is_final:             # CHECK
            return thought_action.answer
        result = call_tool(thought_action.tool, # ACT + OBSERVE
                           thought_action.args, tools)
        memory.append(thought_action)
        memory.append(observation(result))
    return fallback("max steps reached")        # stop condition
```

⚠️ **Trap:** forgetting the **stop conditions**. Every agent needs guards: max steps, max tokens/cost, wall-clock timeout, repeated-action detection, and a confidence/terminate tool. Unbounded loops are the #1 cause of runaway cost.

---

## 1.4 Levels of agency (a useful framing)

Agency is a spectrum, not binary. A clean ladder to cite:

| Level | Name | Who decides control flow | Example |
|-------|------|--------------------------|---------|
| 0 | **Pure LLM** | Human, one shot | Single completion |
| 1 | **LLM + tools (router)** | Code routes; LLM picks within a step | Function-calling chatbot |
| 2 | **Chained workflow** | Code (fixed DAG) | Prompt chaining, RAG |
| 3 | **Single agent (ReAct)** | LLM loops over tools | Coding assistant, research agent |
| 4 | **Multi-agent** | Orchestrator + specialists | Orchestrator-worker research system |
| 5 | **Autonomous / self-directed** | LLM sets sub-goals, spawns agents | Long-running "do my taxes" agent |

💡 Higher agency = more capability but **less predictability, higher cost, harder eval**. Interviewers love when you argue for the *lowest* level that solves the problem.

---

## 1.5 Common agentic workflow patterns (building blocks)

Before "agents," know the **workflow** patterns (from Anthropic's "Building Effective Agents"). Many "agentic" systems are actually these:

1. **Prompt chaining** — decompose a task into a fixed sequence of LLM calls; each step's output feeds the next. Add programmatic gates between steps. *Use when the task cleanly splits into fixed subtasks.*
2. **Routing** — classify input, then dispatch to a specialized prompt/tool/model. *Use for distinct input categories (e.g., support triage).*
3. **Parallelization** — run subtasks concurrently (*sectioning*) or run the same task multiple times for votes (*voting*). *Use for speed or confidence.*
4. **Orchestrator-workers** — a lead LLM dynamically breaks down a task and delegates to worker LLMs. *Use when subtasks aren't known in advance.*
5. **Evaluator-optimizer** — one LLM generates, another critiques and sends feedback; loop. *Use when you have clear eval criteria and iteration helps (e.g., translation, code).*

Then the **agent** itself: open-ended loop with tools, used when the number of steps is unpredictable and you can't hardcode the path.

---

## 1.6 When to use agents (and when NOT to)

🎯 *"When would you NOT build an agent?"* — a favorite question.

**Use an agent when:**
- The number/order of steps can't be known ahead of time.
- The task needs dynamic tool selection and adaptation to feedback.
- The value of solving it justifies higher cost/latency and lower predictability.

**Do NOT use an agent when:**
- A fixed workflow or single prompt achieves the goal (cheaper, faster, testable).
- Errors are costly and hard to reverse without strong guardrails/HITL.
- Latency or cost budgets are tight.
- You can't evaluate or observe what it's doing.

💡 The senior answer: **"Start with the simplest pattern. Add agency only when simpler approaches fail. Complexity is a cost you pay in latency, money, and debuggability."**

---

## 1.7 Why now? What changed

- **Tool/function calling** became reliable in frontier models (structured outputs, parallel calls).
- **Long context** (100K–1M+ tokens) makes multi-step state feasible.
- **Better reasoning** (CoT, RL-trained reasoning models) makes planning viable.
- **Standard protocols** (MCP for tools, A2A for agent-to-agent) reduce integration cost.
- **Cheaper inference** makes multi-call loops economically viable.

---

## 1.8 Key terms glossary

- **Trajectory / rollout** — the full sequence of (thought, action, observation) an agent produced.
- **Scaffolding** — the code/harness around the LLM (loop, parsing, tools).
- **Grounding** — connecting the model to real data/state so outputs are factual.
- **Excessive agency** — security risk where an agent has more permission/autonomy than needed.
- **Human-in-the-loop (HITL)** — a checkpoint where a human approves/edits before the agent proceeds.
- **Context engineering** — curating exactly what goes into the context window (instructions, memory, tools, retrieved data).

---

## Interview questions for this chapter

1. Define an AI agent and contrast it with a workflow and a single LLM call. *(See 1.1)*
2. Draw the agent loop. What are the mandatory stop conditions? *(1.3)*
3. Walk me up the "levels of agency" ladder with an example at each level. *(1.4)*
4. Name the five components of an agent and what each does. *(1.2)*
5. Give three situations where you'd deliberately avoid an agent. *(1.6)*
6. What is "excessive agency" and why does it matter? *(1.8, see Ch 08)*
7. A PM wants to "make our chatbot agentic." How do you decide if that's warranted? *(1.6)*

**Model answer to #7:** Clarify the goal and failure cost. Map current flows — if they're fixed, a router or chain may suffice. Prototype the simplest version, define success metrics and guardrails, and only escalate to a looped agent if dynamic tool selection is genuinely required. Quantify the added latency/cost and how we'll evaluate it before committing.
