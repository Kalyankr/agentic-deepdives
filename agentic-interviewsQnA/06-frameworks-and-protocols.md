# 06 — Frameworks & Protocols

> Goal: compare the major agent frameworks, know when to pick each, and discuss the standard protocols. Interviewers want *informed pragmatism*, not fandom.

---

## 6.1 The landscape at a glance

| Framework | By | Mental model | Sweet spot |
|-----------|-----|-------------|------------|
| **LangGraph** | LangChain | Stateful **graph** (nodes/edges + shared state) | Complex, controllable, cyclic workflows; production |
| **AutoGen / AG2** | Microsoft | **Conversational** agents in (group) chats | Multi-agent collaboration, research, code-exec |
| **CrewAI** | CrewAI | **Role-based "crew"** with tasks & process | Quick role/task teams; readable, opinionated |
| **OpenAI Agents SDK** (← Swarm) | OpenAI | Lightweight **agents + handoffs + guardrails** | Simple, production multi-agent on OpenAI |
| **Semantic Kernel** | Microsoft | **Plugins/skills + planners**, enterprise SDK | .NET/Java/Python enterprise integration |
| **LlamaIndex (Agents/Workflows)** | LlamaIndex | **Data/RAG-centric** agents + event workflows | Knowledge/RAG-heavy agents |
| **Strands / Bedrock / ADK** | AWS / Google | Cloud-native agent SDKs | Tight cloud integration |

💡 You don't need all of them. Know **LangGraph** (control), **AutoGen** (conversation), **CrewAI** (roles), **OpenAI Agents SDK** (lightweight), and **Semantic Kernel** (enterprise) well enough to choose.

---

## 6.2 LangGraph

**Model:** represent the agent system as a **graph**. Nodes = functions/agents; edges (including **conditional** edges) = control flow; a shared, typed **State** object is passed and updated. Supports **cycles** (loops), **persistence/checkpointing**, **human-in-the-loop** interrupts, streaming, and time-travel.

**Why interviewers like it:** maximal **control and observability**. You can express ReAct, plan-execute, reflection, and multi-agent topologies explicitly as graphs.

```python
# Conceptual LangGraph shape
graph = StateGraph(State)
graph.add_node("planner", planner_fn)
graph.add_node("executor", executor_fn)
graph.add_conditional_edges("executor", route_fn, {"continue": "executor", "done": END})
graph.set_entry_point("planner")
app = graph.compile(checkpointer=memory)
```

**Pros:** fine-grained control, durable state, HITL, great for production & complex flows.
**Cons:** more boilerplate; steeper learning curve; you design the graph yourself.

---

## 6.3 AutoGen (Microsoft) / AG2

**Model:** agents are **conversable**; you compose them in conversations, notably **GroupChat** with a **GroupChatManager** that selects the next speaker. First-class **code execution** agents, human-proxy agents, and async event-driven core (v0.4+).

**Sweet spot:** multi-agent **collaboration/brainstorming**, research, and code-writing-and-running loops.

```python
# Conceptual AutoGen group chat
user = UserProxyAgent("user", code_execution_config={...})
coder = AssistantAgent("coder", llm_config=...)
reviewer = AssistantAgent("reviewer", llm_config=...)
chat = GroupChat(agents=[user, coder, reviewer], max_round=12)
manager = GroupChatManager(groupchat=chat, llm_config=...)
user.initiate_chat(manager, message="Build and test a CSV parser")
```

**Pros:** natural multi-agent conversations, strong code-exec story, flexible.
**Cons:** chat-centric control can be less deterministic; token-heavy; orchestration via conversation can be harder to constrain.

---

## 6.4 CrewAI

**Model:** define **Agents** (role, goal, backstory, tools), **Tasks** (description, expected output, owner), and a **Process** (sequential or hierarchical) that runs the **Crew**.

**Sweet spot:** quickly standing up a readable **role-based team** (e.g., Researcher + Writer + Editor). Opinionated and beginner-friendly; also has "Flows" for more control.

```python
researcher = Agent(role="Researcher", goal="...", tools=[search])
writer = Agent(role="Writer", goal="...")
t1 = Task(description="Research X", agent=researcher, expected_output="bullet brief")
t2 = Task(description="Write article", agent=writer, context=[t1])
crew = Crew(agents=[researcher, writer], tasks=[t1, t2], process=Process.sequential)
crew.kickoff()
```

**Pros:** fast to build, intuitive role/task abstraction, good DX.
**Cons:** less low-level control than LangGraph; abstraction can hide failure modes.

---

## 6.5 OpenAI Agents SDK (formerly Swarm)

**Model:** minimal primitives — **Agents** (instructions + tools), **Handoffs** (transfer control to another agent), **Guardrails** (input/output validation), **Sessions** (memory), and built-in tracing. Production-oriented successor to the experimental Swarm.

**Sweet spot:** lightweight **multi-agent routing** (triage → specialist) with little ceremony on the OpenAI stack.

```python
spanish = Agent(name="Spanish", instructions="Respond in Spanish.")
triage = Agent(name="Triage", instructions="Route by language.",
               handoffs=[spanish, english])
Runner.run_sync(triage, "Hola, ¿qué tal?")
```

**Pros:** simple, explicit handoffs/guardrails, tracing built in.
**Cons:** thinner than LangGraph for complex stateful graphs; OpenAI-centric (though model-agnostic-ish).

---

## 6.6 Semantic Kernel (Microsoft)

**Model:** enterprise SDK (C#, Python, Java). **Plugins/skills** (native or prompt functions), **planners** that compose plugins to reach a goal, **memory** connectors, and an **Agent Framework** + **Process Framework** for multi-agent and business processes.

**Sweet spot:** **enterprise** apps needing strong typing, .NET/Java support, security/compliance, and integration with existing systems.

**Pros:** enterprise-grade, multi-language, model-agnostic, good governance story.
**Cons:** heavier; more concepts; less "agent-loop" ergonomic than dedicated agent libs.

---

## 6.7 LlamaIndex

**Model:** **data-first**. Strong on ingestion/indexing/RAG, plus **agents** and an event-driven **Workflows** API. Best when the agent is fundamentally a **knowledge/RAG** system.

**Pros:** best-in-class RAG tooling, many connectors.
**Cons:** agent orchestration less rich than LangGraph for complex control.

---

## 6.8 Choosing a framework (decision guide)

```
Need maximum control, cycles, durable state, HITL, prod?   → LangGraph
Conversational multi-agent / code-exec collaboration?      → AutoGen
Fast role/task team, readable, opinionated?                → CrewAI
Lightweight handoff routing on OpenAI, minimal code?       → OpenAI Agents SDK
Enterprise .NET/Java, governance, existing systems?        → Semantic Kernel
RAG/knowledge is the core of the agent?                    → LlamaIndex
Just one tool-using assistant?                             → maybe NO framework — raw API + a loop
```

⚠️ **Strong senior move:** "For a simple tool-calling agent I'd skip a framework entirely — a `while` loop around the model's function-calling API is ~50 lines, fully transparent, and avoids abstraction lock-in. I add a framework when I need its state/HITL/orchestration features." (Anthropic explicitly recommends starting framework-free.)

---

## 6.9 Protocols recap

- **MCP** — standardize **tools/data/prompts** to models (agent↔tools). USB-C for AI. (Ch 04.)
- **A2A** — standardize **agent↔agent** task delegation across vendors (Agent Cards, tasks over HTTP). (Ch 05.)
- **Function calling / structured outputs** — the model-level primitive everything builds on.

These let you **mix frameworks**: a CrewAI crew can call MCP tools and delegate to a LangGraph agent over A2A.

---

## 6.10 What to actually say in an interview

1. Lead with **concepts** (loop, tools, memory, topology), then map them onto a framework.
2. Show you know **trade-offs**, not just APIs.
3. Mention you'd **prototype framework-free** and adopt a framework for specific needs.
4. Name **observability/eval** (LangSmith, Langfuse, AgentOps, OpenTelemetry) — see Ch 08.

---

## Interview questions for this chapter

1. Compare LangGraph and AutoGen. When do you pick each? *(6.2–6.3)*
2. Why might you build an agent with no framework at all? *(6.8)*
3. How does CrewAI's abstraction differ from LangGraph's? *(6.2, 6.4)*
4. What does the OpenAI Agents SDK add over a raw chat loop? *(6.5)*
5. Which framework for a regulated enterprise on .NET, and why? *(6.6)*
6. How do MCP and A2A let you mix frameworks? *(6.9)*

**Model answer to #1:** LangGraph models the system as an explicit stateful graph — you control every node, edge, loop, and checkpoint, which is ideal for production-grade, complex, or cyclic flows needing HITL and durability. AutoGen models agents as conversational participants, often in a group chat with a manager selecting speakers — ideal for collaborative, exploratory, code-executing multi-agent tasks. Choose LangGraph when you need deterministic control and observability; choose AutoGen when natural agent-to-agent conversation and rapid multi-agent experimentation matter more than tight control.
