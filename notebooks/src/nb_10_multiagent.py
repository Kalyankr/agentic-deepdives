"""Build NB10 — Multi-agent systems."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 10 · Multi-Agent Systems & Orchestration

> Module: **07 · Agentic Systems** (multi-agent) + **08 · Prompt Orchestration**.

**Goal:** coordinate **multiple** LLM calls/agents — routing, orchestrator–workers,
evaluator–optimizer, and handoffs — and know when multi-agent helps vs. hurts. We implement the
core patterns with mock agents so they run offline.

### Learning objectives
1. Implement the canonical orchestration patterns.
2. Decide single- vs multi-agent (coordination cost, error compounding).
3. Understand handoffs, shared state, and parallelization.
"""),
    md(r"""
## 1. The pattern catalog

| Pattern | Shape | Use when |
|---------|-------|----------|
| **Prompt chaining** | step → step → step (checks between) | a task decomposes into fixed subtasks |
| **Routing** | classify → dispatch to a specialist | distinct input types need different handling |
| **Parallelization** | fan-out → aggregate (vote/merge) | independent subtasks or multiple samples |
| **Orchestrator–workers** | a planner spawns & coordinates workers | subtasks unknown up front |
| **Evaluator–optimizer** | generate → critique → revise (loop) | quality improves with feedback |

Composability beats cleverness: most production systems are **workflows of simple agents**.
"""),
    md(r"""
## 2. Routing
A cheap classifier (often a small LLM) sends each request to the right specialized handler —
better quality and lower cost than one giant prompt trying to do everything.
"""),
    code(r"""
def route(query):
    q = query.lower()
    if any(w in q for w in ["refund", "charge", "billing"]): return "billing"
    if any(w in q for w in ["error", "crash", "bug", "broken"]): return "technical"
    return "general"

specialists = {
    "billing":   lambda q: "[billing] I can help with refunds and charges.",
    "technical": lambda q: "[tech] Let's debug that error step by step.",
    "general":   lambda q: "[general] Happy to help!",
}
for q in ["I want a refund", "the app keeps crashing", "what are your hours?"]:
    print(f"{q!r:32} -> {route(q):9} -> {specialists[route(q)](q)}")
"""),
    md(r"""
## 3. Orchestrator–workers

An **orchestrator** decomposes a goal into subtasks, dispatches them to **workers** (possibly in
parallel), then **synthesizes** the results. This is how "deep research" style systems work:
a lead agent plans and spawns sub-agents that each investigate a piece.
"""),
    code(r"""
def worker(subtask):
    # a real worker is itself an agent (NB09); here a stub that "researches" a subtopic
    return f"findings on '{subtask}': (summary of evidence)"

def orchestrator(goal):
    # 1) PLAN: decompose (a real planner LLM would generate these)
    subtasks = [f"{goal} — aspect {i}" for i in ("background", "evidence", "counterpoints")]
    # 2) DISPATCH (could be parallel / async)
    results = [worker(st) for st in subtasks]
    # 3) SYNTHESIZE
    report = "REPORT on: " + goal + "\n" + "\n".join(f" - {r}" for r in results)
    return report

print(orchestrator("impact of KV-cache quantization on long-context serving"))
"""),
    md(r"""
## 4. Evaluator–optimizer (self-improvement loop)

One agent **generates**, another **evaluates** against criteria and returns feedback; the
generator **revises**. Loop until the evaluator is satisfied or a budget is hit. Great for
writing, code, and translation where quality is judgeable.
"""),
    code(r"""
def generator(task, feedback=None):
    draft = f"draft for: {task}"
    if feedback:
        draft += " | revised to address: " + feedback
    return draft

def evaluator(draft):
    # toy rubric: require the word 'revised' to count as good enough
    if "revised" in draft:
        return True, "looks good"
    return False, "add more specifics and structure"

def eval_optimize(task, max_rounds=3):
    feedback = None
    for r in range(max_rounds):
        draft = generator(task, feedback)
        ok, feedback = evaluator(draft)
        print(f"round {r}: ok={ok}  draft={draft!r}")
        if ok:
            return draft
    return draft

eval_optimize("explain attention to a new engineer")
"""),
    md(r"""
## 5. Handoffs & shared state
- **Handoff:** one agent transfers control (and context) to another better suited to continue
  (e.g. triage → specialist). The receiving agent gets the relevant history.
- **Shared state / blackboard:** agents read/write a common store instead of passing everything
  in messages — reduces token cost and keeps a single source of truth.
- **Communication protocols:** structured messages (typed), not free text, to reduce ambiguity.
"""),
    code(r"""
class Blackboard:
    def __init__(self): self.state = {}
    def write(self, key, value): self.state[key] = value
    def read(self, key): return self.state.get(key)

bb = Blackboard()
bb.write("plan", ["research", "draft", "review"])
bb.write("research", "3 sources found")
print("shared state:", bb.state)
# A 'draft' agent reads research, a 'review' agent reads the draft — all via the blackboard.
"""),
    md(r"""
## 6. When NOT to go multi-agent

Multi-agent adds **coordination overhead** and **compounding errors** (NB09). Prefer it only when:
- subtasks are genuinely **parallelizable** (e.g. independent research threads), or
- **separation of concerns** clearly improves reliability (distinct skills/tools/permissions).

Otherwise a single well-designed agent (or a plain workflow) is cheaper, faster, and easier to
evaluate. Anthropic's multi-agent research write-up: gains come mostly from **parallel** work,
at a real **token cost** — measure it.

## 7. Orchestration engineering (Module 08)
- Structured outputs (JSON schema / constrained decoding) **between** steps, with validation + repair.
- **Prefix caching** the stable system prompt/tools; **model cascades** (cheap→expensive).
- Treat prompts as **versioned, tested** software; put **evals in CI** (NB11).
- Durable execution (checkpoint/resume) for long-running multi-step jobs.

## Exercises
1. Make the workers in §3 real NB09 agents and run them with `asyncio.gather` (true parallelism).
2. Add a router in front of an orchestrator; measure tokens/cost vs a single mega-prompt.
3. Build an evaluator–optimizer for code that runs tests as the evaluator.
4. Implement a handoff from a triage agent to billing/technical specialists with context transfer.

## Resources
- *Building Effective Agents* (Anthropic 2024) — the pattern catalog used here.
- Anthropic — *How we built our multi-agent research system* (2025).
- *A Practical Guide to Building Agents* (OpenAI 2025); LangGraph multi-agent docs; AutoGen, CrewAI.
- *DSPy* (programmatic orchestration & prompt optimization).
"""),
]

if __name__ == "__main__":
    write(cells, "10_multi_agent_systems.ipynb")
