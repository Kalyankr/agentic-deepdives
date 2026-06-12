# Projects — Your Agentic AI Portfolio

Four portfolio projects, one per course. Each proves a distinct competency and **runs offline**
(deterministic `MockLLM`) — set `OPENAI_API_KEY` to run any of them against a real model unchanged.

| # | Project | Course | Proves you can… |
|---|---------|--------|-----------------|
| 01 | [Trip Planner](01_trip_planner/) | [01 Prompting](../courses/01-prompting.md) | extract structured constraints, reason an itinerary (CoT), act via tools (ReAct), and self-correct in a feedback loop |
| 02 | [Project-Management Workflow](02_project_management_workflow/) | [02 Workflows](../courses/02-agentic-workflows.md) | build a reusable agent library and orchestrate it (orchestrator-workers) into a project plan |
| 03 | [Research Agent](03_research_agent/) | [03 Building Agents](../courses/03-building-agents.md) | build a stateful agent with tools, agentic RAG, web fallback, structured cited output, and an eval harness |
| 04 | [Paper Company Sales Team](04_sales_team/) | [04 Multi-Agent](../courses/04-multi-agent-systems.md) | design a multi-agent system with orchestration, routing, shared state, and concurrency control |

---

## How each project is organized

```
NN_project_name/
├── README.md       ← brief, requirements, rubric, run instructions
├── solution.py     ← complete, runnable reference (the "answer key"); has a demo in __main__
├── starter.py      ← same structure with core methods left as TODO for you to implement
└── test_*.py       ← tests that pin the expected behavior
```

### Run a reference demo

```bash
cd agentic-ai
uv run python projects/01_trip_planner/solution.py
```

### Do the exercise

1. Implement the `TODO` methods in `starter.py`.
2. Point the test at your work: change `import solution as impl` → `import starter as impl` at the
   top of the project's `test_*.py`.
3. Run it:

```bash
uv run --extra dev pytest projects/01_trip_planner -q
```

The tests pass against `solution.py` out of the box, so you always have a working target to study.

---

## Suggested order

Do each project right after its course + notebook: 01 → 02 → 03 → 04. Together they form a
portfolio that demonstrates the full arc — reliable prompting, composable workflows, real agents,
and coordinated multi-agent systems.
