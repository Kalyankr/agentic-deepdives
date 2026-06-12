# Lab 07 — A ReAct Agent

> Module: [07 · Agentic Systems](../../modules/07-agentic-systems.md)

Build the **ReAct loop** that turns an LLM into an agent: *think → act (call a tool) → observe →
… → answer*. The model ("brain") and the tools are pluggable; a deterministic mock brain lets the
whole thing run **offline**, so you focus on the control flow that every agent framework shares.

## What you implement

The `TODO`s live in [agent.py](agent.py):

| Piece | What it does |
|-------|--------------|
| `parse_action` | read one brain output → `("final", answer)` / `("action", (tool, arg))` / `("error", …)` |
| `Agent.run` | the loop: call brain → parse → run tool → append observation → repeat until done / `max_steps` |

Provided for you: the [tools](tools.py) (`calculator`, `lookup`, each with safe error capture),
the [brain](brain.py) (`mock_brain`, a deterministic ReAct policy + a real-LLM sketch), and the
`Step` / `AgentResult` dataclasses.

## Run

```bash
# spec tests (fail until you implement the TODOs)
uv run pytest -m todo tests/test_lab07_agent.py

# the demo (works once the TODOs pass)
uv run python -m lab07_agent.demo
```

Expected: the math question routes through `calculator` → `84`; the "kv cache" question routes
through `lookup`; an unknown question is answered directly — each printed with its trajectory.

## Why this matters (interview-relevant)

- **The loop is the product.** Models don't "use tools" natively — a loop parses a tool request,
  executes it, and feeds the **observation** back so the model can react. That feedback append is
  the entire trick.
- **Tools fail; agents recover.** Tools return `ERROR: …` text instead of raising, so the agent can
  read the error and retry. Robust error surfaces beat clever prompts.
- **Safety lives in the tools.** The `calculator` allow-lists characters and runs `eval` with empty
  builtins — a concrete example of sandboxing untrusted, model-chosen input.
- **Termination matters.** `max_steps` prevents infinite loops — every production agent needs a
  budget (steps, tokens, wall-clock, or cost).

## Stretch goals

- Add a **`search` tool** backed by Lab 06's retriever — now it's a retrieval-augmented agent.
- Add **per-step token/cost budgets** and stop when exceeded; report the spend in `AgentResult`.
- Implement a **reflection** step (the agent critiques its own answer before finalizing).
- Swap `mock_brain` for a real LLM using the sketch in [brain.py](brain.py) — the loop is unchanged.
- Support **parallel tool calls** in one step and merge the observations.
