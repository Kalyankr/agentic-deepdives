# Notebooks — Learn Every Pattern by Running It

> Four **teaching notebooks** that build the entire Agentic AI Course from scratch and run
> **offline** — no API key, no cost. LLM-driven cells *script* a deterministic `MockLLM` so output
> is reproducible; the mechanics (ReAct loop, SQLite text2SQL, retrieval, concurrency/locking) are
> real, runnable Python. Set `OPENAI_API_KEY` and the **same code** calls a real model.

| # | Notebook | You build / run | Course |
|---|----------|-----------------|--------|
| 01 | [Prompting](01_prompting.ipynb) | R-T-C-E-O prompts, CoT, a runnable **ReAct** loop, JSON contracts, validated chains, a self-correcting feedback loop | [01](../courses/01-prompting.md) |
| 02 | [Agentic Workflows](02_agentic_workflows.ipynb) | a reusable `Agent` + the **five patterns**: chaining, routing, parallelization, evaluator-optimizer, orchestrator-workers | [02](../courses/02-agentic-workflows.md) |
| 03 | [Building Agents](03_building_agents.ipynb) | tools/function-calling, **Pydantic** structured output, a state machine, sliding-window memory, **real SQLite text2SQL**, agentic RAG, long-term memory, an eval harness | [03](../courses/03-building-agents.md) |
| 04 | [Multi-Agent Systems](04_multi_agent_systems.ipynb) | typed messaging + registry, orchestration, routing, a shared-state blackboard, **real concurrency control** (no overselling), multi-agent RAG, a mini sales-team sim | [04](../courses/04-multi-agent-systems.md) |

Run them in order 01 → 04 (each builds on the last), but later ones stand alone.

---

## How to run

```bash
cd agentic-ai
uv sync --extra dev --extra viz        # one-time

uv run jupyter lab                     # open notebooks/*.ipynb, pick the .venv kernel
```

Or open the `notebooks/` folder in VS Code and select the `.venv` interpreter as the kernel.

> **No key needed.** Cells default to the offline `MockLLM`. To use a real model:
> `uv sync --extra openai` and `export OPENAI_API_KEY=sk-...`, then swap any
> `MockLLM(scripted=[...])` for `get_llm()` — the surrounding code is unchanged.

---

## How the notebooks are generated

To keep the JSON valid and the style consistent, each notebook is produced by a small Python
**builder** in [`src/`](src/) using the helpers in [`src/_nbtools.py`](src/_nbtools.py).

```bash
# rebuild every notebook from its builder, then validate the JSON
uv run python notebooks/build_all.py

# execute every code cell to prove the notebooks run offline (catches regressions)
uv run python notebooks/_exec_check.py
```

Edit the builder (`src/nb_XX_*.py`), not the `.ipynb`, then re-run `build_all.py`.

---

## Pairs with

- **Courses** — the lesson-by-lesson deep dives in [`../courses/`](../courses/).
- **Projects** — the four portfolio builds in [`../projects/`](../projects/).
