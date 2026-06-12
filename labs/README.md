# AI/ML Bootcamp — Labs

Runnable, build-first starter code for the [Senior AI/ML Engineer Bootcamp](../README.md).

These labs are **fill-in-the-blanks**: the boilerplate (data, training loops, plotting, CLIs) is
done for you so you can focus on implementing the *conceptual core* (autograd, attention, …).
Each lab ships with a **spec test suite** that is your definition of done.

## Labs

| Lab | Builds | Syllabus module | You implement |
|-----|--------|-----------------|---------------|
| [lab01_micrograd](lab01_micrograd/) | A tiny reverse-mode autograd engine + MLP | [01](../modules/01-deep-learning-foundations.md) | Local gradients (`_backward`) |
| [lab02_nanogpt](lab02_nanogpt/) | A GPT from scratch (train + sample) | [02](../modules/02-transformer-internals.md) | Causal self-attention |
| [lab04_inference_bench](lab04_inference_bench/) | A latency/throughput benchmark for any OpenAI-compatible endpoint (e.g. vLLM) | [04](../modules/04-gpu-architecture-and-inference.md) | Runs as-is; extend it |
| [lab06_rag](lab06_rag/) | A RAG retrieval core: cosine, exact + IVF index, hybrid fusion, metrics | [06](../modules/06-rag-and-vector-databases.md) | Similarity, kNN/IVF search, recall@k & MRR |
| [lab07_agent](lab07_agent/) | A ReAct agent loop with pluggable tools + brain | [07](../modules/07-agentic-systems.md) | Action parsing + the think/act/observe loop |

More labs (evals, fine-tuning) follow the same pattern — add them as you reach those modules.

## Setup (uses [uv](https://docs.astral.sh/uv/))

```bash
cd labs

# Core deps (numpy, httpx) + dev tools (pytest, ruff)
uv sync --extra dev

# Add deep-learning + plotting extras when you start lab02 / want plots
uv sync --extra dev --extra nn --extra viz
```

## The workflow

1. Open a lab's `README.md` and the matching syllabus module.
2. Find the `TODO` / `NotImplementedError` markers — that's your work.
3. Run the spec tests until they pass:

   ```bash
   # Default run = quick smoke tests only (green out of the box)
   uv run pytest

   # The learning specs (fail until you implement the TODOs):
   uv run pytest -m todo
   uv run pytest -m todo tests/test_lab01_micrograd.py   # one lab
   ```

4. Run the demos:

   ```bash
   uv run python -m lab01_micrograd.demo
   uv run python -m lab02_nanogpt.train
   uv run python -m lab02_nanogpt.sample --prompt "To be"
   uv run python -m lab04_inference_bench.benchmark --url http://localhost:8000 --model my-model
   uv run python -m lab06_rag.demo
   uv run python -m lab07_agent.demo
   ```

## Lint

```bash
uv run ruff check .
uv run ruff format .
```

> The `todo`-marked tests are deselected by default (see `pyproject.toml`) so a fresh clone is green.
> They are your spec — run `uv run pytest -m todo` to check your implementations.
