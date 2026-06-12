# Setup

This course runs in two modes. **Offline mode is the default and needs no API key.**

---

## 1. Install `uv` (one time)

`uv` is a fast Python package/venv manager.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## 2. Offline mode (default — no key, no cost)

```bash
cd agentic-ai
uv sync --extra dev
```

Every lesson, notebook, and project runs against the deterministic
[`MockLLM`](shared/llm.py). Outputs are reproducible; you learn the orchestration
without paying for tokens. Verify:

```bash
uv run python -c "from shared.llm import get_llm, user; print(get_llm().chat([user('hello')]))"
# -> [mock-llm reply] ...
```

---

## 3. Online mode (real models)

Install the OpenAI SDK and set a key. **The same code now calls a real model.**

```bash
uv sync --extra dev --extra openai

# macOS / Linux
export OPENAI_API_KEY="sk-..."
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."

uv run python -c "from shared.llm import get_llm, user; print(get_llm().chat([user('Say hi in 3 words.')]))"
```

`get_llm()` auto-detects: if `OPENAI_API_KEY` is set **and** `openai` is installed it
uses [`OpenAILLM`](shared/llm.py); otherwise it falls back to the mock. Force either side:

```python
from shared.llm import get_llm
llm = get_llm(prefer_mock=True)     # always offline
llm = get_llm(prefer_mock=False)    # require a real model (raises if no SDK)
llm = get_llm(model="gpt-4o")       # pick a model
```

> A `.env` file is supported when you install `--extra openai` (it pulls in
> `python-dotenv`). Put `OPENAI_API_KEY=sk-...` in `agentic-ai/.env`
> and call `from dotenv import load_dotenv; load_dotenv()` at the top of your script.
> **Never commit your key.**

### Azure OpenAI or other OpenAI-compatible endpoints

```python
from shared.llm import OpenAILLM
llm = OpenAILLM(model="my-deployment", base_url="https://<resource>.openai.azure.com/openai/v1/",
                api_key="...")
```

---

## 4. Optional extras (only when a lesson needs them)

| Extra | Installs | Used by |
|-------|----------|---------|
| `openai` | `openai`, `python-dotenv` | real models |
| `rag` | `chromadb` | Course 3 agentic RAG, Course 4 multi-agent RAG |
| `db` | `sqlalchemy` | Course 3 SQL database agents |
| `web` | `tavily-python`, `httpx` | Course 3 web-search agents |
| `viz` | `matplotlib` | a few notebook plots |
| `dev` | `pytest`, `ruff`, `jupyter` | tooling + notebooks |

Install several at once:

```bash
uv sync --extra dev --extra openai --extra rag --extra db --extra web --extra viz
```

---

## 5. Run things

```bash
# Notebooks
uv run jupyter lab                       # open notebooks/*.ipynb, pick the .venv kernel

# Rebuild notebooks from their Python builders
uv run python notebooks/build_all.py

# Lint
uv run ruff check .

# Project tests (each project ships its own)
uv run pytest projects/01_trip_planner -q
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: shared` | Run from the `agentic-ai/` folder (it's on `pythonpath`), or `uv run`. |
| `openai is not installed` | `uv sync --extra openai`, or use `get_llm(prefer_mock=True)`. |
| Real model ignored | Confirm `OPENAI_API_KEY` is exported **and** `--extra openai` was synced. |
| Rate limits / 429 | Lower request volume, or switch to offline mode for development. |
| Notebook kernel missing | In VS Code pick the `.venv` interpreter; or `uv run jupyter lab`. |
