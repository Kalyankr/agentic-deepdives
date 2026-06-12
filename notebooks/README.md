# Notebooks — Learn Every Concept by Running It

> Twelve **teaching notebooks** that build the entire curriculum from scratch — transformers,
> training, fine-tuning, alignment, inference, distributed systems, RAG, agents, multi-agent
> systems, evaluations, and prompt orchestration.

Every notebook is **self-contained and runnable offline**. Core demos use **NumPy + the standard
library only** (always run, no GPU, no network, no API keys). PyTorch / matplotlib / live-API code
is shown as clearly-labeled optional or illustrative snippets.

---

## The 12 notebooks

| # | Notebook | You build / derive | Module |
|---|----------|--------------------|--------|
| 01 | [Autograd & Training](01_autograd_and_training.ipynb) | reverse-mode autodiff `Value` engine, MLP, training loop, gradient check | [01](../modules/01-deep-learning-foundations.md) |
| 02 | [Transformers From Scratch](02_transformers_from_scratch.ipynb) | tokenization, scaled-dot-product & multi-head attention, causal mask, RoPE, KV cache, params/FLOPs math | [02](../modules/02-transformer-internals.md) |
| 03 | [Pretraining & Scaling Laws](03_pretraining_and_scaling_laws.ipynb) | cross-entropy/perplexity, data packing, Chinchilla `6ND`, GPU-hours estimation | [02–03](../modules/02-transformer-internals.md) |
| 04 | [Fine-tuning: SFT & PEFT](04_finetuning_sft_and_peft.ipynb) | chat templates + loss masking, LoRA `W'=W+(α/r)BA`, QLoRA, multi-LoRA serving | [03](../modules/03-llm-training-rlhf-dpo.md) |
| 05 | [Alignment: RLHF & DPO](05_alignment_rlhf_and_dpo.ipynb) | Bradley–Terry reward model (trained), PPO+KL, **runnable DPO trainer** with analytic gradient | [03](../modules/03-llm-training-rlhf-dpo.md) |
| 06 | [Inference & Serving](06_inference_and_serving.ipynb) | prefill/decode, samplers, KV-cache memory, roofline throughput, quantization, speculative decoding | [04](../modules/04-gpu-architecture-and-inference.md) |
| 07 | [Distributed Training & Inference](07_distributed_training_and_inference.ipynb) | collectives + ring all-reduce sim, training-memory math, DP/TP/PP/ZeRO/FSDP, strategy chooser | [05](../modules/05-distributed-systems-and-inference.md) |
| 08 | [RAG & Vector Databases](08_rag_and_vector_databases.ipynb) | embeddings + cosine, exact vs **IVF** ANN with recall@k, PQ compression, full RAG pipeline, MRR | [06](../modules/06-rag-and-vector-databases.md) |
| 09 | [AI Agents](09_ai_agents.ipynb) | tools with error recovery, **ReAct loop** (runnable), memory, prompt-injection/MCP safety | [07](../modules/07-agentic-systems.md) |
| 10 | [Multi-Agent Systems](10_multi_agent_systems.ipynb) | routing, orchestrator–workers, evaluator–optimizer, handoffs, blackboard, when *not* to | [07](../modules/07-agentic-systems.md) |
| 11 | [Evaluations](11_evaluations.ipynb) | metrics, **bootstrap CI**, LLM-as-judge + bias fix, Elo, agent/safety eval, **CI gate** | [09](../modules/09-evaluations.md) |
| 12 | [Prompt Orchestration](12_prompt_orchestration.ipynb) | few-shot, **self-consistency** voting, context engineering, structured-output repair, model cascade | [08](../modules/08-prompt-orchestration.md) |

Recommended order is 01 → 12 (each builds on the last), but later notebooks stand alone.

---

## How to run

These notebooks reuse the [labs](../labs/README.md) environment (NumPy is already a dependency).

```bash
# 1) set up the environment once (from the repo root)
cd labs && uv sync --extra viz      # add --extra nn for the optional PyTorch cells

# 2) launch Jupyter against that environment
uv run --with jupyter jupyter lab   # then open ../notebooks/*.ipynb
```

Or open the `notebooks/` folder directly in VS Code and select the `labs/.venv` kernel.

> The optional `viz` extra installs matplotlib (used by a couple of plotting cells). Without it,
> those cells are skipped — every other cell still runs.

---

## How the notebooks are generated

To keep the JSON valid and the style consistent, each notebook is produced by a small Python
**builder** in [`src/`](src/) using the helpers in [`src/_nbtools.py`](src/_nbtools.py).

```bash
# rebuild every notebook from its builder, then validate the JSON
uv run --project labs python notebooks/build_all.py

# (optional) execute every runnable code cell to catch regressions
uv run --project labs python notebooks/_exec_check.py
```

Edit the builder (`src/nb_XX_*.py`), not the `.ipynb`, then re-run `build_all.py`.

---

## Pairs with the labs

The notebooks **teach and derive**; the [labs](../labs/README.md) are **fill-in-the-blank projects**
with spec tests as your definition of done. Suggested loop: read the notebook → implement the
matching lab → check it with `uv run pytest -m todo`.
