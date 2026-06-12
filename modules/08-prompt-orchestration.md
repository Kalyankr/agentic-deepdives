# Module 08 · Prompt Orchestration

> **Goal:** Move from ad-hoc prompting to **engineered, programmatic, optimized** prompt systems — chaining, routing, structured outputs, context engineering, caching, and automatic prompt optimization — treating prompts as versioned, tested software.

**Duration:** ~3 weeks. **Prereqs:** [Module 07](07-agentic-systems.md).

---

## 8.1 Prompting fundamentals (done rigorously)

- Zero-shot, few-shot, in-context learning — and why it works (induction heads, pattern completion)
- **Chain-of-thought**, least-to-most, self-consistency (sample-and-vote)
- Role/system prompts, delimiters, output formatting
- Decoding controls: temperature, top-p/top-k, repetition penalties, min-p; when determinism matters
- Failure modes: prompt sensitivity, position bias, verbosity bias, sycophancy

## 8.2 Context engineering

- The context window as a scarce, managed resource (not a junk drawer)
- What belongs in context: instructions, tools, few-shots, retrieved facts, memory, state
- Compaction: summarization, pruning, structured note-taking
- Ordering effects ("lost in the middle"); putting critical info at the edges
- Token budgeting across system/tools/history/retrieval
- "Context rot" in long sessions and how to mitigate it

## 8.3 Orchestration patterns

- **Prompt chaining** — decompose into sequential calls with checks between
- **Routing** — classify the request, dispatch to a specialized prompt/model
- **Parallelization** — sectioning and voting (map-reduce over an LLM)
- **Orchestrator–workers**, **evaluator–optimizer** loops
- Control flow, branching, retries, fallbacks, schema validation between steps
- These overlap with agent patterns ([Module 07](07-agentic-systems.md)) — here the focus is *deterministic orchestration of LLM calls*

## 8.4 Structured & reliable outputs

- JSON mode, function/tool schemas, **constrained decoding / grammars** (Outlines, XGrammar, JSON Schema)
- Validation with `pydantic`; repair loops on invalid output
- Idempotency and determinism for testable pipelines

## 8.5 Programmatic prompting & optimization

- Prompts as code: templating, versioning, parameterization, separation from logic
- **DSPy** — declare *signatures*, compile/optimize prompts and few-shots automatically; optimizers (MIPRO, bootstrap)
- Automatic prompt optimization (APE, OPRO, evolutionary search)
- Prompt registries, A/B testing prompts, regression testing prompts (ties to [Module 09](09-evaluations.md))

> **Build:** Take one task, write a baseline prompt, then optimize it with **DSPy** (or an APE-style loop) against an eval set. Report the accuracy lift and the discovered prompt/few-shots.

## 8.6 Caching & cost in orchestration

- **Prompt/prefix caching** (provider-side and in vLLM/SGLang) — restructure prompts to maximize cache hits (stable prefix first)
- Semantic caching of responses
- Model routing/cascades: cheap model first, escalate on low confidence
- Batching offline workloads; streaming for UX
- Cost/latency accounting per chain (ties to [Module 11](11-system-design-and-capacity-planning.md))

## 8.7 Frameworks (use, then understand the internals)

- LangChain / LangGraph (graphs of LLM calls + state)
- LlamaIndex (data/RAG orchestration)
- DSPy (programmatic optimization)
- Provider SDKs (OpenAI, Anthropic) and their tool/caching primitives
- Guidance / Outlines (constrained generation)

> Build at least one orchestration **without** a heavy framework so you understand what they abstract.

---

## Module 08 capstone — **An optimized, cached orchestration pipeline**

1. A multi-step pipeline (e.g., classify → retrieve → draft → critique → finalize) with schema-validated structured outputs and fallbacks.
2. **Prefix/response caching** added, with a measured cost/latency reduction and cache-hit rate.
3. A **DSPy/APE-optimized** prompt vs. a hand-written baseline, evaluated on a held-out set with the accuracy delta reported.
4. Prompts versioned and covered by regression tests in CI.

## Exit criteria
- [ ] You treat prompts as tested, versioned software with evals in CI.
- [ ] You can design chaining/routing/parallel orchestration with validation and fallbacks.
- [ ] You can enforce structured outputs via constrained decoding.
- [ ] You can optimize prompts programmatically and cut cost with caching/routing.

## Core sources
- Anthropic & OpenAI prompt engineering / context engineering guides
- *Chain-of-Thought Prompting* — Wei et al., 2022; *Self-Consistency* — Wang et al., 2022
- *DSPy* — Khattab et al., 2023; *MIPRO*
- *OPRO: LLMs as Optimizers* — Yang et al., 2023
- Outlines / XGrammar docs (constrained decoding)
- Anthropic prompt-caching & context-engineering posts
