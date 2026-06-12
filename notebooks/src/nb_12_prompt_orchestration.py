"""Build NB12 — Prompt orchestration & context engineering."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 12 · Prompt Orchestration & Context Engineering

> Module: **08 · Prompt Orchestration**.

**Goal:** get the most out of a *fixed* model — prompting techniques, **context engineering**,
**structured outputs**, and cost patterns (**prefix caching**, **model cascades**). Runnable demos
for self-consistency voting, structured-output validation/repair, and a cost cascade.

### Learning objectives
1. Apply zero/few-shot, **chain-of-thought**, and **self-consistency**.
2. Engineer the context window deliberately (what goes in, what gets compacted).
3. Get **reliable structured output** with validation + repair.
4. Cut cost/latency with prefix caching and cascades.
"""),
    md(r"""
## 1. Core prompting techniques

| Technique | Idea | When |
|-----------|------|------|
| **Zero-shot** | just ask | strong instruction-tuned models, simple tasks |
| **Few-shot** | show 2–5 examples | format/style matters; niche tasks |
| **Chain-of-thought** | "think step by step" | math/logic/multi-step reasoning |
| **Self-consistency** | sample N CoT, **majority vote** | boost reasoning accuracy (costs N×) |
| **Decomposition** | break into sub-prompts | complex tasks (→ chaining, NB10) |

Modern **reasoning models** do much CoT internally — but examples, structure, and clear
instructions still matter a lot.
"""),
    code(r"""
import re, json
from collections import Counter
import numpy as np
rng = np.random.default_rng(0)

# Few-shot prompt builder: examples shape the output format.
def build_few_shot(task, examples, query):
    lines = [task, ""]
    for x, y in examples:
        lines += [f"Input: {x}", f"Output: {y}", ""]
    lines += [f"Input: {query}", "Output:"]
    return "\n".join(lines)

prompt = build_few_shot(
    "Classify sentiment as POS/NEG.",
    [("I love this", "POS"), ("worst ever", "NEG")],
    "this is great",
)
print(prompt)
"""),
    md(r"""
## 2. Self-consistency = sample several reasoning paths, then vote

Instead of one greedy answer, sample $N$ chains at temperature > 0 and take the **majority**.
Trades compute for accuracy on reasoning tasks. Here a mock solver returns noisy answers; voting
recovers the correct one.
"""),
    code(r"""
def noisy_solver():
    # correct answer is 42; the model is right 60% of the time, else a random wrong number
    return 42 if rng.random() < 0.6 else int(rng.integers(0, 100))

def self_consistency(n):
    votes = [noisy_solver() for _ in range(n)]
    return Counter(votes).most_common(1)[0][0]

acc1 = np.mean([noisy_solver() == 42 for _ in range(2000)])
accN = np.mean([self_consistency(7) == 42 for _ in range(2000)])
print(f"single sample accuracy : {acc1:.0%}")
print(f"majority of 7 accuracy : {accN:.0%}   (self-consistency lifts reasoning accuracy)")
"""),
    md(r"""
## 3. Context engineering

The context window is a **budget** you allocate, not a dump. Good systems decide *what* to put in
and *when* to remove it:
- **System prompt:** role, rules, output contract (keep stable for prefix caching, §6).
- **Few-shot examples / retrieved docs (NB08):** only the most relevant; rerank.
- **Tool definitions & results (NB09):** treated as data, not instructions.
- **Compaction:** summarize old turns; drop stale tool output to fight "context rot."

> "Context engineering" is the agent-era successor to prompt engineering: managing the *whole*
> token budget across a trajectory, not crafting one clever string.
"""),
    code(r"""
def pack_context(system, examples, retrieved, history, budget_tokens=1000, approx=4):
    # greedily fill the budget by priority; ~approx chars per token (toy estimate)
    def toks(s): return len(s)//approx
    ctx, used = [], 0
    for label, chunk in [("system", system), *[("ex", e) for e in examples],
                         *[("doc", d) for d in retrieved], *[("turn", h) for h in history]]:
        if used + toks(chunk) > budget_tokens:
            ctx.append("...[older context compacted to fit budget]...")
            break
        ctx.append(chunk); used += toks(chunk)
    print(f"used ~{used}/{budget_tokens} tokens, {len(ctx)} blocks kept")
    return "\n".join(ctx)

_ = pack_context("SYSTEM: you are helpful.", ["ex1"*50, "ex2"*50],
                 ["doc"*400, "doc2"*400], ["u: hi", "a: hello", "u: ..."], budget_tokens=300)
"""),
    md(r"""
## 4. Structured output — make the model machine-readable

For pipelines you need **valid JSON** every time. Three levels:
1. **Ask** for JSON (give the schema) — easy, not guaranteed.
2. **Constrained decoding / JSON mode** — grammar restricts tokens so output *must* parse.
3. **Validate + repair** — parse against a schema; on failure, send the error back to fix.
"""),
    code(r"""
SCHEMA = {"name": str, "age": int}

def validate(obj, schema):
    errs = []
    for k, t in schema.items():
        if k not in obj: errs.append(f"missing '{k}'")
        elif not isinstance(obj[k], t): errs.append(f"'{k}' must be {t.__name__}")
    return errs

def parse_and_repair(raw, schema, repair):
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raw = repair(f"invalid JSON: {e}"); obj = json.loads(raw)
    errs = validate(obj, schema)
    if errs:
        raw = repair("; ".join(errs)); obj = json.loads(raw)
    return obj

# mock model: first emits broken output, then repairs when told the error
state = {"n": 0}
def mock_model(_msg):
    state["n"] += 1
    return '{"name": "Ada", "age": "x"}' if state["n"] == 1 else '{"name": "Ada", "age": 36}'

print(parse_and_repair(mock_model("go"), SCHEMA, repair=mock_model))
"""),
    md(r"""
## 5. Automatic prompt optimization
Stop hand-tuning strings — **optimize** them:
- **DSPy:** declare *signatures* (typed in→out) and *compile* prompts/weights against a metric.
- **APE / OPRO:** an LLM proposes and refines prompts using eval scores as the objective.
- Treat prompts as **versioned, tested artifacts**; let your **eval suite (NB11)** pick winners.
"""),
    md(r"""
## 6. Cost & latency patterns

- **Prefix caching:** the long, *stable* prefix (system prompt + tools + few-shot) is cached by the
  server, so repeated calls only pay for the changing suffix — big savings for agents/chat.
- **Model cascade:** try a **cheap** model first; **escalate** to an expensive one only when the
  cheap answer is low-confidence. Most traffic stays cheap.
- **Batching / parallel** independent calls; **stream** for better perceived latency.
"""),
    code(r"""
# Model cascade: route easy queries to a cheap model, hard ones to an expensive model.
def cheap_model(q):   return ("answer", 0.9 if len(q) < 20 else 0.4)   # (text, confidence)
def expensive_model(q): return ("answer", 0.99)

PRICE = {"cheap": 0.0002, "expensive": 0.003}   # $ per call (illustrative)

def cascade(queries, conf_threshold=0.7):
    total, escalations = 0.0, 0
    for q in queries:
        _, conf = cheap_model(q); total += PRICE["cheap"]
        if conf < conf_threshold:
            expensive_model(q); total += PRICE["expensive"]; escalations += 1
    print(f"{len(queries)} queries: {escalations} escalated, total ${total:.4f} "
          f"(vs ${len(queries)*PRICE['expensive']:.4f} if all-expensive)")

cascade(["hi", "what time is it", "explain the full derivation of DPO with KL constraints please"]*10)
"""),
    md(r"""
## Exercises
1. Implement self-consistency on a real reasoning benchmark; plot accuracy vs N samples (vs cost).
2. Build a JSON-mode pipeline with a real model + a Pydantic schema + repair loop; measure parse rate.
3. Add prefix caching to your agent (NB09) and measure token/cost reduction.
4. Build a confidence-based cascade and report cost vs quality on a held-out set (NB11).

## Resources
- *Chain-of-Thought* (Wei 2022); *Self-Consistency* (Wang 2022); *ReAct* (Yao 2022).
- *DSPy* (Khattab 2023); *APE* (Zhou 2022); *OPRO* (Yang 2023).
- Anthropic — *Effective context engineering*; provider docs on **prompt caching** & **JSON mode**.
- Outlines / Guidance / XGrammar for constrained decoding.
"""),
]

if __name__ == "__main__":
    write(cells, "12_prompt_orchestration.ipynb")
