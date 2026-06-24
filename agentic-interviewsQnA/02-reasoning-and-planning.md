# 02 — Reasoning & Planning

> Goal: explain how LLMs "think" inside agents — CoT, ReAct, Reflexion, Tree of Thoughts, Plan-and-Execute — and the trade-offs of each.

---

## 2.1 Why reasoning matters in agents

An agent's quality is bounded by how well it decides the **next action**. Reasoning techniques are prompting/looping strategies that improve that decision at the cost of more tokens, latency, and money. The core trade-off across this whole chapter:

> 💡 **More deliberate reasoning → higher accuracy & reliability, but more latency & cost.** Pick the lightest technique that hits your quality bar.

---

## 2.2 Chain of Thought (CoT)

**Idea:** prompt the model to produce intermediate reasoning steps before the final answer ("Let's think step by step").

- **Why it works:** allocates more compute to multi-step problems; externalizes intermediate state.
- **Variants:** zero-shot CoT, few-shot CoT, **self-consistency** (sample N chains, majority-vote the answer).
- **In agents:** the "Thought" in ReAct is CoT applied to action selection.

⚠️ CoT is reasoning *without* acting — no external feedback. It can confidently hallucinate a wrong chain. Agents fix this by interleaving actions/observations.

---

## 2.3 ReAct (Reason + Act) — the canonical agent pattern

💡 **ReAct interleaves reasoning traces and actions:** `Thought → Action → Observation → Thought → …`. The model reasons about what to do, takes an action (tool call), observes the result, and reasons again with the new information.

```
Thought 1: I need the current weather in Paris. I'll use the weather tool.
Action 1:  weather_api(city="Paris")
Observation 1: 18°C, light rain.
Thought 2: It's raining. I should recommend an umbrella and indoor options.
Action 2:  finish("Bring an umbrella; here are 3 indoor activities…")
```

**Why ReAct beats pure CoT for agents:**
- Reasoning is **grounded** in real observations (less hallucination).
- The agent can **recover** from errors using feedback.
- The trace is **interpretable** and debuggable.

**Failure modes:**
- Loops/repeated actions when a tool keeps failing → need loop detection + max steps.
- Bad tool-call formatting → use native function calling / structured output instead of regex parsing.
- "Reasoning drift" on long trajectories → summarize/compact context.

🎯 *"Explain ReAct and why it reduces hallucination vs. CoT."* — Because each reasoning step is conditioned on real tool observations rather than the model's priors alone.

---

## 2.4 Reflection & Reflexion (self-critique)

**Reflection:** after producing an output (or failing a task), the agent critiques its own work and revises. The **evaluator-optimizer** workflow is the two-LLM version (generator + critic).

**Reflexion** (the paper): the agent gets feedback (success/failure, error messages, or an evaluator), generates a **verbal self-reflection**, stores it in episodic memory, and uses it on the next attempt — effectively "learning" within a session without weight updates.

```
Attempt 1 → fails tests
Reflect:  "I assumed 1-indexing; the API is 0-indexed. Next time, check indexing."
(store reflection in memory)
Attempt 2 → uses reflection → passes
```

**When to use:** clear success signal (tests pass, validator, rubric) and iteration measurably helps — coding, math, translation, structured generation.

⚠️ Reflection adds 2–N× cost. Skip it when there's no reliable signal to reflect on (you'll just reinforce noise). Cap the number of reflection rounds.

---

## 2.5 Plan-and-Execute (a.k.a. Planner-Executor)

**Idea:** separate **planning** from **doing**. A planner LLM produces a multi-step plan up front; an executor carries out each step (often a ReAct sub-loop); optionally a **replanner** updates the plan as results arrive.

```
Planner:   [1) find revenue 2) find net income 3) compute margin 4) compare YoY]
Executor:  do step 1 → observe → step 2 → ... 
Replanner: results show data missing for 2022 → insert "search alt source"
```

**Pros:** fewer LLM calls for the "what next" decision (plan once vs. reason every step); clearer structure; easier to parallelize independent steps; the explicit plan is auditable.
**Cons:** rigid if the world changes mid-plan (mitigate with replanning); planner errors cascade.

**ReAct vs. Plan-and-Execute (classic comparison):**

| | ReAct | Plan-and-Execute |
|---|-------|------------------|
| Decision cadence | Every step | Once, then execute |
| Adaptivity | High (re-reasons each step) | Lower (needs replanner) |
| Cost/latency | Higher per step | Lower for long tasks |
| Best for | Unpredictable, exploratory | Structured, decomposable |

---

## 2.6 Tree of Thoughts (ToT) & search-based reasoning

**Idea:** instead of one linear chain, explore a **tree** of reasoning branches, evaluate intermediate states, and search (BFS/DFS/beam) toward the best solution. Generalizes CoT to deliberate search with backtracking.

- **Use for:** problems needing exploration/look-ahead (puzzles, planning, game-like tasks).
- **Cost:** expensive — many model calls per node. Usually overkill for production agents; great as an interview talking point on the reasoning-compute spectrum.

Related: **Graph of Thoughts**, **Language Agent Tree Search (LATS)** = ToT + MCTS + reflection + tool use.

---

## 2.7 Other techniques worth naming

- **Self-consistency** — sample multiple reasoning paths, majority vote. Cheap reliability boost.
- **Least-to-most prompting** — solve easy subproblems first, build up.
- **ReWOO** — decouple reasoning from observations to reduce token use (plan all tool calls, then execute) — a token-efficient alternative to ReAct.
- **Reflexion + memory** — verbal RL within a session (see 2.4).
- **"Reasoning models"** (o-series, DeepSeek-R1, etc.) — models RL-trained to produce long internal CoT before answering. They shift reasoning from your scaffold into the model; you prompt them differently (give the goal, not step-by-step micro-instructions).

💡 With reasoning models, **do less prompt-level hand-holding**: state the objective, constraints, and tools, and let the model plan. Over-specified CoT prompts can hurt them.

---

## 2.8 Choosing a reasoning strategy (decision guide)

```
Single, well-defined task, no tools?            → CoT (+ self-consistency if accuracy critical)
Need external info / actions, path unknown?     → ReAct
Long, decomposable task with clear sub-steps?   → Plan-and-Execute (+ replanning)
Clear pass/fail signal & iteration helps?       → add Reflection/Reflexion
Needs exploration/backtracking, cost OK?        → Tree of Thoughts / LATS
Using a reasoning model?                         → minimal scaffolding, give it the goal
```

---

## 2.9 Reasoning vs. acting — the mental model

| Mode | Has external feedback? | Risk |
|------|------------------------|------|
| CoT / ToT | No | Hallucinated chains |
| ReAct | Yes (observations) | Loops, parse errors |
| Reflexion | Yes (eval signal) | Reinforcing noise if signal is weak |

> The progression of the field: **think → think+act → think+act+critique+remember.**

---

## Interview questions for this chapter

1. Compare CoT and ReAct. Why does ReAct reduce hallucination? *(2.2–2.3)*
2. What is Reflexion and when is it worth the extra cost? *(2.4)*
3. ReAct vs. Plan-and-Execute — when do you pick each? *(2.5)*
4. How does self-consistency improve reliability, and what's the cost? *(2.7)*
5. You're using a reasoning model (o-series/R1). How should your prompting change? *(2.7)*
6. Your agent loops forever calling the same failing tool. Diagnose and fix. *(2.3)*
7. When is Tree of Thoughts justified over plain ReAct in production? *(2.6)*

**Model answer to #6:** Add loop/repeat-action detection (hash the last K actions), cap max steps and budget, surface the tool's error text back into context so the model can adapt, add a fallback/terminate tool, and consider a reflection step after repeated failure. If the tool is genuinely broken, the agent should degrade gracefully (ask the user or return partial results), not retry blindly.
