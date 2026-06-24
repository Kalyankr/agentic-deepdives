# Agentic AI — Interview Preparation Kit

A full, structured study pack for **Agentic AI**, **multi-agent systems**, **system design**, and **interview questions**. Built for senior/staff-level ML, AI Engineer, and Applied Scientist interviews.

> Use this as a 2–3 week study plan or a fast refresher the night before. Each chapter is self-contained and ends with interview-style questions.

---

## How to use this kit

1. **Skim the [Cheat Sheet](10-cheatsheet.md)** to map the whole landscape in 10 minutes.
2. **Read chapters 01 → 08** in order. They build on each other.
3. **Drill the [Interview Question Bank](09-interview-questions.md)** — say answers out loud, then check the [Full Spoken Answers](09b-interview-answers-full.md).
4. **Run the [code examples](code/)** so you can whiteboard an agent loop from memory.
5. **Practice 2–3 [system design case studies](07-system-design.md)** end-to-end on a whiteboard.
6. **Review the [flashcards](flashcards.csv)** daily (import into Anki) for spaced repetition.
7. **Run a timed [Mock Interview](11-mock-interview.md)** — solo or with a partner — and score yourself.

---

## Table of contents

| # | Chapter | What you'll be able to do |
|---|---------|---------------------------|
| 01 | [Fundamentals](01-fundamentals.md) | Define agents, the agent loop, autonomy levels, when *not* to use agents |
| 02 | [Reasoning & Planning](02-reasoning-and-planning.md) | Explain CoT, ReAct, Reflexion, ToT, Plan-and-Execute trade-offs |
| 03 | [Memory & Context](03-memory-and-context.md) | Design short/long-term memory, RAG, context engineering |
| 04 | [Tools & Function Calling](04-tools-and-function-calling.md) | Design tools, handle errors, explain MCP and computer-use |
| 05 | [Multi-Agent Systems](05-multi-agent-systems.md) | Choose topologies, orchestration, handoffs, A2A protocols |
| 06 | [Frameworks & Protocols](06-frameworks-and-protocols.md) | Compare LangGraph, AutoGen, CrewAI, Semantic Kernel, OpenAI Agents SDK |
| 07 | [System Design](07-system-design.md) | Drive an agentic system-design interview with a repeatable framework |
| 08 | [Production, Eval & Security](08-production-evaluation-security.md) | Evaluate, observe, guardrail, and secure agents in production |
| 09 | [Interview Question Bank](09-interview-questions.md) | 120+ Q&A: conceptual, design, coding, behavioral |
| 09b | [Full Spoken Answers](09b-interview-answers-full.md) | Every question answered the way you'd say it out loud |
| 09c | [Follow-Up Questions](09c-followup-questions.md) | The deeper 2nd/3rd-level probes + emerging 2025–26 topics |
| 10 | [Cheat Sheet](10-cheatsheet.md) | One-page recall of every key concept |
| 11 | [Mock Interview](11-mock-interview.md) | Run a timed 5-round simulation with follow-ups + scoring rubric |
| 🚨 | [Panic Sheet](15-panic-sheet.md) | 20 facts/numbers to skim in the **5 minutes before the call** |

Code: [`code/`](code/) — runnable Python examples (ReAct from scratch, LangGraph multi-agent, AutoGen group chat, tool calling, reflection loop).

Practice: [`flashcards.csv`](flashcards.csv) — 95 Anki-importable spaced-repetition cards covering every chapter.

---

## Role-specific tracks

The chapters above are the shared core. These tracks tell you **what to emphasize, add role-specific Q&A, and give a 1-week plan** for your target role. Read your track *after* skimming the core.

| Track | For | Emphasis |
|-------|-----|----------|
| 12 | [AI Engineer](12-track-ai-engineer.md) | Build agentic products: tools/MCP, RAG, frameworks, eval, latency/cost, shipping |
| 13 | [Applied Scientist](13-track-applied-scientist.md) | Methods & rigor: evaluation methodology, SFT/RLHF/DPO/RFT, experiments, paper critique |
| 14 | [ML Engineer](14-track-ml-engineer.md) | Infra & serving: inference optimization, vector/retrieval infra, deployment, SLOs, cost-at-scale |

> Not sure which? **AI Engineer** = "make the model useful in a product." **Applied Scientist** = "prove it works and why." **ML Engineer** = "make it run fast, cheap, and reliably at scale." Many roles blend two — skim all three and go deep on your primary.

---

## The 10 ideas you must be able to explain cold

1. **Agent = LLM + loop + tools + memory + goal.** It decides *what to do next* based on feedback from the environment, instead of following a fixed script.
2. **Agentic vs. workflow:** workflows have predetermined code paths; agents dynamically direct their own process and tool use. Use the simplest thing that works.
3. **The agent loop:** `perceive → reason → act → observe → repeat until done`. ReAct is the canonical instance.
4. **Tool use / function calling** is how agents affect the world. Good tool design matters more than clever prompts.
5. **Memory** turns a stateless LLM into a stateful agent: short-term (context window), long-term (vector/DB), episodic/semantic/procedural.
6. **Planning** decomposes goals into steps; **reflection** lets the agent critique and retry. Both trade latency/cost for reliability.
7. **Multi-agent systems** split work across specialized roles (orchestrator-worker, hierarchical, debate). They add coordination cost — justify them.
8. **Context engineering** (not just prompt engineering) is the core skill: get the right information and tools into the window at the right time.
9. **Evaluation is the hard part.** You need trajectory-level and outcome-level metrics, LLM-as-judge, and offline + online eval.
10. **Production = reliability + observability + guardrails + cost/latency control + security** (prompt injection, excessive agency, tool sandboxing).

---

## Quick study plans

### 3-day crash plan
- **Day 1:** Ch 01, 02, 04 + Cheat Sheet. Run the ReAct example.
- **Day 2:** Ch 05, 06 + run the LangGraph example. Skim Ch 03.
- **Day 3:** Ch 07, 08 + drill the question bank (conceptual + design).

### 2-week deep plan
- **Days 1–3:** Ch 01–03, write your own notes.
- **Days 4–6:** Ch 04–05, build a small multi-agent demo.
- **Days 7–9:** Ch 06–07, do 3 system-design case studies on paper.
- **Days 10–12:** Ch 08, set up tracing/eval on your demo.
- **Days 13–14:** Mock interviews using Ch 09.

---

## Conventions

- 💡 = key insight worth memorizing.
- ⚠️ = common interview trap / mistake.
- 🎯 = likely interview question.
- Code is Python 3.10+ and framework-agnostic where possible.

> This material is framework-aware but concept-first. Interviewers care that you understand *why*, not that you memorized one SDK's API.
