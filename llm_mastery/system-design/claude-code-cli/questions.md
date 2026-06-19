# Claude Code (Agentic Coding CLI) — Interview Questions (all levels)

> **Scope:** screening through **senior / staff / principal** ML-systems / Applied-Scientist / agent-engineering interviews. The reference design is [README.md](README.md). `→` = what a strong answer covers.
>
> **Tags:** 🟢 Fundamentals · 🟡 Core · 🔴 Senior/Staff · 🧮 Math/Estimation · 🏗️ Design · 🐞 Debug/Ops
>
> **✅ Answer key:** full worked solutions in [answers.md](answers.md).

---

## 🟢 Fundamentals
1. What is an agentic coding CLI, and how is it different from a chatbot or autocomplete?
2. Describe the agentic loop (think → act → observe). What makes it stop?
3. Why can't you just load the whole codebase into the context window?
4. What is a "tool" here, and what makes a good tool definition?
5. How does the agent find the relevant code in a large repo?
6. Why is *verifying* edits (compiler/tests) so central to the design?
7. What is prompt injection in this setting, and why is it especially dangerous?
8. What is the permission system for, and what does it gate?
9. Why is this workload "prefill-heavy," and why does that matter for cost?
10. What is context compaction, and when does it kick in?

## 🟡 Core design
11. Walk through what happens from "fix this failing test" to a completed fix.
12. Design the tool set (action space) — which tools, and at what granularity?
13. How do you represent code edits so they apply reliably and don't clobber code?
14. Design the context manager: how do you budget the window on each step?
15. Agentic search vs an embedding index (RAG) for code — pick one and justify.
16. Design the permission model — which actions are gated, and how?
17. How do you execute commands safely? Walk the sandboxing spectrum.
18. How do you handle long-horizon tasks that exceed the context window?
19. Design session persistence, resume, and rollback.
20. When and how do you use subagents?

## 🔴 Senior / Staff deep dives
21. The agent gets stuck repeating a failing action. Design the loop guards.
22. Defend against prompt injection from file/web/tool content — end to end.
23. A multi-file refactor (rename a symbol across the repo) — how do you do it correctly?
24. Make the system cheap at scale — where does the cost go, and what do you cut?
25. Design a headless / CI mode that runs the agent unattended on issues.
26. How do you keep the human in control without crippling autonomy?
27. How do you evaluate an agentic coding system?
28. The target is a giant monorepo (billions of tokens). What changes?

## 🧮 Math & estimation
29. Estimate a codebase's size in tokens and compare it to the context window.
30. For a 60-step task, estimate input vs output tokens — what dominates?
31. Estimate the cost of a task with vs without prompt caching.
32. Estimate the wall-clock latency of a multi-step task.
33. Estimate the per-step context budget breakdown.
34. How many steps/tool-calls does a typical task take, and why does it matter?

## 🏗️ Design variations
35. Redesign the same agent as an IDE extension instead of a CLI — what changes?
36. Design a multi-agent system that resolves GitHub issues end-to-end.
37. Add a semantic code-search index (hybrid retrieval) — design it.
38. Design an MCP / plugin system to extend the tool set safely.

## 🐞 Debugging & ops
39. The agent made a wrong, over-broad edit and broke the build. What went wrong, and how do you prevent it?
40. Tasks suddenly cost 5× more tokens than last week. Diagnose.
41. On long tasks the agent keeps "forgetting" earlier decisions. Why, and how do you fix it?
42. A user reports the agent ran a destructive command. Triage and prevent a recurrence.

---

> **How to practice:** anchor every answer on the **loop + the context budget** (LLM-in-a-loop with tools, window ≪ codebase) and the **safety boundary** (permissions · sandbox · injection defense). Check yourself against [answers.md](answers.md) and the [one-page cheat-sheet](cheat-sheet.md).

[← Back to Claude Code HLD](README.md) · [Answer key](answers.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
