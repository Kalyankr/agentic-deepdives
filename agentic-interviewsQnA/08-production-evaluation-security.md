# 08 — Production, Evaluation & Security

> Goal: take agents from demo to production — evaluate them, observe them, guardrail them, secure them, and control cost/latency.

---

## 8.1 Why production agents are hard

⚠️ Agents are **non-deterministic, multi-step, and tool-using**, so:
- A single wrong step compounds across a trajectory (errors multiply).
- The same input can produce different paths.
- Failures are emergent and hard to reproduce.
- Cost/latency vary wildly per request.

> 💡 The hardest part of agents in production isn't building the loop — it's **evaluating, observing, and constraining** it.

---

## 8.2 Evaluation (the most-asked production topic)

### Two levels of evaluation
1. **Outcome / final-response eval** — did it achieve the goal? (task success, correctness, quality).
2. **Trajectory / process eval** — *how* did it get there? (right tools, right order, no wasted steps, no loops).

### What to measure
- **Task success rate** — % of tasks completed correctly (the headline metric).
- **Tool-use accuracy** — correct tool selected, correct args, valid sequence.
- **Faithfulness/groundedness** — answers supported by sources (for RAG).
- **Hallucination rate**.
- **Efficiency** — #steps, #tokens, latency, $/task vs. an "ideal" path.
- **Safety** — policy violations, unsafe actions, injection susceptibility.
- **Quality** — rubric scores (helpfulness, coherence), CSAT.

### How to measure
- **Reference-based** — compare to gold answers/trajectories (exact match, F1, etc.).
- **LLM-as-judge** — a model scores outputs against a rubric. Cheap, scalable; validate against human labels, watch for bias (position, verbosity, self-preference). Use pairwise comparisons + clear rubrics.
- **Human eval** — gold standard for nuanced quality; expensive; use for calibration.
- **Programmatic checks** — did tests pass? schema valid? Did the action succeed? (Best when available.)
- **Benchmarks** — τ-bench/τ²-bench (tool-agent-user), SWE-bench (coding), WebArena (web), GAIA, AgentBench, BFCL (function calling), RAGAS (RAG). Name a couple.

### Offline vs. online
- **Offline:** curated eval set / regression suite run in CI on every change. Build this early; grow it from real failures.
- **Online:** A/B tests, canary, real-time metrics, user feedback, guardrail-violation rates.

### RAG-specific metrics
- **Context precision/recall**, **faithfulness**, **answer relevance** (e.g., RAGAS). Evaluate retrieval and generation separately.

🎯 *"How do you evaluate an agent?"* — Hit: outcome **and** trajectory metrics; task success as headline; LLM-as-judge + programmatic checks + human calibration; offline regression set built from real failures + online A/B; per-step tracing to localize failures.

---

## 8.3 Observability & tracing

You cannot improve what you can't see. **Trace every step** of every trajectory:
- Inputs/outputs of each LLM call (prompt, completion, tokens, cost, latency).
- Each tool call: args, result, errors, duration.
- The full decision path (thoughts/actions/observations), retries, and final outcome.
- Per-agent attribution in multi-agent systems.

Tools: **LangSmith, Langfuse, Arize Phoenix, AgentOps, Helicone, W&B Weave**, and **OpenTelemetry GenAI** semantic conventions (vendor-neutral; increasingly standard).

What to monitor in prod: task success, step count, latency p50/p95, $/request, tool error rates, loop/repeat rate, guardrail triggers, drift in inputs/outputs.

💡 Mention **OpenTelemetry** for spans/traces and a dedicated LLM-observability tool for prompt/cost analytics — shows production maturity.

---

## 8.4 Guardrails

Layered controls around the agent:

**Input guardrails**
- Prompt-injection / jailbreak detection.
- PII detection/redaction; topic/policy filters; off-topic rejection.
- Schema/auth validation.

**Output guardrails**
- Schema validation (structured output must parse).
- Safety/toxicity/compliance checks; PII leakage check.
- Groundedness/citation check (no unsupported claims).
- Action validation — is this tool call allowed for this user/context?

**Behavioral guardrails**
- Max steps, max tokens, wall-clock timeout, $ budget per task.
- Allow-lists for tools/domains; rate limits.
- **HITL approval** before irreversible/high-risk actions.

Tools: NeMo Guardrails, Guardrails AI, Llama Guard, provider moderation APIs. Validate in **code**, not just by asking the model nicely.

---

## 8.5 Security — the OWASP angle (high-value in interviews)

Agents massively expand attack surface. Know the **OWASP Top 10 for LLM Applications** highlights:

| Risk | What it is | Mitigation |
|------|-----------|------------|
| **Prompt injection** (LLM01) | Malicious instructions in user input or **retrieved/tool content** override the agent | Treat all external content as untrusted; separate instructions from data; input filters; least privilege; don't let tool output change system policy |
| **Excessive agency** (LLM06) | Agent has too many permissions/autonomy → harmful actions | Least-privilege tools, scoped credentials, HITL for high-risk, allow-lists, confirmation |
| **Sensitive info disclosure** | Leaking PII/secrets in outputs or logs | Redaction, output filters, scrub logs, RBAC |
| **Supply chain / insecure tools & MCP** | Malicious/compromised tools, plugins, MCP servers | Vet servers, pin versions, sandbox, sign/verify |
| **Insecure output handling** | Trusting LLM output executed downstream (SQL, shell, code) | Validate/escape outputs; never `eval` raw; parameterize queries |
| **Data/model poisoning** | Poisoned training/RAG data | Source vetting, content provenance, anomaly checks |
| **Unbounded consumption** | Cost/DoS via runaway loops | Budgets, rate limits, timeouts |

💡 **Indirect prompt injection** is the signature agent threat: an attacker plants instructions in a web page/email/document the agent will read, hijacking it. Defenses: untrusted-content isolation, least privilege, dual-LLM/sandbox patterns, and **never executing destructive actions without validation/HITL**.

The **"lethal trifecta"** (Simon Willison) — be ready to name it: an agent with (1) access to **private data**, (2) exposure to **untrusted content**, and (3) ability to **externally communicate/exfiltrate** is uniquely dangerous. Break at least one leg.

---

## 8.6 Reliability & cost/latency engineering

**Reliability**
- Retries with backoff; idempotent actions; circuit breakers.
- Checkpoint/resume long trajectories.
- Graceful degradation & fallbacks (cheaper model, cached answer, human).
- Determinism in code where possible (validation, routing, math).

**Cost control**
- **Model routing** — cheap model for easy steps, frontier for hard ones.
- **Caching** — prompt/semantic caching; cache tool results; provider prompt caching for static context.
- **Limit steps/tokens**; summarize/compact context; trim tool outputs.
- **Parallelize** independent work to cut latency (not cost).
- **Batch** where possible; stream partial output for perceived latency.
- Track **$/task** as a first-class metric.

**Latency**
- Stream tokens; show progress for long tasks; speculative/parallel tool calls; smaller models for routing; pre-warm/caches.

---

## 8.7 Human-in-the-loop (HITL)

Patterns:
- **Approve/reject** before an action executes (irreversible/high-risk).
- **Edit** the agent's plan or output before continuing.
- **Escalation** on low confidence / repeated failure / sensitive case.
- **Feedback capture** for eval and continuous improvement.

LangGraph-style interrupts/checkpointing make HITL pausable/resumable. Always gate destructive tools behind HITL or strong validation.

---

## 8.8 Deployment & lifecycle

- **Versioning** — prompts, tools, models, and graphs are all versioned; changes go through the eval/regression suite (treat prompts like code).
- **CI for agents** — run the offline eval set on every PR; block on regressions.
- **Canary/A-B** rollout; monitor online metrics; fast rollback.
- **Continuous improvement** — mine production failures → add to eval set → fix → re-eval (data flywheel).
- **Feedback loops** — user thumbs/edits feed eval and (optionally) fine-tuning/distillation.

---

## 8.9 Common production failure modes (quick list)

Loops/non-termination • cascading errors in multi-agent • context overflow/"lost in the middle" • tool selection errors • hallucinated tool args • prompt injection via tool/web content • cost spikes • latency tail (p95) • silent quality drift • non-reproducible failures (no tracing). For each, have a one-line mitigation ready.

---

## Interview questions for this chapter

1. How do you evaluate an agent? Outcome vs. trajectory metrics. *(8.2)*
2. Pros/cons of LLM-as-judge; how do you trust it? *(8.2)*
3. What do you trace, and which tools do you use? *(8.3)*
4. Explain prompt injection (direct vs. indirect) and defenses. *(8.5)*
5. What is "excessive agency" and how do you mitigate it? *(8.5)*
6. Name the "lethal trifecta." *(8.5)*
7. How do you cut an agent's cost without hurting quality? *(8.6)*
8. How do you stop an agent from looping forever? *(8.4)*
9. Where do you put humans in the loop, and how? *(8.7)*
10. How do you run CI/CD for a prompt/agent change? *(8.8)*

**Model answer to #4:** Prompt injection is when malicious instructions enter the model's context and override intended behavior. *Direct* = the user types them; *indirect* = they're hidden in content the agent retrieves (web page, email, doc, tool output) — the signature agent threat. Defenses: treat all external content as untrusted and never let it alter system instructions; separate instructions from data; filter/scan inputs; enforce least-privilege tools and scoped credentials; require validation/HITL before destructive or exfiltrating actions; and break the "lethal trifecta" by isolating private data, untrusted content, and outbound communication so no single path has all three.
