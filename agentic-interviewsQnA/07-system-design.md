# 07 — Agentic System Design

> Goal: drive an agentic-AI system-design interview with a **repeatable framework**, then apply it to several worked case studies.

---

## 7.1 The interview framework ( A-G-E-N-T-S)

Use this structure out loud so the interviewer can follow you. ~35–45 min flow:

| Step | What you do | Time |
|------|-------------|------|
| **A — Align on requirements** | Clarify goal, users, scope, success metrics, constraints (latency, cost, accuracy, compliance), scale | 5 min |
| **G — Ground rules & risk** | Failure cost, reversibility, HITL needs, safety/compliance, data sensitivity | 3 min |
| **E — Establish the approach** | Workflow vs. single agent vs. multi-agent? Justify the *simplest* design that works | 4 min |
| **N — Nodes & components** | Draw the architecture: models, tools, memory, orchestration, data stores, services | 10 min |
| **T — Tools, memory, context** | Define tools/APIs, memory (short/long, RAG), context strategy | 6 min |
| **S — Safeguards, eval, scale** | Guardrails, evaluation, observability, cost/latency, scaling, deployment | 8 min |
| — Wrap | Trade-offs, what you'd build first (MVP), what you'd monitor | 3 min |

💡 Always **start simple and add agency only as needed**, narrate trade-offs, and tie back to the success metrics.

---

## 7.2 The requirements checklist (ask these)

- **Goal & scope:** what exactly should it accomplish? In/out of scope?
- **Users & volume:** who, how many, QPS, concurrency, peak load?
- **Success metrics:** task success rate, accuracy, latency p50/p95, cost/task, CSAT?
- **Latency budget:** interactive (<2s feel) vs. async (minutes OK)?
- **Cost budget:** $/task ceiling?
- **Accuracy/risk:** cost of a wrong action? Reversible? Regulated?
- **Data:** sources, freshness, privacy/PII, access control.
- **Autonomy:** fully auto vs. HITL approval? Where are the checkpoints?
- **Integrations:** which systems/APIs/tools?

⚠️ Jumping to architecture without these is the #1 way to fail. Spend the first 5 minutes here.

---

## 7.3 Reference architecture (the building blocks to draw)

```
        ┌──────────────────────── Client / API / Channel ───────────────────────┐
        │  (web, chat, voice, email, Slack)                                       │
        └───────────────┬─────────────────────────────────────────────────────────┘
                        ▼
              ┌───────────────────┐    guardrails (input)   ┌──────────────────┐
   request ─▶ │   Gateway / Auth  │ ───────────────────────▶│   Orchestrator    │
              └───────────────────┘                          │  (planner / router)│
                                                             └───┬───────┬───────┘
                          ┌──────────────────────────────────────┘       │ delegate
                          ▼                                               ▼
                  ┌───────────────┐                              ┌────────────────┐
                  │  LLM / Model  │◀── prompts/context ──────────│  Worker agents  │
                  │  router (cheap│                              │ (specialists)   │
                  │  vs strong)   │                              └───────┬────────┘
                  └──────┬────────┘                                      │ tool calls
                         │ tool calls                                    ▼
                         ▼                                       ┌────────────────┐
                  ┌──────────────┐   ┌──────────────┐            │     Tools       │
                  │ Tool layer / │   │ Memory layer │            │ APIs, code-exec,│
                  │ MCP servers  │   │ vector + SQL │            │ search, RAG     │
                  └──────┬───────┘   └──────┬───────┘            └────────────────┘
                         ▼                  ▼
                  ┌──────────────┐   ┌──────────────┐
                  │ External APIs│   │  Data stores │
                  └──────────────┘   └──────────────┘

   Cross-cutting:  Guardrails (in/out)  •  Observability/Tracing  •  Eval harness
                   Caching  •  Rate limits/budgets  •  HITL approval  •  Secrets/RBAC
```

Components you should mention by name:
- **Gateway/auth**, **orchestrator/router**, **model router** (route by difficulty/cost), **tool layer (MCP)**, **memory (vector + relational + cache)**, **guardrails**, **observability/tracing**, **eval harness**, **HITL**, **queue/worker** for async/long tasks.

---

## 7.4 Cross-cutting design decisions

- **Workflow vs. agent vs. multi-agent** — justify with task coupling, parallelism, and context pressure (Ch 01/05).
- **Model strategy** — small/cheap model for routing & simple steps, frontier model for hard reasoning; cache; consider fine-tuning/distillation later.
- **Sync vs. async** — long-running agents → queue + worker + status/streaming, not a blocking request.
- **State & durability** — checkpoint trajectories so you can resume, debug, and do HITL (LangGraph-style persistence).
- **Determinism where possible** — put non-LLM logic (validation, routing rules, math) in code, not the model.
- **Guardrails** — input (injection, PII, policy) and output (schema, safety, grounding) — Ch 08.
- **Cost/latency** — parallelize independent steps, cap steps/tokens, cache, stream partial output.
- **Scaling** — stateless workers behind a queue; autoscale; isolate tool sandboxes; backpressure & rate limits.
- **Observability & eval** — trace every step; offline eval set + online metrics; LLM-as-judge — Ch 08.

---

## 7.5 Worked case study A — Customer Support Agent

**A — Requirements:** resolve customer tickets (billing, tech, account) over chat; deflect L1 volume; escalate safely; <3s first token; must not take destructive account actions without confirmation; integrate CRM, billing, knowledge base; multilingual.

**E — Approach:** **Router + handoff multi-agent** (triage → specialist), single-agent per domain. Not a giant single agent (distinct tools/policies per domain, context isolation), not a heavy debate system (latency/cost).

**N — Architecture:**
```
User → Gateway/auth → Triage agent (intent + language + sentiment)
   ├── Billing agent  (tools: get_invoice, issue_refund*, payment_status)
   ├── Tech agent     (tools: KB RAG search, run_diagnostic, create_bug)
   ├── Account agent  (tools: get_account, update_email*, reset_pw*)
   └── Human escalation (HITL) for low confidence / angry / high-risk
  * destructive tools require confirmation + policy guardrail
```

**T — Tools/memory/context:** RAG over help-center docs (hybrid search + rerank, with citations); per-user memory (profile, ticket history) in SQL; conversation memory with summarization; tools wrapped via MCP.

**S — Safeguards/eval/scale:**
- Guardrails: prompt-injection filter on user text, PII redaction in logs, **confirmation + authz** before refunds/password resets (excessive-agency control), output safety check.
- Eval: offline test set of tickets → resolution accuracy, escalation precision/recall, hallucination rate, CSAT; online: deflection rate, handle time, reopen rate.
- Observability: full trace per ticket; alert on loops/cost spikes.
- Scale: stateless agents behind a queue; cache KB embeddings; autoscale.
- **MVP first:** single FAQ RAG agent + handoff to human; add specialists once metrics are in place.

**Trade-offs:** handoff topology keeps each domain's context clean and policies enforceable, at the cost of routing errors — mitigated by a confidence threshold that falls back to human.

---

## 7.6 Worked case study B — Deep Research Assistant

**A — Requirements:** given a question, produce a cited report by searching the web + internal docs; thorough over fast (async, minutes OK); must cite sources; budget per report.

**E — Approach:** **Orchestrator-worker multi-agent** — subtasks (sub-questions/sources) are **independent & parallel**, each needs an **isolated context** → classic MAS win (Anthropic's research system).

**N — Architecture:**
```
Lead orchestrator: decompose question → spawn N research workers (parallel)
Research worker ×N: search → fetch → extract → summarize (own context) → structured brief
Citation/verify agent: check each claim is supported; drop/flag unsupported
Writer/synthesizer: merge briefs → report with inline citations
Orchestrator: quality gate (critic) → maybe re-dispatch gaps → deliver
```

**T:** tools = web_search, fetch_url, internal RAG, code-exec (for data/charts); memory = scratchpad per worker + shared findings store; context isolation per worker is the whole point.

**S:**
- Budgets: cap workers, depth, total tokens; early-stop when marginal info is low.
- Eval: factuality/citation accuracy (LLM-as-judge + source-check), coverage, report quality rubric; trajectory eval for wasted steps.
- Risk: prompt injection from fetched web pages → treat tool/web content as **untrusted**, sandbox, don't let it override system instructions.
- Cost: parallelism lowers latency but multiplies tokens — monitor $/report.

**Trade-offs:** parallel workers cut latency and isolate context but cost more tokens; a critic/verify stage adds latency but is essential for citations.

---

## 7.7 Worked case study C — Coding Agent (SWE)

**A — Requirements:** take an issue, edit a repo, run tests, open a PR; correctness critical; tests are the success signal; sandboxed.

**E — Approach:** **Single agent (ReAct + reflection)**, *single-threaded* — coding is tightly coupled, so context fragmentation from naive multi-agent hurts (Cognition's argument). Optionally a separate **reviewer** agent on the final diff.

**N/T:** tools = read_file, edit_file, run_tests, run_shell (sandbox), grep, git; memory = repo map + working set of files + summarized history; reflection loop on test failures (Reflexion-style: read error → revise).

**S:** sandbox with least privilege; cap steps/cost; never push without passing tests + HITL approval on the PR; eval = % issues resolved (e.g., SWE-bench-style), test pass rate, regression rate; trace every edit/command.

**Trade-offs:** single-threaded reliability vs. speed; add a reviewer for quality at extra cost; reflection improves pass rate but adds iterations.

---

## 7.8 Worked case study D — Voice Ordering / Transactional Agent

**A:** phone/voice agent takes restaurant orders; low latency critical; must confirm before charging; integrates menu + payments.

**E:** single agent + strict tool schemas + **mandatory confirmation HITL** before payment (irreversible action). Streaming for low latency.

**Highlights:** STT → agent → TTS pipeline; structured order state (not free chat) to avoid errors; guardrails on price/quantity; idempotent payment with confirmation; fallbacks to human on confusion; eval on order accuracy and completion rate.

---

## 7.9 Things that impress interviewers

✅ Start with requirements & success metrics.
✅ Choose the **simplest** sufficient design; justify any agency.
✅ Name **specific** components (router, MCP tool layer, vector + SQL memory, guardrails, tracing, eval harness, queue/worker).
✅ Call out **failure modes** and mitigations proactively.
✅ Discuss **evaluation** and **observability** unprompted.
✅ Address **security** (prompt injection, excessive agency, sandboxing) and **HITL** for irreversible actions.
✅ Give an **MVP-first** rollout and what you'd monitor.
✅ Quantify **cost/latency** trade-offs.

⚠️ Pitfalls: jumping to "multi-agent" by default; ignoring eval/observability; no guardrails on destructive tools; unbounded loops; stuffing everything into one context; hand-waving "the LLM handles it."

---

## Interview prompts to rehearse (full design)

1. Design an agent that books complex multi-leg travel within budget and policy.
2. Design a multi-agent system to automate a marketing content pipeline.
3. Design an agent that monitors infra alerts and remediates (with safety).
4. Design an enterprise "chat with your data" agent over 10M documents.
5. Design an email-triage-and-draft assistant for an exec.
6. Design a financial-analysis agent producing cited investment memos.
7. Design a healthcare intake agent (note the compliance/HITL constraints).

For each: run **A-G-E-N-T-S**, draw the reference architecture, and finish with eval + safeguards + MVP.
