# Module 10 · AI Infrastructure, Production Monitoring & Safety

> **Goal:** Operate LLM/agent systems in production: MLOps/LLMOps, observability and monitoring tuned for LLMs, reliability/SRE practices, and the safety/security systems that frontier labs require. This is the "make it real and keep it safe" module.

**Duration:** ~4 weeks. **Prereqs:** [Module 05](05-distributed-systems-and-inference.md), [Module 09](09-evaluations.md).

---

## 10.1 AI infrastructure & LLMOps

- The lifecycle: data → train → eval → deploy → monitor → iterate
- Model & artifact registries, versioning (models, prompts, datasets, configs)
- Reproducibility: pinned envs, seeds, data lineage
- Experiment tracking (W&B/MLflow), feature/embedding stores
- CI/CD for models and prompts (eval gates from [Module 09](09-evaluations.md))
- Infra as code (Terraform), containers, **Kubernetes** for GPU workloads, autoscaling
- Serving orchestration: KServe, Ray Serve, BentoML, Triton; canary/blue-green/shadow deploys
- GPU fleet management: scheduling, MIG, spot/preemptible, capacity reservations
- Storage & data pipelines: object stores, parallel FS, streaming datasets

## 10.2 Observability & production monitoring

### System metrics (the SRE layer)
- Latency: **TTFT, TPOT/ITL**, end-to-end p50/p95/p99 (tail latency is the SLA)
- Throughput: tokens/sec, requests/sec, concurrency, queue depth
- Utilization: GPU compute %, **HBM memory**, KV-cache occupancy, batch size
- Errors: timeouts, OOMs, rate limits, 5xx; saturation & queueing
- The four golden signals: latency, traffic, errors, saturation

### LLM-specific observability
- **Tracing** every request/agent trajectory: prompts, tool calls, retrievals, tokens, cost, latency per step (OpenTelemetry GenAI conventions)
- Token & **cost accounting** per request/user/feature
- Output quality monitoring: online evals, guardrail metrics, sampled LLM-as-judge
- **Drift detection:** input distribution shift, embedding drift, response-quality drift
- Hallucination/groundedness monitoring, refusal rates, user feedback (thumbs, regenerations)
- Tooling: LangSmith, Arize Phoenix, Langfuse, Helicone, WhyLabs, Datadog LLM Observability

> **Build:** Add end-to-end tracing + a metrics dashboard (Prometheus/Grafana or an LLM-obs tool) to your Module 07 agent / Module 06 RAG service. Track latency percentiles, token cost, and a sampled quality score. Create alerts on TTFT p95 and cost spikes.

## 10.3 Reliability & SRE for LLM systems

- SLIs/SLOs/SLAs, error budgets
- Graceful degradation: fallback models, cached/canned responses, load shedding
- Rate limiting, quotas, admission control, backpressure
- Timeouts, retries with jitter, circuit breakers, idempotency
- Capacity headroom, autoscaling on the right signal (queue depth, not just CPU)
- Incident response, runbooks, postmortems, on-call
- Rollouts: canary + automatic rollback gated on quality/safety metrics

## 10.4 Safety systems (frontier-lab critical)

### Guardrails & content safety
- Input/output classifiers (toxicity, PII, CSAM, violence, self-harm), moderation APIs
- Policy/usage enforcement, jailbreak & prompt-injection detection
- Layered defense: pre-prompt filtering, system-prompt hardening, output filtering, action gating
- Refusal calibration (avoid over- and under-refusal)

### Agent & tool security (OWASP LLM Top 10)
- **Prompt injection** (direct & indirect via retrieved/tool content) — the #1 risk
- Insecure tool use, excessive agency, confused-deputy, data exfiltration
- **Sandboxing** code/tool execution, least privilege, allow-lists, human-in-the-loop for high-impact actions
- Spend/step limits, output validation, secrets handling
- Supply-chain & model-provenance risks

### Alignment & governance
- Anthropic **Responsible Scaling Policy (RSP)** / ASL levels; OpenAI **Preparedness Framework**
- Dangerous-capability evals & deployment gating (ties to [Module 09](09-evaluations.md))
- Model cards, system cards, usage policies, audit logging
- Privacy/compliance: data retention, PII handling, GDPR; tenant isolation

> **Build:** Add a safety layer to your agent: input/output moderation, a prompt-injection detector, a sandboxed tool executor with allow-lists, spend/step caps, and human approval for dangerous actions. Write an attacker's-eye threat model and show your mitigations.

## 10.5 The full production architecture (synthesis)

Put it together: client → gateway (auth, rate limit) → router/load balancer → safety pre-filters → inference cluster (vLLM/TRT-LLM, replicas, autoscaling) → tools/retrieval → safety post-filters → response, with tracing/metrics/logging throughout and an offline eval/feedback loop feeding retraining.

---

## Module 10 capstone — **Productionize and harden**

1. Containerized deployment of your RAG/agent service on Kubernetes (or a managed equivalent) with autoscaling.
2. Full observability: tracing, latency percentiles, token/cost dashboards, drift & quality monitoring, alerts.
3. Reliability: SLOs, fallbacks, rate limiting, retries/circuit breakers; demonstrate graceful degradation under load/failure.
4. Safety layer: moderation, prompt-injection defense, sandboxed tools, spend caps, audit logs; plus a written threat model.
5. A one-page production architecture diagram + runbook.

## Exit criteria
- [ ] You can instrument LLM-specific observability (TTFT/TPOT, cost, quality, drift) and set meaningful alerts.
- [ ] You can define SLOs and build graceful degradation/reliability mechanisms.
- [ ] You can threat-model an agent and implement layered safety (incl. prompt-injection defense + sandboxing).
- [ ] You can describe RSP/Preparedness-style capability gating.

## Core sources
- *Site Reliability Engineering* (Google SRE book) — relevant chapters
- OWASP **Top 10 for LLM Applications**
- Anthropic **Responsible Scaling Policy**; OpenAI **Preparedness Framework**
- OpenTelemetry **GenAI** semantic conventions
- NIST AI Risk Management Framework
- *Designing ML Systems* — Chip Huyen (ops chapters)
