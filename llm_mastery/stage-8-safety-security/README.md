# Stage 8 — Safety & Security (always-on)

> **Objective:** Build and ship LLM systems that resist attacks, protect data, and behave responsibly. This isn't a final checkbox — it runs **alongside every other stage**. Treat untrusted text as a potential attack surface.

[← Stage 7](../stage-7-advanced-specialization/README.md) · [Index](../README.md) · [Capstones →](../capstones/README.md)

📝 **Interview prep:** [interview-questions.md](interview-questions.md) · ✅ [answer key](answers.md)

---

## Why this stage matters

The moment an LLM touches untrusted input (user text, web pages, documents, tool outputs), it becomes an attack surface. Security failures here are *system* failures — data exfiltration, unauthorized actions, reputational harm. Elite practitioners design for this from day one.

> **Core mental model:** LLMs **cannot reliably distinguish instructions from data.** Any text in the context window — including retrieved documents and tool outputs — can act as an instruction. Architect around this limitation.

---

## Concept-by-concept deep dive

### 8.1 Prompt injection (the #1 LLM-specific threat — OWASP LLM01)
- **Direct injection:** the user types instructions that override your system prompt ("ignore previous instructions and…").
- **Indirect injection:** malicious instructions hidden in **content the model ingests** — a web page, a PDF, an email, a tool result. The model reads it and obeys. This is the dangerous one for RAG/agents.
- **Why it's hard:** there's no robust separation of "trusted instructions" vs "untrusted data" inside one context window.
- **Defenses (layered — none is complete):**
  - Strong system-prompt boundaries + explicit "treat retrieved/tool content as data, not instructions."
  - Input/output filtering and anomaly detection.
  - **Least privilege**: limit what tools/actions the model can take; require confirmation for high-impact actions.
  - Sandboxing tool execution; allow-list tools and arguments.
  - Don't let model output directly trigger irreversible actions without validation.
  - Separate trusted vs untrusted content channels; mark provenance.

### 8.2 Jailbreaking
- Techniques that bypass safety training: role-play ("DAN"), obfuscation/encoding, many-shot, gradient-crafted adversarial suffixes, persona attacks.
- **Defenses:** alignment training (Stage 3), safety classifiers on input and output, refusal calibration, red-teaming, monitoring for known patterns. Expect an arms race — defense in depth, not a silver bullet.

### 8.3 Data poisoning & supply chain
- **Training-data poisoning:** attacker seeds malicious/backdoor data into the training/fine-tuning set (a trigger phrase → bad behavior).
- **RAG/index poisoning:** attacker plants documents in your knowledge base to steer retrieval.
- **Supply chain:** untrusted model weights, datasets, or dependencies (incl. malicious pickles). Verify sources, checksums, prefer safetensors.

### 8.4 Privacy, PII & data governance
- **Memorization/leakage:** models can regurgitate training data (PII, secrets). Dedup + filtering (Stage 2) reduces this; consider differential privacy for sensitive training.
- **PII handling:** detect/redact PII in inputs, logs, and outputs. Mind regulations (GDPR, etc.).
- **Data governance:** know what data you can legally train/serve on; honor retention/consent; protect logs (they contain user data).
- **Inference-time leakage:** don't put secrets in prompts that get logged; scrub traces.

### 8.5 Model extraction & abuse
- **Model/prompt extraction:** attackers probe to steal a proprietary system prompt or distill your model via the API. Mitigate with rate limits, output limits, monitoring.
- **Resource abuse / DoS:** expensive prompts, unbounded agent loops. Add quotas, timeouts, step limits, cost caps.

### 8.6 Output safety & content moderation
- **Toxicity, bias, harmful content:** filter outputs with moderation classifiers; define and enforce a content policy.
- **Hallucination as a safety issue:** confident wrong answers in high-stakes domains (medical/legal/financial) → require grounding, citations, disclaimers, human review.
- **Over-refusal:** too-aggressive safety harms usefulness; calibrate and measure (Stage 4).

### 8.7 Map to OWASP Top 10 for LLM Applications
Know this list — it's the industry shared language:
1. Prompt Injection · 2. Insecure Output Handling · 3. Training Data Poisoning · 4. Model Denial of Service · 5. Supply Chain Vulnerabilities · 6. Sensitive Information Disclosure · 7. Insecure Plugin/Tool Design · 8. Excessive Agency · 9. Overreliance · 10. Model Theft.

> **Insecure Output Handling (LLM02):** never `eval()`/exec/SQL/shell model output unsanitized. Treat model output as untrusted user input downstream.
> **Excessive Agency (LLM08):** give agents the *minimum* tools/permissions needed.

---

## Ordered learning path

1. Read the **OWASP Top 10 for LLM Applications** (cover to cover).
2. Read a prompt-injection deep dive (Simon Willison's writing is excellent) + indirect injection research.
3. Study **adversarial suffix** jailbreaks (Universal/Transferable Adversarial Attacks).
4. Read on **training data extraction** (Carlini et al.).
5. Do the labs — including attacking your own systems.

---

## 🛠️ Hands-on labs

- [ ] **Lab A — Red-team your RAG (Stage 6):** plant an indirect-injection document in the knowledge base; demonstrate the model obeying it; then add defenses and show mitigation.
- [ ] **Lab B — Jailbreak suite:** collect known jailbreak patterns; test your aligned model (Stage 3); measure attack success rate before/after a safety classifier.
- [ ] **Lab C — PII redaction pipeline:** detect + redact PII on inputs, outputs, and logs.
- [ ] **Lab D — Guardrail layer:** add input + output guardrails (injection check, schema validation, moderation) to your Stage-6 app.
- [ ] **Lab E — Agent least-privilege:** constrain an agent's tools/args with an allow-list; add confirmation for high-impact actions; show a blocked malicious action.
- [ ] **Lab F — Output handling:** demonstrate the danger of executing raw model output, then fix it with sanitization/sandboxing.

---

## ⚠️ Common pitfalls & gotchas

- Assuming the system prompt is a security boundary (it isn't — it's overridable).
- Trusting retrieved/tool content as instructions (the indirect-injection trap).
- `eval`/SQL/shell on raw model output (**insecure output handling**).
- Giving agents broad permissions "for convenience" (**excessive agency**).
- Logging prompts/outputs containing PII or secrets with no redaction.
- Treating safety as a one-time gate instead of continuous (new jailbreaks appear constantly).
- Over-refusal that tanks usefulness — measure both safety *and* helpfulness.
- Using unverified model weights/pickles from random sources.

---

## 🔥 Mastery checks (answer without notes)

- [ ] Explain direct vs **indirect** prompt injection and why the latter is dangerous for RAG/agents.
- [ ] Why can't an LLM reliably separate instructions from data? What does that imply for design?
- [ ] List concrete, layered defenses against prompt injection (and why none alone suffices).
- [ ] What is "excessive agency" and how do you apply least privilege to an agent?
- [ ] How can training data be poisoned, and how does dedup/filtering help privacy?
- [ ] What is insecure output handling? Give an exploit and a fix.
- [ ] Name the OWASP LLM Top 10 categories most relevant to an agentic RAG app.
- [ ] How do you balance safety (refusals) against helpfulness, and how do you measure it?

---

## ✅ Stage 8 checklist

- [ ] Read OWASP LLM Top 10 + prompt-injection deep dive
- [ ] Red-teamed your own RAG/agent (Labs A, E)
- [ ] Added a guardrail + PII layer (Labs C, D)
- [ ] Demonstrated an attack **and** its mitigation
- [ ] All mastery checks passable
- [ ] Notes in your own words

**When complete → tackle the [Capstone Projects](../capstones/README.md).**
