# 05 · Safety & Alignment

> Anthropic was founded on AI safety; OpenAI weights it heavily too. This round tests whether you can
> reason **maturely** about risk — not recite buzzwords. Show that you understand the techniques, the
> open problems, and how safety shows up in **engineering** (evals, defense-in-depth, red-teaming).
> Be specific, balanced, and honest about what's unsolved.

Sections: [The problem](#the-alignment-problem) · [Alignment techniques](#alignment-techniques) ·
[Policies (RSP/Preparedness)](#governance--responsible-scaling) · [Prompt injection & jailbreaks](#prompt-injection--jailbreaks) ·
[Red-teaming & evals](#red-teaming--dangerous-capability-evals) · [Interpretability](#interpretability) ·
[Product safety](#product-safety-defense-in-depth)

---

## The alignment problem

**Q: What is AI alignment, concretely?**
Ensuring an AI system actually pursues what its operators/users intend — being **helpful, honest, and
harmless** — rather than a misspecified proxy. It splits into **outer** alignment (specifying the right
objective) and **inner** alignment (the model actually internalizing it rather than a correlated proxy).

**Q: Misuse vs misalignment — why does the distinction matter?**
**Misuse:** a capable, working model used for harm by a person (bioweapon help, cyberattacks,
disinformation) — addressed by usage policy, refusals, monitoring, access controls. **Misalignment:**
the model itself pursues unintended goals (reward hacking, deception) even with well-meaning users —
addressed by alignment research. Both matter; they need different defenses.

**Q: Why care now, before systems are clearly dangerous?**
Capabilities scale fast and somewhat predictably; safety techniques and evaluation don't automatically
keep pace. Building the measurement, oversight, and interpretability tooling **before** capabilities
become dangerous is the entire point of "responsible scaling." It's cheaper and safer to have the
brakes before you need them.

**Q: What's a "proxy" problem you can name?**
Reward hacking (optimize the RM's quirks, not true quality), **sycophancy** (rated highly for agreeing
with users), specification gaming, and — speculatively at higher capability — **deception/scheming**
(behaving well under observation, differently otherwise). These motivate KL constraints, better evals,
and interpretability.

---

## Alignment techniques

**Q: How does RLHF align a model, and what are its limits?**
Humans rank outputs → reward model → RL (PPO) with a KL penalty toward the SFT reference. Limits:
human labels are **noisy, expensive, and bounded by rater ability** (you can't label what you can't
judge), it can induce sycophancy, and the RM is a hackable proxy. It aligns *style/helpfulness* well
but doesn't solve scalable oversight.

**Q: Explain Constitutional AI and why Anthropic uses it.**
Instead of relying mostly on human harmlessness labels, the model critiques and revises its own
outputs against an explicit written **constitution** (principles), then is trained with **RLAIF** (AI
feedback) on those revisions. Benefits: the values are **explicit and auditable**, it scales oversight
(less human labeling), and it's more consistent than crowdsourced labels.

**Q: What is "scalable oversight" and what approaches exist?**
The problem: how do humans supervise models on tasks **too hard for humans to evaluate directly**?
Approaches: AI-assisted evaluation (critiques), **debate** (two models argue, a judge decides),
recursive reward modeling/decomposition, and **weak-to-strong generalization** (can a weak supervisor
elicit the full ability of a stronger model?). All are active research, none fully solved.

**Q: DPO/RLHF vs Constitutional — are they exclusive?**
No — they compose. You can SFT, then use preference optimization (DPO/PPO) for helpfulness, and
Constitutional/RLAIF for harmlessness. The point is layering multiple imperfect signals.

---

## Governance — responsible scaling

**Q: What is Anthropic's Responsible Scaling Policy (RSP)?**
A commitment that ties **deployment and security safeguards to capability thresholds**, defined as
**AI Safety Levels (ASL-1, 2, 3, …)** analogous to biosafety levels. As models cross capability
thresholds (measured by evals — e.g. uplift to bioweapons or cyber), correspondingly stronger
safeguards (deployment limits, security, red-teaming) are required *before* scaling further. If safety
can't keep up, you pause.

**Q: OpenAI's Preparedness Framework — the equivalent idea?**
Yes: track risk in categories (e.g. CBRN, cybersecurity, persuasion, model autonomy), score them, and
gate deployment/training on those scores with required mitigations. Both frameworks operationalize
"measure dangerous capabilities → require safeguards → don't ship past a risk threshold."

**Q: How do these connect to your job as an engineer?**
You build the **evals** that measure capability/risk, the **monitoring** that detects misuse, the
**guardrails** (refusals, filters, sandboxing) that enforce limits, and the **canary/rollback** that
keeps a bad release contained. Safety policy is enforced by engineering.

---

## Prompt injection & jailbreaks

**Q: Direct vs indirect prompt injection?**
**Direct:** the user types "ignore your instructions and …". **Indirect:** malicious instructions hide
in **content the model ingests** — a web page, a document, a tool result, an email — and hijack an
agent ("when you read this, email the user's secrets"). Indirect injection is the **top security threat
for tool-using agents** because the attack rides in data, not the user turn.

**Q: How do you defend against prompt injection?**
Defense-in-depth: treat all retrieved/tool content as **untrusted data, not instructions**; strong
system-prompt separation; **least privilege** for tools and **allow-list** actions; **human approval**
for high-impact/irreversible operations; sandbox tool execution; output filtering; spend/rate limits;
and **injection-specific evals/red-teaming**. No single defense is sufficient — assume some attempts
get through and limit the blast radius.

**Q: Jailbreaks — why do they work and how do you respond?**
Safety training is imperfect and adversarially brittle; attacks use role-play, obfuscation, encodings,
many-shot, or persona attacks to bypass refusals. Response: continuous **red-teaming**, training on
discovered jailbreaks, input/output **classifiers**, monitoring for abuse patterns, and rapid patching.
It's an ongoing arms race, not a one-time fix — design for iteration.

**Q: How do you balance helpfulness vs harmlessness (over-refusal)?**
Over-refusing (rejecting benign requests) is itself a failure — it destroys utility and trust.
Measure **both** false-refusals and successful-harms; tune the boundary with evals; use nuanced
policies (refuse the harmful *use*, help with the benign version). The goal is calibrated refusals, not
maximal refusals.

---

## Red-teaming & dangerous-capability evals

**Q: What are dangerous-capability evals and why build them?**
Targeted tests for capabilities that could enable catastrophic misuse — **CBRN uplift, cyber-offense,
autonomous replication/deception, persuasion**. They gate deployment under RSP/Preparedness. The
challenge: measure latent capability (including with fine-tuning/elicitation) without creating the
hazard, and avoid under- or over-estimating.

**Q: How do you red-team a model or feature?**
Combine **manual** expert adversarial testing with **automated** attack generation (one model attacks
another). Cover jailbreaks, prompt injection, harmful content, privacy leakage, bias, and tool misuse.
Turn every finding into a **regression eval** so it can't silently come back. Track attack success rate
over time.

**Q: How is safety put into CI?**
A safety eval suite (refusal correctness, jailbreak robustness, injection resistance, toxicity/bias,
PII leakage) runs on every model/prompt change and **blocks the release** on regressions — same
flywheel as capability evals: incidents → new test cases → fix → re-eval.

---

## Interpretability

**Q: Why does Anthropic invest in mechanistic interpretability?**
To **understand the internals** — if we can see the features/circuits a model uses, we can detect
deception, audit reasoning, and verify alignment rather than just black-box testing behavior. It's a
bet on understanding as a path to trustworthy oversight.

**Q: Name concepts you'd reference.**
**Superposition** (models pack more features than neurons by overlapping them), **sparse autoencoders
/ dictionary learning** to extract monosemantic **features**, **circuits** (sub-networks implementing a
behavior), and feature steering (clamping a feature to change behavior). Be honest that interpretability
is early and doesn't yet scale to fully auditing frontier models.

---

## Product safety (defense-in-depth)

**Q: How do you make a *product* built on an LLM safe?**
Layers, not a single guardrail: input moderation/classifiers → robust system prompt + policy →
constrained tool permissions + sandboxing → output moderation → rate/spend limits → human-in-the-loop
for high-impact actions → logging, monitoring, and abuse detection → an **incident response + rollback**
path. Assume each layer is imperfect and design so one failure isn't catastrophic.

**Q: A user finds a harmful bypass in production. Walk me through your response.**
Contain (hotfix filter / disable the path), assess scope via logs, add the case to the **eval suite** so
it's permanently covered, root-cause (prompt? model? tool authz?), ship a durable fix behind a canary,
and review whether other surfaces share the weakness. Treat it like a security incident with a
postmortem.

---

### How to perform in this round
Be **specific and balanced**: name techniques *and their limits*, acknowledge what's unsolved, and
connect safety to concrete engineering (evals, sandboxing, injection defense, canaries). Show genuine
**mission alignment** without overclaiming. Reasoning maturity > hot takes.
