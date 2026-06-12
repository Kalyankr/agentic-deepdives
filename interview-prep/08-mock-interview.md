# 08 · Mock Interview — Self-Run Loop, Rubrics & Checklists

> The bank ([01](01-coding.md)–[07](07-rapid-fire.md)) is the *material*. This file is the
> **rehearsal harness**: how to run realistic mocks, score yourself like an interviewer, and
> walk in on the day with nothing left to chance.

Knowing the answer and *performing* under time pressure are different skills. Frontier-lab loops
reward **judgment + communication** as much as correctness — that only improves under simulation.
Run these mocks out loud, timeboxed, and recorded.

---

## How to run a mock

**Solo (most days):** set a timer, talk to a webcam/voice recorder as if to an interviewer, no
pausing, no peeking. Watch the recording back against the rubric. The recording is where the
learning is — you'll catch filler words, buried ledes, and skipped assumptions.

**With a partner (weekly):** swap roles. The interviewer stays quiet, gives only the prompt and
one hint if truly stuck, and scores on the rubric. Being the interviewer is itself great prep.

**With an LLM (anytime):** paste a prompt below into a model and tell it: *"You are my interviewer.
Ask one question, stay silent unless I ask for a hint, then score me 1–4 on each rubric dimension
with specific evidence."* Use a different model than your daily driver to avoid style collusion.

> Golden rule: **never stop the clock**. If you blank, say what you'd do to get unstuck and keep
> moving — that's exactly what a real interviewer wants to see.

---

## The full-loop simulation (one sitting ≈ 3.5 h)

Run this once a week in Weeks 3–4. Mirror the real onsite order and take only 5-minute breaks.

| Round | Time | Source | What "pass" looks like |
|------|------|--------|------------------------|
| Coding | 45 min | [01](01-coding.md) | Working code, tested, narrated; clean complexity |
| ML/LLM depth | 45 min | [02](02-ml-and-llm-depth.md) | Derivations from memory; numbers; trade-offs named |
| System design | 45 min | [03](03-system-design.md) | End-to-end design with capacity math + bottlenecks |
| Applied / deep-dive | 45 min | [04](04-applied-llm.md) | Concrete build; evals; cost/latency; failure modes |
| Behavioral + safety | 45 min | [05](05-safety-alignment.md), [06](06-behavioral-mission.md) | Crisp STAR stories; genuine mission fit; safety reflex |

After each round, score yourself (rubric below) **before** moving on. Don't binge-fix mid-loop —
capture notes and debrief at the end, like a real candidate would.

---

## Per-round playbooks

### Coding round (45 min)
- **First 5 min:** restate the problem, give 1–2 examples, state the signature and complexity target,
  name edge cases. Do **not** start typing yet.
- **Middle 30 min:** implement incrementally; narrate decisions; keep it runnable; add a quick test.
- **Last 10 min:** test against your examples + an edge case, state complexity, name one improvement.
- **Anti-patterns:** silent coding, premature optimization, ignoring edge cases, no tests.
- **Mock prompts:** implement top-p sampling; a KV-cache class; `cross_entropy` with label smoothing;
  BPE merge step; `pass@k`; a sliding-window rate limiter. (All have worked solutions in [01](01-coding.md).)

### ML/LLM depth round (45 min)
- Lead with the one-sentence answer, then derive. Use the whiteboard for shapes and FLOPs.
- **Must produce from memory:** attention forward + shapes; `6ND` training FLOPs and the Chinchilla
  `D≈20N` rule; KV-cache size formula; the DPO loss and why it drops the reward model; ZeRO stages.
- **Mock prompts:** "Derive KV-cache bytes for a 70B model at 8k context." · "PPO vs DPO — derive
  DPO's objective." · "Why does FlashAttention help if FLOPs are unchanged?" · "When is tensor vs
  pipeline vs data parallelism the right cut?" (Answers in [02](02-ml-and-llm-depth.md).)

### System-design round (45 min)
- **Min 0–5:** requirements + scale (QPS, context length, latency SLO, budget). Write the numbers down.
- **Min 5–10:** capacity math (GPUs for KV cache + weights; throughput target).
- **Min 10–35:** architecture — request path, batching, caching, retrieval/agents, storage, scaling.
- **Min 35–45:** bottlenecks, failure modes, evals/monitoring, cost levers, and what you'd cut for v1.
- **Mock prompts:** ChatGPT-scale serving; RAG over 100M docs; a multi-tenant agent platform; a
  training cluster for a 70B model. (Full walkthroughs + capacity cheat-sheet in [03](03-system-design.md).)

### Applied / project deep-dive (45 min)
- Either your past work *or* a build task ("design + partly build a customer support agent with evals").
- Hit: data → retrieval/prompting → structured output → eval harness → cost/latency → guardrails.
- Have **metrics** ready (win-rate, recall@k, p95 latency, $/1k requests) and a failure-mode list.
- **Mock prompts:** "Cut RAG hallucinations — what do you measure and change?" · "Design an eval set
  for a coding agent." · "Get structured JSON reliably out of an LLM." (Answers in [04](04-applied-llm.md).)

### Behavioral + safety (45 min)
- 6 STAR stories ready (conflict, failure, leadership, ambiguity, impact, learning). 2 min each, tight.
- "Why Anthropic" in 60 seconds, specific and honest. Have 3 questions to ask back.
- Show a safety reflex unprompted: misuse, prompt injection, eval-before-ship.
- **Mock prompts:** "Tell me about a project that failed." · "Disagreed with your manager — what
  happened?" · "Why safety-focused AI?" · "Threat-model an email-writing agent." ([05](05-safety-alignment.md), [06](06-behavioral-mission.md)).

---

## Scoring rubric (score 1–4 after every round)

Interviewers don't grade "right/wrong" — they map you to a level. Use the same lens on yourself.

| Dimension | 1 — No hire | 2 — Mixed | 3 — Hire | 4 — Strong hire |
|-----------|-------------|-----------|----------|-----------------|
| **Correctness** | Wrong / non-working | Works with major nudges | Correct, minor hints | Correct, anticipates edges |
| **Communication** | Silent or rambling | Explains when asked | Narrates clearly, leads with answer | Teaches; structures the room |
| **Judgment / trade-offs** | None stated | States after prompting | Names trade-offs proactively | Quantifies & picks with reasons |
| **Depth** | Surface-level | Knows the what | Knows the why | Derives from first principles |
| **Quantification** | No numbers | Vague ("more memory") | Right ballpark | Crisp math, sanity-checked |
| **Safety reflex** | Absent | Mentions if asked | Raises it naturally | Threat-models + layered defense |

**Bar:** average **≥ 3.0 with no dimension at 1**, across a full loop, on prompts you haven't pre-rehearsed.
Two dimensions to never let slip below 3: **communication** and **judgment** — they're weighted heavily.

---

## Post-mock retro (fill this out every time)

```
Round: ____________________   Date: __________   Prompt: ____________________
Scores  Correct __ / Comm __ / Judgment __ / Depth __ / Quant __ / Safety __

What went well (keep doing):
1.
2.

Biggest gap (fix before next mock):
1.

One concept to re-drill (link the bank section): ____________________
Did I lead with the answer?  Y / N
Did I state assumptions + what I'd measure?  Y / N
Filler-word / silence moments to cut: ____________________
```

Keep a running list of "biggest gaps." When the same gap appears twice, that's your next study block.

---

## Common failure modes → the fix

- **Buries the answer** → State the conclusion in sentence one, *then* justify.
- **Codes in silence** → Narrate intent before each block; the interviewer is scoring reasoning.
- **No numbers in design** → Always do capacity math first; "~2× memory" beats "more memory."
- **Forgets evals** → End every applied/design answer with "here's what I'd measure and the bar."
- **No safety reflex** → Add one line on misuse/injection/eval-before-ship where it fits naturally.
- **Rehearsed STAR feels canned** → Keep the structure, vary the words; lead with the result.
- **Freezes when stuck** → Say your recovery plan out loud and keep moving; never stop the clock.
- **Over-builds** → State the v1 cut and what you'd defer; scope is a senior signal.

---

## Day-of checklist

**48 hours before**
- [ ] One full-loop mock done this week; gaps logged and re-drilled.
- [ ] Whiteboard attention + the four capacity formulas cold (params, `6ND`, KV-cache, GPU count).
- [ ] 6 STAR stories + "why Anthropic" rehearsed out loud (not just read).
- [ ] 3 questions to ask each interviewer.
- [ ] Re-ran a [notebook](../notebooks/README.md) and a [lab](../labs/README.md) so the code is fresh in the hands.

**Morning of**
- [ ] Skim [07 rapid-fire](07-rapid-fire.md) once (recall warm-up, not new learning).
- [ ] Tech check: camera, mic, screen-share, coding env, water, paper + pen for shapes/math.
- [ ] Quiet space; phone off; timeboxes internalized.

**In each round**
- [ ] Restate the problem and assumptions first.
- [ ] Lead with the answer; quantify; name the trade-off; say what you'd measure.
- [ ] Add a safety line where it fits.
- [ ] Leave time to test (coding) or to cover failure modes + v1 cut (design).
- [ ] Ask your questions; end with genuine interest in the mission.

> The fastest way to make these reflexes automatic is reps: re-run the [notebooks](../notebooks/README.md)
> and [labs](../labs/README.md), then mock until leading-with-the-answer and reasoning-in-numbers feel natural.
