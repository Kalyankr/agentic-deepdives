# Interview Prep — Senior AI Engineer / Applied AI Engineer (Anthropic-level)

> A complete, answer-included question bank for interviewing at **Anthropic, OpenAI**, and
> equivalent frontier labs. This is the companion to [Module 12](../modules/12-capstones-and-interview-prep.md):
> Module 12 is the *strategy*; this folder is the *drill material* — hundreds of real questions
> with senior-level answers.

Every file is **Q → worked answer**. Read the answer, then close it and re-derive out loud. If you
can't reproduce the reasoning (not the words) from a blank page, you don't own it yet.

---

## The two tracks

Anthropic (and peers) hire along two related but distinct ladders. Know which loop you're in.

| | **Applied AI / Forward-Deployed Engineer** | **ML / Research Engineer** |
|---|---|---|
| Mandate | Build with the model: agents, RAG, prompting, evals, customer solutions | Build the model & the systems: training, inference, kernels, infra |
| Coding | Practical LLM coding, API integration, data wrangling, evals | Practical ML coding + DSA, sometimes CUDA/perf |
| Depth round | Prompting, context engineering, agent design, eval design | Transformers, training, RLHF/DPO, parallelism, scaling laws |
| System design | LLM *application* architecture (RAG/agent platforms, latency/cost) | Training & serving infra at scale, capacity planning |
| Shared | **Safety mindset**, clear communication, strong fundamentals, mission fit | same |

> This bank covers **both**. Applied-track readers: prioritize [04-applied-llm](04-applied-llm.md),
> [03-system-design](03-system-design.md) (app side), [05-safety](05-safety-alignment.md), [13-take-home](13-take-home-portfolio.md), and the agent/RAG parts of [01-coding](01-coding.md).
> ML-track readers: prioritize [02-ml-and-llm-depth](02-ml-and-llm-depth.md), [09-papers](09-papers.md), [10-numbers-and-hardware](10-numbers-and-hardware.md), [12-math-stats](12-math-stats.md), [14-cuda-and-kernels](14-cuda-and-kernels.md), the infra side of
> [03-system-design](03-system-design.md), and the from-scratch parts of [01-coding](01-coding.md).
> **Both tracks:** [09 papers](09-papers.md), [10 numbers](10-numbers-and-hardware.md), [11 debugging](11-debugging.md), the [08 mock harness](08-mock-interview.md), [15 glossary](15-glossary.md), [17 company research](17-company-research.md), and — once you have an offer — [16 negotiation](16-negotiation-and-leveling.md).

---

## The question bank

| # | File | Round it prepares you for |
|---|------|---------------------------|
| 01 | [Coding](01-coding.md) | Practical ML coding (attention, sampler, BPE, KV cache, metrics) + DSA flavor |
| 02 | [ML & LLM Depth](02-ml-and-llm-depth.md) | The "knowledge" round — transformers, training, RLHF/DPO, inference, distributed, scaling |
| 03 | [System Design](03-system-design.md) | Full walkthroughs: ChatGPT-scale serving, RAG at scale, agent platform, training cluster — with capacity math |
| 04 | [Applied LLM](04-applied-llm.md) | RAG, agents, prompting/context engineering, structured outputs, evals (Applied-AI core) |
| 05 | [Safety & Alignment](05-safety-alignment.md) | Alignment, RSP/Preparedness, prompt injection, jailbreaks, red-teaming |
| 06 | [Behavioral & Mission](06-behavioral-mission.md) | "Why Anthropic", STAR stories, collaboration, questions to ask |
| 07 | [Rapid-Fire Flashcards](07-rapid-fire.md) | 120+ one-line recall checks across everything |
| 08 | [Mock Interview](08-mock-interview.md) | Self-run loop simulation, scoring rubrics, retro template, day-of checklist |
| 09 | [Key Papers](09-papers.md) | Annotated must-know papers — discuss each in 2 min (depth & research rounds) |
| 10 | [Numbers & Hardware](10-numbers-and-hardware.md) | Memorizable constants + capacity math + a night-before cheat sheet |
| 11 | [Debugging](11-debugging.md) | Find-the-bug round: broken training/attention/sampling code + fixes |
| 12 | [Math, Stats & Classic ML](12-math-stats.md) | Probability, statistics, the math behind training, pre-LLM ML, metrics |
| 13 | [Take-Home & Portfolio](13-take-home-portfolio.md) | Acing the take-home + project deep-dive; what to build, reviewer rubric |
| 14 | [CUDA & GPU Kernels](14-cuda-and-kernels.md) | Performance/ML-systems round: execution model, roofline, tiling, FlashAttention, Triton |
| 15 | [Glossary & Quick-Reference](15-glossary.md) | A–Z one-line definitions of every term in the bank |
| 16 | [Negotiation & Leveling](16-negotiation-and-leveling.md) | Getting the right level + negotiating comp/equity (private-lab mechanics) |
| 17 | [Company & Lab Research](17-company-research.md) | Know the lab: research framework, Anthropic profile, questions to ask, red flags |
| — | [flashcards.csv](flashcards.csv) | The 132 cards as CSV — import into Anki for spaced repetition |

---

## A typical loop (what each stage tests)

1. **Recruiter screen** — motivation, background, logistics. (See [06](06-behavioral-mission.md); research the lab with [17](17-company-research.md).)
2. **Technical phone screen** — 1 coding problem (practical/ML-flavored) + a few depth questions.
3. **Take-home or live coding** — build a small working thing (an eval, a mini-agent, a retrieval
   pipeline) or implement a core algorithm. Judged on correctness, clarity, tests, and judgment.
   ([13](13-take-home-portfolio.md))
4. **Onsite (virtual), ~4–6 rounds:**
   - **Coding** (1–2): practical, often ML-flavored; sometimes DSA or **find-the-bug**. ([01](01-coding.md), [11](11-debugging.md))
   - **ML/LLM depth** (1): whiteboard attention, derive FLOPs/KV-cache, explain DPO, discuss a paper. ([02](02-ml-and-llm-depth.md), [09](09-papers.md), [10](10-numbers-and-hardware.md))
   - **System design** (1): design a real LLM system end-to-end with numbers. ([03](03-system-design.md), [10](10-numbers-and-hardware.md))
   - **ML systems / performance** (sometimes, ML-track): GPU execution model, roofline, kernels. ([14](14-cuda-and-kernels.md))
   - **Applied / project deep-dive** (1): your past work or a realistic build task. ([04](04-applied-llm.md), [13](13-take-home-portfolio.md))
   - **Behavioral / mission / safety** (1): values fit, collaboration, safety reasoning. ([05](05-safety-alignment.md), [06](06-behavioral-mission.md))
5. **Offer & negotiation** — leveling, comp, and equity. Don't skip this stage. ([16](16-negotiation-and-leveling.md))

> Frontier labs weight **judgment and communication** as heavily as raw correctness. Think out loud,
> state assumptions, quantify trade-offs, and name what you'd measure.

---

## How to use this bank (4-week sprint)

- **Week 1 — Fundamentals:** [02](02-ml-and-llm-depth.md) + the from-scratch problems in [01](01-coding.md), with [12 math/stats](12-math-stats.md) and the [10 numbers card](10-numbers-and-hardware.md). Whiteboard attention and a KV cache daily until automatic.
- **Week 2 — Applied + Design:** [04](04-applied-llm.md) + [03](03-system-design.md); add [11 debugging](11-debugging.md) drills. Do one full system-design mock out loud each day.
- **Week 3 — Safety + Behavioral + Papers:** [05](05-safety-alignment.md) + [06](06-behavioral-mission.md) + [09 key papers](09-papers.md). Write out 6 STAR stories; rehearse the "why Anthropic" answer; practice the 2-min paper pitch; research the lab and fill a [17 prep sheet](17-company-research.md).
- **Week 4 — Integrate + mock:** daily mixed [07](07-rapid-fire.md) / [flashcards.csv](flashcards.csv) (Anki); 3 full mock interviews using the [08 mock-interview harness](08-mock-interview.md) (rubrics + retro); polish a [13 portfolio project](13-take-home-portfolio.md); record yourself and tighten.

> ML-systems/performance track: fold [14 CUDA & kernels](14-cuda-and-kernels.md) into Weeks 1–2.
> Skim [15 glossary](15-glossary.md) as a daily warm-up, and read [16 negotiation](16-negotiation-and-leveling.md) **before** your first offer call.

### The drills you must pass cold
- Whiteboard multi-head **causal attention** in < 15 min (forward + shapes).
- Given a model spec, estimate **params, FLOPs/token, KV-cache, and serving GPU count** in < 10 min ([10](10-numbers-and-hardware.md)).
- Implement **top-k / top-p sampling** and a **KV cache** from scratch.
- **Find the bug** in a broken training loop / attention impl on sight ([11](11-debugging.md)).
- Design **ChatGPT-scale serving** end-to-end in 45 min with capacity + cost.
- Explain **DPO vs PPO** and **Constitutional AI** to both a non-expert and an expert.
- Discuss any of the **eight core papers** in 2 minutes ([09](09-papers.md)).
- **Threat-model an agent** and propose layered defenses.

---

## Ground rules for great answers

1. **Lead with the answer, then justify.** Interviewers are busy; don't bury the lede.
2. **Quantify.** "~2× memory" beats "more memory." Senior engineers reason in numbers.
3. **Name the trade-off.** Every real answer is "X vs Y; I'd pick X here because …".
4. **State assumptions out loud** and revise them when corrected — that's a positive signal.
5. **Say what you'd measure** to validate the choice (latency, cost, recall, win-rate, CI eval).
6. **Show a safety reflex** where relevant (misuse, injection, eval-before-ship) — it's a core value.

The companion runnable code lives in the [notebooks](../notebooks/README.md) and [labs](../labs/README.md);
re-running those is the fastest way to make these answers muscle memory.
