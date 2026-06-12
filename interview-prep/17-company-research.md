# 17 · Company & Lab Research (Know Where You're Interviewing)

> The single most common avoidable mistake: walking in without having **researched the lab and used its
> product**. Frontier labs hire for *mission fit*, and "Why us?" is asked in almost every loop. This
> file is a **research framework** plus a durable profile of Anthropic — so your answers are specific,
> current, and genuine.

> Models, funding, valuations, and org details change **constantly**. Treat specifics here as a
> starting point and **verify the latest before your interview** (the lab's blog, newsroom, and recent
> interviews). Doing that refresh *is* the prep.

Pairs with [06-behavioral-mission](06-behavioral-mission.md) ("Why Anthropic", values, questions to ask)
and [09-papers](09-papers.md) (the research itself).

---

## The research framework (works for any lab)

Spend 3–4 focused hours before an onsite:

1. **Use the product, seriously.** Spend real time in their assistant/API. Build a tiny thing. Form an
   opinion: what's great, what's weak, what you'd improve. *"I used X to do Y and noticed Z"* is the
   single most convincing thing you can say.
2. **Read the research.** Skim their blog and 3–5 flagship papers ([09-papers](09-papers.md)). Know
   their *thesis* — what they believe about how to build AI and why.
3. **Read the leadership's own words.** Founder essays, recent podcast/interview appearances, and
   manifestos reveal strategy and values far better than press coverage.
4. **Understand the business.** Who pays them? (API devs, enterprises, consumer subs, partnerships.)
   What's the go-to-market? This shows you think like an owner, not just a coder.
5. **Know recent launches.** The last few model releases, major features, and announcements. Set a news
   alert for the two weeks before your loop.
6. **Map the differentiation.** How do they position vs peers? What's their distinct bet?
7. **Prepare specific questions** tied to their actual work (see below) — generic questions signal you
   didn't do the homework.

> Output of this work: 2–3 sentences of genuine, specific motivation; one informed product opinion; one
> paper you can discuss; and 3 sharp questions. That's a top-decile "Why us?" answer.

---

## Anthropic — a durable profile

**What it is.** An AI safety and research company founded in **2021** by a group of former OpenAI
researchers, including **Dario Amodei** (CEO) and **Daniela Amodei** (President). It's structured as a
**Public Benefit Corporation (PBC)** — legally allowed to weigh its mission alongside shareholder
returns — and is governed with safety as a first-class concern.

**The mission.** Build AI that is **safe, beneficial, and reliable**, and ensure humanity navigates the
transition to powerful AI well. Anthropic's framing: frontier capabilities and safety research must
advance **together**, and being at the frontier is necessary to do safety research that matters.

**The product — Claude.** A family of LLMs offered via a consumer app, an API, and cloud-partner
marketplaces, plus developer tooling (e.g. Claude Code). The model line has historically used tiers
(commonly **Haiku / Sonnet / Opus**, smallest→largest). *Verify the current model names, versions, and
headline capabilities before interviewing — they update often.*

**The research program (this is the heart of Anthropic).**
- **Constitutional AI / RLAIF** — aligning models with AI feedback guided by a written **constitution**,
  rather than only human harmlessness labels ([09-papers](09-papers.md), [05-safety](05-safety-alignment.md)).
- **Mechanistic interpretability** — reverse-engineering model internals: **superposition**, **sparse
  autoencoders / dictionary learning**, "Towards/Scaling Monosemanticity", and circuit analysis. This
  is a flagship, distinctive effort.
- **Responsible Scaling Policy (RSP)** — tying concrete safeguards to capability thresholds (**ASL**
  levels); a public commitment that capability triggers stronger safeguards ([05-safety](05-safety-alignment.md)).
- **Alignment science** — scalable oversight, **Sleeper Agents** (deceptive behavior surviving safety
  training), **many-shot jailbreaking**, red-teaming, and dangerous-capability evals.
- **MCP (Model Context Protocol)** — an **open standard**, introduced by Anthropic, for connecting models
  to tools and data sources ([04-applied-llm](04-applied-llm.md)).

**Values / culture (well-documented).** Safety-first and **empirical** ("AI is a young, empirical
science"); a **"race to the top"** philosophy — set a high safety bar and pull the field up by example;
high trust, high candor, mission-driven. Read Dario Amodei's essays (e.g. the "Core Views on AI Safety"
post and his longer essays on the upside of powerful AI) and recent interviews for the current framing.

**How they think about the frontier.** Capability and safety co-evolve; interpretability and evals
gate deployment; public policy engagement matters. If you can speak to *why* safety work requires being
at the frontier, you're speaking their language.

---

## How Anthropic tends to interview (and what it implies)

- **Mission fit is real, not a formality.** Expect genuine probing on why *safety-focused* AI, and on
  how you reason about the impact of your work. Have a sincere answer ([06-behavioral-mission](06-behavioral-mission.md)).
- **Safety reflex everywhere.** Bring up misuse, prompt injection, and eval-before-ship naturally in
  technical rounds ([05-safety](05-safety-alignment.md)) — it's a core value, not a separate box.
- **Communication & judgment** weigh as heavily as raw correctness. Think out loud, quantify, name
  trade-offs ([README ground rules](README.md)).
- **Empirical mindset.** "What would you measure?" is a recurring theme — evals, CIs, and honest
  uncertainty land well ([04-applied-llm](04-applied-llm.md), [12-math-stats](12-math-stats.md)).

---

## Questions that show you did the homework

Generic ("What's the culture like?") signals nothing. Tie questions to their actual work:

- "How does the **interpretability** work feed back into deployment decisions in practice?"
- "As capabilities cross **RSP/ASL** thresholds, how do day-to-day engineering priorities change?"
- "Where does **Constitutional AI** struggle today, and what's the next direction?"
- "How do product and safety teams resolve tension when an eval says 'not yet' but customers want a feature?"
- "What does success for someone in this role look like in the first 6–12 months?"
- "How is **MCP** adoption shaping the agent/tooling roadmap?"

> Always have **3 genuine questions per interviewer**. The best ones reveal you've read the research and
> used the product.

---

## Researching *other* labs (transferable checklist)

The same framework applies to OpenAI, Google DeepMind, Meta FAIR, Mistral, xAI, Cohere, and others. For
each, know — and **verify current**:

- **Thesis & differentiation** — what's their distinct bet (e.g. safety-first, open-weights, research-lab
  vs product-company, scale-maximalism)?
- **Flagship models & products** — the current line and how it's sold (API, consumer, enterprise, open weights).
- **Signature research** — the work they're known for (RLHF, scaling, multimodality, agents, interpretability).
- **Recent launches** — the last quarter of announcements.
- **Safety posture** — their public framework (e.g. OpenAI's **Preparedness Framework** vs Anthropic's
  **RSP**) and how they talk about risk ([05-safety](05-safety-alignment.md)).
- **Business model & backers** — who funds them, who pays them, key partnerships.
- **Culture signals** — leadership essays, eng blog, open-source footprint.

> A clean comparison you can articulate ("Lab A bets on X and positions safety as Y; Lab B emphasizes Z")
> signals maturity — *without* badmouthing anyone. Be respectful about every lab; the field is small.

---

## Red flags to avoid

- **Never used the product.** Disqualifying for a product-facing role; always have hands-on impressions.
- **Generic praise** ("you're the leader in AI") with no specifics — reads as flattery.
- **Can't name their research** or a single recent launch.
- **Mission as a script.** Insincere mission talk is transparent; speak to what genuinely draws you.
- **Stale facts.** Citing an old model or a defunct detail shows you didn't refresh. *Verify before you go.*
- **Trash-talking competitors.** Critique ideas respectfully; never the people or the org.

---

## A 1-page prep sheet to fill in (per company)

```
COMPANY: ____________________   ROLE/LEVEL: ____________   INTERVIEW DATE: ________
Mission / thesis (1 sentence):
Current flagship models/products (verified ____):
3 papers/posts I can discuss:  1) ________  2) ________  3) ________
My hands-on product impression (what I built/tried + 1 opinion):
Most recent launch/announcement:
Differentiation vs peers (1 line):
Safety posture / framework:
Business model (who pays):
My genuine "Why here?" (2–3 sentences):
3 questions to ask:  1) ________  2) ________  3) ________
```

> The payoff: motivation that sounds like *you*, an informed product opinion, a paper you can riff on,
> and questions that prove you've done the work. That combination is rare — and it's the difference
> between "strong technically" and "clearly wants to be **here**."
