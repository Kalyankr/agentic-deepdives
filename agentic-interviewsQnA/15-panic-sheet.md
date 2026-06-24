# 🚨 Panic Sheet — read this in the 5 minutes before your call

Twenty things that make you sound senior. Don't memorize prose — glance, recall the **number** or **one-liner**, move on.

> How to use: skim top-to-bottom once (3–4 min). Then re-read only the **bold** bits. Breathe. You know this.

---

## The 5 one-liners that win rounds

1. **What is an agent?** "An LLM in a loop with tools and memory, pursuing a goal — it *decides the next action*, observes the result, and repeats until done or stopped." The decision-making is what separates an agent from a pipeline.
2. **When NOT to use an agent.** "If the steps are known and fixed, write a workflow — it's cheaper, faster, and testable. Reach for an agent only when the path is *dynamic and can't be hard-coded*." Saying this unprompted is a senior signal.
3. **ReAct in one breath.** "Reason → Act → Observe, looped. The model thinks, calls a tool, reads the result, and re-plans with that evidence." It grounds reasoning in real tool output instead of hallucinating.
4. **RAG vs fine-tuning.** "RAG for *knowledge* (facts that change, need citations); fine-tuning for *behavior* (format, tone, skill). They're complementary, not competing."
5. **Eval is the moat.** "I evaluate **outcome** (did it succeed?) *and* **trajectory** (did it take a sane path?). Without trajectory eval you can't debug *why* it failed." Lead with this in any production question.

---

## Numbers to drop (rough, defensible)

6. **Latency budget.** Each agent step ≈ **1–3 s** (LLM call + tool). A 5-step task ≈ **5–15 s**. If users wait >10 s, **stream tokens / show intermediate steps** or go async. Parallelize independent tool calls.
7. **Context window ≠ free.** Cost and latency scale with tokens; quality *degrades* past a point ("**lost in the middle**" — models attend best to the **start and end** of context). Don't dump everything in — retrieve and rank.
8. **Token cost intuition.** Output tokens cost **~2–4×** input tokens. Long agent loops blow up because every step re-sends the growing transcript → use **summarization/compaction** of old turns.
9. **Retrieval top-k.** Start **k = 3–5** chunks, chunk size **~256–512 tokens** with overlap. Re-rank if recall is the problem; trim if the model gets distracted.
10. **Temperature.** **0–0.2** for tool-use/deterministic agents and eval; higher only for ideation. Most production agents run **low temp**.
11. **Cache.** Prompt/KV caching can cut cost **~50–90%** on the stable prefix (system prompt + tools). Always mention caching when asked about cost.

---

## Security — say "the lethal trifecta"

12. **The lethal trifecta** (Simon Willison): an agent is dangerous when it has **(1) access to private data + (2) exposure to untrusted content + (3) ability to exfiltrate (external comms)**. Remove *any one leg* and the data-theft risk collapses.
13. **Prompt injection** is the #1 agent threat (OWASP LLM01). Untrusted text (web page, email, doc, tool output) carries instructions the model obeys. Mitigate: **treat tool output as data not instructions, sandbox, allow-list tools, human-in-the-loop for high-impact actions, least privilege.**
14. **Excessive agency** (OWASP LLM06): too many permissions / unconstrained tools. Fix with **scoped credentials, approval gates on irreversible actions** (spend, delete, send), and dry-run modes.
15. **Never put secrets in the prompt.** Tools call authenticated APIs server-side; the model gets results, not keys.

---

## Training acronyms — one line each

16. **SFT** = supervised fine-tune on demonstrations (teaches format/behavior). **RLHF** = optimize a reward model from human preferences with PPO (aligns to preferences, complex). **DPO** = same preference data, *no separate reward model / no RL loop* — simpler, stable, popular. **RFT/RLVR** = RL from **verifiable rewards** (math/code where you can *check* correctness). **LoRA** = freeze the base, train tiny low-rank adapters → cheap, swappable fine-tuning.
17. **One-liner:** "**DPO** if I have preference pairs and want simplicity; **RFT** if I have a *verifiable* signal; **SFT** first to establish the behavior."

---

## Multi-agent — only when asked, and with a caveat

18. **Topologies:** **orchestrator-worker** (a planner delegates — the default), **hierarchical** (managers of managers), **network/handoff** (peers pass control, e.g. swarm), **group chat/debate** (parallel perspectives → judge). Name orchestrator-worker first.
19. **The caveat that sounds senior:** "Multi-agent adds **coordination cost, latency, and failure modes**. I'd start with a **single agent + good tools**, and split into agents only when roles are genuinely distinct or context/permissions need isolation." (Cognition's *Don't Build Multi-Agents* vs Anthropic's research-system view — knowing both sides scores points.)

---

## Reliability — the closing flourish

20. **Make agents reliable:** **bound the loop** (max steps + timeout), **validate tool I/O against schemas**, **retry with backoff on transient errors**, **checkpoint state** so you can resume, **human approval for high-impact actions**, and **log the full trajectory** (every thought/action/observation) for eval and debugging. "Determinism at the edges, autonomy in the middle."

---

### If your mind goes blank, say one of these

- "Let me restate the requirements and constraints first." *(buys time, shows structure)*
- "I'd start simple — a single agent with these 3 tools — then add complexity only if eval shows I need it."
- "I'd measure that with both an outcome metric and a trajectory check, using an LLM-as-judge calibrated against a small human-labeled set."
- "The trade-off here is **latency/cost vs. accuracy** — I'd tune based on the product's tolerance."

---

> 📌 Three phrases to land at least once: **"lethal trifecta," "outcome *and* trajectory eval," "start with a workflow, reach for an agent only when the path is dynamic."** Now close the sheet. You're ready.

← Back to [README](README.md) · deeper drills in [Mock Interview](11-mock-interview.md) · pick your [role track](README.md#role-specific-tracks)
