# 🃏 Claude Code (Agentic Coding CLI) — One-Page Cheat-Sheet

> Last-minute recall card for the [full HLD](README.md). Drill the bold bits.

## The one idea
An **LLM in a loop with tools**, driving the **real environment**: **think → act → observe → repeat**, stop when there's **no tool call**. The hard parts aren't the chat — they're **autonomy under a context budget far smaller than the codebase**, **reliable edits**, and **acting safely on the user's machine**.

## The loop (memorize)
```
state = [system, tools, memory, task]
loop: resp = LLM(state)            # prefill-heavy, streamed
      if resp.tool_calls: run (gate risky) → append observation
                          if near_limit: compact;  if over budget/no-progress: break
      else: return answer          # no tool call ⇒ done
```
**Stops on:** no tool call · budget cap (steps/tokens/$) · user interrupt · unrecoverable error. **Errors are observations** → the model self-corrects (ground truth from compiler/tests).

## The defining number
**Codebase ≫ window.** 10K files × 2K ≈ **20M tokens ≈ 100×** a 200K window; monorepo = **billions ≈ 10,000×**. → **Never preload. Retrieve just-in-time** (grep → read ranges).

## Context management = the core skill
| Region | Contents | Action |
|---|---|---|
| **Prefix** (cached) | system + tools + steering | keep first, stable → cache it |
| **Working set** | files currently relevant | read **ranges**, drop stale |
| **Recent dialogue** | last turns + observations | summarize/compact |
- **Compact** at ~80%: summarize done/decisions/state, drop verbose output, **offload to a memory file**.
- **Subagents** = fresh window for a sub-task → return a summary (isolate noise).

## Agentic search vs RAG (key contrast)
- **Agentic** (grep/glob/LSP + reasoning): **live, exact, follows refs, zero infra**; costs steps. ← **default**
- **Embedding index** (RAG): great for **NL/huge** repos; suffers **staleness + chunking + plausible-wrong**.
- **Retrieval is a *tool*, not the architecture.** Contrast: RAG platform indexes a corpus; an agent **navigates a live filesystem**.

## Reliable edits
**Surgical `old_string → new_string`** (unique-anchored, match **exactly once**) — *not* whole-file rewrites. Ambiguous/whitespace mismatch ⇒ **fail loud + retry**. Then **verify**: `get_errors` + tests + re-read. Cross-file ⇒ **LSP rename** + git staging (atomic, revertible).

## Tools (action space)
`read_file(range)` · `grep` · `glob`/`list_dir` (read-only) · `edit` · `create_file` (write) · `bash` (danger) · `get_errors` · `web_fetch` (untrusted). **Right granularity**; **descriptions = the model's docs**; **few & sharp** (every def costs prefix tokens); extend via **MCP/plugins**.

## Safety (first-class subsystem)
- **Permissions** — risk taxonomy: read-only **auto** · writes **diff/auto-in-workspace** · `bash` **gated** · **always-confirm** for `rm -rf`/force-push/secrets. Granular grants + modes (interactive ↔ auto-edit ↔ autonomous-in-sandbox).
- **Sandbox** — workspace scoping → containers (mounted repo, no host, limits) → **egress control** → read-only mounts + secret redaction.
- **Prompt injection** — untrusted file/web/tool text + tools = **RCE/exfiltration**. Defense: **trust boundary** (content = data, not commands) · **approval at the action boundary** · sandbox/egress · secret hygiene · detect+alert. (OWASP-LLM **injection + excessive agency**.)

## Cost & latency (prefill-heavy!)
- 60 steps ≈ **2.4M input** vs **60K output** → **input ~40×**. Optimize **prefill, not decode**.
- **#1 lever = prompt caching** the stable prefix (~10× cheaper reads, lower TTFT).
- **Model routing** (big = plan/edit, small = compact/rank/classify) · token frugality · **step/$ caps**.
- **Latency** = sequential LLM round-trips + slow tests → **stream** (felt latency = TTFT) + **parallel read-only tools** + **interrupt**.

## Long-horizon + persistence
**Compaction** + **external memory/TODO** + **checkpoints** (conversation + file changes → resume/rollback) + **subagents**. Resumed session reloads **summary + memory**, not raw transcript.

## Failure modes → fixes
| Failure | Fix |
|---|---|
| Infinite/oscillating loop | budget caps · **no-progress detection** · escalate |
| Context overflow / "forgetting" | **compact + memory file + checkpoints** |
| Hallucinated API/file | **verify** via grep/`get_errors`/tests |
| Edit won't apply / clobbers | unique anchor · **fail-loud** · diff preview · LSP |
| Destructive command | **always-confirm** · allow-list · sandbox |
| Injection compromise | trust boundary · approval · **egress control** |
| Cost blow-up | caching · routing · budget caps |

## Evaluation
**SWE-bench-style**: real repo @ bug commit, **hidden tests pass** (`pass@1`), **containerized**. + process metrics (success, **steps/$ per task**, edit-apply rate, human-rescue %) + human review + **safety evals** (injection, destructive-refusal, secret-leak).

## Roadmap
**MVP** loop + core tools + permission prompts + streaming → **Growth** caching + compaction + sessions/checkpoints + diff UX + steering file → **Scale** sandbox + subagents + routing + **headless/CI** + MCP + audit/policy → **Frontier** long-horizon autonomy, multi-agent teams, unattended issue-solving.

## 30-second pitch
"It's an **LLM in a think-act-observe loop** with tools, working in a window **100×–10,000× smaller than the repo** — so the engineering is **context management** (just-in-time retrieval, compaction, memory, subagents), **reliable edits** (surgical patches **verified** by compiler/tests), and **safety** (permissions + sandbox + injection defense). It's a **prefill-heavy client** of an inference service, so **prompt caching** is the top cost lever."

---

[← HLD](README.md) · [Q&A](questions.md) · [Answers](answers.md) · [Index](../../README.md)
