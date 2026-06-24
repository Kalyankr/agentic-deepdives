# 04 — Tools & Function Calling

> Goal: design good tools, understand function/tool calling mechanics, error handling, and the new protocols (MCP) plus computer-use agents.

---

## 4.1 Tools are how agents affect the world

A **tool** is any function the agent can invoke: an API call, DB query, code execution, web search, retrieval, sending email, or **another agent**. Without tools, an LLM can only talk; tools give it **hands**.

💡 In practice, **tool design matters more than prompt cleverness.** Clear, well-scoped tools with good schemas and error messages do more for reliability than elaborate prompts.

---

## 4.2 How function/tool calling works (mechanics)

Modern LLMs support **native function calling**. The flow:

```
1. You pass the model: messages + a list of tool schemas (name, description, JSON-schema params).
2. The model decides to call a tool and returns a STRUCTURED tool_call (name + JSON args).
   (It does NOT execute anything — it just emits the intent.)
3. YOUR code executes the function and returns the result as a "tool" message.
4. The model reads the result and continues (answer or another tool call).
```

Key points to state in an interview:
- The **model never runs code itself** — your harness executes and returns results. (Security boundary lives here.)
- Args come back as **structured JSON** validated against your schema → fewer parse errors than regex on free text.
- **Parallel tool calls**: many models can request several tools at once for independent work.
- **Structured outputs / JSON mode / constrained decoding** guarantee schema-valid output.

Minimal schema (OpenAI-style):
```json
{
  "name": "get_weather",
  "description": "Get current weather for a city.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
      "unit": {"type": "string", "enum": ["c", "f"], "default": "c"}
    },
    "required": ["city"]
  }
}
```

---

## 4.3 Principles of good tool design

🎯 *"How do you design tools for an agent?"* — high-signal answer:

1. **Clear names & descriptions** — the description is a *prompt*; the model picks tools from it. Be explicit about *when* to use it and *when not to*.
2. **Narrow, well-scoped** — one tool, one job. Avoid god-tools with 15 params.
3. **Strong typed schemas** — enums over free strings; required vs. optional; sane defaults. Constrain the input space.
4. **Helpful return values** — return what the model needs to decide the next step, not raw dumps. Include status and, on failure, an actionable error message.
5. **Idempotency & safety** — make writes idempotent where possible; gate destructive actions behind confirmation/HITL.
6. **Few, good tools > many overlapping tools** — overlapping tools cause selection confusion. Consolidate.
7. **Error as feedback** — return errors the model can act on ("city not found; try a country code"), not stack traces.
8. **Token-aware outputs** — paginate/summarize large results; return handles/IDs for big artifacts.

⚠️ Trap: dumping a 10k-token API response into context. Filter/summarize tool outputs before returning.

---

## 4.4 Error handling & reliability

- **Validate args** before executing; return a clear validation error so the model can retry correctly.
- **Timeouts & retries** with backoff on transient failures (network, rate limits).
- **Idempotency keys** for actions with side effects to avoid duplicates on retry.
- **Graceful degradation** — fallback tool or "ask the human" when a tool is down.
- **Circuit breakers / budgets** — cap calls per tool to stop runaway loops.
- **Sandboxing** — run code execution and untrusted actions in isolated, least-privilege environments.

---

## 4.5 Model Context Protocol (MCP)

💡 **MCP is an open standard (Anthropic, Nov 2024) that standardizes how applications provide tools, data, and prompts to LLMs** — "a USB-C port for AI": connect any MCP-compatible client to any MCP server without bespoke integrations.

**Architecture:**
- **MCP Host** — the AI app (e.g., Claude Desktop, an IDE, your agent).
- **MCP Client** — lives in the host, maintains a 1:1 connection to a server.
- **MCP Server** — exposes capabilities to the model.

**MCP primitives (servers expose):**
- **Tools** — model-callable functions (actions).
- **Resources** — readable data/context (files, DB rows, docs).
- **Prompts** — reusable prompt templates/workflows.

**Why it matters:** N hosts × M tools integration explodes; MCP makes it N + M. Build a tool once as an MCP server, use it everywhere. Transports: stdio (local) and HTTP/SSE (remote).

⚠️ Security note: MCP servers can be a prompt-injection / supply-chain surface. Vet servers, scope permissions, and don't auto-trust tool descriptions or returned content.

---

## 4.6 Computer-use / GUI agents

A frontier capability: the agent controls a computer like a human — **screenshots in, mouse/keyboard actions out** (Anthropic Computer Use, OpenAI Operator/CUA, browser agents).

- **Loop:** screenshot → model picks action (click x,y / type / scroll) → execute → new screenshot → repeat.
- **Use when** there's no API — legacy apps, arbitrary websites.
- **Challenges:** slow, brittle to UI changes, expensive (vision tokens), and a big **security risk** (can do anything a user can). Always sandbox + HITL for sensitive actions.

---

## 4.7 Code execution as a tool

Letting the agent **write and run code** is one of the most powerful tools (data analysis, math, file manipulation, orchestrating other tools).
- Run in a **sandbox** (container/VM, no network unless needed, least privilege).
- Capture stdout/stderr and feed back as observations (enables reflection on errors).
- Pattern: "**code as actions**" — the agent emits Python that calls tools, instead of single JSON tool calls (e.g., CodeAct). More expressive, composes multiple tools per step.

---

## 4.8 Tools vs. retrieval vs. agents-as-tools

- **Retrieval** is a *read* tool (semantic memory access).
- **Other agents** can be wrapped as tools ("agent-as-tool") → basis for hierarchical multi-agent systems (Ch 05).
- **APIs** are *act* tools (read or write external systems).

> The unifying idea: anything the agent can *call and get an observation from* is a tool — including sub-agents.

---

## Interview questions for this chapter

1. Walk through the function-calling loop. Who actually executes the function? *(4.2)*
2. Give five principles of good tool design. *(4.3)*
3. How do you handle tool errors so the agent can recover? *(4.4)*
4. What is MCP and what problem does it solve? Name its three primitives. *(4.5)*
5. When would you use a computer-use agent over API tools, and what are the risks? *(4.6)*
6. The agent keeps picking the wrong tool. How do you debug/fix? *(4.3)*
7. Why can returning huge tool outputs hurt an agent, and what do you do instead? *(4.3)*

**Model answer to #6:** Usually a tool-selection problem rooted in descriptions/overlap. Tighten each tool's description to say exactly when to use vs. avoid it; remove or merge overlapping tools; add enums/required params to constrain misuse; add few-shot examples of correct selection; and inspect traces to see whether the model misreads the schema or the task. If two tools are genuinely confusable, consolidate them into one with a mode parameter.
