"""The agent's "brain" — fully provided.

A brain is just a callable that, given the conversation `history`, returns the next
ReAct text block (`Thought` + `Action`/`Final Answer`). `mock_brain` is a tiny
*deterministic* policy so the agent runs offline and the spec tests are reproducible.
A real LLM brain (sketch at the bottom) returns the same shape of text.
"""

from __future__ import annotations

import re

_MATH_EXPR = re.compile(r"[0-9][0-9+\-*/(). ]*[0-9)]")
_KEYWORDS = ("kv cache", "rag", "dpo", "lora")


def mock_brain(history: list[dict]) -> str:
    """A deterministic ReAct policy good enough to demo and test the loop.

    Rules:
      * if a tool observation already exists → finalize with that observation.
      * else if the question looks like arithmetic → call the calculator.
      * else if it mentions a known keyword → call lookup.
      * else → answer directly (no tool).
    """
    question = history[0]["content"].lower()
    observations = [m["content"] for m in history if m["role"] == "tool"]
    if observations:
        return f"Thought: I have what I need.\nFinal Answer: {observations[-1]}"

    if re.search(r"\d", question) and re.search(r"[+\-*/]", question):
        match = _MATH_EXPR.search(question)
        if match:
            expr = match.group().strip()
            return f"Thought: I should compute this.\nAction: calculator\nAction Input: {expr}"

    for kw in _KEYWORDS:
        if kw in question:
            return f"Thought: I should look this up.\nAction: lookup\nAction Input: {kw}"

    return "Thought: I can answer this directly.\nFinal Answer: I don't have a tool for that."


# --- Real LLM brain (sketch) -------------------------------------------------
# Wiring a real model is a drop-in replacement — same input (history) and output
# (a ReAct text block). The system prompt teaches the protocol parse_action expects:
#
#   def llm_brain(history, client, tools):
#       system = (
#           "You are a ReAct agent. On each turn output either:\n"
#           "  Thought: <reasoning>\n  Action: <tool>\n  Action Input: <arg>\n"
#           "or, when done:\n  Thought: <reasoning>\n  Final Answer: <text>\n"
#           f"Available tools: {[t.name for t in tools.values()]}"
#       )
#       msgs = [{"role": "system", "content": system}, *history]
#       resp = client.messages.create(model="claude-...", max_tokens=512, messages=msgs)
#       return resp.content[0].text
#
# Everything else (parse_action, the loop, the tools) stays exactly the same.
