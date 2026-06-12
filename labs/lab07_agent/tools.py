"""Tools the agent can call — fully provided.

A `Tool` wraps a plain Python function with **safe error handling**: instead of
raising, a failing tool returns an `ERROR: ...` string so the agent can read it,
recover, and try again. That recover-from-text behavior is central to real agents.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_ARITHMETIC = re.compile(r"[0-9+\-*/(). ]+")


class Tool:
    """A named, described callable with built-in error capture."""

    def __init__(self, name: str, description: str, func: Callable[[str], object]):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, arg: str) -> str:
        try:
            return str(self.func(arg))
        except Exception as exc:  # surface the failure to the agent as recoverable text
            return f"ERROR: {exc}. Check your input and try again."

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"


def _calculator(expr: str) -> object:
    """Evaluate a *strictly arithmetic* expression.

    The regex allow-list + empty builtins are a sandbox: only digits, operators,
    parentheses, dot and spaces get through, so `eval` can't reach anything else.
    """
    expr = expr.strip()
    if not _ARITHMETIC.fullmatch(expr):
        raise ValueError("only arithmetic characters (0-9 + - * / ( ) . space) are allowed")
    return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — sandboxed by the allow-list above


_KB = {
    "kv cache": "The KV cache stores past keys/values so decoding doesn't recompute attention.",
    "rag": "RAG retrieves documents and grounds the model's answer in them with citations.",
    "dpo": "DPO aligns a model directly from preference pairs, with no separate reward model.",
    "lora": "LoRA freezes the base weights and trains small low-rank adapter matrices.",
}


def _lookup(query: str) -> str:
    """Look up a fact by keyword from a tiny in-memory knowledge base."""
    q = query.lower()
    for key, val in _KB.items():
        if key in q:
            return val
    return "No entry found for that query."


def default_tools() -> dict[str, Tool]:
    """The toolbox the demo and tests use."""
    return {
        "calculator": Tool(
            "calculator",
            "Evaluate an arithmetic expression, e.g. '12 * (3 + 4)'.",
            _calculator,
        ),
        "lookup": Tool(
            "lookup",
            "Look up an ML fact by keyword (kv cache, rag, dpo, lora).",
            _lookup,
        ),
    }
