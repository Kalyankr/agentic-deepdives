"""Unified LLM client for the Agentic AI Course.

The entire course is written against **one** small interface so every notebook,
example, and project runs in two modes with *identical* calling code:

* **Online** — if ``OPENAI_API_KEY`` is set and the ``openai`` package is
  installed, calls a real model (default ``gpt-4o-mini``).
* **Offline** — otherwise falls back to :class:`MockLLM`, a deterministic,
  no-network stand-in. You can *script* its replies for reproducible teaching
  demos, or rely on its heuristics for ad-hoc calls.

Why a wrapper? Agentic systems are mostly *orchestration* — routing, chaining,
memory, tool use. You can learn and test 90% of that without ever paying for a
token. When you are ready for real reasoning, set one env var and the same code
talks to OpenAI.

Typical use::

    from shared.llm import get_llm, system, user

    llm = get_llm()                       # real if key present, else mock
    reply = llm.chat([system("You are concise."), user("Say hi.")])
    print(reply)                          # -> str

    # Reproducible offline demo: queue exact replies.
    from shared.llm import MockLLM
    llm = MockLLM(scripted=["Paris", "4"])
    llm.chat([user("Capital of France?")])   # -> "Paris"
    llm.chat([user("2 + 2?")])               # -> "4"
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Message helpers — a "message" is just {"role": ..., "content": ...}.
# These tiny constructors keep call-sites readable and match the OpenAI schema.
# --------------------------------------------------------------------------- #
Message = dict[str, str]


def system(content: str) -> Message:
    """A system message: sets the model's role, rules, and persona."""
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    """A user message: the human turn / task input."""
    return {"role": "user", "content": content}


def assistant(content: str) -> Message:
    """An assistant message: a prior model turn (used to build history)."""
    return {"role": "assistant", "content": content}


def tool(content: str) -> Message:
    """A tool/observation message: the result of a tool call fed back to the model."""
    return {"role": "tool", "content": content}


# --------------------------------------------------------------------------- #
# JSON extraction — LLMs love to wrap JSON in prose or ```json fences.
# This is used everywhere structured output matters (Course 3 especially).
# --------------------------------------------------------------------------- #
def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model response.

    Handles raw JSON, ```json fenced blocks, and JSON embedded in prose.
    Raises ``ValueError`` if nothing parses — caller decides how to repair/retry.
    """
    text = text.strip()
    # 1) Strip a ```json ... ``` (or plain ``` ... ```) fence if present.
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # 2) Fast path: the whole thing is JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 3) Fall back to the first balanced {...} or [...] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No parseable JSON found in: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    """Token accounting so you can reason about cost even with the mock."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0


class BaseLLM:
    """Common surface for both backends.

    Subclasses implement :meth:`_complete`. Everything else (``chat``, ``json``)
    is shared so notebooks never branch on which backend is live.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7) -> None:
        self.model = model
        self.temperature = temperature
        self.usage = Usage()

    # -- the one method subclasses must provide ----------------------------- #
    def _complete(self, messages: list[Message], **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError

    # -- public API used everywhere in the course -------------------------- #
    def chat(self, messages: list[Message], **kwargs: Any) -> str:
        """Send a message list, return the assistant's text reply."""
        self.usage.calls += 1
        # crude token estimate (~4 chars/token) — good enough for budgeting demos
        self.usage.prompt_tokens += sum(len(m["content"]) for m in messages) // 4
        out = self._complete(messages, **kwargs)
        self.usage.completion_tokens += len(out) // 4
        return out

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Convenience: a single user prompt -> text reply."""
        return self.chat([user(prompt)], **kwargs)

    def json(self, messages: list[Message], **kwargs: Any) -> Any:
        """Chat, then parse the reply as JSON (with one repair retry)."""
        text = self.chat(messages, **kwargs)
        try:
            return extract_json(text)
        except ValueError:
            repair = [*messages, assistant(text), user("Return ONLY valid JSON, nothing else.")]
            return extract_json(self.chat(repair, **kwargs))


# --------------------------------------------------------------------------- #
# Mock backend — deterministic, offline, no dependencies.
# --------------------------------------------------------------------------- #
@dataclass
class MockLLM(BaseLLM):
    """Deterministic offline model.

    Two modes, in priority order:

    1. **Scripted** — pass ``scripted=[...]`` (or call :meth:`queue`) and each
       ``chat`` pops the next reply. Perfect for reproducible notebook output:
       the *code path* is real, only the "intelligence" is canned.
    2. **Heuristic** — if the queue is empty, inspect the prompt and synthesize a
       plausible reply (echoes JSON requests, picks a routing label, emits a
       ReAct ``Final Answer``, etc.). Good enough to exercise control flow.
    """

    scripted: list[str] = field(default_factory=list)
    model: str = "mock-llm"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        BaseLLM.__init__(self, model=self.model, temperature=self.temperature)
        self._queue: list[str] = list(self.scripted)

    def queue(self, *responses: str) -> MockLLM:
        """Append scripted replies; returns self so calls can chain."""
        self._queue.extend(responses)
        return self

    def _complete(self, messages: list[Message], **kwargs: Any) -> str:
        if self._queue:
            return self._queue.pop(0)
        return self._heuristic(messages, **kwargs)

    # -- heuristic fallbacks ------------------------------------------------ #
    def _heuristic(self, messages: list[Message], **kwargs: Any) -> str:
        sys_txt = " ".join(m["content"] for m in messages if m["role"] == "system").lower()
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        low = last.lower()
        prompt_all = (sys_txt + " " + low).strip()

        # 1) JSON / structured-output requests -> return a small valid object.
        if "json" in prompt_all or "schema" in prompt_all:
            return json.dumps(self._mock_json(prompt_all))

        # 2) Routing / classification -> echo one of the offered labels.
        labels = re.findall(r"['\"]([a-z_][a-z0-9_ ]{2,30})['\"]", prompt_all)
        if "categor" in prompt_all or "route" in prompt_all or "classif" in prompt_all:
            if labels:
                return labels[0]

        # 3) ReAct scaffolding -> emit a terminating step (no real tool use).
        if "thought:" in prompt_all or "final answer" in prompt_all:
            return f"Thought: I can answer directly.\nFinal Answer: [mock] re: {last[:60]}"

        # 4) Default: a deterministic, clearly-labeled echo.
        return f"[mock-llm reply] I received {len(messages)} message(s). Last: {last[:120]}"

    @staticmethod
    def _mock_json(prompt: str) -> dict[str, Any]:
        """Best-effort JSON object whose keys mirror any field names in the prompt."""
        keys = re.findall(r"['\"]([a-zA-Z_][a-zA-Z0-9_]{1,30})['\"]\s*:", prompt)
        if keys:
            return {k: f"mock_{k}" for k in dict.fromkeys(keys)}
        return {"result": "mock", "confidence": 0.5, "ok": True}


# --------------------------------------------------------------------------- #
# OpenAI backend — thin wrapper over the official SDK.
# --------------------------------------------------------------------------- #
class OpenAILLM(BaseLLM):
    """Real model via the ``openai`` SDK. Reads ``OPENAI_API_KEY`` from the env.

    Supports OpenAI and any OpenAI-compatible endpoint (Azure OpenAI, local
    servers) by passing ``base_url`` / ``api_key`` through ``client_kwargs``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        **client_kwargs: Any,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - guarded by get_llm()
            raise ImportError(
                "openai is not installed. Run `uv sync --extra openai` "
                "or use MockLLM for offline work."
            ) from exc
        self._client = OpenAI(**client_kwargs)

    def _complete(self, messages: list[Message], **kwargs: Any) -> str:
        resp = self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            **{k: v for k, v in kwargs.items() if k not in {"model", "temperature"}},
        )
        # Mirror real token usage into our counter for honest cost reporting.
        if resp.usage:
            self.usage.prompt_tokens += resp.usage.prompt_tokens
            self.usage.completion_tokens += resp.usage.completion_tokens
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Factory — the single entry point used across the course.
# --------------------------------------------------------------------------- #
def openai_available() -> bool:
    """True if we can actually reach a real model (key set AND SDK installed)."""
    if not os.getenv("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def get_llm(prefer_mock: bool | None = None, **kwargs: Any) -> BaseLLM:
    """Return the best available backend.

    Parameters
    ----------
    prefer_mock:
        ``None`` (default) auto-selects: real model if a key + SDK are present,
        else the mock. Pass ``True`` to force offline, ``False`` to require real.
    kwargs:
        Forwarded to the chosen backend (e.g. ``model``, ``temperature``).
    """
    if prefer_mock is None:
        prefer_mock = not openai_available()
    if prefer_mock:
        # MockLLM uses dataclass fields; only forward what it accepts.
        return MockLLM(
            model=kwargs.get("model", "mock-llm"),
            temperature=kwargs.get("temperature", 0.0),
        )
    return OpenAILLM(**kwargs)


__all__ = [
    "BaseLLM",
    "MockLLM",
    "OpenAILLM",
    "Usage",
    "assistant",
    "extract_json",
    "get_llm",
    "openai_available",
    "system",
    "tool",
    "user",
]
