"""Spec for Lab 07 — the ReAct agent loop.

These FAIL until you implement `parse_action` and `Agent.run` in lab07_agent/agent.py.

    uv run pytest -m todo tests/test_lab07_agent.py
"""

from __future__ import annotations

import pytest

from lab07_agent.agent import Agent, parse_action
from lab07_agent.brain import mock_brain
from lab07_agent.tools import default_tools

pytestmark = pytest.mark.todo


def test_parse_final_answer():
    kind, payload = parse_action("Thought: done.\nFinal Answer: 42")
    assert kind == "final"
    assert payload == "42"


def test_parse_action_tool_call():
    kind, payload = parse_action("Thought: go.\nAction: calculator\nAction Input: 2 + 2")
    assert kind == "action"
    assert payload == ("calculator", "2 + 2")


def test_parse_error_on_garbage():
    kind, _ = parse_action("just some prose with no protocol")
    assert kind == "error"


def test_agent_uses_calculator():
    agent = Agent(default_tools(), mock_brain)
    result = agent.run("what is 12 * (3 + 4)?")
    assert result.answer.strip() == "84"


def test_agent_uses_lookup():
    agent = Agent(default_tools(), mock_brain)
    result = agent.run("explain the kv cache please")
    assert "KV cache" in result.answer


def test_agent_respects_max_steps():
    def never_finishes(history):
        return "Thought: loop forever.\nAction: calculator\nAction Input: 1 + 1"

    result = Agent(default_tools(), never_finishes, max_steps=3).run("go")
    assert "max steps" in result.answer.lower()


def test_agent_records_trajectory():
    agent = Agent(default_tools(), mock_brain)
    result = agent.run("what is 2 + 2?")
    kinds = [s.kind for s in result.steps]
    assert "action" in kinds
    assert "observation" in kinds
    assert kinds[-1] == "answer"
