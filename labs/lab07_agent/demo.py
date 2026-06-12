"""Agent demo — runs once you implement the loop in `agent.py`.

    uv run python -m lab07_agent.demo

Watch the ReAct loop pick a tool, observe the result, and finalize an answer.
"""

from __future__ import annotations

from lab07_agent.agent import Agent
from lab07_agent.brain import mock_brain
from lab07_agent.tools import default_tools

QUESTIONS = [
    "what is 12 * (3 + 4)?",
    "explain the kv cache",
    "what's the capital of France?",
]


def main() -> None:
    agent = Agent(default_tools(), mock_brain, max_steps=6)
    for q in QUESTIONS:
        print(f"\n=== Q: {q}")
        result = agent.run(q)
        for step in result.steps:
            print(f"  [{step.kind:11}] {step.content}")
        print(f"  -> ANSWER: {result.answer}")


if __name__ == "__main__":
    main()
