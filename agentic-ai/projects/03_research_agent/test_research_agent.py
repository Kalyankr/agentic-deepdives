"""Tests for the Research Agent.

Imports the reference ``solution`` so they pass out of the box. Change to ``import starter as impl``
to grade your own work.
"""

import solution as impl  # <- change to `import starter as impl` to test your implementation

from shared.llm import MockLLM


def test_retrieve_finds_relevant_fact():
    hits = impl.retrieve("roguelike Supergiant")
    assert any("Hades" in h for h in hits)


def test_answer_from_local_kb_is_structured_and_cited():
    llm = MockLLM(
        scripted=[
            '{"sufficient": true, "better_query": ""}',
            '{"answer": "Hades is a roguelike.", "sources": ["GAME_FACTS"], "confidence": 0.9}',
        ]
    )
    agent = impl.ResearchAgent(llm)
    ans = agent.answer("What is Hades?")
    assert isinstance(ans, impl.Answer)
    assert ans.sources and 0 <= ans.confidence <= 1
    assert agent.memory[-1]["web"] is False


def test_answer_falls_back_to_web_when_kb_insufficient():
    llm = MockLLM(
        scripted=[
            '{"sufficient": false, "better_query": "Elden Ring"}',
            '{"sufficient": false, "better_query": "Elden Ring"}',
            '{"answer": "Elden Ring is an action RPG.", "sources": ["web"], "confidence": 0.8}',
        ]
    )
    agent = impl.ResearchAgent(llm)
    ans = agent.answer("Tell me about Elden Ring")
    assert "RPG" in ans.answer
    assert agent.memory[-1]["web"] is True


def test_evaluate_returns_mean_score():
    agent_llm = MockLLM(
        scripted=[
            '{"sufficient": true, "better_query": ""}',
            '{"answer": "Hades is a roguelike.", "sources": ["GAME_FACTS"], "confidence": 0.9}',
        ]
    )
    judge = MockLLM(scripted=['{"score": 5}'])
    score = impl.evaluate(
        impl.ResearchAgent(agent_llm),
        [{"question": "What is Hades?", "reference": "roguelike"}],
        judge,
    )
    assert score == 5.0
