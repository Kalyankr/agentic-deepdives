"""Research Agent — STARTER.

Implement the ``TODO`` methods. The KB, retrieval, web stub, and ``Answer`` contract are given.
Point the test at this file and run:

    uv run --extra dev pytest projects/03_research_agent -q

Concepts: agentic RAG, retrieval-sufficiency decision, web fallback, structured output, memory,
evaluation (see courses/03-building-agents.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from shared.llm import BaseLLM, extract_json, system, user

GAME_FACTS: list[str] = [
    "Chrono Trigger (1995, SNES) is a role-playing game famous for time travel.",
    "Half-Life (1998, PC) is a first-person shooter set in the Black Mesa research facility.",
    "Celeste (2018, PC/Switch) is a precision platformer about climbing Celeste Mountain.",
    "Hades (2020, PC/Switch) is a roguelike dungeon crawler by Supergiant Games.",
    "The Legend of Zelda: Ocarina of Time (1998, N64) popularized 3D action-adventure design.",
]


def retrieve(query: str, k: int = 2) -> list[str]:
    terms = [w for w in query.lower().split() if len(w) > 2]
    scored = sorted(GAME_FACTS, key=lambda d: -sum(t in d.lower() for t in terms))
    return [d for d in scored if sum(t in d.lower() for t in terms) > 0][:k]


def web_search(query: str) -> list[dict]:
    return [
        {
            "title": "Elden Ring",
            "url": "https://example.com/elden-ring",
            "content": "Elden Ring (2022, multi-platform) is an action RPG by FromSoftware.",
        }
    ]


class Answer(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


@dataclass
class ResearchAgent:
    llm: BaseLLM
    memory: list[dict] = field(default_factory=list)
    max_reformulations: int = 1

    def _sufficient(self, question: str, context: list[str]) -> dict:
        """TODO: judge if context is enough. Return {sufficient: bool, better_query: str}."""
        raise NotImplementedError("TODO")

    def answer(self, question: str) -> Answer:
        """TODO: retrieve -> judge -> (reformulate once) -> (web fallback) -> structured Answer.
        Append {q, a, web} to self.memory before returning."""
        raise NotImplementedError("TODO")


def evaluate(agent: ResearchAgent, cases: list[dict], judge_llm: BaseLLM) -> float:
    """TODO: run the agent on each case, score 1-5 vs reference with judge_llm, return the mean."""
    raise NotImplementedError("TODO")
