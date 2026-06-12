"""Research Agent for the video-game industry — reference solution (Course 03).

A **stateful research agent** that combines the Course-03 capabilities:

* **Agentic RAG** over a local game-facts knowledge base (retrieve -> judge sufficiency -> retry).
* **Web-search fallback** when the local KB is insufficient (an agentic retrieval *decision*).
* **Structured, cited output** (Pydantic ``Answer``).
* **Session memory** (remembers prior questions in the conversation).
* An **evaluation harness** over a fixed question set.

Runs offline: retrieval is a real keyword scorer, the web search is a deterministic stub, and the
LLM's sufficiency/answer turns are scripted with a ``MockLLM``. Swap in ``get_llm()`` + ``--extra
rag``/``--extra web`` for ChromaDB + Tavily in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from shared.llm import BaseLLM, MockLLM, extract_json, system, user

# --------------------------------------------------------------------------- #
# Local knowledge base (the "game facts" vector store, here a keyword index).
# --------------------------------------------------------------------------- #
GAME_FACTS: list[str] = [
    "Chrono Trigger (1995, SNES) is a role-playing game famous for time travel.",
    "Half-Life (1998, PC) is a first-person shooter set in the Black Mesa research facility.",
    "Celeste (2018, PC/Switch) is a precision platformer about climbing Celeste Mountain.",
    "Hades (2020, PC/Switch) is a roguelike dungeon crawler by Supergiant Games.",
    "The Legend of Zelda: Ocarina of Time (1998, N64) popularized 3D action-adventure design.",
]


def retrieve(query: str, k: int = 2) -> list[str]:
    """Keyword-overlap retrieval over the local KB (stand-in for vector similarity)."""
    terms = [w for w in query.lower().split() if len(w) > 2]
    scored = sorted(GAME_FACTS, key=lambda d: -sum(t in d.lower() for t in terms))
    top = [d for d in scored if sum(t in d.lower() for t in terms) > 0][:k]
    return top


def web_search(query: str) -> list[dict]:
    """Deterministic offline stub. In production: Tavily/SerpAPI returning {title,url,content}."""
    return [
        {
            "title": "Elden Ring - Wikipedia",
            "url": "https://example.com/elden-ring",
            "content": "Elden Ring (2022, multi-platform) is an action RPG by FromSoftware.",
        }
    ]


# --------------------------------------------------------------------------- #
# Structured, cited answer contract.
# --------------------------------------------------------------------------- #
class Answer(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


@dataclass
class ResearchAgent:
    """Stateful agent: decides whether local retrieval suffices, else searches the web."""

    llm: BaseLLM
    memory: list[dict] = field(default_factory=list)  # session memory of past Q/A
    max_reformulations: int = 1

    def _sufficient(self, question: str, context: list[str]) -> dict:
        raw = self.llm.chat(
            [
                system(
                    'Reply JSON {"sufficient": bool, "better_query": str}. '
                    "Is the context enough to answer the question well?"
                ),
                user(f"Q: {question}\nCONTEXT: {context}"),
            ]
        )
        return extract_json(raw)

    def answer(self, question: str) -> Answer:
        # 1) retrieve from the local KB and judge sufficiency
        context = retrieve(question)
        verdict = self._sufficient(question, context)

        # 2) one reformulation attempt if the first retrieval was insufficient
        if not verdict["sufficient"] and self.max_reformulations:
            better = retrieve(verdict.get("better_query") or question)
            verdict = self._sufficient(question, better)
            if verdict["sufficient"]:
                context = better

        # 3) fall back to web search when the KB still can't answer
        used_web = False
        if not verdict["sufficient"]:
            context = [h["content"] for h in web_search(question)]
            used_web = True

        # 4) produce a structured, cited answer
        raw = self.llm.chat(
            [
                system(
                    "Answer ONLY from the context. Reply JSON "
                    '{"answer": str, "sources": [str], "confidence": number in [0,1]}.'
                ),
                user(f"Q: {question}\nCONTEXT: {context}"),
            ]
        )
        result = Answer(**extract_json(raw))
        # 4) remember the interaction (session memory)
        self.memory.append({"q": question, "a": result.answer, "web": used_web})
        return result


# --------------------------------------------------------------------------- #
# Evaluation harness.
# --------------------------------------------------------------------------- #
def evaluate(agent: ResearchAgent, cases: list[dict], judge_llm: BaseLLM) -> float:
    """Mean correctness score over a fixed test set (LLM-as-judge, 1-5)."""
    scores = []
    for case in cases:
        ans = agent.answer(case["question"])
        verdict = extract_json(
            judge_llm.chat(
                [
                    system('Score 1-5 vs reference. Reply JSON {"score": int}.'),
                    user(f"Q: {case['question']}\nANSWER: {ans.answer}\nREF: {case['reference']}"),
                ]
            )
        )
        scores.append(verdict["score"])
    return sum(scores) / len(scores)


def _demo() -> None:
    # KB question (sufficient on first retrieval) then a web-fallback question.
    agent_llm = MockLLM(
        scripted=[
            '{"sufficient": true, "better_query": ""}',  # local KB is enough
            '{"answer": "Hades (2020) is a roguelike by Supergiant Games.", '
            '"sources": ["GAME_FACTS"], "confidence": 0.9}',
            '{"sufficient": false, "better_query": "Elden Ring release"}',  # not in KB
            '{"sufficient": false, "better_query": "Elden Ring"}',  # still insufficient -> web
            '{"answer": "Elden Ring (2022) is an action RPG by FromSoftware.", '
            '"sources": ["https://example.com/elden-ring"], "confidence": 0.8}',
        ]
    )
    agent = ResearchAgent(agent_llm)
    a1 = agent.answer("What kind of game is Hades?")
    print("Q1 ->", a1.answer, "| sources:", a1.sources)
    a2 = agent.answer("Tell me about Elden Ring")
    print("Q2 ->", a2.answer, "| used web:", agent.memory[-1]["web"])
    print("memory has", len(agent.memory), "interactions")


if __name__ == "__main__":
    _demo()
