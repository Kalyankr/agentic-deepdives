"""Project-Management Workflow — STARTER.

Implement the ``TODO`` methods. The agent classes' structure is given; fill in the bodies, then
build ``run_project_workflow``. Point the test at this file and run:

    uv run --extra dev pytest projects/02_project_management_workflow -q

Concepts: a reusable agent library + orchestrator-workers + evaluator-optimizer
(see courses/02-agentic-workflows.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.llm import BaseLLM, extract_json, system, user


@dataclass
class DirectPromptAgent:
    llm: BaseLLM

    def respond(self, prompt: str) -> str:
        """TODO: send the prompt as a single user message and return the reply."""
        raise NotImplementedError("TODO")


@dataclass
class PersonaAgent:
    llm: BaseLLM
    persona: str

    def respond(self, prompt: str, context: str = "") -> str:
        """TODO: system=persona, user=prompt (+ optional context). Return the reply."""
        raise NotImplementedError("TODO")


@dataclass
class KnowledgeAgent:
    llm: BaseLLM
    persona: str
    knowledge: str

    def respond(self, prompt: str) -> str:
        """TODO: instruct the model to use ONLY self.knowledge, then answer the prompt."""
        raise NotImplementedError("TODO")


@dataclass
class EvaluationAgent:
    llm: BaseLLM
    criteria: str
    max_rounds: int = 3

    def evaluate(self, worker: PersonaAgent, task: str) -> tuple[str, list[dict]]:
        """TODO: generate -> critique (JSON {pass, feedback}) -> revise, until pass or max_rounds.
        Return (final_draft, history)."""
        raise NotImplementedError("TODO")


@dataclass
class RoutingAgent:
    llm: BaseLLM
    routes: dict[str, PersonaAgent]

    def route(self, query: str) -> str:
        """TODO: classify the query into one of self.routes (JSON {route}) and dispatch."""
        raise NotImplementedError("TODO")


@dataclass
class OrchestratorAgent:
    llm: BaseLLM

    def plan(self, goal: str) -> list[dict]:
        """TODO: ask for 3-6 ordered subtasks as JSON {subtasks:[{id,title,role}]}."""
        raise NotImplementedError("TODO")


@dataclass
class ProjectPlan:
    goal: str
    tasks: list[dict] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"PROJECT: {self.goal}", ""]
        for t in self.tasks:
            lines.append(f"  [{t['id']}] {t['title']} ({t['role']})")
            lines.append(f"       -> {t['output']}")
        return "\n".join(lines)


def run_project_workflow(
    goal: str,
    llm: BaseLLM,
    worker_llms: dict[int, BaseLLM] | None = None,
    eval_llm: BaseLLM | None = None,
) -> ProjectPlan:
    """TODO: plan subtasks -> assign a PersonaAgent worker per task -> (optional) evaluate ->
    collect into a ProjectPlan and return it."""
    raise NotImplementedError("TODO")
