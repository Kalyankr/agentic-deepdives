"""AI-Powered Agentic Workflow for Project Management — reference solution (Course 02).

Two deliverables, matching the project brief:

1. **A reusable agent library** — small, composable agent types you can reuse on any workflow:
   ``DirectPromptAgent``, ``PersonaAgent``, ``KnowledgeAgent``, ``EvaluationAgent``,
   ``RoutingAgent``, ``OrchestratorAgent``.
2. **A project-management workflow** that wires them together (orchestrator-workers +
   evaluator-optimizer) to turn a one-line project goal into a structured plan.

Runs offline with a scripted ``MockLLM``; swap in ``get_llm()`` for a real model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.llm import BaseLLM, MockLLM, extract_json, system, user


# --------------------------------------------------------------------------- #
# The reusable agent library.
# --------------------------------------------------------------------------- #
@dataclass
class DirectPromptAgent:
    """Simplest agent: pass the prompt straight to the model (no persona)."""

    llm: BaseLLM

    def respond(self, prompt: str) -> str:
        return self.llm.chat([user(prompt)])


@dataclass
class PersonaAgent:
    """Augmented prompt: a system persona shapes tone + expertise."""

    llm: BaseLLM
    persona: str

    def respond(self, prompt: str, context: str = "") -> str:
        content = prompt if not context else f"{prompt}\n\nCONTEXT:\n{context}"
        return self.llm.chat([system(self.persona), user(content)])


@dataclass
class KnowledgeAgent:
    """Knowledge-augmented: the agent may use ONLY the supplied knowledge (grounding)."""

    llm: BaseLLM
    persona: str
    knowledge: str

    def respond(self, prompt: str) -> str:
        sys = f"{self.persona}\n\nUse ONLY this knowledge; do not invent facts:\n{self.knowledge}"
        return self.llm.chat([system(sys), user(prompt)])


@dataclass
class EvaluationAgent:
    """Evaluator-optimizer: loop a worker against explicit criteria until it passes."""

    llm: BaseLLM
    criteria: str
    max_rounds: int = 3

    def evaluate(self, worker: PersonaAgent, task: str) -> tuple[str, list[dict]]:
        draft = worker.respond(task)
        history: list[dict] = []
        for _ in range(self.max_rounds):
            verdict = extract_json(
                self.llm.chat(
                    [
                        system(
                            f"You are a strict reviewer. Criteria: {self.criteria}. "
                            'Reply JSON {"pass": bool, "feedback": "specific + actionable"}.'
                        ),
                        user(f"CANDIDATE:\n{draft}"),
                    ]
                )
            )
            history.append({"draft": draft, **verdict})
            if verdict["pass"]:
                return draft, history
            draft = worker.respond(f"{task}\n\nRevise using this feedback:\n{verdict['feedback']}")
        return draft, history


@dataclass
class RoutingAgent:
    """Classify a request and dispatch to the right specialist agent."""

    llm: BaseLLM
    routes: dict[str, PersonaAgent]

    def route(self, query: str) -> str:
        labels = list(self.routes)
        decision = self.llm.chat(
            [
                system(f'Classify into one of {labels}. Reply JSON {{"route": "<label>"}}.'),
                user(query),
            ]
        )
        choice = extract_json(decision).get("route", labels[-1])
        choice = choice if choice in self.routes else labels[-1]
        return self.routes[choice].respond(query)


@dataclass
class OrchestratorAgent:
    """Action planning: decompose a high-level goal into ordered, delegable subtasks."""

    llm: BaseLLM

    def plan(self, goal: str) -> list[dict]:
        raw = self.llm.chat(
            [
                system(
                    "Break the project goal into 3-6 ordered subtasks. Reply JSON: "
                    '{"subtasks": [{"id": int, "title": str, "role": str}]}'
                ),
                user(goal),
            ]
        )
        return extract_json(raw)["subtasks"]


# --------------------------------------------------------------------------- #
# The project-management workflow (orchestrator-workers + evaluation).
# --------------------------------------------------------------------------- #
@dataclass
class ProjectPlan:
    goal: str
    tasks: list[dict] = field(default_factory=list)  # each: {id, title, role, output, passed}

    def render(self) -> str:
        lines = [f"PROJECT: {self.goal}", ""]
        for t in self.tasks:
            mark = "ok" if t.get("passed", True) else "needs-review"
            lines.append(f"  [{t['id']}] {t['title']} ({t['role']}) [{mark}]")
            lines.append(f"       -> {t['output']}")
        return "\n".join(lines)


def run_project_workflow(
    goal: str,
    llm: BaseLLM,
    worker_llms: dict[int, BaseLLM] | None = None,
    eval_llm: BaseLLM | None = None,
) -> ProjectPlan:
    """Orchestrate: plan subtasks -> assign a worker per task -> evaluate -> assemble a plan."""
    orchestrator = OrchestratorAgent(llm)
    subtasks = orchestrator.plan(goal)
    plan = ProjectPlan(goal=goal)
    for st in subtasks:
        worker_llm = (worker_llms or {}).get(st["id"], llm)
        worker = PersonaAgent(worker_llm, persona=f"You are a {st['role']}. Be concrete and brief.")
        if eval_llm is not None:
            evaluator = EvaluationAgent(eval_llm, criteria="actionable, specific, on-topic")
            output, history = evaluator.evaluate(worker, st["title"])
            passed = history[-1]["pass"] if history else True
        else:
            output, passed = worker.respond(st["title"]), True
        plan.tasks.append({**st, "output": output, "passed": passed})
    return plan


def _demo() -> None:
    # Orchestrator plan + two worker outputs, all scripted for a reproducible offline run.
    orch_llm = MockLLM(
        scripted=[
            '{"subtasks": ['
            '{"id": 1, "title": "Define scope and milestones", "role": "project manager"},'
            '{"id": 2, "title": "Identify technical risks", "role": "tech lead"}]}'
        ]
    )
    worker_llms = {
        1: MockLLM(scripted=["Scope: ship MVP in 6 weeks; milestones at weeks 2/4/6."]),
        2: MockLLM(scripted=["Risks: auth integration, data migration, load at launch."]),
    }
    plan = run_project_workflow("Launch a customer support portal", orch_llm, worker_llms)
    print(plan.render())


if __name__ == "__main__":
    _demo()
