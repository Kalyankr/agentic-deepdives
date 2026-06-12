"""Tests for the Project-Management Workflow.

Imports the reference ``solution`` so they pass out of the box. To grade your own work, change
the import to ``import starter as impl``.
"""

import solution as impl  # <- change to `import starter as impl` to test your implementation

from shared.llm import MockLLM


def test_direct_prompt_agent_returns_text():
    agent = impl.DirectPromptAgent(MockLLM(scripted=["hello world"]))
    assert agent.respond("say hi") == "hello world"


def test_orchestrator_plans_subtasks():
    llm = MockLLM(
        scripted=[
            '{"subtasks": [{"id": 1, "title": "A", "role": "pm"},'
            '{"id": 2, "title": "B", "role": "dev"}]}'
        ]
    )
    subtasks = impl.OrchestratorAgent(llm).plan("ship a feature")
    assert [s["id"] for s in subtasks] == [1, 2]
    assert subtasks[0]["title"] == "A"


def test_evaluation_agent_loops_until_pass():
    worker = impl.PersonaAgent(MockLLM(scripted=["v1 draft", "v2 improved"]), persona="writer")
    evaluator = impl.EvaluationAgent(
        MockLLM(
            scripted=[
                '{"pass": false, "feedback": "add detail"}',
                '{"pass": true, "feedback": "ok"}',
            ]
        ),
        criteria="detailed",
    )
    final, history = evaluator.evaluate(worker, "write a blurb")
    assert final == "v2 improved"
    assert len(history) == 2 and history[-1]["pass"] is True


def test_workflow_assembles_plan():
    orch = MockLLM(scripted=['{"subtasks": [{"id": 1, "title": "scope", "role": "pm"}]}'])
    workers = {1: MockLLM(scripted=["6-week MVP plan"])}
    plan = impl.run_project_workflow("launch portal", orch, workers)
    assert plan.goal == "launch portal"
    assert plan.tasks[0]["output"] == "6-week MVP plan"
