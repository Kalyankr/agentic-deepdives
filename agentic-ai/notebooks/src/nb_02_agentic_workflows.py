"""Build NB02 — Agentic Workflows (the five patterns)."""

from _nbtools import BOOTSTRAP, code, md, write

cells = [
    md(r"""
# 02 · Agentic Workflows — The Five Patterns

> Course 02 of the **Agentic AI Course**. Pairs with [`courses/02-agentic-workflows.md`](../courses/02-agentic-workflows.md).

**Runs offline** (deterministic `MockLLM`; set `OPENAI_API_KEY` for a real model). You'll build a
reusable `Agent` class, then the five canonical workflow patterns from Anthropic's *Building
Effective Agents*:

1. **Prompt Chaining** · 2. **Routing** · 3. **Parallelization** · 4. **Evaluator-Optimizer** ·
5. **Orchestrator-Workers**
"""),
    BOOTSTRAP,
    md(r"""
## 0. A reusable `Agent` building block

One node = a persona + an LLM, callable as `text -> text`. All five patterns wire `Agent`
instances together. We let each agent carry its **own** model so we can script replies per agent.
"""),
    code(r"""
from __future__ import annotations
from dataclasses import dataclass, field
from shared.llm import BaseLLM, MockLLM, get_llm, system, user, extract_json

@dataclass
class Agent:
    name: str
    instructions: str
    llm: BaseLLM = field(default_factory=get_llm)   # real model if a key is set, else mock
    def run(self, task: str, context: str = "") -> str:
        content = task if not context else f"{task}\n\nCONTEXT:\n{context}"
        return self.llm.chat([system(self.instructions), user(content)])

# A specialized agent is just an instance with different instructions:
hello = Agent("greeter", "Greet the user warmly in one line.",
              llm=MockLLM(scripted=["Hey there, welcome aboard!"]))
print(hello.run("a new user just signed up"))
"""),
    md(r"""
## 1. Prompt Chaining

A fixed sequence where each step consumes the previous output. A **gate** validates each hand-off.
"""),
    code(r"""
def chain(task, steps, gate=None):
    data = task
    for agent in steps:
        data = agent.run(data)
        print(f"  [{agent.name}] -> {data}")
        if gate and not gate(data):
            raise ValueError(f"{agent.name} failed the gate")
    return data

outline = Agent("outliner", "...", llm=MockLLM(scripted=["- tests catch bugs\n- tests document"]))
draft   = Agent("writer",   "...", llm=MockLLM(scripted=["Unit tests catch bugs early and act as living docs."]))
polish  = Agent("editor",   "...", llm=MockLLM(scripted=["Unit tests catch regressions early and double as living documentation."]))

print("RESULT:", chain("benefits of unit tests", [outline, draft, polish], gate=lambda t: len(t) > 0))
"""),
    md(r"""
## 2. Routing

A cheap classifier picks the path; the chosen specialist does the work. Always include a safe
fallback for mis-routes.
"""),
    code(r"""
def router(query, routes, classifier_llm):
    labels = list(routes)
    decision = classifier_llm.chat([
        system(f"Classify into one of {labels}. Reply JSON {{\"route\": \"<label>\"}}."),
        user(query)])
    choice = extract_json(decision).get("route", labels[-1])
    choice = choice if choice in routes else labels[-1]    # safe fallback
    print(f"  routed to: {choice}")
    return routes[choice].run(query)

routes = {
    "billing":   Agent("billing",   "Handle billing.",  llm=MockLLM(scripted=["Your refund posts in 3-5 days."])),
    "technical": Agent("technical", "Debug issues.",    llm=MockLLM(scripted=["Try clearing the cache, then re-login."])),
    "general":   Agent("general",   "Answer politely.", llm=MockLLM(scripted=["Happy to help!"])),
}
classifier = MockLLM(scripted=['{"route": "billing"}'])
print("ANSWER:", router("I want a refund on my invoice", routes, classifier))
"""),
    md(r"""
## 3. Parallelization

Run **independent** sub-tasks concurrently (LLM calls are I/O-bound, so threads give a real
speedup), then **aggregate** with a synthesizer.
"""),
    code(r"""
from concurrent.futures import ThreadPoolExecutor

def parallel(task, workers, synthesizer):
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        findings = list(pool.map(lambda a: f"[{a.name}] {a.run(task)}", workers))
    for f in findings:
        print(" ", f)
    return synthesizer.run(task, context="\n".join(findings))

workers = [
    Agent("risk",  "Financial risks only.", llm=MockLLM(scripted=["Currency exposure is high."])),
    Agent("legal", "Legal issues only.",    llm=MockLLM(scripted=["Indemnity clause is one-sided."])),
    Agent("ops",   "Operational only.",     llm=MockLLM(scripted=["Delivery SLA is unrealistic."])),
]
synth = Agent("synth", "Merge into a prioritized brief.",
              llm=MockLLM(scripted=["Top risks: 1) one-sided indemnity 2) FX exposure 3) tight SLA."]))
print("\nBRIEF:", parallel("Review the Q3 vendor contract", workers, synth))
"""),
    md(r"""
## 4. Evaluator-Optimizer

A generator proposes; an evaluator critiques against explicit criteria; repeat until it passes.
The evaluator must return **specific, actionable** feedback.
"""),
    code(r"""
def evaluator_optimizer(task, generator, evaluator, max_rounds=3):
    draft = generator.run(task)
    for r in range(1, max_rounds + 1):
        verdict = extract_json(evaluator.run(task, context=f"CANDIDATE: {draft}"))
        print(f"  round {r}: pass={verdict['pass']} draft={draft!r}")
        if verdict["pass"]:
            return draft, "accepted"
        draft = generator.run(f"{task}\nRevise using: {verdict['feedback']}")
    return draft, "max_rounds"

gen = Agent("writer", "Write a tagline.",
            llm=MockLLM(scripted=["Paper for every office need.",          # round 1 (rejected)
                                  "Paper that works as hard as you do."]))   # round 2 (accepted)
crit = Agent("critic", "Brand editor. Strict.",
             llm=MockLLM(scripted=['{"pass": false, "feedback": "too generic; add benefit + punch"}',
                                   '{"pass": true, "feedback": "great"}']))
print("\nFINAL:", evaluator_optimizer("tagline for a paper company", gen, crit))
"""),
    md(r"""
## 5. Orchestrator-Workers

The orchestrator *dynamically* plans subtasks at runtime, delegates each to a worker, then a
synthesizer combines results. Unlike chaining, the subtasks aren't known in advance.
"""),
    code(r"""
def orchestrate(goal, orchestrator, worker_llm_factory, synthesizer):
    plan = extract_json(orchestrator.run(goal))
    results = []
    for st in plan["subtasks"]:
        worker = Agent(f"worker-{st['id']}", "Focused analyst.", llm=worker_llm_factory(st["id"]))
        out = worker.run(st["instruction"])
        results.append(f"[{st['id']}] {out}")
        print(" ", results[-1])
    return synthesizer.run(goal, context="\n".join(results))

orchestrator = Agent("planner", "Plan subtasks.", llm=MockLLM(scripted=[
    '{"subtasks": [{"id": 1, "instruction": "summarize recent news"},'
    ' {"id": 2, "instruction": "list top competitors"}]}']))
worker_replies = {1: MockLLM(scripted=["Demand for recycled paper is rising."]),
                  2: MockLLM(scripted=["Main rivals: Dunder Mifflin, Prestige Worldwide."])}
synth = Agent("synth", "Combine into a report.",
              llm=MockLLM(scripted=["Market brief: recycled-paper demand up; rivals are DM & PW."]))
print("\nREPORT:", orchestrate("market brief for a paper company",
                               orchestrator, lambda i: worker_replies[i], synth))
"""),
    md(r"""
## Choosing a pattern

| If the task… | Use |
|---|---|
| has fixed ordered steps | Prompt Chaining |
| splits into known categories | Routing |
| has independent parts / needs voting | Parallelization |
| needs iterative quality improvement | Evaluator-Optimizer |
| needs runtime-planned decomposition | Orchestrator-Workers |
| genuinely can't be predetermined | a real Agent (Course 3) |

Patterns **compose**: a router can dispatch to a chain; an orchestrator's workers can each be
evaluator-optimizer loops.

**Project:** [Project-Management Workflow](../projects/02_project_management_workflow/) ·
**Next:** [03 · Building Agents](03_building_agents.ipynb).
"""),
]

write(cells, "02_agentic_workflows.ipynb")
