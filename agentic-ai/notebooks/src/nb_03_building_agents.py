"""Build NB03 — Building Agents (tools, structured output, state, memory, RAG, eval)."""

from _nbtools import BOOTSTRAP, code, md, write

cells = [
    md(r"""
# 03 · Building Agents

> Course 03 of the **Agentic AI Course**. Pairs with [`courses/03-building-agents.md`](../courses/03-building-agents.md).

**Runs offline.** LLM-driven cells script a `MockLLM` for reproducibility; the SQL, state-machine,
memory, and retrieval mechanics are **real, runnable Python**. You'll build: tools/function calling,
structured outputs (Pydantic), a state machine, short-term memory, a text2SQL agent (real SQLite),
agentic RAG (real keyword retrieval), long-term memory, and an evaluation harness.
"""),
    BOOTSTRAP,
    md(r"""
## 1. Tools / function calling

A tool = name + description + arg schema + **forgiving error messages** (the model reads errors and
retries). Errors return as text, never crash the loop.
"""),
    code(r"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable
from shared.llm import MockLLM, get_llm, system, user, extract_json

@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., object]
    schema: dict
    def __call__(self, **kwargs) -> str:
        try:
            return str(self.func(**kwargs))
        except Exception as e:
            return f"ERROR: {e}. Check arguments against the schema and retry."

def _calculator(expression: str):
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        raise ValueError("only arithmetic characters allowed")
    return eval(expression, {"__builtins__": {}}, {})

calculator = Tool("calculator", "Evaluate arithmetic, e.g. '2*(3+4)'.", _calculator,
                  {"type": "object", "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]})
print(calculator(expression="2*(3+4)"))            # 14
print(calculator(expression="__import__('os')"))   # safely rejected as ERROR
"""),
    md(r"""
## 2. Structured outputs with Pydantic

Make the model return JSON that conforms to a schema, then **validate** it into a typed object.
The schema you send is the same contract you validate against — no drift.
"""),
    code(r"""
import json
from pydantic import BaseModel, Field, ValidationError

class Ticket(BaseModel):
    title: str
    priority: str = Field(description="low | medium | high")
    tags: list[str] = []
    estimate_hours: float

def extract_ticket(text, model):
    schema = json.dumps(Ticket.model_json_schema())
    raw = model.chat([system(f"Extract a ticket as JSON matching:\n{schema}\nReturn ONLY JSON."),
                      user(text)])
    try:
        return Ticket(**extract_json(raw))
    except (ValidationError, ValueError) as e:
        return f"validation failed: {e}"

m = MockLLM(scripted=['{"title": "Login broken", "priority": "high", '
                      '"tags": ["auth"], "estimate_hours": 3}'])
t = extract_ticket("Users can't log in since the deploy.", m)
print(type(t).__name__, "->", t)
print("priority:", t.priority, "| hours:", t.estimate_hours)
"""),
    md(r"""
## 3. State management (a state machine)

Model agent state explicitly. Illegal transitions are **rejected** — preventing the "LLM wanders
off the workflow" failure.
"""),
    code(r"""
from dataclasses import dataclass, field
from enum import Enum

class State(str, Enum):
    GREETING = "greeting"; COLLECTING = "collecting"; CONFIRMING = "confirming"; DONE = "done"

TRANSITIONS = {
    State.GREETING: {State.COLLECTING},
    State.COLLECTING: {State.CONFIRMING},
    State.CONFIRMING: {State.COLLECTING, State.DONE},
    State.DONE: set(),
}

@dataclass
class OrderSession:
    state: State = State.GREETING
    cart: list[str] = field(default_factory=list)
    def transition(self, to):
        if to not in TRANSITIONS[self.state]:
            raise ValueError(f"illegal transition {self.state.value} -> {to.value}")
        self.state = to

s = OrderSession()
s.transition(State.COLLECTING); s.cart.append("A4 paper")
s.transition(State.CONFIRMING); s.transition(State.DONE)
print("final:", s.state.value, s.cart)
try:
    s.transition(State.COLLECTING)        # DONE is terminal -> rejected
except ValueError as e:
    print("blocked:", e)
"""),
    md(r"""
## 4. Short-term memory (sliding window)

Keep the last *k* turns so the model stays coherent without blowing the context window.
"""),
    code(r"""
from collections import deque
from shared.llm import system as sys_msg

class ChatMemory:
    def __init__(self, persona, k_turns=4):
        self.persona = persona
        self.turns = deque(maxlen=k_turns)   # automatically drops the oldest turn
    def add(self, role, content):
        self.turns.append({"role": role, "content": content})
    def messages(self):
        return [sys_msg(self.persona), *self.turns]

mem = ChatMemory("You are a barista bot.", k_turns=4)
for role, text in [("user", "I'd like a latte."), ("assistant", "Hot or iced?"),
                   ("user", "Iced, oat milk."), ("assistant", "Got it."),
                   ("user", "Make it a large.")]:
    mem.add(role, text)
print("window keeps", len(mem.turns), "turns:")
for t in mem.messages():
    print(" ", t["role"], "->", t["content"])
"""),
    md(r"""
## 5. Database agent — text2SQL (real SQLite)

Give the model the schema, get a **read-only** SELECT, execute it for real, then summarize. The
guardrail rejects anything that isn't a SELECT.
"""),
    code(r"""
import sqlite3

conn = sqlite3.connect(":memory:")
conn.executescript('''
CREATE TABLE games(id INTEGER, title TEXT, platform TEXT, year INTEGER, genre TEXT);
INSERT INTO games VALUES
 (1,'Chrono Trigger','SNES',1995,'RPG'),
 (2,'Half-Life','PC',1998,'FPS'),
 (3,'Celeste','PC',2018,'Platformer'),
 (4,'Hades','PC',2020,'Roguelike');
''')
SCHEMA = "TABLE games(id, title, platform, year, genre)"

def text2sql(question, model, conn):
    sql = model.chat([
        system(f"Schema:\n{SCHEMA}\nWrite ONE read-only SQLite SELECT. Return ONLY SQL."),
        user(question)]).strip().strip("`").removeprefix("sql").strip()
    if not sql.lower().lstrip().startswith("select"):
        return "ERROR: only SELECT permitted"
    rows = conn.execute(sql).fetchall()
    print("  SQL:", sql)
    return rows

sql_model = MockLLM(scripted=["SELECT title, year FROM games WHERE platform='PC' ORDER BY year"])
print("RESULT:", text2sql("Which PC games do we have, oldest first?", sql_model, conn))

danger = MockLLM(scripted=["DROP TABLE games"])
print("BLOCKED:", text2sql("delete everything", danger, conn))   # guardrail stops it
"""),
    md(r"""
## 6. Agentic RAG (real keyword retrieval + a reflection loop)

Plain RAG retrieves once and answers. **Agentic** RAG judges whether the context is sufficient and
**reformulates + retries** if not. (Here we use a keyword scorer so it runs with no vector-DB
install; swap in ChromaDB via `--extra rag`.)
"""),
    code(r"""
DOCS = [
    "Chrono Trigger (1995) is a SNES role-playing game with time travel.",
    "Hades (2020) is a roguelike where you escape the underworld.",
    "Celeste (2018) is a precision platformer about climbing a mountain.",
    "Half-Life (1998) is a first-person shooter set in Black Mesa.",
]

def retrieve(query, k=2):
    scored = sorted(DOCS, key=lambda d: -sum(w in d.lower() for w in query.lower().split()))
    return scored[:k]

def agentic_rag(question, model, max_retries=2):
    query = question
    for attempt in range(max_retries + 1):
        ctx = retrieve(query)
        verdict = extract_json(model.chat([
            system('Reply JSON {"sufficient": bool, "better_query": str}.'),
            user(f"Q: {question}\nCONTEXT: {ctx}")]))
        print(f"  attempt {attempt}: query={query!r} sufficient={verdict['sufficient']}")
        if verdict["sufficient"]:
            return model.chat([system("Answer from context only."),
                               user(f"Q: {question}\nCONTEXT: {ctx}")])
        query = verdict["better_query"]
    return "Not enough information."

# First retrieval is judged insufficient -> reformulate -> sufficient -> answer.
rag_model = MockLLM(scripted=[
    '{"sufficient": false, "better_query": "roguelike underworld game"}',
    '{"sufficient": true, "better_query": ""}',
    "Hades (2020) is the roguelike about escaping the underworld.",
])
print("ANSWER:", agentic_rag("what's the underworld escape game?", rag_model))
"""),
    md(r"""
## 7. Long-term memory (persists across sessions)

Write memories to a store; at session start, **recall** the relevant ones by similarity and inject
them. Three kinds: **semantic** (facts), **episodic** (events), **procedural** (how-to).
"""),
    code(r"""
class LongTermMemory:
    def __init__(self):
        self._store = []
    def remember(self, text):
        self._store.append(text)
    def recall(self, context, k=2):
        return sorted(self._store,
                      key=lambda m: -sum(w in m.lower() for w in context.lower().split()))[:k]

ltm = LongTermMemory()
ltm.remember("User is vegetarian.")                    # semantic
ltm.remember("User booked Tokyo last month.")          # episodic
ltm.remember("User prefers terse, bullet-point replies.")  # procedural
print("recalled for a food query:", ltm.recall("recommend a restaurant dish"))
"""),
    md(r"""
## 8. Agent evaluation

Stochastic, multi-step systems need a **fixed test set** + LLM-as-judge for fuzzy quality, plus
deterministic checks (right tool? valid JSON? under budget?). Track the mean score as a CI gate.
"""),
    code(r"""
def judge(question, answer, reference, model):
    return extract_json(model.chat([
        system('Score 1-5 vs reference. Reply JSON {"score": int, "reason": str}.'),
        user(f"Q: {question}\nANSWER: {answer}\nREF: {reference}")]))

def eval_suite(agent_fn, cases, judge_model):
    scores = []
    for c in cases:
        ans = agent_fn(c["question"])
        v = judge(c["question"], ans, c["reference"], judge_model)
        print(f"  {c['question'][:30]:32} score={v['score']}")
        scores.append(v["score"])
    return sum(scores) / len(scores)

cases = [{"question": "underworld escape game?", "reference": "Hades"},
         {"question": "SNES time-travel RPG?", "reference": "Chrono Trigger"}]
agent_fn = lambda q: "Hades" if "underworld" in q else "Chrono Trigger"
judge_model = MockLLM(scripted=['{"score": 5, "reason": "correct"}',
                                '{"score": 5, "reason": "correct"}'])
print("\nmean score:", eval_suite(agent_fn, cases, judge_model), "/ 5")
"""),
    md(r"""
## Recap

You built every capability of a real agent: **tools, structured output, state, short/long-term
memory, SQL + retrieval, agentic RAG, and evaluation.** Combine them and you have a research agent.

**Project:** [Research Agent](../projects/03_research_agent/) ·
**Next:** [04 · Multi-Agent Systems](04_multi_agent_systems.ipynb).
"""),
]

write(cells, "03_building_agents.ipynb")
