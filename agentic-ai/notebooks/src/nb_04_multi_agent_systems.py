"""Build NB04 — Multi-Agent Systems (architecture, orchestration, coordination, RAG)."""

from _nbtools import BOOTSTRAP, code, md, write

cells = [
    md(r"""
# 04 · Multi-Agent Systems

> Course 04 of the **Agentic AI Course**. Pairs with [`courses/04-multi-agent-systems.md`](../courses/04-multi-agent-systems.md).

**Runs offline.** You'll build typed inter-agent messaging + a registry, a coordinator
(sequential / parallel / conditional), a routing agent, a shared-state **blackboard**, real
**concurrency control** (a lock that prevents overselling under threads), multi-agent RAG, and a
tiny end-to-end **sales-team** simulation that previews the capstone project.
"""),
    BOOTSTRAP,
    md(r"""
## 1. Typed messaging + a registry

Agents exchange typed `Message` envelopes; a `Registry` lets a coordinator address agents by name.
"""),
    code(r"""
from __future__ import annotations
from dataclasses import dataclass, field
from shared.llm import BaseLLM, MockLLM, get_llm, system, user, extract_json

@dataclass
class Message:
    sender: str
    recipient: str
    content: str

@dataclass
class Agent:
    name: str
    instructions: str
    llm: BaseLLM = field(default_factory=get_llm)
    def handle(self, msg: Message) -> Message:
        reply = self.llm.chat([system(self.instructions), user(msg.content)])
        return Message(self.name, msg.sender, reply)

class Registry:
    def __init__(self):
        self.agents: dict[str, Agent] = {}
    def register(self, agent):
        self.agents[agent.name] = agent
        return agent
    def send(self, msg):
        return self.agents[msg.recipient].handle(msg)

reg = Registry()
reg.register(Agent("inventory", "Report stock numerically.",
                   llm=MockLLM(scripted=["A4 paper: 1200 reams in stock."])))
print(reg.send(Message("coord", "inventory", "How many reams of A4?")).content)
"""),
    md(r"""
## 2. Orchestration: sequential, parallel, conditional

The coordinator decides *how* specialists run. Real systems mix all three.
"""),
    code(r"""
from concurrent.futures import ThreadPoolExecutor

class Coordinator:
    def __init__(self, registry):
        self.reg = registry
    def sequential(self, steps, task):
        data = task
        for name in steps:
            data = self.reg.send(Message("coord", name, data)).content
        return data
    def parallel(self, names, task):
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            msgs = list(pool.map(lambda n: self.reg.send(Message("coord", n, task)), names))
        return [m.content for m in msgs]
    def conditional(self, task, predicate, if_true, if_false):
        return self.reg.send(Message("coord", if_true if predicate(task) else if_false, task)).content

# Two scripted replies each: one for the parallel call, one for the conditional call.
reg.register(Agent("pricing", "Be brief.", llm=MockLLM(scripted=["A4 is $5/ream.", "A4 is $5/ream."])))
reg.register(Agent("shipping", "Be brief.", llm=MockLLM(scripted=["Ships in 2 days.", "Rush ships overnight."])))
coord = Coordinator(reg)
print("parallel:", coord.parallel(["pricing", "shipping"], "info for A4 order"))
print("conditional:", coord.conditional("urgent rush order",
      predicate=lambda t: "urgent" in t, if_true="shipping", if_false="pricing"))
"""),
    md(r"""
## 3. Routing agent

Inspect each request and forward to the right specialist (multi-agent version of Course 2 routing).
Pass agents **only what they need** (least context).
"""),
    code(r"""
class Router:
    def __init__(self, llm, specialists: dict[str, str]):
        self.llm = llm
        self.specialists = specialists
    def route(self, request):
        catalog = "\n".join(f"- {n}: {d}" for n, d in self.specialists.items())
        decision = self.llm.chat([
            system(f"Route to ONE specialist:\n{catalog}\nReply JSON {{\"agent\": \"<name>\"}}."),
            user(request)])
        return extract_json(decision).get("agent")

router = Router(MockLLM(scripted=['{"agent": "pricing"}']),
                {"pricing": "quotes & prices", "shipping": "delivery & logistics"})
print("routed to:", router.route("How much for 500 reams?"))
"""),
    md(r"""
## 4. Shared state (the blackboard)

One source of truth all agents read/write, so the team never makes contradictory assumptions.
"""),
    code(r"""
@dataclass
class SharedState:
    facts: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    def update(self, key, value, by):
        self.facts[key] = value
        self.history.append(f"{by} set {key}={value!r}")
    def snapshot(self):
        return "; ".join(f"{k}={v}" for k, v in self.facts.items())

state = SharedState()
state.update("customer_budget", 500, by="sales")
state.update("in_stock", True, by="inventory")
print("snapshot:", state.snapshot())
print("audit:", state.history)
"""),
    md(r"""
## 5. State coordination under concurrency (real threads)

When agents act **concurrently** they can race on shared state. A lock serializes the critical
section so we never oversell the last item. This cell genuinely spawns threads.
"""),
    code(r"""
import threading
from concurrent.futures import ThreadPoolExecutor

class CoordinatedInventory:
    def __init__(self, stock):
        self.stock = stock
        self._lock = threading.Lock()
    def reserve(self, item, qty, agent):
        with self._lock:                          # only one agent in the critical section
            have = self.stock.get(item, 0)
            if have < qty:
                return f"CONFLICT: {agent} wanted {qty}, only {have} left"
            self.stock[item] = have - qty
            return f"OK: {agent} reserved {qty} ({self.stock[item]} left)"

inv = CoordinatedInventory({"A4": 10})
# Five agents each try to reserve 3 reams of A4 at the same time (demand 15 > stock 10).
with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(lambda i: inv.reserve("A4", 3, f"agent{i}"), range(5)))
for r in results:
    print(" ", r)
print("final stock:", inv.stock, "(never negative thanks to the lock)")
"""),
    md(r"""
## 6. Multi-agent RAG

Specialized retrievers each own a source; a synthesizer combines + judges + cites.
"""),
    code(r"""
def multi_agent_rag(query, retrievers, synthesizer):
    evidence = [f"[{name}]\n{retrieve(query)}" for name, retrieve in retrievers.items()]
    return synthesizer.handle(
        Message("coord", synthesizer.name,
                f"Question: {query}\n\nEVIDENCE:\n" + "\n\n".join(evidence))).content

retrievers = {
    "catalog": lambda q: "A4 80gsm, A3 100gsm, recycled A4.",
    "pricing": lambda q: "A4 $5/ream, A3 $9/ream, recycled $6/ream.",
}
synth = Agent("synth", "Combine evidence; cite sources.",
              llm=MockLLM(scripted=["Recycled A4 is available at $6/ream [catalog][pricing]."]))
print(multi_agent_rag("price for recycled A4?", retrievers, synth))
"""),
    md(r"""
## 7. Putting it together — a tiny sales-team simulation

A coordinator handles a customer order end-to-end: **route -> check stock -> quote -> reserve**,
all over shared state. This is a miniature of the [Paper Company](../projects/04_sales_team/)
capstone.
"""),
    code(r"""
def handle_order(item, qty, inventory, quote_llm):
    # 1) check stock (shared, thread-safe resource)
    reservation = inventory.reserve(item, qty, agent="sales")
    if reservation.startswith("CONFLICT"):
        return f"Sorry, we can't fulfill that: {reservation}"
    # 2) quote via a pricing agent
    quote = quote_llm.chat([system("Quote total price. Be brief."),
                            user(f"{qty} reams of {item} at $5/ream")])
    return f"{reservation}\nQuote: {quote}"

inv2 = CoordinatedInventory({"A4": 100})
pricing = MockLLM(scripted=["Total: $250 for 50 reams of A4."])
print(handle_order("A4", 50, inv2, pricing))
print("---")
print(handle_order("A4", 80, inv2, pricing))   # only 50 left now -> conflict
"""),
    md(r"""
## Recap

You can now design a multi-agent system: **typed messaging + registry**, **orchestration**
(sequential/parallel/conditional), **routing**, **shared state**, **concurrency control**, and
**multi-agent RAG** — and you simulated a coordinated sales team end-to-end.

**Capstone:** [Paper Company Sales Team](../projects/04_sales_team/). You've now
covered the whole course — build the [four projects](../projects/) to prove it.
"""),
]

write(cells, "04_multi_agent_systems.ipynb")
