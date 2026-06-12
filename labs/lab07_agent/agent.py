"""The ReAct agent loop for Lab 07 — implement the `TODO`s.

An **agent** is an LLM in a loop: *think → act (call a tool) → observe → … → answer*.
The model ("brain") is pluggable; a deterministic mock brain in `brain.py` lets the
spec tests run offline. You implement the two pieces that make the loop work:
`parse_action` (read the brain's output) and `Agent.run` (the loop itself).

Run the spec:  uv run pytest -m todo tests/test_lab07_agent.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lab07_agent.tools import Tool

# The text protocol the brain speaks (ReAct). The brain emits either:
#
#     Thought: <reasoning>
#     Action: <tool_name>
#     Action Input: <argument>
#
# ...or, when it's done:
#
#     Thought: <reasoning>
#     Final Answer: <text>


@dataclass
class Step:
    """One entry in the agent's trajectory (for logging / evaluation)."""

    kind: str  # "action" | "observation" | "answer"
    content: str


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)


def parse_action(text: str) -> tuple[str, object]:
    """Parse one brain output into a typed decision.

    Returns one of:
      * `("final", answer_str)`            — the brain emitted a Final Answer
      * `("action", (tool_name, arg))`     — the brain wants to call a tool
      * `("error", text)`                  — nothing parseable was found

    TODO:
      1. if "Final Answer:" is in `text`: return ("final", <text after it, stripped>).
      2. else look for an Action + Action Input. A regex like
             r"Action:\\s*(\\w+)\\s*Action Input:\\s*(.+)"
         with `re.DOTALL` captures the tool name and argument; return
         ("action", (name.strip(), arg.strip())).
      3. otherwise return ("error", text).
    """
    raise NotImplementedError("Implement parse_action — see the TODO above")


class Agent:
    """A minimal ReAct agent: loop the brain + tools until an answer or a step cap."""

    def __init__(self, tools: dict[str, Tool], brain, max_steps: int = 6):
        self.tools = tools  # name -> Tool
        self.brain = brain  # callable(history) -> str
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        """Run the ReAct loop and return the final answer plus the trajectory.

        TODO — implement the loop:
          1. history = [{"role": "user", "content": question}];  steps = []
          2. repeat up to self.max_steps times:
             a. text = self.brain(history)
             b. append {"role": "assistant", "content": text} to history
             c. kind, payload = parse_action(text)
             d. if kind == "final":
                    record Step("answer", payload); return AgentResult(payload, steps)
             e. if kind == "action":
                    tool_name, arg = payload
                    obs = self.tools[tool_name](arg) if tool_name in self.tools \\
                          else f"ERROR: unknown tool {tool_name}"
                    record Step("action", f"{tool_name}({arg})") and Step("observation", obs)
                    append {"role": "tool", "content": obs} to history
             f. if kind == "error":
                    return AgentResult("(could not parse brain output)", steps)
          3. if the loop finishes without a Final Answer:
                 return AgentResult("(max steps reached)", steps)

        Tip: appending the tool observation to `history` is what lets the brain
        "see" the result on the next iteration — that's the whole trick.
        """
        raise NotImplementedError("Implement the ReAct loop — see the TODO above")
