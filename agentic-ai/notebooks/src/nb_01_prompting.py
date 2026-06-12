"""Build NB01 — Prompting for Effective LLM Reasoning and Planning."""

from _nbtools import BOOTSTRAP, code, md, write

cells = [
    md(r"""
# 01 · Prompting for Effective LLM Reasoning and Planning

> Course 01 of the **Agentic AI Course**. Pairs with [`courses/01-prompting.md`](../courses/01-prompting.md).

**Runs offline.** Every cell uses the course's `shared.llm` client. With no API key it uses a
deterministic `MockLLM` (we *script* its replies so output is reproducible); set `OPENAI_API_KEY`
and the **same code** calls a real model.

### You will build
1. Role-based prompts (the R-T-C-E-O template)
2. Chain-of-Thought reasoning
3. A **ReAct** agent loop (reason + act with a tool) — fully runnable
4. Instruction refinement to a JSON contract
5. A validated **prompt chain**
6. A self-correcting **feedback loop**
"""),
    BOOTSTRAP,
    code(r"""
from shared.llm import get_llm, system, user, assistant, MockLLM, extract_json

# Auto-selects a real model if OPENAI_API_KEY is set, else the offline mock.
llm = get_llm()
print("backend:", type(llm).__name__)
print(llm.chat([system("You are concise."), user("What is agentic AI in one line?")]))
"""),
    md(r"""
## 1. Role-Based Prompting — the R-T-C-E-O template

The five highest-leverage prompt components: **R**ole, **T**ask, **C**ontext, **E**xamples,
**O**utput format. A reusable composer keeps prompts disciplined.
"""),
    code(r"""
def rtceo(role, task, context="", examples="", output_format=""):
    sys = f"ROLE: {role}"
    if output_format:
        sys += f"\n\nOUTPUT FORMAT (follow exactly):\n{output_format}"
    msg = f"TASK: {task}"
    if context:
        msg += f"\n\nCONTEXT:\n{context}"
    if examples:
        msg += f"\n\nEXAMPLES:\n{examples}"
    return [system(sys), user(msg)]

# Scripted reply so this prints the same thing offline; delete `scripted=` for a real model.
demo = MockLLM(scripted=["An algorithm is a recipe: a list of clear steps that always reach the "
                         "same result, like sorting cards smallest to largest."])
messages = rtceo(
    role="You are Ada Lovelace. Stay in character.",
    task="Explain what an algorithm is to a curious child.",
    output_format="2-3 warm sentences, no modern jargon.",
)
print(demo.chat(messages))
"""),
    md(r"""
## 2. Chain-of-Thought (CoT)

Ask the model to **think step by step** and end with a delimited answer your program can parse.
CoT trades tokens for accuracy on multi-step problems.
"""),
    code(r"""
cot_llm = MockLLM(scripted=[
    "Step 1: 12 pens = 4 groups of 3.\nStep 2: each group costs $4.\n"
    "Step 3: 4 x $4 = $16.\nANSWER: $16"
])
out = cot_llm.chat([
    system("Solve step by step. Show reasoning, then end with 'ANSWER: <value>'."),
    user("A store sells pens at 3 for $4. How much do 12 pens cost?"),
])
print(out)
answer = out.split("ANSWER:")[-1].strip()
print("\nparsed answer ->", answer)   # programs consume the delimited answer, not the prose
"""),
    md(r"""
## 3. ReAct = Reason + Act (a runnable agent loop)

CoT only *thinks*. **ReAct** interleaves **Thought -> Action -> Observation**, letting the model
call a tool and read the result. The loop is just *string protocol + parser + tool dispatcher*.
"""),
    code(r"""
import re

def calculator(expr: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
        return "ERROR: only arithmetic allowed"
    return str(eval(expr, {"__builtins__": {}}, {}))   # sandboxed: no builtins

TOOLS = {"calculator": calculator}

REACT_SYSTEM = system(
    "Use this exact format each turn:\n"
    "Thought: <reasoning>\nAction: <tool>\nAction Input: <input>\n"
    "or when done:\nThought: <reasoning>\nFinal Answer: <answer>\n"
    f"Tools: {list(TOOLS)}"
)

def react(question, model, max_steps=5, verbose=True):
    history = [REACT_SYSTEM, user(question)]
    for _ in range(max_steps):
        text = model.chat(history)
        history.append(assistant(text))
        if verbose:
            print(text, "\n---")
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()
        m = re.search(r"Action:\s*(\w+)\s*Action Input:\s*(.+)", text, re.S)
        if not m:
            return "(no action parsed)"
        name, arg = m.group(1).strip(), m.group(2).strip()
        obs = TOOLS.get(name, lambda _: "ERROR: unknown tool")(arg)
        history.append({"role": "tool", "content": f"Observation: {obs}"})
        if verbose:
            print(f"Observation: {obs}\n===")
    return "(max steps reached)"

# Script the model's two ReAct turns so the loop is reproducible offline.
brain = MockLLM(scripted=[
    "Thought: I should compute this.\nAction: calculator\nAction Input: 12 * (3 + 4)",
    "Thought: I have the result.\nFinal Answer: 84",
])
print("FINAL:", react("What is 12 * (3 + 4)?", brain))
"""),
    md(r"""
With a real model you would write `react(question, get_llm())` — the loop is **identical**, only
the brain changes. Notice the agent never computed `84` itself; the **tool** did, and the model
just orchestrated. That separation is the whole idea behind agents.
"""),
    md(r"""
## 4. Instruction refinement -> a JSON contract

The moment a later step parses the reply, you must pin the **output shape**. `extract_json` pulls
JSON out even if the model wraps it in prose or ```json fences.
"""),
    code(r"""
REFINED = system(
    "ROLE: registered dietitian.\nTASK: analyze the recipe; flag allergens; estimate calories.\n"
    "RULES: use ONLY the ingredients given.\n"
    'OUTPUT: JSON like {"allergens": [...], "calories": <int>, "vegan": <bool>}'
)
dietitian = MockLLM(scripted=[
    'Here is the analysis:\n```json\n'
    '{"allergens": ["peanuts"], "calories": 410, "vegan": true}\n```'
])
raw = dietitian.chat([REFINED, user("Recipe: oats, almond milk, banana, peanut butter.")])
data = extract_json(raw)            # robust: strips the prose + code fence
print(type(data), data)
print("allergens ->", data["allergens"])
"""),
    md(r"""
## 5. Prompt chaining with a validation gate

Decompose into stages; **validate the hand-off** between them with Pydantic so errors surface
early instead of compounding.
"""),
    code(r"""
from pydantic import BaseModel, ValidationError

class Claim(BaseModel):
    policy_id: str
    amount: float
    category: str

def stage_extract(text, model):
    raw = model.chat([
        system('Extract JSON {"policy_id": str, "amount": number, "category": str}.'),
        user(text)])
    return Claim(**extract_json(raw))          # raises if the gate contract is violated

def stage_route(claim, model):
    return model.chat([
        system("Reply with one of: AUTO_APPROVE, REVIEW, DENY."),
        user(claim.model_dump_json())]).strip()

def triage(text, model):
    try:
        claim = stage_extract(text, model)
    except (ValidationError, ValueError) as e:
        return f"REJECTED at extraction: {e}"
    return stage_route(claim, model)

good = MockLLM(scripted=['{"policy_id": "P-12", "amount": 240.0, "category": "auto"}',
                         "AUTO_APPROVE"])
print("valid claim   ->", triage("Policy P-12, $240, fender scratch", good))

bad = MockLLM(scripted=['{"policy_id": "P-9", "category": "auto"}'])   # missing 'amount'
print("broken claim  ->", triage("malformed claim text", bad))
"""),
    md(r"""
**Why gates matter:** chains compound errors. A 90%-reliable stage run 4x is only ~66% reliable
end-to-end. A gate converts silent corruption into a loud, recoverable failure.
"""),
    md(r"""
## 6. Self-correcting feedback loop

The model writes code, an **objective** check (unit tests) runs it, and the *failures are fed back*
so it can debug itself. This is the seed of Course 2's Evaluator-Optimizer pattern.
"""),
    code(r"""
def run_tests(code_str, tests):
    ns = {}
    try:
        exec(code_str, ns)
        for inp, expected in tests:
            got = ns["solve"](inp)
            assert got == expected, f"solve({inp!r})={got!r}, expected {expected!r}"
        return True, "all passed"
    except Exception as e:
        return False, str(e)          # the error string IS the feedback signal

def self_correcting(task, tests, model, max_iters=3):
    history = [system("Write Python. Return ONLY a def solve(x): ... function."), user(task)]
    code_str = ""
    for attempt in range(1, max_iters + 1):
        code_str = model.chat(history)
        ok, report = run_tests(code_str, tests)
        print(f"attempt {attempt}: {'PASS' if ok else 'FAIL'} - {report}")
        if ok:
            return code_str, attempt
        history += [assistant(code_str),
                    user(f"Tests failed: {report}\nFix it. Return ONLY the function.")]
    return code_str, max_iters

tests = [(2, 4), (3, 9), (5, 25)]
# First reply has a bug (uses x*2); after seeing the failure, the second reply fixes it (x**2).
coder = MockLLM(scripted=["def solve(x):\n    return x * 2",
                          "def solve(x):\n    return x ** 2"])
final_code, n = self_correcting("Return the square of x.", tests, coder)
print(f"\nconverged in {n} attempts:\n{final_code}")
"""),
    md(r"""
## Recap

| Technique | One-liner |
|-----------|-----------|
| Role prompting | a persona conditions tone + expertise |
| CoT | think step by step, parse a delimited answer |
| ReAct | think **and act** via tools in a loop |
| Refinement | tune R-T-C-E-O to a JSON contract |
| Chaining + gates | validate every hand-off; errors compound |
| Feedback loop | revise using an **objective** signal |

These are the atoms of every agent in the rest of the program.
**Project:** [Trip Planner](../projects/01_trip_planner/) ·
**Next notebook:** [02 · Agentic Workflows](02_agentic_workflows.ipynb).
"""),
]

write(cells, "01_prompting.ipynb")
