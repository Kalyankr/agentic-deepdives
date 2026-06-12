"""Trip Planner — reference solution (Course 01).

A multi-stage travel assistant that demonstrates the four Course-01 skills end to end:

1. **Structured extraction** — turn a free-text request into a validated ``TravelRequest``
   (role prompt + JSON contract, parsed with Pydantic).
2. **Tool use (ReAct-style)** — a deterministic planner queries ``find_activities`` /
   ``get_weather`` tools to assemble a day-by-day itinerary.
3. **Prompt chaining with a gate** — ``validate`` is the gate between "plan" and "accept".
4. **Feedback loop** — if validation fails (over budget, wrong #days, missing interests),
   ``revise`` adjusts the plan and we re-validate until it passes.

Runs offline: the LLM is used only for extraction + a narrative summary, and the demo scripts a
``MockLLM`` so output is reproducible. Set ``OPENAI_API_KEY`` and pass ``get_llm()`` for a real run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from shared.llm import BaseLLM, MockLLM, extract_json, system, user


# --------------------------------------------------------------------------- #
# The structured contract for a trip request.
# --------------------------------------------------------------------------- #
class TravelRequest(BaseModel):
    destination: str
    days: int
    interests: list[str]
    budget_usd: float


# --------------------------------------------------------------------------- #
# Tools — plain functions the planner "acts" through (no LLM, fully deterministic).
# In production these would hit real activity/weather APIs.
# --------------------------------------------------------------------------- #
ACTIVITIES: dict[str, dict[str, list[tuple[str, float]]]] = {
    "Rivertown": {
        "food": [("Street food tour", 30.0), ("Cooking class", 60.0)],
        "history": [("Old town walking tour", 0.0), ("Museum day pass", 25.0)],
        "outdoors": [("River kayak", 45.0), ("Sunset hill hike", 0.0)],
        "art": [("Gallery crawl", 15.0), ("Pottery workshop", 50.0)],
    }
}


def find_activities(destination: str, interest: str) -> list[tuple[str, float]]:
    """Tool: return (activity, cost) options for an interest in a destination."""
    return ACTIVITIES.get(destination, {}).get(interest, [])


def get_weather(destination: str, day: int) -> str:
    """Tool: deterministic stub forecast (sunny/rain cycle) so the demo is reproducible."""
    return ["sunny", "cloudy", "light rain"][day % 3]


# --------------------------------------------------------------------------- #
# The itinerary data structure.
# --------------------------------------------------------------------------- #
@dataclass
class DayPlan:
    day: int
    weather: str
    activity: str
    cost: float


@dataclass
class Itinerary:
    destination: str
    days: list[DayPlan] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(d.cost for d in self.days)

    def interests_covered(self) -> set[str]:
        names = {d.activity for d in self.days}
        covered = set()
        for interest, opts in ACTIVITIES.get(self.destination, {}).items():
            if any(name in names for name, _ in opts):
                covered.add(interest)
        return covered


# --------------------------------------------------------------------------- #
# Stage 1 — extraction (the LLM stage).
# --------------------------------------------------------------------------- #
def extract_request(text: str, llm: BaseLLM) -> TravelRequest:
    """Role prompt + JSON contract -> validated TravelRequest (raises on bad output)."""
    raw = llm.chat(
        [
            system(
                "You are a travel intake assistant. Extract the trip request as JSON with keys: "
                '{"destination": str, "days": int, "interests": [str], "budget_usd": number}. '
                "Return ONLY JSON."
            ),
            user(text),
        ]
    )
    return TravelRequest(**extract_json(raw))


# --------------------------------------------------------------------------- #
# Stage 2 — plan (deterministic tool use).
# --------------------------------------------------------------------------- #
def build_itinerary(req: TravelRequest) -> Itinerary:
    """Assemble one activity per day, rotating through the requested interests.

    Picks the *cheapest* option for each interest first (so revision has room to cut later only
    if needed). This is the "act via tools" stage: every choice comes from ``find_activities``.
    """
    itinerary = Itinerary(destination=req.destination)
    interests = req.interests or ["food"]
    for day in range(req.days):
        interest = interests[day % len(interests)]
        options = find_activities(req.destination, interest)
        if not options:
            itinerary.days.append(
                DayPlan(day + 1, get_weather(req.destination, day), "Free day", 0.0)
            )
            continue
        name, cost = min(options, key=lambda o: o[1])  # cheapest match
        itinerary.days.append(DayPlan(day + 1, get_weather(req.destination, day), name, cost))
    return itinerary


# --------------------------------------------------------------------------- #
# Stage 3 — validate (the gate).
# --------------------------------------------------------------------------- #
def validate(itinerary: Itinerary, req: TravelRequest) -> list[str]:
    """Return a list of constraint violations (empty == the plan passes the gate)."""
    problems: list[str] = []
    if len(itinerary.days) != req.days:
        problems.append(f"expected {req.days} days, got {len(itinerary.days)}")
    if itinerary.total_cost > req.budget_usd:
        problems.append(f"over budget: ${itinerary.total_cost:.0f} > ${req.budget_usd:.0f}")
    missing = set(req.interests) - itinerary.interests_covered()
    available = set(ACTIVITIES.get(req.destination, {}))
    missing &= available  # only flag interests the destination can actually satisfy
    if missing:
        problems.append(f"interests not covered: {sorted(missing)}")
    return problems


# --------------------------------------------------------------------------- #
# Stage 4 — revise (closes the feedback loop).
# --------------------------------------------------------------------------- #
def revise(itinerary: Itinerary, req: TravelRequest) -> Itinerary:
    """One revision step: if over budget, swap the most expensive day for a free/cheaper option."""
    if itinerary.total_cost <= req.budget_usd:
        return itinerary
    worst = max(itinerary.days, key=lambda d: d.cost)
    interest = req.interests[(worst.day - 1) % len(req.interests)] if req.interests else "food"
    cheaper = sorted(find_activities(req.destination, interest), key=lambda o: o[1])
    if cheaper and cheaper[0][1] < worst.cost:
        worst.activity, worst.cost = cheaper[0]
    else:
        worst.activity, worst.cost = "Free day", 0.0
    return itinerary


def plan_trip(text: str, llm: BaseLLM, max_revisions: int = 5) -> tuple[Itinerary, list[str]]:
    """Full pipeline: extract -> build -> (validate -> revise)* -> return plan + any problems."""
    try:
        req = extract_request(text, llm)
    except (ValidationError, ValueError) as exc:
        raise ValueError(f"could not understand the request: {exc}") from exc
    itinerary = build_itinerary(req)
    problems = validate(itinerary, req)
    for _ in range(max_revisions):
        if not problems:
            break
        itinerary = revise(itinerary, req)
        problems = validate(itinerary, req)
    return itinerary, problems


def summarize(itinerary: Itinerary, llm: BaseLLM) -> str:
    """Optional narrative summary (LLM stage). Falls back gracefully with the mock."""
    lines = [f"Day {d.day} ({d.weather}): {d.activity} — ${d.cost:.0f}" for d in itinerary.days]
    return llm.chat(
        [
            system("You are an upbeat travel concierge. Summarize the itinerary in 2 sentences."),
            user("\n".join(lines) + f"\nTotal: ${itinerary.total_cost:.0f}"),
        ]
    )


def _demo() -> None:
    # Script the extraction so the demo is fully reproducible offline.
    llm = MockLLM(
        scripted=[
            '{"destination": "Rivertown", "days": 4, '
            '"interests": ["food", "history", "outdoors"], "budget_usd": 100}',
            "Get ready for a delicious, history-rich long weekend in Rivertown — "
            "with a sunset hike to cap it off, all under budget!",
        ]
    )
    request = "I want 4 days in Rivertown for food, history and the outdoors, budget around $100."
    itinerary, problems = plan_trip(request, llm)
    print("REQUEST:", request, "\n")
    for d in itinerary.days:
        print(f"  Day {d.day} ({d.weather}): {d.activity} — ${d.cost:.0f}")
    print(f"\n  total: ${itinerary.total_cost:.0f}   problems: {problems or 'none — plan passes!'}")
    print("\nSUMMARY:", summarize(itinerary, llm))


if __name__ == "__main__":
    _demo()
