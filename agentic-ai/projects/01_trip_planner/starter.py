"""Trip Planner — STARTER.

Implement the functions marked ``TODO``. The data models and tools are provided. When done, point
the test at this file (``import starter as impl`` in ``test_trip_planner.py``) and run:

    uv run --extra dev pytest projects/01_trip_planner -q

Study ``solution.py`` only after you've tried. Concepts: structured extraction, tool use,
prompt chaining with a validation gate, and a feedback loop (see courses/01-prompting.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from shared.llm import BaseLLM, extract_json, system, user

# ----------------------- provided: contract + tools ----------------------- #


class TravelRequest(BaseModel):
    destination: str
    days: int
    interests: list[str]
    budget_usd: float


ACTIVITIES: dict[str, dict[str, list[tuple[str, float]]]] = {
    "Rivertown": {
        "food": [("Street food tour", 30.0), ("Cooking class", 60.0)],
        "history": [("Old town walking tour", 0.0), ("Museum day pass", 25.0)],
        "outdoors": [("River kayak", 45.0), ("Sunset hill hike", 0.0)],
        "art": [("Gallery crawl", 15.0), ("Pottery workshop", 50.0)],
    }
}


def find_activities(destination: str, interest: str) -> list[tuple[str, float]]:
    return ACTIVITIES.get(destination, {}).get(interest, [])


def get_weather(destination: str, day: int) -> str:
    return ["sunny", "cloudy", "light rain"][day % 3]


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


# ----------------------------- your work --------------------------------- #


def extract_request(text: str, llm: BaseLLM) -> TravelRequest:
    """TODO: prompt the LLM for JSON, then parse + validate into a TravelRequest.

    Steps:
      1. Send a system message that demands JSON with keys destination/days/interests/budget_usd.
      2. Use ``extract_json`` on the reply.
      3. Return ``TravelRequest(**parsed)``.
    """
    raise NotImplementedError("TODO: implement extract_request")


def build_itinerary(req: TravelRequest) -> Itinerary:
    """TODO: build one DayPlan per day, rotating through req.interests.

    For each day pick the cheapest activity for that interest via ``find_activities`` and attach
    ``get_weather``. If an interest has no activities, use a 'Free day' at $0.
    """
    raise NotImplementedError("TODO: implement build_itinerary")


def validate(itinerary: Itinerary, req: TravelRequest) -> list[str]:
    """TODO: return a list of constraint violations (empty list == passes the gate).

    Check: correct number of days, total_cost within budget, and all *available* requested
    interests are covered.
    """
    raise NotImplementedError("TODO: implement validate")


def revise(itinerary: Itinerary, req: TravelRequest) -> Itinerary:
    """TODO: one revision step. If over budget, replace the most expensive day with a cheaper
    (or free) option for that day's interest. Return the (mutated) itinerary."""
    raise NotImplementedError("TODO: implement revise")


def plan_trip(text: str, llm: BaseLLM, max_revisions: int = 5) -> tuple[Itinerary, list[str]]:
    """TODO: extract -> build -> (validate -> revise)*; return (itinerary, problems)."""
    raise NotImplementedError("TODO: implement plan_trip")
