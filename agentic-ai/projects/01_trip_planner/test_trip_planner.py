"""Tests for the Trip Planner.

These import the reference ``solution`` so they pass out of the box. To grade your own work,
change the line below to ``import starter as impl``.
"""

import solution as impl  # <- change to `import starter as impl` to test your implementation

from shared.llm import MockLLM


def _llm():
    # Scripted extraction so tests are deterministic and need no API key.
    return MockLLM(
        scripted=[
            '{"destination": "Rivertown", "days": 4, '
            '"interests": ["food", "history", "outdoors"], "budget_usd": 100}'
        ]
    )


def test_extract_request_parses_structured_fields():
    req = impl.extract_request("4 days, food/history/outdoors, $100", _llm())
    assert req.destination == "Rivertown"
    assert req.days == 4
    assert "food" in req.interests
    assert req.budget_usd == 100


def test_build_itinerary_has_one_plan_per_day():
    req = impl.TravelRequest(
        destination="Rivertown", days=4, interests=["food", "history", "outdoors"], budget_usd=100
    )
    itin = impl.build_itinerary(req)
    assert len(itin.days) == 4
    assert all(d.activity for d in itin.days)


def test_validate_flags_over_budget():
    req = impl.TravelRequest(destination="Rivertown", days=1, interests=["food"], budget_usd=10)
    itin = impl.Itinerary("Rivertown", [impl.DayPlan(1, "sunny", "Cooking class", 60.0)])
    problems = impl.validate(itin, req)
    assert any("budget" in p for p in problems)


def test_plan_trip_returns_passing_plan_within_budget():
    itin, problems = impl.plan_trip("4 days food/history/outdoors $100", _llm())
    assert problems == []
    assert itin.total_cost <= 100
    assert len(itin.days) == 4
