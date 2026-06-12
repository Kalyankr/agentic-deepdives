# Project 01 · Trip Planner

> **Course:** [01 · Prompting for Effective LLM Reasoning and Planning](../../courses/01-prompting.md)
> · **Notebook:** [01_prompting.ipynb](../../notebooks/01_prompting.ipynb)

Build an **agentic travel assistant** that turns a free-text trip request into a validated,
on-budget, day-by-day itinerary — and **self-corrects** until the plan satisfies every constraint.

This project exercises all four Course-01 skills:

| Skill | Where it appears |
|-------|------------------|
| Structured extraction (role prompt + JSON contract) | `extract_request` |
| Tool use (ReAct-style "act") | `build_itinerary` calling `find_activities` / `get_weather` |
| Prompt chaining with a validation **gate** | `validate` between build and accept |
| **Feedback loop** | `revise` → re-`validate` until it passes |

---

## Requirements

1. **Extract** a `TravelRequest` (`destination`, `days`, `interests`, `budget_usd`) from free text
   using the LLM, parsed and validated with Pydantic.
2. **Plan** a `DayPlan` for each day by *acting through tools* — pick activities matching the
   traveler's interests and attach a weather forecast.
3. **Validate** the itinerary against the constraints: right number of days, within budget, and
   every (available) requested interest covered. Return a list of violations.
4. **Revise** in a loop: when validation fails (e.g. over budget), adjust the plan and re-validate
   until it passes or you hit `max_revisions`.
5. (Optional) **Summarize** the final itinerary with a concierge-style narrative.

---

## Run

```bash
cd agentic-ai

# See the finished reference in action (offline, no key):
uv run python projects/01_trip_planner/solution.py

# Do the exercise: implement starter.py, then point the test at it and run:
uv run --extra dev pytest projects/01_trip_planner -q
```

Switch the test target by editing the first import in
[`test_trip_planner.py`](test_trip_planner.py): `import solution as impl` → `import starter as impl`.

To run against a **real model**, install `--extra openai`, set `OPENAI_API_KEY`, and replace the
scripted `MockLLM` with `get_llm()`.

---

## Grading rubric

| Criterion | Pass | Strong |
|-----------|------|--------|
| Extraction | parses valid requests | handles vague text, validates types, fails loudly on bad input |
| Itinerary build | one activity/day | respects interests, uses tools, sensible variety |
| Validation gate | catches over-budget | catches all three constraint types with clear messages |
| Feedback loop | terminates | converges to a passing plan and explains the trade-offs made |
| Code quality | runs | typed, documented, `uv run ruff check` clean |

---

## Stretch goals

- Add a real **ReAct loop** (let the LLM choose activities turn-by-turn) instead of the
  deterministic planner.
- Add a weather rule (no outdoor activity on "light rain" days) and let the loop fix violations.
- Add a second traveler with conflicting interests and reconcile them.
