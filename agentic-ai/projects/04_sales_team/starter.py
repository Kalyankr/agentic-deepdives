"""Paper Company Sales Team — STARTER.

Implement the ``TODO`` methods to build the multi-agent sales system. The shared ``Inventory``
(with its lock) and the price table are provided. Run:

    uv run python projects/04_sales_team/starter.py
    uv run --extra dev pytest projects/04_sales_team -q

Concepts: specialist agents, routing, orchestration, shared state, concurrency control
(see courses/04-multi-agent-systems.md).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from shared.llm import BaseLLM, extract_json, system, user

PRICE_PER_REAM = {"A4": 5.0, "A3": 9.0, "recycled A4": 6.0}


@dataclass
class Inventory:
    stock: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, item: str, qty: int) -> bool:
        with self._lock:
            if self.stock.get(item, 0) < qty:
                return False
            self.stock[item] -= qty
            return True

    def available(self, item: str) -> int:
        return self.stock.get(item, 0)


@dataclass
class InventoryAgent:
    inventory: Inventory

    def check(self, item: str, qty: int) -> str:
        """TODO: return availability text; flag 'INSUFFICIENT: ...' when short."""
        raise NotImplementedError("TODO")

    def reserve(self, item: str, qty: int) -> bool:
        return self.inventory.reserve(item, qty)


@dataclass
class QuotingAgent:
    def quote(self, item: str, qty: int) -> float:
        """TODO: price * qty, with a bulk discount (e.g. 10% off at qty >= 100)."""
        raise NotImplementedError("TODO")


@dataclass
class FinanceAgent:
    def approve(self, total: float, customer_budget: float | None) -> bool:
        """TODO: approve when there's no budget or total <= budget."""
        raise NotImplementedError("TODO")


@dataclass
class SalesCoordinator:
    llm: BaseLLM
    inventory_agent: InventoryAgent
    quoting_agent: QuotingAgent
    finance_agent: FinanceAgent

    def parse(self, text: str) -> dict:
        raw = self.llm.chat(
            [
                system(
                    'Extract JSON {"intent": "order|quote|stock", "item": str, '
                    '"qty": int, "budget": number|null}.'
                ),
                user(text),
            ]
        )
        return extract_json(raw)

    def message(self, summary: str) -> str:
        return self.llm.chat([system("You are a friendly sales rep. One sentence."), user(summary)])

    def handle(self, text: str) -> dict:
        """TODO: route by intent, then orchestrate check-stock -> quote -> finance -> reserve.
        Return a dict with a 'status' key (info/quoted/confirmed/rejected)."""
        raise NotImplementedError("TODO")
