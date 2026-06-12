"""Paper Company Sales Team — reference solution (Course 04, capstone).

A complete **multi-agent system** for a paper company's sales operation. It combines every
Course-04 idea:

* **Architecture** — specialist agents (inventory, quoting, finance) + a coordinator.
* **Routing** — a request is classified (order / quote / stock question) and dispatched.
* **Orchestration** — an order flows through check-stock -> quote -> finance-approve -> reserve.
* **Shared state + concurrency control** — a lock on the inventory prevents overselling when many
  orders are processed at once.

Runs offline: the business logic is deterministic; the LLM (scripted ``MockLLM``) handles request
parsing + the customer-facing message. Swap in ``get_llm()`` for a real model.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from shared.llm import BaseLLM, MockLLM, extract_json, system, user

# --------------------------------------------------------------------------- #
# Shared, thread-safe state: the inventory (the contended resource).
# --------------------------------------------------------------------------- #
PRICE_PER_REAM = {"A4": 5.0, "A3": 9.0, "recycled A4": 6.0}


@dataclass
class Inventory:
    """The blackboard for stock. The lock makes concurrent reserves safe (no overselling)."""

    stock: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, item: str, qty: int) -> bool:
        with self._lock:  # critical section: only one agent reserves at a time
            if self.stock.get(item, 0) < qty:
                return False
            self.stock[item] -= qty
            return True

    def available(self, item: str) -> int:
        return self.stock.get(item, 0)


# --------------------------------------------------------------------------- #
# Specialist agents.
# --------------------------------------------------------------------------- #
@dataclass
class InventoryAgent:
    inventory: Inventory

    def check(self, item: str, qty: int) -> str:
        have = self.inventory.available(item)
        return f"{have} {item} available" if have >= qty else f"INSUFFICIENT: only {have} {item}"

    def reserve(self, item: str, qty: int) -> bool:
        return self.inventory.reserve(item, qty)


@dataclass
class QuotingAgent:
    def quote(self, item: str, qty: int) -> float:
        price = PRICE_PER_REAM.get(item)
        if price is None:
            raise ValueError(f"no price for {item!r}")
        discount = 0.9 if qty >= 100 else 1.0  # bulk discount
        return round(price * qty * discount, 2)


@dataclass
class FinanceAgent:
    def approve(self, total: float, customer_budget: float | None) -> bool:
        return customer_budget is None or total <= customer_budget


# --------------------------------------------------------------------------- #
# The coordinator: routes + orchestrates the specialists over shared state.
# --------------------------------------------------------------------------- #
@dataclass
class SalesCoordinator:
    llm: BaseLLM
    inventory_agent: InventoryAgent
    quoting_agent: QuotingAgent
    finance_agent: FinanceAgent

    def parse(self, text: str) -> dict:
        """Use the LLM to turn a free-text request into a structured order."""
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
        """Customer-facing natural-language reply (the LLM's narrative layer)."""
        return self.llm.chat([system("You are a friendly sales rep. One sentence."), user(summary)])

    def handle(self, text: str) -> dict:
        order = self.parse(text)
        item, qty = order["item"], int(order["qty"])

        # ROUTING by intent ---------------------------------------------------
        if order["intent"] == "stock":
            return {"status": "info", "detail": self.inventory_agent.check(item, qty)}

        # quote always computed for order/quote intents
        try:
            total = self.quoting_agent.quote(item, qty)
        except ValueError as exc:
            return {"status": "rejected", "detail": str(exc)}

        if order["intent"] == "quote":
            return {"status": "quoted", "item": item, "qty": qty, "total": total}

        # ORCHESTRATION for an actual order: stock -> finance -> reserve -------
        if self.inventory_agent.check(item, qty).startswith("INSUFFICIENT"):
            return {"status": "rejected", "detail": self.inventory_agent.check(item, qty)}
        if not self.finance_agent.approve(total, order.get("budget")):
            return {"status": "rejected", "detail": f"over budget: ${total} > ${order['budget']}"}
        if not self.inventory_agent.reserve(item, qty):  # final, atomic, under lock
            return {"status": "rejected", "detail": "stock taken by a concurrent order"}
        return {
            "status": "confirmed",
            "item": item,
            "qty": qty,
            "total": total,
            "message": self.message(f"Confirmed {qty} {item} for ${total}."),
        }


def _demo() -> None:
    inv = Inventory(stock={"A4": 120, "A3": 40})
    # Script: parse two orders (the second oversized) + one confirmation message.
    llm = MockLLM(
        scripted=[
            '{"intent": "order", "item": "A4", "qty": 100, "budget": 600}',
            "All set — 100 reams of A4 are on the way!",
            '{"intent": "order", "item": "A4", "qty": 50, "budget": 500}',
        ]
    )
    coord = SalesCoordinator(llm, InventoryAgent(inv), QuotingAgent(), FinanceAgent())

    print("order 1:", coord.handle("I'd like 100 reams of A4, budget $600."))
    print("order 2:", coord.handle("Now 50 more reams of A4."))  # only 20 left -> rejected
    print("final stock:", inv.stock)

    # Concurrency: 6 simultaneous orders for the last A3 stock — lock prevents overselling.
    inv2 = Inventory(stock={"A3": 10})
    llm2 = MockLLM(scripted=['{"intent": "order", "item": "A3", "qty": 3, "budget": 1000}'] * 12)
    coord2 = SalesCoordinator(llm2, InventoryAgent(inv2), QuotingAgent(), FinanceAgent())
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: coord2.handle("3 reams of A3"), range(6)))
    confirmed = sum(r["status"] == "confirmed" for r in results)
    left = inv2.available("A3")
    print(f"\nconcurrent: {confirmed} confirmed, {6 - confirmed} rejected; A3 left = {left}")


if __name__ == "__main__":
    _demo()
