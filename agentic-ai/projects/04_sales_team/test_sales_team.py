"""Tests for the Paper Company multi-agent sales team.

Imports the reference ``solution`` so they pass out of the box. Change to ``import starter as impl``
to grade your own work.
"""

from concurrent.futures import ThreadPoolExecutor

import solution as impl  # <- change to `import starter as impl` to test your implementation

from shared.llm import MockLLM


def _coord(stock, scripted):
    inv = impl.Inventory(stock=dict(stock))
    llm = MockLLM(scripted=list(scripted))
    coord = impl.SalesCoordinator(
        llm, impl.InventoryAgent(inv), impl.QuotingAgent(), impl.FinanceAgent()
    )
    return coord, inv


def test_bulk_discount_applied():
    assert impl.QuotingAgent().quote("A4", 100) == 450.0  # 5 * 100 * 0.9
    assert impl.QuotingAgent().quote("A4", 10) == 50.0  # no discount


def test_confirmed_order_decrements_stock():
    coord, inv = _coord(
        {"A4": 200},
        ['{"intent": "order", "item": "A4", "qty": 100, "budget": 600}', "Thanks for your order!"],
    )
    res = coord.handle("100 reams of A4, budget 600")
    assert res["status"] == "confirmed"
    assert inv.available("A4") == 100


def test_order_rejected_when_over_budget():
    coord, _ = _coord({"A4": 200}, ['{"intent": "order", "item": "A4", "qty": 100, "budget": 100}'])
    res = coord.handle("100 reams of A4, budget 100")
    assert res["status"] == "rejected"


def test_stock_intent_is_info_only():
    coord, inv = _coord({"A4": 5}, ['{"intent": "stock", "item": "A4", "qty": 1, "budget": null}'])
    res = coord.handle("how many A4 do you have?")
    assert res["status"] == "info"
    assert inv.available("A4") == 5  # unchanged


def test_concurrency_never_oversells():
    inv = impl.Inventory(stock={"A3": 10})
    llm = MockLLM(scripted=['{"intent": "order", "item": "A3", "qty": 3, "budget": 1000}'] * 24)
    coord = impl.SalesCoordinator(
        llm, impl.InventoryAgent(inv), impl.QuotingAgent(), impl.FinanceAgent()
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: coord.handle("3 reams of A3"), range(8)))
    confirmed = sum(r["status"] == "confirmed" for r in results)
    assert confirmed == 3  # 3 * 3 = 9 <= 10; a 4th (12) would oversell
    assert inv.available("A3") == 1
    assert inv.available("A3") >= 0  # never negative
