"""Spec for Lab 01 — micrograd gradients.

These FAIL until you implement the `_backward` closures in lab01_micrograd/engine.py.
They are your definition of done.

    uv run pytest -m todo tests/test_lab01_micrograd.py
"""

from __future__ import annotations

import math

import pytest

from lab01_micrograd.engine import Value

pytestmark = pytest.mark.todo


def test_add_mul_relu_graph():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c  # 4.0
    e = d.relu()  # 4.0
    e.backward()
    # de/dd = 1 (d>0); dd/da = b = -3; dd/db = a = 2; dd/dc = 1
    assert e.data == pytest.approx(4.0)
    assert a.grad == pytest.approx(-3.0)
    assert b.grad == pytest.approx(2.0)
    assert c.grad == pytest.approx(1.0)


def test_relu_blocks_negative_gradient():
    x = Value(-5.0)
    y = (x * 2.0).relu()  # input to relu is -10 -> 0, gradient blocked
    y.backward()
    assert y.data == pytest.approx(0.0)
    assert x.grad == pytest.approx(0.0)


def test_gradient_accumulation_for_reused_value():
    a = Value(3.0)
    b = a + a  # uses `a` twice
    b.backward()
    assert a.grad == pytest.approx(2.0)


def test_power_rule():
    x = Value(3.0)
    y = x**2
    y.backward()
    assert y.data == pytest.approx(9.0)
    assert x.grad == pytest.approx(6.0)  # d(x^2)/dx = 2x = 6


def test_tanh_gradient():
    x = Value(0.5)
    y = x.tanh()
    y.backward()
    assert x.grad == pytest.approx(1 - math.tanh(0.5) ** 2)


def test_exp_gradient():
    x = Value(1.0)
    y = x.exp()
    y.backward()
    assert x.grad == pytest.approx(math.e)


def test_division_chain():
    a = Value(4.0)
    b = Value(2.0)
    c = a / b  # 2.0
    c.backward()
    # dc/da = 1/b = 0.5 ; dc/db = -a/b^2 = -1.0
    assert c.data == pytest.approx(2.0)
    assert a.grad == pytest.approx(0.5)
    assert b.grad == pytest.approx(-1.0)


def test_matches_torch_on_complex_expression():
    torch = pytest.importorskip("torch")

    # our engine
    a = Value(-4.0)
    b = Value(2.0)
    d = (a * b + b**3).relu()
    e = (d * 2 + (a + b).tanh()) * d
    e.backward()

    # torch reference
    ta = torch.tensor(-4.0, requires_grad=True)
    tb = torch.tensor(2.0, requires_grad=True)
    td = (ta * tb + tb**3).relu()
    te = (td * 2 + (ta + tb).tanh()) * td
    te.backward()

    assert e.data == pytest.approx(te.item(), rel=1e-5)
    assert a.grad == pytest.approx(ta.grad.item(), rel=1e-5)
    assert b.grad == pytest.approx(tb.grad.item(), rel=1e-5)
