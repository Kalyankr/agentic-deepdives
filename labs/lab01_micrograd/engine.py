"""A tiny reverse-mode autograd engine (a la Karpathy's micrograd).

This is the keystone exercise for Module 01. Forward operations are implemented
for you. Your job is to implement the **local gradient** for each operation by
filling in the `_backward` closures marked with `TODO`.

The chain rule: each node stores how the output's gradient flows back to its
inputs. `Value.backward()` topologically sorts the graph and calls every node's
`_backward` in reverse, so each closure only needs to implement *one* local step:

    out = f(self, other)
    # given d(loss)/d(out) == out.grad, accumulate into self.grad / other.grad

Remember to **accumulate** (`+=`), never overwrite, so a value reused in multiple
places sums its gradients correctly.

Run the spec:  uv run pytest -m todo tests/test_lab01_micrograd.py
"""

from __future__ import annotations

import math


class Value:
    """A scalar value in the computation graph that tracks its gradient."""

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = ""):
        self.data: float = float(data)
        self.grad: float = 0.0
        # internal: backward closure for this node and its parents in the graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # --- core ops -------------------------------------------------------

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # TODO: d(out)/d(self) = 1, d(out)/d(other) = 1
            #   self.grad  += ... * out.grad
            #   other.grad += ... * out.grad
            raise NotImplementedError("Implement the gradient of addition")

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # TODO: product rule.
            #   self.grad  += other.data * out.grad
            #   other.grad += self.data  * out.grad
            raise NotImplementedError("Implement the gradient of multiplication")

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> Value:
        assert isinstance(other, (int, float)), "only supports int/float powers"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            # TODO: power rule: d(self**n)/d(self) = n * self**(n-1)
            #   self.grad += (other * self.data ** (other - 1)) * out.grad
            raise NotImplementedError("Implement the gradient of power")

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward() -> None:
            # TODO: gradient flows only where the input was positive.
            #   self.grad += (out.data > 0) * out.grad
            raise NotImplementedError("Implement the gradient of relu")

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # TODO: d(tanh(x))/dx = 1 - tanh(x)**2
            #   self.grad += (1 - t * t) * out.grad
            raise NotImplementedError("Implement the gradient of tanh")

        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            # TODO: d(exp(x))/dx = exp(x) == out.data
            #   self.grad += out.data * out.grad
            raise NotImplementedError("Implement the gradient of exp")

        out._backward = _backward
        return out

    # --- the engine -----------------------------------------------------

    def backward(self) -> None:
        """Run reverse-mode autodiff from this node (seed grad = 1.0)."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # --- conveniences (derived from the core ops; no new grad logic) ----

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other: float) -> Value:
        return Value(other) + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __rtruediv__(self, other: float) -> Value:
        return Value(other) * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
