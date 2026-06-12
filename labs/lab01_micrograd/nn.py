"""A minimal neural-network library built on the `Value` autograd engine.

This is fully implemented — it exercises the `Value` ops you write in `engine.py`.
Once your `_backward` closures are correct, `demo.py` will train an MLP end-to-end.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from lab01_micrograd.engine import Value


class Module:
    """Base class: a container of trainable `Value` parameters."""

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(self, n_in: int, nonlin: bool = True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Iterable[Value | float]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x: Iterable[Value | float]) -> list[Value] | Value:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """A simple multilayer perceptron. Last layer is linear (no nonlinearity)."""

    def __init__(self, n_in: int, n_outs: list[int]):
        sizes = [n_in, *n_outs]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=i != len(n_outs) - 1) for i in range(len(n_outs))
        ]

    def __call__(self, x: Iterable[Value | float]):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
