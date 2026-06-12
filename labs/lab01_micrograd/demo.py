"""Train an MLP on a toy 2D classification task using only your autograd engine.

Run after implementing the `_backward` closures in `engine.py`:

    uv run python -m lab01_micrograd.demo

You should see the loss fall and accuracy climb toward ~1.0.
"""

from __future__ import annotations

import random

from lab01_micrograd.engine import Value
from lab01_micrograd.nn import MLP


def make_blobs(n: int = 100, seed: int = 1337) -> tuple[list[list[float]], list[int]]:
    """Two Gaussian blobs → binary labels in {-1, +1}."""
    rng = random.Random(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    for _ in range(n // 2):
        xs.append([rng.gauss(-1.0, 0.5), rng.gauss(-1.0, 0.5)])
        ys.append(-1)
        xs.append([rng.gauss(1.0, 0.5), rng.gauss(1.0, 0.5)])
        ys.append(1)
    return xs, ys


def hinge_loss(model: MLP, xs: list[list[float]], ys: list[int], reg: float = 1e-4) -> Value:
    # max-margin (SVM) loss + L2 regularization
    scores = [model(x) for x in xs]
    data_loss = sum((1 + -yi * si).relu() for yi, si in zip(ys, scores, strict=True)) * (
        1.0 / len(ys)
    )
    reg_loss = reg * sum((p * p for p in model.parameters()), Value(0.0))
    return data_loss + reg_loss


def accuracy(model: MLP, xs: list[list[float]], ys: list[int]) -> float:
    correct = sum((model(x).data > 0) == (yi > 0) for x, yi in zip(xs, ys, strict=True))
    return correct / len(ys)


def main() -> None:
    xs, ys = make_blobs(100)
    model = MLP(2, [16, 16, 1])
    print(f"params: {len(model.parameters())}")

    epochs = 50
    for epoch in range(epochs):
        loss = hinge_loss(model, xs, ys)
        model.zero_grad()
        loss.backward()

        # simple SGD with a decaying learning rate
        lr = 0.1 - 0.09 * epoch / epochs
        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  loss {loss.data:.4f}  acc {accuracy(model, xs, ys):.2%}")


if __name__ == "__main__":
    main()
