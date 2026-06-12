# Lab 01 — micrograd

> Module: [01 · Deep Learning Foundations](../../modules/01-deep-learning-foundations.md)

Build a tiny **reverse-mode autograd engine** from scratch and train an MLP with it.
This single exercise demystifies all of deep learning: once you've written `.backward()`
yourself, every framework is just a faster version of this.

## Your task

Open [engine.py](engine.py) and implement the **local gradient** inside each `_backward`
closure (look for `TODO` / `NotImplementedError`):

- `__add__`, `__mul__`, `__pow__`, `relu`, `tanh`, `exp`

The graph traversal (`backward()`), the `nn` library, and the demo are already written.

### The one rule that trips everyone up
**Accumulate** gradients (`+=`), never assign (`=`). A value used in several places must
sum the gradient contributions from each — that's the multivariate chain rule.

## Check your work

```bash
uv run pytest -m todo tests/test_lab01_micrograd.py
```

The tests cover each operation's gradient plus gradient **accumulation** for reused values.

## Run the demo

```bash
uv run python -m lab01_micrograd.demo
```

Expected: loss decreases and accuracy climbs toward ~100% on the toy blobs.

## Stretch goals

- Add `log`, `sigmoid`, and a numerically-stable `softmax + cross-entropy`.
- Swap the SVM hinge loss for cross-entropy.
- Add momentum / Adam to the optimizer.
- Visualize the decision boundary (install the `viz` extra and use matplotlib).
