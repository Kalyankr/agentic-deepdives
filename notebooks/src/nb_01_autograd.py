"""Build NB01 — Autograd & training from scratch (numpy only)."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 01 · Autograd & Training From Scratch

> Module: **01 · Deep Learning Foundations** · Prereq for everything else.

**Goal:** understand backpropagation so deeply that PyTorch feels like "a faster version of
what I already built." We implement a tiny **reverse-mode autograd engine** in pure NumPy,
train a neural net with it, then connect every piece to how real frameworks work.

### Learning objectives
1. Express any computation as a **graph** of simple ops.
2. Derive and implement the **local gradient** of each op (the chain rule).
3. Build the **backward pass** (topological order + gradient accumulation).
4. Train an MLP with **SGD/Adam** and read the loss curve like an engineer.

### Why this matters at a frontier lab
Every training bug (NaNs, dead ReLUs, exploding grads) is a story about gradients flowing
through a graph. If you've built the graph yourself, you can debug anything.
"""),
    md(r"""
## 1. The computation graph

A neural net is just a big function built from tiny ops: `+`, `*`, `matmul`, `relu`, … .
The trick of **reverse-mode autodiff** is: do a forward pass to compute the output, then
walk the graph **backwards** applying the chain rule, so each node only needs to know its
*own* local derivative.

For an op `out = f(a, b)` the chain rule says, given the upstream gradient `dL/d out`:

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial out}\cdot\frac{\partial out}{\partial a},\qquad
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial out}\cdot\frac{\partial out}{\partial b}.$$

The single rule everyone gets wrong: **accumulate** (`+=`) gradients, never overwrite — a
value used in several places sums the contributions from each path.
"""),
    code(r"""
import numpy as np

class Value:
    # A scalar node in the computation graph that tracks its gradient.
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None   # local backward closure
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  += 1.0 * out.grad   # d(a+b)/da = 1
            other.grad += 1.0 * out.grad   # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += other.data * out.grad   # product rule
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float))
        out = Value(self.data ** n, (self,), f"**{n}")
        def _backward():
            self.grad += (n * self.data ** (n - 1)) * out.grad   # power rule
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")
        def _backward():
            self.grad += (out.data > 0) * out.grad   # gradient passes only where positive
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # 1) topological order so every node comes after its inputs
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        # 2) seed dL/dL = 1, then apply local backward in reverse
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # conveniences (no new gradient logic — built from the ops above)
    def __neg__(self):       return self * -1
    def __sub__(self, o):    return self + (-o if isinstance(o, Value) else Value(-o))
    def __radd__(self, o):   return self + o
    def __rmul__(self, o):   return self * o
    def __truediv__(self, o):return self * (o ** -1 if isinstance(o, Value) else Value(o) ** -1)
    def __repr__(self):      return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

print("Value engine ready")
"""),
    md(r"""
## 2. Sanity-check the gradients against finite differences

Before trusting any autograd, verify it. The numerical gradient
$\frac{f(x+h)-f(x-h)}{2h}$ should match what `backward()` produces.
"""),
    code(r"""
def f(xv):
    # an arbitrary scalar function of x
    a = xv * xv          # x^2
    b = a + xv * 3.0     # x^2 + 3x
    return (b + 1.0).tanh()

x = Value(0.7)
y = f(x)
y.backward()
analytic = x.grad

h = 1e-6
num = (f(Value(0.7 + h)).data - f(Value(0.7 - h)).data) / (2 * h)
print(f"analytic grad = {analytic:.6f}")
print(f"numerical grad = {num:.6f}")
assert abs(analytic - num) < 1e-4, "gradients disagree!"
print("gradients match ->", "OK")
"""),
    md(r"""
## 3. A minimal neural-network library

With the engine working, an MLP is just neurons (`w·x + b` then a nonlinearity) stacked
into layers. Same pattern PyTorch uses: a `Module` owning `parameters()`.
"""),
    code(r"""
import random
random.seed(1337)

class Neuron:
    def __init__(self, n_in, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, n_in, n_outs):
        sizes = [n_in] + n_outs
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin=i != len(n_outs)-1)
                       for i in range(len(n_outs))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

model = MLP(2, [16, 16, 1])
print("parameters:", len(model.parameters()))
"""),
    md(r"""
## 4. Train it on a toy 2-class problem

Two Gaussian blobs, labels in {-1, +1}. We use the **max-margin (hinge) loss** with L2
regularization, the same loss an SVM uses — it's a clean way to see learning happen.
"""),
    code(r"""
def make_blobs(n=100):
    xs, ys = [], []
    for _ in range(n // 2):
        xs.append([random.gauss(-1, 0.5), random.gauss(-1, 0.5)]); ys.append(-1)
        xs.append([random.gauss( 1, 0.5), random.gauss( 1, 0.5)]); ys.append( 1)
    return xs, ys

xs, ys = make_blobs(100)

def total_loss(reg=1e-4):
    scores = [model(x) for x in xs]
    data_loss = sum((1 + -yi * si).relu() for yi, si in zip(ys, scores)) * (1.0 / len(ys))
    reg_loss = reg * sum((p * p for p in model.parameters()), Value(0.0))
    acc = sum((si.data > 0) == (yi > 0) for si, yi in zip(scores, ys)) / len(ys)
    return data_loss + reg_loss, acc

history = []
for epoch in range(60):
    loss, acc = total_loss()
    model.zero_grad()
    loss.backward()
    lr = 0.1 - 0.09 * epoch / 60          # simple LR decay
    for p in model.parameters():
        p.data -= lr * p.grad             # vanilla SGD step
    history.append((loss.data, acc))
    if epoch % 10 == 0 or epoch == 59:
        print(f"epoch {epoch:2d}  loss {loss.data:.4f}  acc {acc:.0%}")
"""),
    md(r"""
## 5. Plot the loss curve (optional — needs matplotlib)

Reading loss curves is a core skill: a healthy curve drops fast then flattens. Spikes mean
LR too high; a flat line means no learning (check grads aren't zero / disconnected graph).
"""),
    code(r"""
try:
    import matplotlib.pyplot as plt
    losses = [h[0] for h in history]
    accs   = [h[1] for h in history]
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(losses, label="loss");  ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax2 = ax1.twinx(); ax2.plot(accs, color="tab:green", label="acc"); ax2.set_ylabel("accuracy")
    plt.title("Training from scratch with our own autograd"); plt.tight_layout(); plt.show()
except ImportError:
    print("matplotlib not installed (uv sync --extra viz) — skipping plot")
    print("final:", history[-1])
"""),
    md(r"""
## 6. From our engine to PyTorch / optimizers

Everything you built maps 1:1 onto real frameworks:

| You built | PyTorch | Notes |
|-----------|---------|-------|
| `Value` + `_backward` | `Tensor` + `autograd` | theirs is tensor-valued & C++/CUDA |
| `backward()` topo-sort | `loss.backward()` | identical idea |
| manual `p.data -= lr*p.grad` | `torch.optim.SGD` | optimizers wrap this |
| `zero_grad()` | `optimizer.zero_grad()` | grads accumulate, so you must reset |

**Optimizers** improve the raw SGD step:
- **Momentum:** $v \leftarrow \beta v + g;\ \ \theta \leftarrow \theta - \eta v$ (smooths noise).
- **Adam/AdamW:** per-parameter adaptive step from running estimates of the 1st/2nd moment of
  the gradient. **AdamW** (decoupled weight decay) is the LLM default.
- **Schedules:** linear **warmup** then **cosine decay** — transformers are unstable without warmup.

### Debugging playbook (memorize)
1. **Overfit one batch** first — if you can't, the model/loss is wrong.
2. Watch the **gradient norm**: explodes → lower LR / clip; ~0 → broken graph or dead ReLUs.
3. Check **shapes** and that the loss actually depends on every parameter.
"""),
    md(r"""
## 7. Exercises
1. Add `exp`, `log`, and a numerically-stable `softmax + cross-entropy` to `Value`.
2. Replace hinge loss with cross-entropy; compare convergence.
3. Implement **momentum** and **Adam** as drop-in optimizers; compare loss curves.
4. Add **gradient clipping** and show it tames a too-high learning rate.

## Resources
- Karpathy — *The spelled-out intro to backpropagation* (micrograd) and *Neural Networks: Zero to Hero*.
- *Deep Learning* (Goodfellow et al.) ch. 6 & 8; CS231n notes on optimization & backprop.
- Kingma & Ba, 2014 — **Adam**; Loshchilov & Hutter, 2017 — **AdamW**.
- The lab version with spec tests: `labs/lab01_micrograd/`.
"""),
]

if __name__ == "__main__":
    write(cells, "01_autograd_and_training.ipynb")
