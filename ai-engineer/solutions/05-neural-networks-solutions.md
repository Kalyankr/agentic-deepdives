# Chapter 5 — Neural Networks from Scratch · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-2-deep-learning/05-neural-networks-from-scratch.md)

---

## Interview answers

### Q: "What does `.backward()` actually do?"

It computes $\partial L/\partial \theta$ for every parameter by running the **chain rule backward through the computation graph**. Concretely: (1) the forward pass builds a DAG of operations; (2) `.backward()` does a **reverse-topological traversal** of that DAG; (3) at each node it multiplies the incoming gradient ($\partial L/\partial \text{out}$) by that op's **local derivative** and **accumulates** (`+=`) into each input's `.grad`. The accumulation is what lets a value feeding multiple consumers correctly sum its gradient contributions. Seed it with $\partial L/\partial L = 1$ at the loss, traverse in reverse, and every leaf ends up holding its gradient. That's all PyTorch's autograd is, at scale.

### Q: "Derive the gradient of softmax + cross-entropy."

It collapses to $p - \text{onehot}(y)$. The softmax Jacobian is $\partial p_i/\partial z_j = p_i(\delta_{ij}-p_j)$; chaining it through $L=-\sum_k y_k\log p_k$ gives $\partial L/\partial z_j = p_j - y_j$. The cross terms cancel because $\sum_k y_k = 1$. (Full algebra in the [Chapter 2 solutions](02-mathematics-solutions.md#q-derive-the-gradient-of-cross-entropy--softmax).) Interpretation: "prediction minus target" — zero gradient when right and confident, large when confidently wrong.

### Q: "Why do residual connections help training?"

A residual block computes $y = x + F(x)$. Its derivative w.r.t. the input is $1 + F'(x)$ — the **$+1$ creates a gradient highway** that lets the signal flow back through dozens or hundreds of layers without shrinking to zero. Without residuals, gradients pass through many multiplications and **vanish** in deep nets; with them, even if $F'(x)$ is tiny, the identity term keeps a strong gradient path. This is what makes 100+ layer nets — and every transformer — trainable.

### Q: "Why ReLU over sigmoid?"

Sigmoid **saturates**: for large $|x|$ its derivative $\sigma(x)(1-\sigma(x)) \to 0$, so stacked sigmoids multiply many near-zero derivatives and gradients vanish. ReLU's derivative is exactly **1 for $x>0$**, so the gradient passes through undiminished and deep nets actually learn. ReLU is also cheap (a `max`) and induces sparsity. Its one flaw — "dying ReLUs" for persistently negative inputs — is patched by LeakyReLU/GELU/SiLU, but the core reason it replaced sigmoid is **non-saturation → no vanishing gradient**.

### Q: "What's the bug if your model won't learn at all?"

Run through the usual suspects, cheapest first: **forgot `zero_grad()`** (gradients accumulate across steps and explode), **bad initialization** (activations vanish/explode from step 0), **learning rate** way too high (diverges to NaN) or too low (no movement), or a **shape/broadcasting bug** silently corrupting the loss. The fastest single test is to **overfit one batch** — if the model can't drive a single batch to ~0 loss, it's a wiring/code bug, not hyperparameters.

---

## Exercise solutions

These build on the chapter's `Value` autograd engine. Here it is, self-contained, so every solution below runs as-is.

```python
import math, random

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data; self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children); self._op = _op
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o), '+')
        def _b(): self.grad += out.grad; o.grad += out.grad
        out._backward = _b; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o), '*')
        def _b(): self.grad += o.data*out.grad; o.grad += self.data*out.grad
        out._backward = _b; return out
    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Value(self.data**k, (self,), f'**{k}')
        def _b(): self.grad += (k * self.data**(k-1)) * out.grad
        out._backward = _b; return out
    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')
        def _b(): self.grad += (out.data > 0) * out.grad
        out._backward = _b; return out
    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,), 'tanh')
        def _b(): self.grad += (1 - t*t) * out.grad
        out._backward = _b; return out
    def exp(self):
        e = math.exp(self.data); out = Value(e, (self,), 'exp')
        def _b(): self.grad += e * out.grad
        out._backward = _b; return out
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo): v._backward()
    def __neg__(self): return self * -1
    def __radd__(self, o): return self + o
    def __sub__(self, o): return self + (-o)
    def __rsub__(self, o): return o + (-self)
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * o**-1
```

### Exercise 1 — `TrainableMLP` overfits a single batch to ~0 loss

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    def params(self): return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout): self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def params(self): return [p for n in self.neurons for p in n.params()]

class TrainableMLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
    def params(self): return [p for l in self.layers for p in l.params()]

random.seed(0)
model = TrainableMLP(3, [4, 4, 1])
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y)**2 for p, y in zip(preds, ys))
    for p in model.params(): p.grad = 0.0      # zero_grad FIRST
    loss.backward()
    for p in model.params(): p.data -= 0.05 * p.grad
    if step % 20 == 0: print(step, loss.data)

print("final loss:", loss.data)                 # -> ~1e-4, essentially 0
```

**Result:** loss falls from ~4 to ~$10^{-4}$ — the model memorizes the 4-example batch, confirming the full forward/backward/update loop is wired correctly. (If it *couldn't* overfit, you'd have a bug — that's the diagnostic from the last interview answer.)

### Exercise 2 — Extend `Value` with `tanh`, `exp`, `**`, verified by finite differences

`tanh`, `exp`, and `**` are already added above. Verify each local gradient against a central finite difference:

```python
def check(fn, x0, h=1e-6):
    x = Value(x0); y = fn(x); y.backward()
    num = (fn(Value(x0 + h)).data - fn(Value(x0 - h)).data) / (2*h)
    return x.grad, num

for name, fn in [("tanh", lambda v: v.tanh()),
                 ("exp",  lambda v: v.exp()),
                 ("**3",  lambda v: v**3)]:
    ana, num = check(fn, 0.7)
    print(f"{name:5s} analytic={ana:.6f} numeric={num:.6f} ok={abs(ana-num)<1e-4}")
```

**Result:** for each op the analytic gradient matches the numeric one to ~1e-6 (`ok=True`): $\frac{d}{dx}\tanh = 1-\tanh^2$, $\frac{d}{dx}e^x=e^x$, $\frac{d}{dx}x^3=3x^2$. Finite-difference checking is the standard way to trust a hand-written backward.

### Exercise 3 — Add momentum, then Adam, to the from-scratch optimizer

```python
def train(opt='sgd', steps=100, lr=0.05):
    random.seed(0)
    model = TrainableMLP(3, [4, 4, 1])
    ps = model.params()
    m = [0.0]*len(ps); v = [0.0]*len(ps)        # momentum / Adam state
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps+1):
        loss = sum((model(x) - y)**2 for x, y in zip(xs, ys))
        for p in ps: p.grad = 0.0
        loss.backward()
        for i, p in enumerate(ps):
            g = p.grad
            if opt == 'sgd':
                p.data -= lr * g
            elif opt == 'momentum':
                m[i] = 0.9*m[i] + g; p.data -= lr * m[i]
            elif opt == 'adam':
                m[i] = b1*m[i] + (1-b1)*g
                v[i] = b2*v[i] + (1-b2)*g*g
                mh = m[i]/(1-b1**t); vh = v[i]/(1-b2**t)
                p.data -= lr * mh/(vh**0.5 + eps)
    return loss.data

for opt in ('sgd', 'momentum', 'adam'):
    print(f"{opt:9s} final loss: {train(opt):.6f}")
```

**Result:** momentum reaches low loss in fewer steps than plain SGD by accumulating velocity through consistent gradient directions; Adam converges fastest and most robustly to the learning rate by normalizing per-parameter step sizes — the same behavior you saw on the ill-conditioned bowl in [Chapter 2](02-mathematics-solutions.md#exercise-4--gradient-descent-vs-adam-on-an-ill-conditioned-bowl), now inside a real network.

### Exercise 4 — Dropout reduces the train/val gap

```python
import numpy as np

def dropout(x, p, training=True):
    if not training or p == 0: return x
    mask = (np.random.rand(*x.shape) > p) / (1 - p)   # inverted dropout: scale at train time
    return x * mask

# Tiny overparameterized MLP on noisy data, NumPy for speed:
rng = np.random.default_rng(0)
Xtr, ytr = rng.standard_normal((40, 20)), rng.integers(0, 2, 40).astype(float)
Xva, yva = rng.standard_normal((200, 20)), rng.integers(0, 2, 200).astype(float)

def run(p_drop):
    W1 = rng.standard_normal((20, 256))*0.1; W2 = rng.standard_normal((256, 1))*0.1
    for _ in range(400):
        h = np.maximum(0, Xtr @ W1); h = dropout(h, p_drop)
        pred = (h @ W2).ravel(); err = pred - ytr
        gW2 = h.T @ err[:, None] / len(Xtr)
        gh = (err[:, None] @ W2.T) * (h > 0)
        gW1 = Xtr.T @ gh / len(Xtr)
        W1 -= 0.05*gW1; W2 -= 0.05*gW2
    tr = np.mean((np.maximum(0, Xtr@W1)@W2).ravel() - ytr)**2
    va = np.mean((np.maximum(0, Xva@W1)@W2).ravel() - yva)**2
    return tr, va

for p in (0.0, 0.5):
    tr, va = run(p); print(f"dropout {p}: train={tr:.3f} val={va:.3f} gap={va-tr:.3f}")
```

**Result:** with `p=0` the overparameterized net fits the 40 training points tightly but generalizes worse (large train/val gap); with `p=0.5` the net can't co-adapt to memorize noise, so the **gap shrinks**. Dropout is a regularizer: randomly zeroing activations (and scaling the survivors) forces redundant, robust features. Remember to **disable it at eval time**.

### Exercise 5 — Manual backward == autograd gradients

```python
import numpy as np

# Two-layer MLP, forward + MANUAL backward in NumPy:
rng = np.random.default_rng(0)
x = rng.standard_normal((1, 3)); y = np.array([[1.0]])
W1 = rng.standard_normal((3, 4)); W2 = rng.standard_normal((4, 1))

z1 = x @ W1; a1 = np.tanh(z1); z2 = a1 @ W2; loss = ((z2 - y)**2).sum()
dz2 = 2*(z2 - y)
dW2 = a1.T @ dz2
da1 = dz2 @ W2.T
dz1 = da1 * (1 - np.tanh(z1)**2)
dW1 = x.T @ dz1

# Same computation with the Value autograd engine:
def mse_grad_autograd():
    W1v = [[Value(W1[i, j]) for j in range(4)] for i in range(3)]
    W2v = [Value(W2[i, 0]) for i in range(4)]
    xv  = [Value(x[0, i]) for i in range(3)]
    a1v = [sum(xv[i]*W1v[i][j] for i in range(3)).tanh() for j in range(4)]
    z2v = sum(a1v[j]*W2v[j] for j in range(4))
    loss_v = (z2v - Value(1.0))**2
    loss_v.backward()
    gW1 = np.array([[W1v[i][j].grad for j in range(4)] for i in range(3)])
    gW2 = np.array([[W2v[j].grad] for j in range(4)])
    return gW1, gW2

gW1_auto, gW2_auto = mse_grad_autograd()
print("W1 match:", np.allclose(dW1, gW1_auto, atol=1e-9))   # True
print("W2 match:", np.allclose(dW2, gW2_auto, atol=1e-9))   # True
```

**Result:** the hand-derived backprop and the autograd engine produce **identical gradients** (to machine precision). That equality is the whole point of Part II: autograd isn't magic — it's the same chain rule you'd write by hand, just automated over the graph. Once you've confirmed this once, `.backward()` is fully demystified.

---

[← Chapter 4 solutions](04-cs-fundamentals-solutions.md) · [Solutions index](README.md) · [Next: Chapter 6 solutions →](06-transformer-solutions.md)
