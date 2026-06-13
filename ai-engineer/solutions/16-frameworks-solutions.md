# Chapter 16 — Frameworks: PyTorch & JAX · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-4-systems/16-frameworks.md)

---

## Interview answers

### Q: "Eager vs graph execution — tradeoffs?"

- **Eager** (default PyTorch): ops run **immediately**, line by line. Easy to write, **debug** (set breakpoints, print tensors), and supports dynamic control flow. Cost: no whole-program view, so the framework can't fuse/optimize across ops, and per-op launch overhead adds up.
- **Graph** (compiled): capture the computation as a **graph first**, then optimize it — **fuse kernels**, eliminate overhead, generate fast code. Faster, but harder to debug and awkward with data-dependent control flow.

`torch.compile` and `jax.jit` aim to give you **both**: write eager-style code, get graph-level speed under the hood.

### Q: "What does `torch.compile` do?"

It JIT-compiles your eager PyTorch: **captures** a graph (TorchDynamo), **optimizes/fuses** ops, and **generates** fast kernels (often **Triton**) via the Inductor backend — typically a 1.3–2× speedup with a one-line change and no rewrite. When it hits something it can't capture (data-dependent Python, unsupported ops) it inserts a **graph break**, falling back to eager for that part. So you keep eager's ergonomics and recover most of graph mode's performance.

### Q: "PyTorch vs JAX?"

- **PyTorch**: **imperative/eager**, object-oriented (`nn.Module`), the **dominant ecosystem** (models, tooling, community), best debuggability. The safe default and what most jobs use.
- **JAX**: **functional** (pure functions, immutable state) with **composable transforms** — `grad`, `jit`, `vmap`, `pmap`. Excels on **TPUs** and at elegant **scaling**, favored in research (DeepMind). Steeper learning curve.

The senior answer: **learn PyTorch deeply first** (it's the industry default), add **JAX as a differentiator**, and be able to argue the tradeoffs both ways — imperative familiarity & ecosystem vs. functional purity & composable transforms.

### Q: "What is `vmap`?"

**Automatic vectorization.** You write a function for a **single example**, and `vmap` transforms it into one that runs efficiently over a **batch** — without manually adding/managing batch dimensions. It maps the computation over an axis and lets the compiler vectorize it. Benefits: cleaner code (no batch-dim bookkeeping), fewer broadcasting bugs, and you can compose it (e.g., `vmap(grad(f))` for per-example gradients). It's one of JAX's signature conveniences.

### Q: "Why does JAX use explicit PRNG keys?"

Because JAX functions are **pure** — no hidden global state. A global RNG (like NumPy's) is mutable shared state, which breaks reproducibility and parallelism (order-dependent, unsafe under `jit`/`vmap`/`pmap`). Instead you pass an explicit **key**; to get new randomness you **split** it (`jax.random.split`). This makes randomness **explicit, reproducible, and parallel-safe** — the same key always gives the same draw, and you can vectorize/parallelize without races. It's more verbose but principled.

### Q: "What does `nn.Module` give you?"

It's the backbone of PyTorch model-building. For free you get: **parameter registration** (auto-tracked for the optimizer and `.parameters()`), **device movement** (`.to(device)` moves everything), **train/eval mode** (`.train()`/`.eval()` toggles dropout/batchnorm), **state save/load** (`state_dict`), and **composability** (nest modules into trees). It turns the from-scratch parameter bookkeeping you did in Part II into a clean, reusable abstraction — the unit the whole ecosystem builds on.

---

## Exercise solutions

### Exercise 1 — Re-implement the Chapter 6 GPT as `nn.Module`s

The full `nn.Module` GPT is the [Chapter 6 Exercise 5 solution](06-transformer-solutions.md#exercise-5--char-level-gpt-on-tinyshakespeare-pytorch). The point of *this* exercise is to confirm the framework version reproduces your from-scratch behavior:

```python
import torch
torch.manual_seed(0)

# Same architecture, same seed, same data -> the nn.Module version should track
# your from-scratch NumPy/Value transformer's loss curve closely.
model = GPT()                                 # the nn.Module GPT from Ch.6
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
losses = []
for step in range(500):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

print("start loss:", round(losses[0], 3), "→ end loss:", round(losses[-1], 3))
# Both implementations start near ln(V) and converge on the same structure-learning curve.
```

**Result:** the `nn.Module` GPT learns the same task with the same loss trajectory as your from-scratch version — because the framework is doing **exactly** the autograd + parameter bookkeeping you built by hand in Part II, just industrialized. `nn.Module` removed the boilerplate (manual parameter lists, manual grad zeroing) without changing the math. That equivalence is the confidence-builder: frameworks aren't magic.

### Exercise 2 — `torch.compile` speedup + a deliberate graph break

```python
import torch, time

def bench(fn, x, n=50):
    for _ in range(3): fn(x)                  # warmup (compile happens here)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn(x)
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

model = GPT().cuda().eval()
compiled = torch.compile(model)
x = torch.randint(0, V, (8, 64), device="cuda")

with torch.no_grad():
    print(f"eager   : {bench(lambda z: model(z)[0], x):.2f} ms")
    print(f"compiled: {bench(lambda z: compiled(z)[0], x):.2f} ms")   # typically faster

# Trigger a graph break: data-dependent Python control flow can't be captured.
@torch.compile
def with_break(z):
    if z.sum() > 0:           # .item()-style data dependence forces a break
        return z * 2
    return z - 1
import torch._dynamo as dyno
dyno.reset()
with_break(torch.randn(4, device="cuda"))
print("graph break explanations:", len(dyno.explain(with_break)(torch.randn(4, device="cuda")).break_reasons))
```

**Result:** `torch.compile` runs the same model **faster** (fusion + Triton codegen) after a one-time compile warmup, with no code rewrite. Introducing **data-dependent control flow** forces a **graph break** — TorchDynamo can't trace through the Python branch, so it splits the graph and falls back to eager there (visible via `torch._dynamo.explain`). The lesson: keep hot paths free of Python-level data dependence so the compiler can capture one big graph.

### Exercise 3 — JAX linear regression with `grad` + `jit` (speed with/without)

```python
# pip install jax jaxlib
import jax, jax.numpy as jnp, time

key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (1000, 10))
true_w = jax.random.normal(jax.random.PRNGKey(1), (10,))
y = X @ true_w + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (1000,))

def loss_fn(w, X, y):
    return jnp.mean((X @ w - y) ** 2)

grad_fn = jax.grad(loss_fn)                    # autodiff transform

def step(w, X, y, lr=0.1):
    return w - lr * grad_fn(w, X, y)

step_jit = jax.jit(step)                       # compile the step

def train(step_impl, n=1000):
    w = jnp.zeros(10)
    t0 = time.perf_counter()
    for _ in range(n):
        w = step_impl(w, X, y)
    w.block_until_ready()                      # JAX is async; force completion
    return time.perf_counter() - t0

train(step_jit, 5)                             # warmup compile
print(f"no jit: {train(step, 1000):.3f} s")
print(f"jit   : {train(step_jit, 1000):.3f} s")   # markedly faster
print("recovered w close to true:", bool(jnp.allclose(step_jit(true_w, X, y), true_w, atol=0.05)))
```

**Result:** `jax.grad` gives the gradient as a **transform** of the pure loss function (no `.backward()`), and `jax.jit` compiles the whole update step with XLA — fusing ops and removing Python overhead for a large speedup, especially over many iterations. Note the **functional style**: `w` is passed in and returned, never mutated in place. `block_until_ready()` is needed because JAX dispatch is asynchronous.

### Exercise 4 — `jax.vmap` equals an explicit loop

```python
import jax, jax.numpy as jnp

def predict(w, x):                 # single-example function: w·x
    return jnp.dot(w, x)

w = jnp.array([1.0, 2.0, 3.0])
batch = jnp.arange(15.0).reshape(5, 3)        # 5 examples

explicit = jnp.stack([predict(w, x) for x in batch])   # manual loop
vectorized = jax.vmap(predict, in_axes=(None, 0))(w, batch)  # vmap over axis 0 of x

print("vmap == explicit loop:", bool(jnp.allclose(explicit, vectorized)))   # True
print(vectorized)              # [ 8. 26. 44. 62. 80.]
```

**Result:** `vmap` turns the single-example `predict` into a batched one **automatically** — identical results to the hand-written loop, but vectorized (and `jit`/`grad`-composable). `in_axes=(None, 0)` says "broadcast `w`, map over axis 0 of `x`." You write the clean per-example math and get efficient batching for free, eliminating a whole class of batch-dimension bugs.

### Exercise 5 — The same MLP training loop in PyTorch and JAX (reflection)

```python
# ---------- PyTorch: imperative, stateful, in-place updates ----------
import torch, torch.nn as nn
torch.manual_seed(0)
modelt = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.SGD(modelt.parameters(), lr=0.1)
Xt, yt = torch.randn(64, 10), torch.randn(64, 1)
for _ in range(100):
    loss = ((modelt(Xt) - yt) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()      # MUTATE params in place
print("pytorch final loss:", round(loss.item(), 4))

# ---------- JAX: functional, pure, explicit state threading ----------
import jax, jax.numpy as jnp
def init(key):
    k1, k2 = jax.random.split(key)
    return [jax.random.normal(k1, (10, 32)) * 0.1, jnp.zeros(32),
            jax.random.normal(k2, (32, 1)) * 0.1, jnp.zeros(1)]
def forward(p, x):
    h = jnp.maximum(0, x @ p[0] + p[1]); return h @ p[2] + p[3]
def loss_fn(p, x, y): return jnp.mean((forward(p, x) - y) ** 2)
Xj, yj = jnp.array(Xt.numpy()), jnp.array(yt.numpy())
p = init(jax.random.PRNGKey(0)); gfn = jax.jit(jax.grad(loss_fn))
for _ in range(100):
    g = gfn(p, Xj, yj)
    p = [w - 0.1 * dw for w, dw in zip(p, g)]          # RETURN new params (no mutation)
print("jax final loss:", round(float(loss_fn(p, Xj, yj)), 4))
```

**Reflection (the write-up that makes a good blog post):**

- **State**: PyTorch **mutates** parameters in place (`opt.step()`); JAX **returns new** parameters each step (pure, immutable). PyTorch hides state in objects; JAX threads it explicitly.
- **Gradients**: PyTorch builds a graph during the forward pass and calls `.backward()`; JAX treats `grad` as a **function transform** applied to a pure function — no tape, no `.backward()`.
- **Randomness**: PyTorch uses a global seed; JAX passes/splits explicit **keys**.
- **Speed**: JAX's `jit` compiles the whole step with XLA; PyTorch matches it with `torch.compile`.
- **Feel**: PyTorch reads like imperative Python (familiar, easy to debug); JAX reads like math (pure functions you compose with `grad`/`jit`/`vmap`).

**Result:** both converge to the same low loss because the underlying math is identical — the difference is **programming model**, not capability. Being fluent in both, and able to articulate the tradeoff above, is exactly the "T-shaped" differentiator the book recommends.

---

[← Chapter 15 solutions](15-gpu-programming-solutions.md) · [Solutions index](README.md) · [Next: Chapter 17 solutions →](17-serving-mlops-solutions.md)
