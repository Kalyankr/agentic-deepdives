# Chapter 2 — Mathematical Foundations · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-1-foundations/02-mathematics.md)

---

## Interview answers

### Q: "Why subtract the max before softmax?"

**Numerical stability.** Softmax is $p_i = e^{z_i} / \sum_k e^{z_k}$. If any logit $z_i$ is large (say 1000), $e^{1000}$ overflows to `inf` in floating point and you get `nan`. Softmax is **shift-invariant**: subtracting any constant $c$ from every logit leaves the result unchanged, because

$$\frac{e^{z_i - c}}{\sum_k e^{z_k - c}} = \frac{e^{-c}e^{z_i}}{e^{-c}\sum_k e^{z_k}} = \frac{e^{z_i}}{\sum_k e^{z_k}}.$$

Choosing $c = \max_k z_k$ makes the largest exponent $e^0 = 1$, so nothing overflows and the result is mathematically identical. It costs one `max` and one subtraction — every production softmax does this.

### Q: "Derive the gradient of cross-entropy + softmax."

With softmax probabilities $p = \text{softmax}(z)$ and one-hot target $y$ (hot at class $c$), the cross-entropy is $L = -\sum_k y_k \log p_k$. The gradient w.r.t. the logits is the famously clean

$$\boxed{\frac{\partial L}{\partial z_j} = p_j - y_j.}$$

**The derivation** (worth being able to do on a whiteboard) — first the softmax Jacobian:

$$\frac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j),$$

then chain through the loss:

$$\frac{\partial L}{\partial z_j} = -\sum_k \frac{y_k}{p_k}\frac{\partial p_k}{\partial z_j} = -\sum_k \frac{y_k}{p_k}p_k(\delta_{kj}-p_j) = -\sum_k y_k(\delta_{kj}-p_j) = -y_j + p_j\underbrace{\sum_k y_k}_{=1} = p_j - y_j.$$

The interpretation: the gradient is just "predicted minus actual." If the model is right and confident, $p \approx y$ and the gradient vanishes; if it's confidently wrong, the gradient is large. This is why softmax + cross-entropy is the default classification head — its backward pass is trivial and well-scaled.

### Q: "Why Adam over SGD for transformers?"

Transformer gradients are **heterogeneous in scale** across parameters — embeddings, attention projections, LayerNorm gains, and the LM head all have very different gradient magnitudes. Plain SGD uses one global learning rate, so a step size that's stable for one group is too big or too small for another.

Adam keeps a **per-parameter** estimate of the first moment (mean, $m$) and second moment (uncentered variance, $v$) of the gradient and steps with $m/(\sqrt{v}+\epsilon)$. Dividing by $\sqrt{v}$ rescales each parameter's step to roughly unit size, so it automatically handles the conditioning. **AdamW** additionally *decouples* weight decay from the gradient update (decay is applied directly to the weights, not folded into the gradient and then rescaled by $v$), which fixes a subtle bug in the original Adam+L2 and is the de-facto optimizer for LLMs.

### Q: "What is KL divergence and where have you used it?"

KL divergence measures how different one distribution $P$ is from a reference $Q$:

$$D_{\text{KL}}(P \,\|\, Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} \ge 0,$$

zero iff $P = Q$. It is **asymmetric**: $D_{\text{KL}}(P\|Q) \neq D_{\text{KL}}(Q\|P)$ — forward KL is "mean-seeking" (covers all of $P$'s mass), reverse KL is "mode-seeking" (locks onto one mode). Places I've used it:

- **RLHF**: a KL penalty against the frozen reference model keeps the policy from drifting/reward-hacking (the "leash," Chapter 9).
- **DPO**: the loss is *derived* from the KL-constrained RLHF objective in closed form.
- **Distillation**: minimize KL between student and teacher output distributions.
- **VAEs**: KL to a prior regularizes the latent space.

### Q: "What's perplexity?"

Perplexity is the exponential of the average cross-entropy (in nats):

$$\text{PPL} = \exp\!\Big(-\tfrac{1}{N}\sum_i \log p(x_i)\Big) = \exp(\text{cross-entropy}).$$

Intuitively it's the **average branching factor** — the effective number of equally-likely choices the model is hesitating between at each token. PPL $= 1$ means perfect certainty; PPL $= V$ (vocab size) means a uniform random guess. Lower is better, and it's the standard intrinsic metric for language-model quality.

---

## Exercise solutions

### Exercise 1 — Softmax and shift-invariance

```python
import numpy as np

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)   # the stability trick
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

z = np.array([2.0, 1.0, 0.1])
print(softmax(z))                 # [0.659, 0.242, 0.099]
print(np.allclose(softmax(z), softmax(z + 100)))   # True  -> shift-invariant

# Without the max-subtraction, large logits overflow:
def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()
print(softmax_naive(np.array([1000.0, 1.0])))   # [nan nan] -> why we subtract the max
```

**Result:** `softmax(z)` and `softmax(z + 100)` are equal to floating-point tolerance, confirming shift-invariance. The naive version produces `nan` on large logits — the concrete reason the max-subtraction is mandatory.

### Exercise 2 — Cross-entropy: confident-correct vs confident-wrong

```python
import numpy as np

def cross_entropy(probs, target_idx):
    return -np.log(probs[target_idx] + 1e-12)   # epsilon guards log(0)

confident_correct = np.array([0.97, 0.02, 0.01])   # true class = 0
confident_wrong   = np.array([0.01, 0.02, 0.97])   # true class = 0

print(cross_entropy(confident_correct, 0))   # ~0.030  -> low loss
print(cross_entropy(confident_wrong,   0))   # ~4.605  -> high loss
print(cross_entropy(np.array([1/3]*3), 0))   # ~1.099  -> uniform guess = log(3)
```

**Result:** confident-and-correct → loss ≈ 0.03; confident-and-wrong → loss ≈ 4.6; a uniform guess → $\ln 3 \approx 1.10$. Cross-entropy punishes confident mistakes *hard* (the $-\log$ blows up as $p \to 0$), which is exactly the pressure you want during training.

### Exercise 3 — Rank-$r$ SVD approximation → LoRA

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
A = rng.standard_normal((100, 80))           # full-rank random matrix
U, S, Vt = np.linalg.svd(A, full_matrices=False)

def low_rank_approx(r):
    return (U[:, :r] * S[:r]) @ Vt[:r, :]     # keep top-r singular triplets

ranks = range(1, len(S) + 1)
errors = [np.linalg.norm(A - low_rank_approx(r), 'fro') for r in ranks]

# Eckart–Young: the truncation error equals the norm of the dropped singular values
check = [np.sqrt(np.sum(S[r:]**2)) for r in ranks]
print(np.allclose(errors, check))            # True

plt.plot(list(ranks), errors); plt.xlabel('rank r'); plt.ylabel('Frobenius error')
plt.title('Low-rank approximation error vs r'); plt.show()
```

**Result:** error decreases monotonically as $r$ grows and hits ~0 at the true rank. By the **Eckart–Young theorem**, the best rank-$r$ approximation is exactly the top-$r$ singular triplets, and its error is $\sqrt{\sum_{k>r}\sigma_k^2}$ — which the code verifies.

**Connection to LoRA:** if a matrix can be well-approximated by something low-rank, you can store/learn it as a product of two thin matrices $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times d}$ instead of a full $d\times d$. LoRA *bets* that the fine-tuning **update** $\Delta W$ is approximately low-rank, so it freezes $W$ and learns only $\Delta W = BA$ with small $r$ — turning millions of trainable parameters into thousands (Chapter 11).

### Exercise 4 — Gradient descent vs Adam on an ill-conditioned bowl

$f(x,y) = x^2 + 10y^2$ has gradient $\nabla f = (2x,\,20y)$. The $y$ direction is 10× more curved, so a single learning rate that's stable in $y$ crawls in $x$ (and vice-versa) — classic ill-conditioning.

```python
import numpy as np

def grad(p):
    x, y = p
    return np.array([2*x, 20*y])

def gd(lr=0.05, steps=50):
    p = np.array([1.0, 1.0])
    for _ in range(steps):
        p = p - lr * grad(p)
    return p

def adam(lr=0.1, steps=50, b1=0.9, b2=0.999, eps=1e-8):
    p = np.array([1.0, 1.0]); m = np.zeros(2); v = np.zeros(2)
    for t in range(1, steps + 1):
        g = grad(p)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g*g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        p = p - lr * mhat / (np.sqrt(vhat) + eps)
    return p

print("GD  :", gd(),   "‖p‖ =", np.linalg.norm(gd()))     # slow in x, near 0 in y
print("Adam:", adam(), "‖p‖ =", np.linalg.norm(adam()))   # both coords ~0 faster
```

**Result:** after 50 steps, GD with a single learning rate has driven $y$ to ~0 but $x$ lags (or, with a larger LR, $y$ oscillates/diverges). Adam normalizes each coordinate by its own $\sqrt{v}$, so it makes balanced progress in both directions and reaches the minimum faster — a 2-D illustration of why adaptive methods win on transformer-scale heterogeneous gradients.

### Exercise 5 — Derive $\partial\,\text{softmax}(x)_i/\partial x_j$ and verify numerically

**Analytic Jacobian:**

$$\frac{\partial p_i}{\partial x_j} = p_i(\delta_{ij} - p_j) = \begin{cases} p_i(1 - p_i) & i = j \\ -p_i p_j & i \neq j \end{cases}$$

```python
import numpy as np

def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def softmax_jacobian(z):
    p = softmax(z)
    return np.diag(p) - np.outer(p, p)          # p_i δ_ij − p_i p_j

z = np.array([0.5, -1.0, 2.0])
J = softmax_jacobian(z)

# Finite-difference check: column j ≈ (softmax(z + eps e_j) − softmax(z)) / eps
eps = 1e-6
J_num = np.zeros((3, 3))
for j in range(3):
    dz = np.zeros(3); dz[j] = eps
    J_num[:, j] = (softmax(z + dz) - softmax(z)) / eps

print(np.allclose(J, J_num, atol=1e-4))   # True
```

**Result:** the closed-form Jacobian $\text{diag}(p) - pp^\top$ matches the finite-difference estimate to ~1e-4. This Jacobian is the building block that, when composed with the cross-entropy gradient, collapses to the clean $p - y$ from the first interview question.

---

[← Chapter 1 solutions](01-introduction-solutions.md) · [Solutions index](README.md) · [Next: Chapter 3 solutions →](03-programming-solutions.md)
