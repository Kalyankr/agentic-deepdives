# 11 · Debugging & Code-Review Round

> A common (and revealing) round: the interviewer shows broken ML code — a training loop that won't
> converge, an attention impl with a subtle leak, a sampler that crashes — and asks **"what's wrong?"**
> It tests whether you actually understand the mechanics, not just the vocabulary.

**How to run the drill:** read each *symptom + buggy snippet*, say the bug and fix **out loud** before
revealing. The NumPy snippets are runnable — paste them and confirm the failure, then the fix.

**A reusable debugging method (say this in the round):**
1. **Reproduce** + read the actual error/shape.
2. **Localize** — print shapes, dtypes, and a few values; bisect the pipeline.
3. **Hypothesize** the smallest cause; **check** it; **fix**; **add a regression test**.
4. Sanity baselines: can the model **overfit one batch**? Is the **loss at init ≈ `ln(vocab)`**?

---

## A. Numerical & attention bugs (runnable)

### 1. Softmax overflows on large logits
**Symptom:** `nan`/`inf` in the loss with big activations.
```python
import numpy as np
def softmax(x):
    e = np.exp(x)              # BUG: exp(1000) -> inf
    return e / e.sum()
softmax(np.array([1000.0, 1001.0, 1002.0]))   # -> [nan, nan, nan]
```
**Bug:** no max-subtraction → overflow. **Fix:** subtract the max (shift-invariant).
```python
def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
```

### 2. Attention has no causal mask (future leakage)
**Symptom:** train loss great, generation garbage — the model "cheated" by seeing the future.
```python
def attn(Q, K, V):                       # BUG: no mask -> position t attends to t+1..T
    s = Q @ K.T / np.sqrt(Q.shape[-1])
    w = softmax_rows(s)
    return w @ V
```
**Fix:** add `-inf` to future positions **before** softmax (not 0, and not after).
```python
def attn(Q, K, V):
    T = Q.shape[0]
    s = Q @ K.T / np.sqrt(Q.shape[-1])
    s = s + np.triu(np.full((T, T), -np.inf), k=1)   # mask strictly-upper triangle
    w = softmax_rows(s)
    return w @ V
```
*Gotchas:* masking with `0` instead of `-inf` doesn't remove the position; masking *after* softmax requires renormalizing.

### 3. Softmax over the wrong axis
**Symptom:** rows don't sum to 1; attention weights look uniform/odd.
```python
w = np.exp(s) / np.exp(s).sum(axis=0, keepdims=True)   # BUG: normalizes columns
```
**Fix:** normalize across **keys** = last axis: `axis=-1` (per query row).

### 4. Temperature applied to probabilities, not logits
**Symptom:** "temperature" barely changes diversity; distribution no longer sums to 1.
```python
p = softmax(logits)
p = p / temperature        # BUG: scaling probabilities is not temperature
token = sample(p)
```
**Fix:** temperature scales **logits** before softmax: `p = softmax(logits / temperature)`.

### 5. Top-p (nucleus) without renormalizing
**Symptom:** rare crashes / biased sampling — truncated probs don't sum to 1.
```python
probs[~keep] = 0.0
token = np.random.choice(len(probs), p=probs)   # BUG: p doesn't sum to 1 -> error/bias
```
**Fix:** renormalize the surviving nucleus before sampling: `probs = probs / probs.sum()`.

### 6. Off-by-one targets in LM training
**Symptom:** loss collapses toward 0 almost immediately — the model is "predicting" the input.
```python
logits = model(tokens)                  # (T, V)
loss = cross_entropy(logits, tokens)    # BUG: position t predicts token t (itself)
```
**Fix:** shift — logits at `t` predict token `t+1`.
```python
loss = cross_entropy(logits[:-1], tokens[1:])
```

---

## B. Training-loop bugs (PyTorch idioms)

### 7. Forgetting `zero_grad()`
**Symptom:** loss is unstable / explodes; gradients silently accumulate across steps.
```python
for x, y in loader:
    loss = loss_fn(model(x), y)
    loss.backward()        # BUG: grads add to last step's grads
    optimizer.step()
```
**Fix:** clear grads each step: call `optimizer.zero_grad()` (or `set_to_none=True`) **before** `backward()`.

### 8. Feeding probabilities into `CrossEntropyLoss`
**Symptom:** trains slowly / underfits — softmax applied twice.
```python
logits = model(x)
probs = torch.softmax(logits, dim=-1)
loss = nn.CrossEntropyLoss()(probs, y)   # BUG: expects raw LOGITS, not probabilities
```
**Fix:** pass **logits** directly — `CrossEntropyLoss` does `log_softmax` internally.

### 9. Evaluating in train mode / building the graph
**Symptom:** eval metrics jitter run-to-run; eval OOMs.
```python
def evaluate(model, loader):
    for x, y in loader:                  # BUG: dropout/BN still in train mode; autograd on
        preds = model(x)
```
**Fix:** `model.eval()` + wrap in `torch.no_grad()` (then `model.train()` after).

### 10. Gradient clipping after the optimizer step
**Symptom:** clipping "does nothing"; loss still spikes.
```python
loss.backward()
optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # BUG: too late
```
**Fix:** clip **between** `backward()` and `step()` (grads must exist and not be applied yet).

---

## C. Data & evaluation bugs

### 11. Preprocessing leaks test statistics
**Symptom:** great offline metrics, worse in production — info from test leaked into training.
```python
X = scaler.fit_transform(X)              # BUG: fit on ALL data
X_tr, X_te = split(X)
```
**Fix:** `fit` on **train only**, then `transform` train and test with those stats. (Same rule for
vocab, imputation, target encoding, and dedup across splits.)

### 12. Reporting a single eval number with no uncertainty
**Symptom:** you "beat baseline by 1.5%" — but it's within noise.
**Fix:** report a **confidence interval** (bootstrap or Wald). On `n` examples, the SE of an accuracy
`p` is `sqrt(p(1-p)/n)`; a difference smaller than ~`2·SE` isn't significant. See
[12-math-stats](12-math-stats.md). (This exact gate is implemented in the [CI-eval notebook](../notebooks/README.md).)

---

## The bug catalog (skim before the round)

| Area | Classic bug | Tell |
|------|-------------|------|
| Numerics | softmax without max-subtraction | `nan` loss |
| Attention | missing/late causal mask | train ✓, gen ✗ |
| Shapes | softmax/normalize on wrong axis | rows ≠ 1 |
| Sampling | temperature on probs; top-p not renormalized | no diversity / crash |
| LM data | targets not shifted by one | loss → 0 instantly |
| Training | missing `zero_grad`; clip after `step` | unstable loss |
| Loss | probs into `CrossEntropyLoss` | slow/underfit |
| Eval | no `eval()`/`no_grad()` | jitter / OOM |
| Data | scaler/vocab fit before split | offline≫online |
| Mixed precision | fp16 without loss scaling | underflow `nan` |
| Stats | single number, no CI | "win" is noise |
| Repro | no seed / nondeterministic ops | unrepeatable runs |

> Strong signal in this round: **state the symptom→cause link** ("loss is `nan` from step 1 → suspect
> numerics or LR"), propose the **minimal fix**, and add a **test or sanity check** so it can't regress.
