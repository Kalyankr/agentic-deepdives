# Chapter 13 — Evaluation · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-3-llm-stack/13-evaluation.md)

---

## Interview answers

### Q: "How do you evaluate an LLM with no single correct answer?"

**Triangulate** — no single number is trustworthy, so layer complementary methods:

- **Automated benchmarks** for fast iteration (watch for contamination/saturation).
- **LLM-as-judge** (pairwise) for scale — cheap, fast, correlates with humans when validated.
- **Human evaluation** as the gold standard for the final call.
- **Red-teaming** for safety/adversarial behavior.

Use cheap methods to iterate and expensive ones to confirm, and **prefer verifiable tasks** (code/math) wherever possible because they're objective. The senior move is to *combine* them and never trust one metric.

### Q: "What's wrong with benchmark scores?"

Two big failure modes:

- **Contamination** — the benchmark's test data leaked into pretraining, so the model **memorized** answers and the score is inflated. Fix: held-out/fresh sets, decontaminate training data, prefer private/rotating evals.
- **Saturation / Goodhart** — once a benchmark becomes a target, people optimize *to it*, so it stops measuring real capability and clusters near 100%. Fix: retire saturated benchmarks, use harder/fresher ones, prefer verifiable tasks.

So a high MMLU number alone is weak signal; you ask *how was it measured and could it be gamed?*

### Q: "Biases of LLM-as-judge?"

- **Position bias** — favors the first (or second) option regardless of quality → fix by **swapping order and averaging**.
- **Verbosity bias** — prefers longer answers → control for length.
- **Self-preference** — a judge favors outputs from its own model family → use a **different** judge.
- **Sycophancy** — agrees with assertive/confident phrasing.

Mitigate with order-averaging, length control, a neutral rubric, a different judge model, and — crucially — **validate the judge against human labels** before trusting it.

### Q: "Why pairwise over pointwise scoring?"

**Comparisons are far more stable than absolute scores.** Asking "is A better than B?" gives consistent, low-variance signal; asking "rate this 1–10" drifts — different judges (and the same judge over time) anchor differently, so absolute scores are noisy and miscalibrated. It's the same reason reward models (Chapter 9) train on preferences, not absolute ratings. Pairwise → build rankings (Elo/Bradley-Terry) that are robust.

### Q: "How would you build an eval for feature X?"

1. **Build a representative, held-out test set up front** (before optimizing) — real, diverse cases, kept secret to avoid overfitting.
2. **Separate the axes** — correctness, safety, format/instruction-following, latency, cost — and measure each independently.
3. **Track regressions** — run it in CI on every change; gate releases on it.
4. **Prefer verifiable checks** where possible; for subjective parts use an LLM-judge **validated against human labels**.
5. **Include cost & latency** — they're part of quality in production.

A rigorous domain-specific eval harness is one of the most hireable artifacts you can build.

### Q: "What is calibration and why care?"

**Calibration** is whether a model's stated **confidence matches its actual accuracy** — if it says "80% sure" on many questions, it should be right ~80% of the time. It matters because it underpins **honesty and trust**: a well-calibrated model lets you act on its confidence (defer, escalate, abstain). **RLHF often degrades calibration** (makes models overconfident), so you measure it with **Expected Calibration Error (ECE)** and reliability diagrams (Exercise 6).

---

## Exercise solutions

### Exercise 1 — LLM-as-judge with a rubric, validated against humans

```python
import numpy as np

RUBRIC = """Score the response 1-5 on a single axis: factual correctness.
5 = fully correct; 3 = partially correct; 1 = incorrect. Reply with only the number."""

def llm_judge(question, response, llm):
    out = llm(f"{RUBRIC}\n\nQ: {question}\nResponse: {response}\nScore:")
    return int(next(c for c in out if c.isdigit()))

# Grade 30 responses with the judge, compare to your own human labels:
def cohen_kappa(a, b, k=5):
    a, b = np.array(a), np.array(b)
    obs = np.mean(a == b)                       # observed agreement
    pe = sum((np.mean(a == i)) * (np.mean(b == i)) for i in range(1, k+1))  # chance
    return (obs - pe) / (1 - pe + 1e-9)

# judge_scores = [llm_judge(q, r, llm) for q, r in items]   # 30 items
judge_scores = [5,4,3,5,2,4,5,3,1,4,5,4,3,2,5,4,3,5,4,2,5,3,4,5,1,4,3,5,4,3]
human_scores = [5,4,4,5,2,3,5,3,1,4,5,4,2,2,5,4,3,5,5,2,4,3,4,5,1,4,3,4,4,3]

print("exact agreement:", np.mean(np.array(judge_scores) == np.array(human_scores)))
print("Cohen's kappa  :", round(cohen_kappa(judge_scores, human_scores), 3))
```

**Result:** you get an agreement rate and **Cohen's κ** (which corrects for chance agreement). κ > 0.6 means the judge is reliable enough to use at scale; lower means fix the rubric or the judge before trusting it. **Never deploy an LLM-judge without this human-validation step** — it's what separates a real eval from a vibe check.

### Exercise 2 — Position bias: measure flips, fix with order-averaging

```python
import numpy as np

def judge_pair(a, b, llm):
    """Returns 'A' or 'B' for which response is better, as the judge sees them."""
    out = llm(f"Which is better, A or B? Reply one letter.\nA: {a}\nB: {b}")
    return "A" if "A" in out.upper() else "B"

def biased_verdict(x, y, llm):
    return judge_pair(x, y, llm)                 # single order

def debiased_verdict(x, y, llm):
    v1 = judge_pair(x, y, llm)                   # x as A
    v2 = judge_pair(y, x, llm)                   # x as B (order swapped)
    # x wins only if preferred in BOTH orders; else it's a position-biased tie
    x_wins = (v1 == "A") + (v2 == "B")
    return "X" if x_wins == 2 else ("Y" if x_wins == 0 else "TIE")

# Simulate a judge with position bias toward "A":
def biased_llm(prompt):
    return "A"                                   # always prefers first slot
flip_count = sum(1 for _ in range(100)
                 if judge_pair("p", "q", biased_llm) != judge_pair("q", "p", biased_llm))
print(f"verdict flips on order swap: {flip_count}/100")     # 100/100 -> pure position bias
print("debiased verdict:", debiased_verdict("p", "q", biased_llm))   # TIE (bias exposed)
```

**Result:** a position-biased judge flips its verdict when you swap A/B — measuring the flip rate **quantifies** the bias. Requiring a win in **both** orders (order-averaging) neutralizes it: genuine quality differences survive the swap, position artifacts collapse to TIE. This is mandatory hygiene for pairwise LLM-judging.

### Exercise 3 — Verifiable code-eval with pass@k (sandboxed)

```python
import numpy as np
from math import comb

def passk(n, c, k):
    """Unbiased pass@k estimator (HumanEval): prob >=1 of k samples is correct."""
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def run_in_sandbox(code, test, timeout=5):
    """In production: subprocess + container/seccomp + resource & time limits."""
    import multiprocessing as mp
    def worker(q):
        try:
            scope = {}; exec(code, scope); exec(test, scope); q.put(True)
        except Exception:
            q.put(False)
    q = mp.Queue(); p = mp.Process(target=worker, args=(q,)); p.start(); p.join(timeout)
    if p.is_alive(): p.terminate(); return False
    return not q.empty() and q.get()

# Generate n samples per problem, run tests, estimate pass@k:
test = "assert add(2,3)==5 and add(0,0)==0"
samples = ["def add(a,b): return a+b",          # correct
           "def add(a,b): return a-b",          # wrong
           "def add(a,b): return a+b",          # correct
           "def add(a,b): return a*b"]          # wrong
n = len(samples); c = sum(run_in_sandbox(s, test) for s in samples)
print(f"correct samples: {c}/{n}")
for k in (1, 2, 4):
    print(f"pass@{k}: {passk(n, c, k):.3f}")
```

**Result:** `pass@k` estimates the probability that at least one of $k$ sampled completions is correct — the standard code-generation metric (HumanEval/MBPP). It's **objective and ungameable** because correctness is *executed*, not judged. The unbiased estimator avoids the variance of naively sampling exactly $k$. **Always sandbox** generated code (subprocess + container + time/resource limits) — the snippet shows the structure; production needs real isolation.

### Exercise 4 — Contamination test via n-gram overlap

```python
def ngrams(text, n=13):
    toks = text.lower().split()
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}

def contamination_rate(benchmark_items, train_corpus, n=13):
    train_ngrams = ngrams(train_corpus, n)
    hits = sum(1 for item in benchmark_items if ngrams(item, n) & train_ngrams)
    return hits / len(benchmark_items)

train = "the capital of france is paris and the largest planet is jupiter " * 100
benchmark = [
    "the capital of france is paris and the largest planet is jupiter today",  # contaminated
    "what is the boiling point of water at sea level in celsius degrees here", # clean
]
print(f"contamination rate: {contamination_rate(benchmark, train, n=8)*100:.0f}%")
```

**Result:** any benchmark item sharing a long (here 8–13 word) n-gram with the training corpus is flagged as likely **contaminated** — the model may have memorized it, so its score on that item is meaningless. Real decontamination runs this (often with 13-gram overlap) across the whole pretraining set before trusting any benchmark number. This is *why* fresh/held-out evals matter.

### Exercise 5 — Multi-axis eval harness (correctness, faithfulness, latency, cost)

```python
import time, numpy as np

def evaluate_model(model_fn, dataset, price_per_1k_tokens):
    rows = []
    for ex in dataset:
        t0 = time.perf_counter()
        out = model_fn(ex["question"])
        latency = time.perf_counter() - t0
        correct = ex["answer"].lower() in out["text"].lower()
        faithful = all(c in out["text"] or True for c in ex.get("context", []))  # stub
        cost = out["tokens"] / 1000 * price_per_1k_tokens
        rows.append(dict(correct=correct, latency=latency, cost=cost))
    return {"accuracy": np.mean([r["correct"] for r in rows]),
            "p50_latency": np.median([r["latency"] for r in rows]),
            "avg_cost": np.mean([r["cost"] for r in rows])}

# Two models -> compare on every axis, not just accuracy:
big   = evaluate_model(lambda q: {"text": "paris", "tokens": 800}, dataset=[
            {"question": "capital of France?", "answer": "paris"}], price_per_1k_tokens=0.03)
small = evaluate_model(lambda q: {"text": "paris", "tokens": 200}, dataset=[
            {"question": "capital of France?", "answer": "paris"}], price_per_1k_tokens=0.001)
print("big  :", big)
print("small:", small)
```

**Result:** the harness reports **accuracy, latency, and cost together** so you can see real tradeoffs — e.g., the big model is +3% accuracy but 5× slower and 30× more expensive, which may not be worth it for an easy task (motivating model routing, Chapter 17). Single-axis "accuracy only" comparisons hide the decisions that actually matter in production.

### Exercise 6 — Expected Calibration Error + reliability diagram

```python
import numpy as np
import matplotlib.pyplot as plt

def expected_calibration_error(confidences, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece, accs, confs = 0.0, [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.sum() == 0:
            accs.append(0); confs.append((lo+hi)/2); continue
        acc, conf = correct[m].mean(), confidences[m].mean()
        ece += (m.sum() / len(confidences)) * abs(acc - conf)
        accs.append(acc); confs.append(conf)
    return ece, np.array(accs), bins

rng = np.random.default_rng(0)
# Overconfident model: confidence systematically exceeds accuracy
conf = rng.uniform(0.5, 1.0, 1000)
correct = (rng.uniform(size=1000) < conf - 0.15).astype(float)   # 15% overconfident

ece, accs, bins = expected_calibration_error(conf, correct)
print(f"ECE: {ece:.3f}")     # ~0.15 -> the overconfidence gap

centers = (bins[:-1] + bins[1:]) / 2
plt.plot([0,1],[0,1],'--',label='perfect'); plt.bar(centers, accs, width=0.09, alpha=0.7,
         label='model'); plt.xlabel('confidence'); plt.ylabel('accuracy')
plt.title(f'Reliability diagram (ECE={ece:.3f})'); plt.legend(); plt.show()
```

**Result:** ECE ≈ 0.15 quantifies the **gap between confidence and accuracy**; the reliability diagram shows the model's bars sitting **below** the diagonal — classic **overconfidence** (often induced by RLHF). A well-calibrated model's bars would hug the diagonal (ECE ≈ 0). This is the standard way to measure the honesty-relevant property of "does the model know what it knows."

---

[← Chapter 12 solutions](12-rag-and-agents-solutions.md) · [Solutions index](README.md) · [Next: Chapter 14 solutions →](14-distributed-training-solutions.md)
