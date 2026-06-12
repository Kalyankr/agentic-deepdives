"""Build NB11 — Evaluations."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 11 · Evaluations

> Module: **09 · Evaluations** — the single most important skill at a frontier lab.

> *You can't improve what you can't measure.* Evals turn "feels better" into a number, gate
> releases, drive RLHF/DPO, and quantify safety. We implement metrics, a **bootstrap CI**,
> an **LLM-as-judge** (with a **bias** demo), **Elo** ranking, and agent/safety eval ideas.

### Learning objectives
1. Compute capability metrics and **confidence intervals**.
2. Build and **validate** an LLM-as-judge; account for its biases.
3. Evaluate RAG and agents; design safety/red-team evals.
4. Wire evals into **CI** to block regressions.
"""),
    md(r"""
## 1. Classic metrics (and their limits)

- **accuracy / exact match** — for closed answers.
- **F1 / token overlap** — partial credit for spans.
- **pass@k** — for code: probability at least one of k samples passes the tests.
- **perplexity** — intrinsic LM quality (NB03).
- **calibration (ECE)** — does the model's confidence match its accuracy?

BLEU/ROUGE exist for generation but correlate poorly with quality — prefer task metrics or judges.
"""),
    code(r"""
import numpy as np
from math import comb
rng = np.random.default_rng(0)

def exact_match(pred, gold): return float(pred.strip().lower() == gold.strip().lower())

def token_f1(pred, gold):
    p, g = pred.lower().split(), gold.lower().split()
    common = 0
    gg = list(g)
    for t in p:
        if t in gg: common += 1; gg.remove(t)
    if common == 0: return 0.0
    prec, rec = common/len(p), common/len(g)
    return 2*prec*rec/(prec+rec)

def pass_at_k(n, c, k):
    # unbiased estimator (Codex): n samples, c correct, prob a random k contains >=1 correct
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

print("exact_match:", exact_match("Paris", "paris"))
print("token_f1   :", round(token_f1("the cat sat", "a cat sat down"), 3))
print("pass@1,5 (10 samples, 3 correct):", round(pass_at_k(10,3,1),3), round(pass_at_k(10,3,5),3))
"""),
    md(r"""
## 2. Statistical rigor — don't ship a number without a CI

Eval scores are estimates from finite samples. Report a **confidence interval** (e.g. via
**bootstrap**) and check **significance** before claiming model A beats B. A 1-point gain on
100 examples is usually noise.
"""),
    code(r"""
def bootstrap_ci(scores, n_boot=5000, alpha=0.05):
    scores = np.asarray(scores, float)
    means = [rng.choice(scores, len(scores), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
    return scores.mean(), lo, hi

results = (rng.random(100) < 0.72).astype(float)   # model got 72% on 100 examples
m, lo, hi = bootstrap_ci(results)
print(f"accuracy = {m:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
print("-> with n=100 the CI is wide; a 2-point 'win' may not be real.")
"""),
    md(r"""
## 3. LLM-as-judge — the workhorse for open-ended quality

For open-ended outputs there's no exact answer, so we ask a strong model to grade — **pairwise**
("which is better, A or B?") or with a **rubric**. It scales, but it's **biased**: position bias
(favoring the first option), verbosity bias, self-preference. **Always validate the judge** and
mitigate (e.g. average over both orderings).
"""),
    code(r"""
# Mock judge with a built-in POSITION BIAS to demonstrate the failure + the fix.
def biased_judge(answer_first, answer_second, position_bias=0.3):
    # "true" quality is the length of the answer here (toy); judge also over-prefers position 1
    q1, q2 = len(answer_first), len(answer_second)
    score1 = q1 + position_bias * 100      # unfair boost to whoever is shown first
    return "A" if score1 >= q2 else "B"

A = "concise correct answer"
B = "a much longer and more detailed and thorough correct answer"

raw = biased_judge(A, B)                     # A shown first
swapped = biased_judge(B, A)                 # B shown first
print("A first  -> winner:", raw)
print("B first  -> winner:", swapped, "(verdict flipped! that's position bias)")

def debiased_judge(x, y):
    v1 = biased_judge(x, y)                   # x first
    v2 = biased_judge(y, x)                   # y first
    # count a win only if consistent across both orders; else call it a tie
    x_wins = (v1 == "A") + (v2 == "B")
    return "X" if x_wins == 2 else ("Y" if x_wins == 0 else "tie")

print("order-averaged verdict (X=concise, Y=long):", debiased_judge(A, B))
"""),
    md(r"""
## 4. Ranking many models — Elo / Bradley–Terry

Chatbot-Arena-style evaluation collects pairwise human (or judge) preferences and fits an **Elo**
(equivalently Bradley–Terry) rating. Useful when there's no absolute score, only "A beat B."
"""),
    code(r"""
def elo_update(ra, rb, a_won, k=32):
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))    # expected score for A
    sa = 1.0 if a_won else 0.0
    return ra + k*(sa - ea), rb + k*((1-sa) - (1-ea))

# simulate: model A truly wins 65% of the time
ra, rb = 1000, 1000
for _ in range(500):
    ra, rb = elo_update(ra, rb, a_won=rng.random() < 0.65)
print(f"final Elo  A={ra:.0f}  B={rb:.0f}  (gap reflects ~65% win rate)")
"""),
    md(r"""
## 5. Evaluating RAG & agents
- **RAG:** retrieval (recall@k, MRR, nDCG) + generation (**faithfulness/groundedness**, answer
  relevance, citation accuracy). Frameworks: RAGAS, TruLens.
- **Agents:** **task success rate**, trajectory quality, tool-call accuracy, **steps/cost/latency**
  to completion. Benchmarks: **SWE-bench**, **τ-bench**, GAIA, WebArena. Use sandboxed,
  reproducible environments; fix seeds.
"""),
    code(r"""
# Agent eval: success rate with a CI + average cost/steps — the dashboard you actually ship.
def eval_agent_suite(tasks):
    succ = np.array([t["passed"] for t in tasks], float)
    m, lo, hi = bootstrap_ci(succ)
    print(f"tasks={len(tasks)}  success={m:.0%}  95% CI [{lo:.0%},{hi:.0%}]")
    print(f"avg steps={np.mean([t['steps'] for t in tasks]):.1f}  avg cost=${np.mean([t['cost'] for t in tasks]):.3f}")

eval_agent_suite([
    {"passed": 1, "steps": 4, "cost": 0.02}, {"passed": 0, "steps": 8, "cost": 0.05},
    {"passed": 1, "steps": 3, "cost": 0.01}, {"passed": 1, "steps": 6, "cost": 0.04},
    {"passed": 1, "steps": 5, "cost": 0.03},
])
"""),
    md(r"""
## 6. Safety & red-team evals
- **Harmlessness:** toxicity, bias, **refusal correctness** (over- vs under-refusal).
- **Red-teaming:** manual + automated adversarial prompts, **jailbreak** robustness.
- **Prompt-injection** resistance for tool-using agents (NB09).
- **Honesty / hallucination / sycophancy** probes.
- **Dangerous-capability** evals gating deployment (Anthropic **RSP** / OpenAI **Preparedness**).

## 7. Evals in CI — block regressions automatically
The flywheel: **prod logs → error analysis → new eval cases → fix → re-eval**. Run the suite on
every model/prompt change; **fail the build** on a statistically significant drop.
"""),
    code(r"""
def ci_gate(baseline, candidate, n=1000, threshold=-0.02):
    # returns nonzero "exit code" if candidate regresses beyond threshold (with CI awareness)
    b = (rng.random(n) < baseline).astype(float)
    c = (rng.random(n) < candidate).astype(float)
    delta = c.mean() - b.mean()
    # crude significance: is the drop bigger than ~2 standard errors?
    se = np.sqrt(b.var()/n + c.var()/n)
    regressed = delta < threshold and delta < -2*se
    print(f"baseline={b.mean():.2f} candidate={c.mean():.2f} delta={delta:+.2f} -> {'FAIL (block PR)' if regressed else 'PASS'}")
    return int(regressed)

ci_gate(0.80, 0.81)   # noise -> PASS
ci_gate(0.80, 0.70)   # real regression -> FAIL
"""),
    md(r"""
## Exercises
1. Build an eval harness that runs a suite against any endpoint, caches results, and reports CIs.
2. Validate a real LLM-as-judge against ~50 human labels; measure agreement + position bias.
3. Add a RAGAS-style faithfulness metric to your NB08 RAG.
4. Write a 20-task agent suite with automated checks; add it to CI as a release gate.

## Resources
- *HELM* (Liang 2022); *MT-Bench / Chatbot Arena* (Zheng 2023).
- *SWE-bench* (Jimenez 2023); *τ-bench* (Yao 2024); *GAIA* (Mialon 2023); RAGAS.
- Anthropic **Responsible Scaling Policy**; OpenAI **Preparedness Framework**.
- `inspect_ai` (UK AISI), `lm-evaluation-harness`; Hamel Husain — *Your AI Product Needs Evals*.
"""),
]

if __name__ == "__main__":
    write(cells, "11_evaluations.ipynb")
