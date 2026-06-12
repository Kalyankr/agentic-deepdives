"""Build NB05 — Alignment: RLHF & DPO."""

from _nbtools import code, md, write

cells = [
    md(r"""
# 05 · Alignment — RLHF & DPO

> Module: **03 · LLM Training** (alignment) — core to how ChatGPT and Claude are made.

A capable SFT model still isn't reliably **helpful, harmless, and honest**. **Preference
optimization** aligns it to human (or AI) preferences. We cover the classic **RLHF** pipeline
(reward model + PPO) and the simpler **DPO**, implementing a runnable DPO trainer in NumPy.

### Learning objectives
1. Train a **reward model** from preference pairs (Bradley–Terry).
2. Understand the **PPO** objective and the all-important **KL-to-reference** penalty.
3. Derive and **implement DPO**; see preferences learned without any RL loop.
4. Explain **Constitutional AI / RLAIF** and **reward hacking**.
"""),
    md(r"""
## 1. The RLHF pipeline (InstructGPT / ChatGPT)

```
1) SFT        : base -> instruction-following policy  (NB04)
2) Reward model: learn r(prompt, response) from human "A is better than B" pairs
3) RL (PPO)   : optimize the policy to maximize r, with a KL penalty to the SFT model
```

The **KL penalty** is the safety valve: it keeps the policy close to the trusted SFT model so
it doesn't drift into degenerate, **reward-hacking** text that scores high but is gibberish.
"""),
    md(r"""
## 2. Reward modeling — the Bradley–Terry loss

Given a preference pair (chosen $y_w$ ≻ rejected $y_l$) for a prompt $x$, the reward model
$r_\phi$ is trained so the chosen response scores higher. The probability that $y_w$ beats
$y_l$ is a logistic function of the reward gap:

$$P(y_w \succ y_l) = \sigma\!\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big),\qquad
\mathcal{L} = -\log\sigma\!\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)$$
"""),
    code(r"""
import numpy as np
sigmoid = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)

# Toy reward model r(features) = w . features. Each response is a feature vector.
# "Quality" is a hidden direction; chosen responses have higher true quality.
D = 6
true_quality = rng.standard_normal(D)
def make_pair():
    a, b = rng.standard_normal(D), rng.standard_normal(D)
    # label which is truly better under the hidden quality direction
    return (a, b) if true_quality @ a > true_quality @ b else (b, a)  # (chosen, rejected)

w = np.zeros(D)              # reward model parameters
lr = 0.1
for step in range(400):
    yw, yl = make_pair()
    margin = w @ yw - w @ yl
    # dL/dw = -(1 - sigmoid(margin)) * (yw - yl)
    grad = -(1 - sigmoid(margin)) * (yw - yl)
    w -= lr * grad

# evaluate: how often does the learned RM agree with the true preference?
agree = np.mean([ (w @ (p:=make_pair())[0]) > (w @ p[1]) for _ in range(2000)])
print(f"reward model agreement with true preference: {agree:.1%}")
print("cosine to true quality direction:", np.dot(w, true_quality)/(np.linalg.norm(w)*np.linalg.norm(true_quality)))
"""),
    md(r"""
## 3. PPO in one paragraph

PPO optimizes the policy $\pi_\theta$ to maximize expected reward while staying close to the
reference $\pi_\text{ref}$:

$$\max_\theta\ \mathbb{E}\big[\,r_\phi(x,y)\,\big] - \beta\,\mathrm{KL}\!\big(\pi_\theta(\cdot|x)\,\|\,\pi_\text{ref}(\cdot|x)\big)$$

It uses a clipped surrogate objective and advantage estimates (GAE) for stable updates.
PPO is powerful but **operationally heavy**: 4 models in memory (policy, ref, reward, value),
sensitive to hyperparameters, prone to reward hacking. That motivated DPO.
"""),
    md(r"""
## 4. DPO — Direct Preference Optimization

**Key result (Rafailov 2023):** the RLHF objective above has a closed-form optimal policy, and
substituting it turns the whole problem into a simple **classification loss on preference pairs** —
no reward model, no RL loop. The implicit reward is $\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$, and:

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\Big[\big(\log\tfrac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}\big) - \big(\log\tfrac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\big)\Big]\right)$$

In words: increase the policy's log-prob of the **chosen** response and decrease it for the
**rejected** one, *measured relative to the frozen reference* (that ratio is the KL control,
baked in). Let's implement and train it.
"""),
    code(r"""
# DPO demo. The "policy" assigns log-probs to responses via a linear scorer over features.
# Reference log-probs are fixed (a frozen copy). We optimize theta with the DPO loss.
D = 6
beta = 0.5
theta = np.zeros(D)                 # policy parameters (start == reference => zero margin)
ref   = rng.standard_normal(D) * 0.3  # frozen reference scorer

def logp(params, y):                # stand-in for log pi(y|x)
    return params @ y

pairs = [ (lambda p: (p[0], p[1]))( ( rng.standard_normal(D), rng.standard_normal(D) ) ) for _ in range(64) ]
# define ground-truth preference by a hidden quality vector
q = rng.standard_normal(D)
pairs = [ (a, b) if q@a > q@b else (b, a) for (a, b) in pairs ]   # (chosen, rejected)

def dpo_loss_and_grad(theta):
    loss, grad = 0.0, np.zeros(D)
    for yw, yl in pairs:
        # h = beta * [ (logp_theta(yw) - logp_ref(yw)) - (logp_theta(yl) - logp_ref(yl)) ]
        h = beta * ((logp(theta, yw) - logp(ref, yw)) - (logp(theta, yl) - logp(ref, yl)))
        loss += np.logaddexp(0, -h)          # softplus(-h) = -log sigmoid(h)
        dldh = sigmoid(h) - 1                 # d/dh of -log sigmoid(h)
        grad += dldh * beta * (yw - yl)
    n = len(pairs)
    return loss/n, grad/n

lr = 0.2
for step in range(300):
    loss, grad = dpo_loss_and_grad(theta)
    theta -= lr * grad
    if step % 50 == 0:
        acc = np.mean([ logp(theta, yw) > logp(theta, yl) for yw, yl in pairs ])
        print(f"step {step:3d}  dpo_loss {loss:.4f}  prefers-chosen {acc:.0%}")
acc = np.mean([ logp(theta, yw) > logp(theta, yl) for yw, yl in pairs ])
print(f"final: policy prefers the chosen response on {acc:.0%} of pairs")
"""),
    md(r"""
The DPO loss falls and the policy learns to prefer chosen over rejected responses — **no reward
model, no rollouts, no RL**. That simplicity (plus stability) is why DPO and its variants are so
widely used.

### The DPO family (what each fixes)
- **IPO** — adds a margin to avoid over-fitting to deterministic preferences.
- **KTO** — learns from *unpaired* good/bad labels (prospect-theory style), no pairs needed.
- **ORPO** — folds preference into SFT (no separate reference), one-stage.
- **SimPO** — reference-free, length-normalized.

**DPO vs PPO:** DPO is simpler/cheaper/stabler; well-tuned **PPO (online)** can still reach a
higher ceiling because it explores with fresh samples. Many labs do both.
"""),
    md(r"""
## 5. AI feedback & Constitutional AI (Anthropic)

Human labels are slow and expensive. **RLAIF** uses an LLM to generate preference labels.
**Constitutional AI** goes further: the model **critiques and revises** its own outputs against
a written set of principles ("a constitution"), generating preference data *from AI feedback*,
then trains on it (SFT + RL). This is central to Anthropic's harmlessness approach and reduces
reliance on human red-teamers for every example.

## 6. Reward hacking (Goodhart's law)
"When a measure becomes a target, it ceases to be a good measure." Optimizing hard against a
proxy reward yields high-scoring but bad outputs (verbose, sycophantic, format-gaming). Defenses:
KL penalty, reward-model ensembles, early stopping on a **gold** eval, and constrained optimization.

## Exercises
1. Run real **DPO** with `trl` on an SFT model + a preference set (UltraFeedback/HH-RLHF);
   measure win-rate vs SFT with an LLM judge (NB11).
2. Add a length penalty to the toy DPO and watch sycophancy/verbosity change.
3. Train the toy reward model with an **ensemble** and show reduced reward hacking.
4. Sketch a 3-principle "constitution" and a self-critique prompt loop.

## Resources
- *InstructGPT* (Ouyang 2022); *DPO* (Rafailov 2023); *PPO* (Schulman 2017).
- *Constitutional AI* (Bai 2022, Anthropic); *RLAIF* (2023).
- *KTO*, *IPO*, *ORPO*, *SimPO* papers; HF `trl` docs; *DeepSeek-R1* (RL for reasoning, 2025).
"""),
]

if __name__ == "__main__":
    write(cells, "05_alignment_rlhf_and_dpo.ipynb")
