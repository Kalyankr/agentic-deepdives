# Chapter 21 — Deep Reinforcement Learning · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-6-frontier/21-deep-rl.md)

---

## Interview answers

### Q: "What is an MDP?"

A **Markov Decision Process** is the formalism for sequential decision-making: a tuple $(\mathcal S,\mathcal A,P,R,\gamma)$ — states, actions, transition dynamics $P(s'\mid s,a)$, reward $R(s,a)$, and discount $\gamma$. The **Markov property** means the next state depends only on the current state and action, not the full history. The goal is a **policy** $\pi(a\mid s)$ maximizing **expected discounted return** $\mathbb E[\sum_t \gamma^t r_t]$. The discount $\gamma<1$ makes the infinite sum finite and encodes a preference for sooner rewards.

### Q: "Value-based vs policy-based RL?"

**Value-based** (Q-learning, DQN) learns the optimal action-value $Q^*(s,a)$ and acts greedily ($\arg\max_a Q$). It's sample-efficient and great for **discrete, small** action spaces, but the $\arg\max$ is awkward for continuous or enormous action spaces. **Policy-based** (REINFORCE, PPO) parameterizes and optimizes the policy *directly* via gradient ascent on expected return — it handles continuous/huge action spaces and stochastic policies naturally, at the cost of higher variance. **Actor-critic** combines them: a policy (actor) plus a value estimate (critic) as a variance-reducing baseline. LLM RLHF is policy-based because the "action space" is the entire vocabulary at every step.

### Q: "Derive the policy gradient."

We want $\nabla_\theta J(\theta)=\nabla_\theta\mathbb E_{\tau\sim\pi_\theta}[R(\tau)]$. Write the expectation as an integral over trajectories weighted by $p_\theta(\tau)$, then use the **log-derivative trick** $\nabla_\theta p_\theta = p_\theta\,\nabla_\theta\log p_\theta$:
$$\nabla_\theta J=\mathbb E_{\tau}\big[R(\tau)\,\nabla_\theta\log p_\theta(\tau)\big].$$
Since $\log p_\theta(\tau)=\sum_t\log\pi_\theta(a_t\mid s_t)+\text{(dynamics terms independent of }\theta)$, the dynamics drop out:
$$\nabla_\theta J=\mathbb E\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,R\Big].$$
Interpretation: **push up the log-probability of actions, scaled by the return they led to.** Subtracting a baseline $b(s)$ (ideally $V(s)$) leaves the gradient unbiased but cuts variance, turning $R$ into the **advantage** $A=R-V$.

### Q: "Why does PPO clip?"

Vanilla policy gradients allow a single update to move the policy arbitrarily far, which can collapse it — and we want to **reuse** each batch for several epochs, which makes that worse. PPO constrains the update with the **probability ratio** $r_t=\pi_\theta/\pi_{\theta_\text{old}}$ and a clipped objective $\min(r_tA_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)A_t)$. The `clip` caps how much the ratio can change in one step (an approximate **trust region**), and the `min` makes the bound **pessimistic** — it removes the incentive to push the ratio past the clip. The result is a robust first-order method that gets most of TRPO's stability without the second-order machinery.

### Q: "Map PPO to RLHF."

The **policy** is the LLM; a **state** is the prompt plus tokens generated so far; an **action** is the next token. The **reward** is the reward model's score on the finished response (a sparse, terminal reward) **minus a per-token KL penalty** to the frozen reference model — that KL is exactly PPO's trust region made explicit, preventing reward hacking and drift. The four models in an RLHF rig are: the **actor** (policy), the **reference** (KL anchor), the **reward model** (signal), and the **critic/value** network (PPO's baseline). So RLHF is PPO applied to a (near) one-step, terminal-reward bandit over text.

### Q: "What is GRPO and why use it?"

**Group Relative Policy Optimization** (DeepSeek) drops the **critic/value network**. For each prompt it samples a *group* of $G$ responses and uses the **group's mean reward as the baseline**: a response's advantage is its reward standardized against the group, $A_i=(r_i-\text{mean})/\text{std}$. This removes a whole network (≈ halving memory and eliminating critic-training instability), which matters enormously when doing RL on large reasoning models. It's a clean variance-reduction trick that made R1-style reasoning training practical.

### Q: "What is the credit-assignment problem?"

Figuring out **which earlier actions deserve credit/blame for a delayed reward** — the move that lost the game may have happened 30 turns before the loss signal. RL copes via **discounting** (nearer actions get more credit), **value bootstrapping** (the critic propagates value backward through the Bellman equation), and **GAE** (a bias-variance-tuned advantage estimate). In RLHF the reward is fully terminal (one score for the whole response), so credit assignment across tokens is especially hard — part of why these methods are finicky.

---

## Exercise solutions

### Exercise 1 — Tabular Q-learning on a gridworld

```python
import numpy as np

class GridWorld:                                   # 4x4, start top-left, goal bottom-right
    def __init__(self, n=4): self.n = n
    def reset(self): self.s = 0; return 0
    def step(self, a):
        r, c = divmod(self.s, self.n)
        if a == 0: r = max(0, r - 1)               # up
        if a == 1: r = min(self.n - 1, r + 1)      # down
        if a == 2: c = max(0, c - 1)               # left
        if a == 3: c = min(self.n - 1, c + 1)      # right
        self.s = r * self.n + c
        done = self.s == self.n * self.n - 1
        return self.s, (1.0 if done else -0.01), done   # small step penalty

env = GridWorld(); nS, nA = 16, 4
Q = np.zeros((nS, nA)); gamma, alpha, eps = 0.95, 0.1, 0.2
for ep in range(3000):
    s = env.reset(); done = False
    while not done:
        a = np.random.randint(nA) if np.random.rand() < eps else Q[s].argmax()
        s2, r, done = env.step(a)
        Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s, a])
        s = s2

policy = Q.argmax(1).reshape(4, 4)
print("greedy policy (0=up,1=down,2=left,3=right):\n", policy)
```

**Result:** the greedy policy converges to "move down/right toward the goal." Larger $\gamma$ makes the agent value the distant goal more (sharper path); too-small $\epsilon$ can leave parts of the grid unexplored (stale $Q$ values), while too-large $\epsilon$ slows convergence — the exploration/exploitation tradeoff in miniature.

### Exercise 2 — REINFORCE, then add a baseline

```python
import torch, torch.nn as nn, numpy as np
# Pseudocode-level: env = CartPole (gym). Policy outputs action logits.
policy = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
value  = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))   # baseline
opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), 3e-3)

def returns_to_go(rews, gamma=0.99):
    out, G = [], 0.0
    for r in reversed(rews): G = r + gamma * G; out.insert(0, G)
    return torch.tensor(out)

# per episode: collect (states, actions, logps, rewards)
def update(states, logps, rewards):
    R = returns_to_go(rewards)
    V = value(torch.stack(states)).squeeze(-1)
    adv = (R - V).detach()                              # baseline subtracted
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg_loss = -(torch.stack(logps) * adv).sum()        # policy gradient
    v_loss = ((R - V) ** 2).mean()                     # fit the critic
    loss = pg_loss + 0.5 * v_loss
    opt.zero_grad(); loss.backward(); opt.step()
```

**Result:** plain REINFORCE (no baseline) solves CartPole but with **noisy, slow** learning and high run-to-run variance. Adding the value baseline (using $A=R-V$) **markedly reduces gradient variance**, so it converges faster and more reliably — a concrete demonstration of why every modern method uses advantages, not raw returns.

### Exercise 3 — PPO clipped loss

```python
import torch
def ppo_update(policy, value, opt, states, actions, old_logp, returns, adv, eps=0.2, epochs=4):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(epochs):                              # REUSE the batch — PPO's win
        logits = policy(states)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ratio = (logp - old_logp).exp()
        clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
        pg_loss = -torch.min(ratio * adv, clipped).mean()
        v_loss = ((value(states).squeeze(-1) - returns) ** 2).mean()
        ent = dist.entropy().mean()                      # encourage exploration
        loss = pg_loss + 0.5 * v_loss - 0.01 * ent
        opt.zero_grad(); loss.backward(); opt.step()
```

**Result:** PPO trains CartPole/LunarLander **stably across seeds** while reusing each batch for several epochs (sample-efficient). Ablating the clip (e.g. $\epsilon\to\infty$, i.e. vanilla multi-epoch PG) causes occasional **policy collapse** — one oversized update destroys performance — which is precisely the failure the clip prevents.

### Exercise 4 — GAE bias/variance

```python
def gae(rewards, values, gamma=0.99, lam=0.95):
    adv, gae_t = [], 0.0
    values = values + [0.0]                              # bootstrap terminal
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]   # TD error
        gae_t = delta + gamma * lam * gae_t
        adv.insert(0, gae_t)
    return adv
```

**Result:** $\lambda=0$ gives the one-step TD advantage — **low variance, high bias** (relies heavily on the critic). $\lambda=1$ gives the full Monte-Carlo advantage — **unbiased, high variance**. $\lambda\approx0.95$ interpolates and is the empirical sweet spot used in PPO: most of the variance reduction, little bias. Measuring `np.var(adv)` across the three confirms variance rises sharply as $\lambda\to1$.

### Exercise 5 — Minimal RLHF loop (and reward hacking)

```python
# A tiny char/word LM as policy; reward = +1 if the word "great" appears, else 0;
# KL penalty keeps it near the reference (frozen copy of the initial policy).
def rlhf_reward(text, ref_logp, policy_logp, beta=0.1):
    task_reward = 1.0 if "great" in text else 0.0
    kl = (policy_logp - ref_logp).sum()                 # per-sequence KL estimate
    return task_reward - beta * kl                      # reward under a KL leash
```

**Result:** with the KL penalty, the model learns to use "great" *naturally* while staying fluent. **Remove the KL term** (`beta=0`) and it **reward-hacks** — degenerating into "great great great great…" because that maximizes the naive reward at the cost of coherence. This reproduces, in miniature, the central RLHF failure mode from Chapter 9 and shows *why* the KL leash (PPO's trust region) is non-negotiable.

---

[← Chapter 20 solutions](20-diffusion-multimodal-solutions.md) · [Solutions index](README.md) · [Next: Chapter 22 solutions →](22-interpretability-solutions.md)
