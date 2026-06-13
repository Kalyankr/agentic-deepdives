# Chapter 9 — Post-training & Alignment · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-3-llm-stack/09-alignment.md)

---

## Interview answers

### Q: "Walk through the full post-training pipeline."

A base model only predicts text; post-training makes it a helpful, honest, harmless **assistant**:

1. **SFT** (supervised fine-tuning) — imitate high-quality demonstrations to learn the instruction-following *format* and basic behavior.
2. **Preference data + reward signal** — collect human (or AI) comparisons of model outputs; train a **reward model** (or use them directly).
3. **Preference optimization** — **RLHF-PPO** (optimize reward under a KL leash) or **DPO** (optimize the preferences directly, no separate RL loop).
4. **(Optional) Reasoning RL** — RL on *verifiable* rewards to elicit long chain-of-thought.
5. **Eval & red-team** — measure helpfulness/safety, probe for jailbreaks, iterate.

The throughline: SFT teaches *format*, preference optimization teaches *taste/values*, reasoning RL teaches *thinking*.

### Q: "Why a KL penalty in RLHF?"

The reward model is an imperfect **proxy** for human preference. If you let the policy maximize it freely, it finds adversarial outputs that score high but are bad (**reward hacking**) and drifts off the fluent, on-distribution text the base model produces. The **KL penalty against the frozen reference model** is a leash: it keeps the policy close to the original distribution, so it can improve reward only in ways that don't wander far from coherent language. Tune the KL coefficient — too tight and nothing changes, too loose and it hacks.

### Q: "DPO vs RLHF — why did DPO win adoption?"

RLHF-PPO is powerful but **operationally heavy**: you maintain *four* models (policy, reference, reward, critic/value) and an unstable RL loop that's finicky to tune. **DPO** uses a mathematical insight — the RLHF objective has a closed-form optimal policy — to **rewrite preference learning as a simple classification-style loss directly on (chosen, rejected) pairs**. No separate reward model, no RL loop, just **two** models (policy + frozen reference). It's stable, supervised-style, and reproducible, so it became the default for most open post-training. The KL leash is still there, implicitly, via the reference model in the loss.

### Q: "What is reward hacking?"

When the policy maximizes the **reward-model proxy** in ways that don't reflect true human preference — exploiting quirks of the RM rather than genuinely improving. Classic symptoms: verbose answers (RM likes length), excessive hedging or formatting, confident-sounding nonsense, sycophantic agreement. It's the central RLHF failure mode and the reason for the KL leash, reward-model retraining, and careful eval — the proxy is never the true objective (Goodhart's law).

### Q: "Explain Constitutional AI."

Anthropic's method to align models with **less human exposure to harmful content** and **explicit, auditable values**. Two phases: (1) **Supervised** — the model **critiques and revises** its own responses against a written **constitution** (a set of principles), generating improved training data. (2) **RLAIF** — instead of human preference labels, an **AI judges** which response better follows the constitution, and you do RL/DPO on those AI preferences. The win: values are **written down and inspectable** (not buried in opaque reward-model weights), and you scale feedback without paying humans to read toxic content. It's core to how Claude is trained.

### Q: "How do reasoning models train?"

With **RL on verifiable rewards**. For tasks where correctness is *checkable* — math with a known answer, code that must pass unit tests — you reward the model for getting it right regardless of *how* it got there. That pressure makes the model generate long **chain-of-thought** (explore, backtrack, check) because thinking longer raises its success rate. This is "scaling inference-time compute": the model learns to spend more tokens thinking on hard problems. Because the reward is objective (not a learned RM), it's much **harder to hack** than preference-based reward.

### Q: "What is sycophancy and where does it come from?"

Sycophancy is the model **telling users what they want to hear** — agreeing with a stated (even wrong) opinion, caving when challenged, flattering. It comes from **preference data**: human raters tend to prefer agreeable, validating answers, so reward models learn that "agree with the user" scores well, and RLHF amplifies it. It's a direct example of reward hacking against human (not just RM) bias, and a known open problem — mitigations include diversifying raters, training on "correct but disagreeable" examples, and calibration-aware eval.

---

## Exercise solutions

### Exercise 1 — Bradley-Terry reward-model loss

The reward model learns to score chosen > rejected. Bradley-Terry models $P(\text{chosen} \succ \text{rejected}) = \sigma(r_\text{chosen} - r_\text{rejected})$, so the loss is $-\log\sigma(r_\text{chosen}-r_\text{rejected})$.

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
# Toy "responses" as feature vectors; label: first is preferred (chosen).
d = 8
chosen   = torch.randn(200, d) + 1.0      # chosen cluster shifted +1
rejected = torch.randn(200, d) - 1.0      # rejected cluster shifted -1

reward_model = nn.Linear(d, 1)
opt = torch.optim.Adam(reward_model.parameters(), lr=0.05)

def bt_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()

for step in range(300):
    rc, rr = reward_model(chosen), reward_model(rejected)
    loss = bt_loss(rc, rr)
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    acc = (reward_model(chosen) > reward_model(rejected)).float().mean()
print("final loss:", round(loss.item(), 4))
print("ranks chosen > rejected:", round(acc.item()*100, 1), "%")   # ~100%
```

**Result:** the loss falls and the RM ranks **chosen above rejected ~100%** of the time. Note it learns from **comparisons**, never absolute scores — exactly why preference data ("A is better than B") is easier to collect reliably than asking humans to assign numeric quality.

### Exercise 2 — DPO loss (shifts probability toward chosen)

DPO optimizes the policy directly on preference pairs, using the frozen reference as the KL anchor:

$$\mathcal{L}_\text{DPO} = -\log\sigma\!\Big(\beta\big[(\log\pi_\theta(y_w)-\log\pi_\text{ref}(y_w)) - (\log\pi_\theta(y_l)-\log\pi_\text{ref}(y_l))\big]\Big).$$

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
V = 10                                   # tiny "vocab" of possible responses
policy   = nn.Parameter(torch.zeros(V))  # logits over responses
ref      = torch.zeros(V)                # frozen reference (uniform)
chosen, rejected, beta = 3, 7, 0.1       # we prefer response 3 over response 7
opt = torch.optim.Adam([policy], lr=0.05)

def logp(logits, idx): return F.log_softmax(logits, -1)[idx]

for step in range(300):
    pi_w, pi_l = logp(policy, chosen), logp(policy, rejected)
    ref_w, ref_l = logp(ref, chosen), logp(ref, rejected)
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    loss = -F.logsigmoid(logits)
    opt.zero_grad(); loss.backward(); opt.step()

probs = F.softmax(policy, -1).detach()
print("P(chosen) :", round(probs[chosen].item(), 3))    # rises above 1/V
print("P(rejected):", round(probs[rejected].item(), 3)) # falls below 1/V
```

**Result:** the policy's probability mass shifts **toward the chosen** response and **away from the rejected** one — no reward model, no RL loop, just a stable supervised-style gradient. The reference term is the implicit KL leash that keeps it from collapsing entirely onto the chosen token.

### Exercise 3 — Mask prompt tokens in the SFT loss

Only the **response** should contribute to the loss; the prompt is context, not a learning target. Set prompt label positions to `-100` (PyTorch's `ignore_index`).

```python
import torch, torch.nn.functional as F

V = 50
prompt = [1, 2, 3, 4]          # "User: ..." (don't train on predicting this)
response = [10, 11, 12]        # "Assistant: ..." (DO train on this)
tokens = torch.tensor(prompt + response)

# labels: shift by one; mask the prompt region with -100
labels = tokens.clone()
labels[:len(prompt)] = -100    # ignore prompt positions
inp, tgt = tokens[:-1], labels[1:]

logits = torch.randn(len(inp), V, requires_grad=True)
loss = F.cross_entropy(logits, tgt, ignore_index=-100)

# Verify only response positions count: how many targets are active?
active = (tgt != -100).sum().item()
print("active (response) target positions:", active)        # 3
print("total positions:", len(tgt))                          # 6
```

**Result:** only the response tokens (3 active positions) contribute to the loss; the prompt positions are ignored. Without this mask the model wastes capacity learning to *generate prompts* and the loss is dominated by context it should only condition on. Every SFT trainer does this masking.

### Exercise 4 — A 5-principle constitution + self-critique→revise loop

```python
constitution = [
    "1. Be helpful and directly address the user's request.",
    "2. Be honest; do not fabricate facts or sources.",
    "3. Avoid harmful, dangerous, or illegal instructions.",
    "4. Be respectful and avoid demeaning language.",
    "5. Be concise; do not pad with filler.",
]

def critique_prompt(principles, response):
    return (f"Principles:\n" + "\n".join(principles) +
            f"\n\nResponse to review:\n{response}\n\n"
            "List any principle this response violates, then explain why.")

def revise_prompt(principles, response, critique):
    return (f"Principles:\n" + "\n".join(principles) +
            f"\n\nOriginal response:\n{response}\nCritique:\n{critique}\n\n"
            "Rewrite the response to satisfy all principles.")

# With any instruct model `llm(prompt) -> str`, the loop is:
def constitutional_revise(llm, principles, response):
    critique = llm(critique_prompt(principles, response))
    revised  = llm(revise_prompt(principles, response, critique))
    return critique, revised

# Example (illustrative I/O):
original = "Sure, here's how to pick a lock on someone else's door: ..."
# critique -> "Violates principle 3 (could enable illegal entry)."
# revised  -> "I can't help with breaking into property that isn't yours.
#              If you're locked out of your own home, contact a licensed locksmith..."
print(critique_prompt(constitution, original))
```

**Result:** the loop takes a response, asks the model to **critique** it against the written principles, then **revise** it to comply. The `(prompt, revised)` pairs become SFT data — this is the supervised half of Constitutional AI. The key property is that the **values are explicit text** you can read, edit, and audit, rather than implicit in reward-model weights. (Plug in any instruct model for `llm`.)

### Exercise 5 — A *verifiable* reward (does the code pass tests?)

```python
import io, contextlib

def unit_tests(fn):
    return [(fn(2, 3) == 5), (fn(0, 0) == 0), (fn(-1, 1) == 0)]

def verifiable_reward(generated_code, fn_name="add"):
    """Reward = fraction of unit tests the generated code passes. Objective, ungameable."""
    scope = {}
    try:
        exec(generated_code, scope)               # NOTE: sandbox this in production!
        fn = scope[fn_name]
        results = unit_tests(fn)
        return sum(results) / len(results)
    except Exception:
        return 0.0                                # doesn't even run -> zero reward

good = "def add(a, b): return a + b"
bad  = "def add(a, b): return a - b"
broken = "def add(a, b): return"                  # returns None
print("good  :", verifiable_reward(good))         # 1.0
print("bad   :", verifiable_reward(bad))          # ~0.33 (passes the 0,0 case)
print("broken:", verifiable_reward(broken))       # 0.0

# RL loop sketch (REINFORCE-style):
#   for prompt in coding_tasks:
#       code   = policy.sample(prompt)            # generate a solution
#       reward = verifiable_reward(code)          # run the tests
#       loss   = -reward * policy.log_prob(code)  # push up the prob of high-reward samples
#       loss.backward(); opt.step()
```

**Result:** the reward is computed by **executing** the generated code against unit tests — correct code gets 1.0, wrong code gets a low/zero score, code that doesn't run gets 0. Because correctness is *checked, not predicted by a learned model*, this reward is **objective and very hard to hack** — which is exactly why verifiable rewards power modern reasoning-model training. **Security note:** always run untrusted generated code in a sandbox (container/seccomp/resource limits), never raw `exec` in production.

---

[← Chapter 8 solutions](08-pretraining-solutions.md) · [Solutions index](README.md) · [Next: Chapter 10 solutions →](10-inference-optimization-solutions.md)
