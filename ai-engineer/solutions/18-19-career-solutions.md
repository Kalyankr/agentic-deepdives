# Chapters 18–19 — Career · Solutions & Model Answers

[← Solutions index](README.md) · [Read Chapter 18](../part-5-career/18-specialization.md) · [Read Chapter 19](../part-5-career/19-getting-hired.md)

The career chapters have no coding exercises. What they *do* have is the most important "exam" in the book: the **6-question self-check** in §19.6 — the bar for being interview-ready at a serious lab. Below are full model answers to all six, plus guidance on the Chapter 18 "which track?" decision.

---

## Chapter 18 — "Which specialization should I choose?"

This is a real interview question ("where do you want to focus?") and a real career decision. The framework: be **T-shaped** — broad literacy across the whole stack (Parts I–IV), with **one deep vertical** you can go to the frontier on. Pick the track that matches what you *can't stop doing*.

| Track | You own | Proof-of-work that gets you hired |
|---|---|---|
| **Research / Training** | Pretraining, scaling laws, alignment, novel architectures | **Reproduce a paper** (DPO, a scaling-law run); a training-dynamics write-up; a small novel result |
| **Inference / Systems** | Kernels, quantization, serving, distributed | A **benchmarked Triton/CUDA kernel**; a vLLM-style serving optimization; an MFU improvement |
| **Applied / Agents** | RAG, agents, evals, product | A **real RAG/agent app** with a rigorous eval harness and a security story (prompt-injection defense) |

**How to answer in an interview:** name the track, give the **evidence** (a specific artifact you built in that vertical), and connect it to the team's work. "I gravitate to inference/systems — I wrote a fused SiLU Triton kernel and a KV-cache implementation that 3×'d my toy GPT's generation, and I find myself reading FlashAttention-style papers for fun. That's why your serving team is the role I want." Same structure as the Chapter 1 motivation answer: **track + artifact + why this team**.

**How to choose if unsure:** look at which chapters you did the *optional* extra work in, which exercises you extended beyond the ask, and which problems you think about unprompted. Your behavior already knows.

---

## Chapter 19 — The 6-question self-check (full model answers)

> If you can confidently do all six from memory, you're ready to interview at a serious AI lab.

### 1. Implement multi-head attention from scratch, from memory, and explain every line.

Be able to write this without references and narrate each step. The complete worked implementation is in the [Chapter 6 solutions](06-transformer-solutions.md#exercise-3--multi-head-attention-output-dim--d_model); the **narration** that proves understanding:

- **Project** $X$ to $Q,K,V$ — three learned linear maps; these are "what I'm looking for / what I contain / what I'll hand over."
- **Split into heads** — reshape to `(n_heads, T, d_head)` so each head attends in its own subspace.
- **Scores** $QK^\top/\sqrt{d_k}$ — dot-product similarity, scaled so softmax doesn't saturate ([why](02-mathematics-solutions.md)).
- **Causal mask** — set future positions to $-\infty$ *before* softmax so a decoder can't peek.
- **Softmax** — turn scores into weights that sum to 1 (a distribution over keys).
- **Aggregate** $WV$ — each token's output is a weighted average of value vectors.
- **Concat + output projection** — recombine the heads and mix them with $W_o$.

The tell of mastery: you can explain *why* each piece exists (scaling, masking, multiple heads), not just type it.

### 2. Derive backprop for a 2-layer MLP on a whiteboard.

For $z_1 = xW_1,\; a_1 = \sigma(z_1),\; z_2 = a_1 W_2,\; L = \tfrac12\|z_2 - y\|^2$, apply the chain rule backward:

$$\frac{\partial L}{\partial z_2} = z_2 - y, \quad \frac{\partial L}{\partial W_2} = a_1^\top \frac{\partial L}{\partial z_2}, \quad \frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial z_2} W_2^\top,$$
$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \odot \sigma'(z_1), \quad \frac{\partial L}{\partial W_1} = x^\top \frac{\partial L}{\partial z_1}.$$

Narrate it as: "gradient at the output is prediction−target; push it back through $W_2$ (transpose to route it), multiply by the activation's local derivative, push through $W_1$." The runnable verification (manual backward == autograd) is in the [Chapter 5 solutions](05-neural-networks-solutions.md#exercise-5--manual-backward--autograd-gradients). Know it cold — it's the single most common ML whiteboard question.

### 3. Explain DPO vs RLHF and *why* DPO is more stable.

**RLHF-PPO**: train a reward model on preferences, then RL-optimize the policy to maximize reward under a KL penalty against a frozen reference. Powerful but **four models** (policy, reference, reward, value) and a finicky, unstable RL loop prone to **reward hacking**.

**DPO**: a mathematical result shows the KL-constrained RLHF objective has a **closed-form optimal policy**, which lets you rewrite preference learning as a **direct classification-style loss** on (chosen, rejected) pairs — no reward model, no RL loop, just **two models**.

**Why more stable:** no separately-trained reward proxy to hack, no high-variance RL/sampling loop to tune — it's a smooth supervised gradient, with the KL leash preserved implicitly via the reference model. Full derivation context + a runnable DPO loss in the [Chapter 9 solutions](09-alignment-solutions.md#exercise-2--dpo-loss-shifts-probability-toward-chosen).

### 4. Explain why FlashAttention is faster without changing the math.

It's **IO-aware**, not approximate. Standard attention writes the full $n\times n$ score matrix to **HBM** and reads it back multiple times (for softmax, then the $V$ multiply) — and HBM traffic, not FLOPs, is the bottleneck. FlashAttention **tiles** Q/K/V into fast **shared memory** and uses **online softmax** (running max & sum) to compute attention **block-by-block without ever materializing the $n\times n$ matrix in HBM**. Same exact output, but memory-linear and far fewer slow HBM round-trips → much faster, and it unlocks long context. The online-softmax core is implemented and verified in the [Chapter 15 solutions](15-gpu-programming-solutions.md#exercise-5--online-softmax-the-flashattention-trick-in-numpy). The one-liner: *"it respects the memory hierarchy."*

### 5. Design an end-to-end system to serve an LLM at scale, naming the key tradeoffs.

Walk the request path and name a tradeoff at each hop:

- **Engine**: vLLM/TGI for **continuous batching** + **paged KV** (throughput↔latency tradeoff via batch size).
- **Phases**: prefill (compute-bound, sets **TTFT**) vs decode (memory-bound, sets **TPOT**) — optimize separately.
- **Memory**: **GQA** + **KV quantization** + paging to fit more concurrent requests (quality↔memory).
- **Model**: **quantization** (int8/int4) and **routing** small↔big (cost↔quality).
- **Latency tricks**: **speculative decoding** (free, identical output), **prefix caching**, streaming.
- **Infra**: Docker + K8s, GPU autoscaling on queue depth, warm pools (cost↔cold-start).
- **Build vs buy**: API vs self-host **crossover** (Chapter 17 exercise 6).
- **Ops**: eval-gated CI/CD, monitor TTFT/TPOT/quality/**drift**/cost, trace RAG/agents.

The senior signal is naming the **tradeoff** at each decision, not just listing components. Backed by the [Chapter 10](10-inference-optimization-solutions.md) and [Chapter 17 solutions](17-serving-mlops-solutions.md).

### 6. Walk through a project you built, defending every design decision under pressure.

There's no "answer key" — this is *your* artifact. Prepare with this structure:

- **Problem & why it matters** — what real need it serves.
- **Key decisions, each with the alternative you rejected and why** — "I used RAG not fine-tuning because the knowledge changes weekly"; "rank-16 LoRA because rank-4 underfit the style in my eval"; "GQA-8 to keep the KV cache under X GB at my context length."
- **How you measured success** — your **eval harness** (this is the highest-signal part; see [Chapter 13 solutions](13-evaluation-solutions.md)).
- **What broke and what you learned** — failures, debugging, the fix.
- **What you'd do next** — shows you think past the demo.

Interviewers will push ("why not X?") — for every decision, know the tradeoff and the evidence. This question, plus the Chapter 1 motivation question, is why the whole book insists on **building real things and writing them up**: the project *is* the proof of work.

---

## The meta-answer

These six map to chapters **6, 5, 9, 15, 10/17, and 19**. If any feels shaky, that chapter (and its solutions file) is where to drill. Being able to do all six **from memory, under pressure, defending each choice** is the operational definition of "cracked" — and the point of everything you built working through this book.

---

[← Chapter 17 solutions](17-serving-mlops-solutions.md) · [Solutions index](README.md) · [Next: Chapter 20 solutions →](20-diffusion-multimodal-solutions.md)
