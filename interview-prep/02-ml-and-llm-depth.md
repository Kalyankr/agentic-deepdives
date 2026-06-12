# 02 · ML & LLM Depth

> The "knowledge" round. Expect rapid back-and-forth on transformers, training, alignment,
> inference, and distributed systems — often with **whiteboard math** (params, FLOPs, KV-cache) and
> "why" follow-ups. Answers below are interview-length: the claim, the reason, the number, the
> trade-off.

Sections: [Transformers](#transformers) · [Pretraining & Scaling](#pretraining--scaling) ·
[Fine-tuning](#fine-tuning--peft) · [Alignment](#alignment-rlhf--dpo) · [Inference](#inference) ·
[Distributed](#distributed-training--inference)

---

## Transformers

**Q: Why did attention replace RNNs/LSTMs?**
RNNs process tokens sequentially (no parallelism over time) and struggle with long-range
dependencies (vanishing gradients, a fixed-size hidden state bottleneck). Self-attention lets every
token directly attend to every other token in `O(1)` path length and is **fully parallel** across
positions during training — which is what made scaling to billions of params on GPUs feasible.

**Q: Explain Q, K, V.**
Each token emits a **query** ("what am I looking for"), a **key** ("what I offer"), and a **value**
("what I'll contribute if matched"). Attention weight `i→j = softmax(qᵢ·kⱼ/√d)`; the output for `i`
is the weighted sum of `vⱼ`. It's a differentiable, content-based soft dictionary lookup.

**Q: Why divide by √d_k?**
`qᵢ·kⱼ` is a sum of `d_k` products; if entries are ~unit variance, the dot product has variance
`≈ d_k`. Large magnitudes push softmax into saturation (near one-hot), where gradients vanish.
Dividing by `√d_k` keeps variance ~1 and gradients healthy.

**Q: Multi-head — why, and does it cost more?**
`h` heads each operate in a `d_model/h` subspace, so different heads can specialize (positional,
syntactic, coreference…). Total FLOPs ≈ one full-width head, because width is split across heads.
More heads ⇒ more relational "channels" per layer at ~constant cost.

**Q: GQA / MQA?**
The KV cache scales with the number of **KV heads**. **MQA** uses 1 shared K/V head (smallest cache,
some quality loss); **GQA** groups query heads to share `g` KV heads — a tunable middle ground
(LLaMA-2/3 use GQA). It can shrink the KV cache ~4–8×, directly increasing the batch size (throughput)
you can serve at a given context length.

**Q: Positional encoding — absolute vs RoPE vs ALiBi?**
Attention is permutation-invariant, so position must be injected. **Absolute** (learned/sinusoidal)
adds a position vector — simple but extrapolates poorly. **RoPE** *rotates* Q/K by a position-dependent
angle so the dot product depends on **relative** position; it's the modern default and extends to
longer context (with interpolation/scaling like YaRN). **ALiBi** adds a linear distance bias to scores
— cheap and good at length extrapolation.

**Q: What does FlashAttention change?**
Same math, different memory access. Vanilla attention materializes the `T×T` score matrix in HBM
(`O(T²)` memory, bandwidth-bound). FlashAttention **tiles** Q/K/V into SRAM and computes softmax
**online** (running max/sum), never writing the full matrix — `O(T)` memory and far fewer HBM
round-trips, so it's much faster and enables longer context. It's exact, not an approximation.

**Q: Derive the parameter count of a decoder block.**
Per layer: attention projections (Q,K,V,O) ≈ `4·d²`; FFN (`d→4d→d`) ≈ `8·d²`. So **~12·d² per
layer**. Total ≈ `12·n_layer·d²` + embeddings (`vocab·d`, tied with the output head). For a 7B model
this back-of-envelope lands in the right ballpark — they may ask you to plug in numbers.

**Q: FLOPs per token? Training cost?**
Forward ≈ **2·N** FLOPs/token (N = params; each weight is one multiply-add = 2 FLOPs). Backward ≈ 2×
forward, so training ≈ **6·N** FLOPs/token ⇒ total compute `C ≈ 6·N·D` for `D` tokens. This single
formula drives scaling-law and GPU-hour estimates.

**Q: KV-cache size?**
`bytes = 2 · n_layer · n_kv_heads · d_head · seq_len · batch · dtype_bytes` (the `2` is K and V).
It grows **linearly with batch × context length** and often dominates inference memory at long
context — the reason for GQA, paged attention, and KV quantization.

---

## Pretraining & Scaling

**Q: What is the pretraining objective and why does it produce general ability?**
Self-supervised **next-token prediction** (cross-entropy) over web-scale text. To predict the next
token well across diverse text, the model must implicitly learn grammar, facts, reasoning patterns,
and even some skills — "compression as intelligence." No labels needed, so it scales to trillions of
tokens.

**Q: Chinchilla / compute-optimal scaling — the key result?**
For a fixed compute budget `C ≈ 6ND`, there's an optimal split between model size `N` and tokens `D`.
Chinchilla found models were **undertrained**: optimal is roughly **~20 tokens per parameter** (train
smaller, on more data, than GPT-3 did). Caveat: that's *training*-optimal; for **inference**-heavy
deployment you often train a *smaller* model on *more* tokens (over the Chinchilla point) to cut
serving cost (LLaMA's choice).

**Q: Scaling laws in one sentence?**
Loss falls as a **power law** in model size, data, and compute (`L ≈ L∞ + (·)·N^-α + (·)·D^-β`) —
smooth and predictable over many orders of magnitude, which lets labs forecast a big run from small
ones.

**Q: "Emergent abilities" — what's the nuance?**
Some capabilities appear to jump at a scale threshold. The senior take: many "emergences" are partly
an artifact of **discontinuous metrics** (exact-match) — switch to a smooth metric and the curve is
often continuous. So: real phenomenon, but be careful attributing it purely to magic vs. measurement.

**Q: What matters most in pretraining data?**
**Quality and dedup** over raw quantity: aggressive dedup (prevents memorization and wasted compute),
quality filtering (classifier/heuristics), and the **mixture** (code, math, multilingual, books).
Data curation is one of the highest-leverage levers on final quality.

**Q: bf16 vs fp16 for training?**
**bf16** has the same exponent range as fp32 (8 bits) but fewer mantissa bits — it rarely overflows,
so it usually needs **no loss scaling** and is the default on modern accelerators. **fp16** has more
precision but a small range, so it needs dynamic **loss scaling** to avoid underflow/overflow.

**Q: Why warmup + cosine decay, and gradient clipping?**
Warmup avoids destabilizing the model with large early steps (random init + large LR ⇒ divergence);
cosine decay anneals to a small LR for fine convergence. **Grad clipping** (by global norm) caps the
occasional huge gradient (loss spike) that would otherwise blow up training.

---

## Fine-tuning & PEFT

**Q: SFT — what is it and what's the loss-masking subtlety?**
Supervised fine-tuning on (prompt, response) pairs with the same next-token loss, but you **mask the
prompt tokens** out of the loss and train only on the **assistant** tokens (using the model's chat
template to mark turns). Training on prompt tokens teaches the model to generate user turns — not what
you want.

**Q: LoRA — the math and why it works.**
Freeze `W` and learn a low-rank update: `W' = W + (α/r)·B·A`, with `A∈ℝ^{r×k}`, `B∈ℝ^{d×r}`, `r ≪
d`. Only `A,B` train (often <1% of params). It works because the **task adaptation** lives in a
low-rank subspace. Benefits: tiny checkpoints, swappable adapters, and you can **merge** `BA` into `W`
at inference for zero added latency.

**Q: QLoRA?**
Quantize the frozen base weights to **4-bit (NF4)** and train LoRA adapters in bf16 on top. The base
never updates, so 4-bit is fine; this fits 30–70B fine-tuning on a single GPU. Tricks: NF4 quant,
double quantization, paged optimizers.

**Q: Multi-LoRA serving?**
Because adapters are small and the base is shared, one server can hold many LoRA adapters and route
each request to its adapter (e.g. S-LoRA), serving many fine-tunes cheaply from one base model.

---

## Alignment (RLHF & DPO)

**Q: Walk through the RLHF pipeline.**
1) **SFT** a base model on demonstrations. 2) Collect **preference data** (humans rank responses).
3) Train a **reward model** (RM) to score responses. 4) **RL** (PPO) optimizes the policy to maximize
RM reward **minus a KL penalty** to the SFT reference, so it improves without drifting into gibberish
the RM over-rates.

**Q: How is the reward model trained?**
**Bradley–Terry**: `P(A≻B) = σ(r(A) − r(B))`. Train the RM to maximize the log-likelihood of human
preferences, i.e. push `r(chosen) > r(rejected)`. It's a regression-from-comparisons model: it learns
relative quality, not an absolute scale.

**Q: Why the KL penalty in PPO?**
Without it the policy **reward-hacks** the RM — exploiting quirks to get high reward while degrading
real quality. The `β·KL(π‖π_ref)` term keeps the policy close to the trusted SFT model; `β` trades
off optimization vs. staying on-distribution.

**Q: Derive the DPO intuition. Why does it remove the RM and PPO?**
RLHF's KL-constrained objective has a **closed-form optimal policy**:
`π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)`. Invert it to express the reward in terms of the policy:
`r(x,y) = β·log(π(y|x)/π_ref(y|x)) + const`. Substitute into the Bradley–Terry preference likelihood
and the partition function cancels, giving a simple **classification loss directly on the policy**:

`L_DPO = −log σ( β·[log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)] )`.

So DPO trains on preference pairs with a supervised-style loss — **no separate reward model, no RL
loop** — making it simpler and more stable. PPO can still reach higher ceilings with online data.

**Q: DPO vs PPO — when each?**
DPO: simpler, cheaper, stable, great when you have a fixed preference dataset. PPO/online RL: more
moving parts but can exploit an explicit RM and **on-policy** exploration, often a higher ceiling at
scale. Many teams start with DPO and move to online methods if needed.

**Q: Name DPO variants and what they fix.**
**IPO** (avoids DPO overfitting to deterministic preferences), **KTO** (learns from
**unpaired** good/bad labels, à la prospect theory), **ORPO** (folds preference into SFT — no
reference model), **SimPO** (reference-free, length-normalized). All target cost/stability/data-format
pain points of vanilla DPO.

**Q: Constitutional AI / RLAIF?**
Replace much of the human feedback with **AI feedback** guided by a written **constitution** (a set of
principles). The model critiques and revises its own outputs against the principles (RLAIF). It scales
oversight and makes the value targets explicit and auditable — central to Anthropic's approach.

**Q: Reward hacking and sycophancy — what and how to mitigate?**
**Reward hacking:** the policy games the proxy (RM) rather than the true goal. **Sycophancy:** telling
users what they want to hear because raters rewarded it. Mitigations: KL constraint, better/ensembled
RMs, adversarial and ongoing data collection, and **evals** that specifically probe these failure
modes before shipping.

---

## Inference

**Q: Prefill vs decode — why treat them differently?**
**Prefill** processes the whole prompt in parallel — **compute-bound** (big matmuls, high GPU
utilization), and sets **TTFT**. **Decode** generates one token at a time, each step reading the whole
model + KV cache for one token — **memory-bandwidth-bound**, low arithmetic intensity, and sets
**TPOT**. Optimizations differ per phase (and disaggregated serving even runs them on separate pools).

**Q: Why is decode memory-bandwidth bound? Implication?**
Per token you move ~all weights (and KV) from HBM but do little compute (batch ~1 of math), so you're
limited by **bandwidth**, not FLOPs. Implication: **batch more requests** to amortize the weight reads
(continuous batching), and **quantize** to move fewer bytes — both raise tokens/sec.

**Q: Continuous batching & PagedAttention?**
**Continuous batching** (vLLM) swaps finished sequences out and new ones in at each step instead of
waiting for the whole batch — huge throughput win under mixed-length traffic. **PagedAttention**
stores the KV cache in fixed-size **pages** (like OS virtual memory), eliminating fragmentation and
enabling sharing (e.g. a common prompt prefix), so you fit more concurrent sequences.

**Q: Quantization — what, and the trade-offs?**
Store/compute weights (and optionally activations / KV cache) in lower precision (INT8, INT4).
Weight-only INT8 is ~lossless and halves weight memory + bandwidth; INT4 (GPTQ/AWQ) is ~4× smaller
with small quality loss. Activation quant is harder (outliers). KV-cache quant cuts the dominant
long-context memory. Trade-off: memory/throughput vs. accuracy — always **eval** after.

**Q: Speculative decoding?**
A small **draft** model proposes `k` tokens cheaply; the big **target** model **verifies** them in one
parallel forward pass and accepts the longest correct prefix. It's **lossless** (output matches
sampling the target) and gives ~2–3× speedup when acceptance is high. Variants: Medusa heads,
EAGLE, n-gram drafts.

**Q: How do you estimate decode throughput (roofline)?**
Decode is bandwidth-bound, so `tokens/sec ≈ HBM_bandwidth / bytes_read_per_token`, where bytes/token
≈ model weight bytes (+ KV). Bigger batch amortizes the weight read across more tokens until you
become compute- or capacity-bound. This is the back-of-envelope they want for serving questions.

---

## Distributed Training & Inference

**Q: Estimate training memory per parameter.**
With mixed-precision Adam: bf16 weights (2) + grads (2) + fp32 master weights (4) + Adam `m`,`v`
(4+4) ≈ **~16–20 bytes/param** — *before* activations. So a 7B model needs ~112–140 GB just for
states ⇒ it **won't** fit on one 80 GB GPU ⇒ you must shard. Activations add more (cut via gradient
checkpointing).

**Q: Compare the parallelism strategies.**
- **Data Parallel (DP/DDP):** replicate model, split the batch, all-reduce grads. Simple; needs the
  model to fit per GPU.
- **Tensor Parallel (TP):** split individual matmuls across GPUs (intra-layer). Heavy communication
  ⇒ keep **within a node** (NVLink).
- **Pipeline Parallel (PP):** split layers into stages across nodes; micro-batches hide the "bubble."
- **Expert Parallel (EP):** route MoE experts to different GPUs.
- **3D parallelism:** combine TP (within node) × PP (across nodes) × DP (across replicas) for the
  largest models.

**Q: ZeRO / FSDP — what problem, what stages?**
DDP wastefully replicates optimizer states, grads, and params on every GPU. **ZeRO** shards them:
**stage 1** optimizer states, **stage 2** + grads, **stage 3** + params (≈ **FSDP**). Stage 3 makes
per-GPU memory ≈ `1/N` of model states (gather params just-in-time per layer), trading extra
communication for the ability to train huge models.

**Q: The collective operations?**
**All-reduce** (sum grads across all, everyone gets the result — DP), **all-gather** (collect shards —
FSDP forward), **reduce-scatter** (reduce then split — FSDP backward), **broadcast**, **all-to-all**
(MoE routing). Ring all-reduce moves ~`2×` the param bytes per GPU, **independent of N**, which is why
DP scales well.

**Q: How do you serve a model too big for one GPU?**
**Tensor-parallel within a node** (fast NVLink) to split the weights; **replicate** that group across
nodes behind a load balancer for QPS; add **pipeline parallel** across nodes only if a single node
still can't hold it (e.g. 400B+). Decode latency favors TP; throughput favors more replicas.

**Q: What dominates the loss-curve risk at scale (stability)?**
Loss spikes/divergence from bad batches, precision issues, or LR too high. Mitigations: bf16,
grad clipping, warmup, careful init, sometimes skipping/replaying bad batches and z-loss. A single
diverged run can waste enormous compute — stability engineering is a real senior skill.

---

### How to perform in this round
- **Whiteboard the formula, then plug numbers.** They want `6ND`, `2N`/token, KV-cache bytes, and
  ~16 B/param at your fingertips.
- **Always end on the trade-off** ("DPO is simpler but PPO's ceiling can be higher because…").
- **Connect to serving reality** — quality claims should bottom out in latency/cost/quality you'd
  measure.
