# Stage 2 — Pretraining at Scale

> **Objective:** Understand how frontier base models are actually trained, and be able to compute resource/memory budgets by hand. You won't train a 70B model, but you must reason like someone who could.

[← Stage 1](../stage-1-transformer-internals/README.md) · [Index](../README.md) · Next: [Stage 3 — Alignment](../stage-3-adaptation-alignment/README.md)

📝 **Interview prep:** [interview-questions.md](interview-questions.md) · ✅ [answer key](answers.md)

---

## Why this stage matters

Pretraining is where capability comes from. Fine-tuning only *steers* a model; it can't add knowledge that pretraining didn't create. Understanding scaling laws, data, and distributed training is what separates people who *use* models from people who *reason about* them.

---

## Mental model

> A base model is a giant next-token predictor trained on a huge, carefully-curated text corpus, optimized with AdamW under tight memory/compute constraints, sharded across many GPUs.

Three levers determine the outcome: **data (quality + quantity)**, **compute (FLOPs)**, and **model size (params)**. Scaling laws tell you how to balance them.

---

## Concept-by-concept deep dive

### 2.1 The pretraining objective
- **Causal language modeling:** predict token *t+1* from tokens *≤ t*. Loss = mean cross-entropy over all positions.
- **Perplexity** = `exp(loss)`. Intuition: the model's "effective branching factor" — how many tokens it's effectively choosing among. Lower is better.
- Every position contributes a training signal → pretraining is extremely sample-efficient *per sequence*.

### 2.2 Data: the real bottleneck
Modern quality comes mostly from data work, not architecture tweaks.
- **Sources:** web crawl (CommonCrawl), code, books, Wikipedia, papers, curated datasets.
- **Cleaning:** language ID, boilerplate/HTML removal, quality classifiers, perplexity filtering.
- **Deduplication:** exact + near-dup (MinHash/SimHash). Dedup materially improves quality and reduces memorization.
- **Decontamination:** remove eval-benchmark text from training data (or your evals lie).
- **Data mixtures:** the ratio of code/web/books/math matters a lot. Mixture is a tuned hyperparameter.
- **Curriculum:** some training orders/phasing (e.g., high-quality data late) help.

> Takeaway: "garbage in, garbage out" is *the* law of pretraining. Data > architecture.

### 2.3 Scaling laws
- **Kaplan et al. (2020):** loss falls predictably as a power law in params, data, and compute.
- **Chinchilla (2022):** for a fixed compute budget, there's a **compute-optimal** balance — roughly **~20 tokens per parameter**. Earlier models (GPT-3) were *undertrained* (too big, too little data).
- **Practical implication:** given FLOPs budget `C ≈ 6·N·D` (N=params, D=tokens), pick N and D to minimize loss. Inference cost later may push you to "over-train" a smaller model (cheaper to serve).

> Be able to use `C ≈ 6ND` to back out tokens or params from a compute budget.

### 2.4 Optimization
- **AdamW:** Adam (per-parameter adaptive LR via 1st + 2nd moment estimates) with **decoupled weight decay**. The 2 moment buffers are why optimizer state is large.
- **LR schedule:** linear/cosine **warmup** (stabilizes early training) → **cosine decay** to a small final LR.
- **Gradient clipping:** clip global grad norm (e.g., 1.0) to survive loss spikes.
- **Batch size:** large global batch (via gradient accumulation) → smoother gradients; relates to critical batch size.

### 2.5 Numerical precision
- **bf16 vs fp16:** bf16 has the same exponent range as fp32 (just fewer mantissa bits), so it rarely overflows → **no loss scaling needed**, more stable. fp16 has more precision but tiny range → needs dynamic loss scaling. Modern training defaults to **bf16**.
- **Mixed precision:** compute in low precision, keep a master copy / accumulations in fp32 where needed.

### 2.6 The memory budget (know this cold)
For a model with **N** parameters trained in mixed precision with AdamW, rough VRAM:
- **Parameters:** 2 bytes (bf16) × N
- **Gradients:** 2 bytes × N
- **Optimizer state (AdamW):** fp32 master weights (4N) + momentum (4N) + variance (4N) = **12 bytes × N**
- **Activations:** depends on batch × seq × layers (often the dominant, variable term)

> Example: a **7B** model → params 14GB + grads 14GB + optimizer 84GB ≈ **112GB** *before activations*. This is why you need sharding or PEFT. Be able to reproduce this arithmetic.

### 2.7 Distributed training
- **Data parallelism (DP):** replicate model, split the batch, all-reduce gradients. Simple; replicates memory.
- **Tensor parallelism (TP):** split individual matmuls across GPUs (within a layer). High communication; intra-node.
- **Pipeline parallelism (PP):** split layers into stages across GPUs; micro-batches to keep stages busy (mind the "bubble").
- **ZeRO (DeepSpeed):** shard optimizer state (stage 1), + gradients (stage 2), + parameters (stage 3) across DP ranks → removes redundancy of plain DP.
- **FSDP (PyTorch):** Fully Sharded Data Parallel — shards params/grads/optimizer, gathers on demand. The modern default.
- **Gradient checkpointing:** recompute activations in backward instead of storing them → trades compute for memory.
- **3D parallelism:** combine DP × TP × PP for the largest models.

---

## Ordered learning path

1. Read the **Chinchilla** paper — internalize compute-optimal scaling.
2. Read the **LLaMA / LLaMA-2** papers — data + practical architecture choices.
3. Read **GPT-3** for the few-shot scaling story (and as a contrast to Chinchilla).
4. Read the **ZeRO** paper + skim **PyTorch FSDP** docs.
5. Do the memory-math and scaling-law labs.

---

## 🛠️ Hands-on labs

- [ ] **Lab A — Train with real tricks:** take your Stage-1 GPT, add bf16, gradient accumulation, gradient clipping, cosine LR + warmup, and gradient checkpointing. Confirm stable training.
- [ ] **Lab B — Mini scaling law:** train 3–4 model sizes; plot final loss vs params (log-log). See the power law emerge.
- [ ] **Lab C — VRAM estimator:** write a script: input (N, dtype, optimizer, batch, seq, layers) → output predicted memory broken into params/grads/optimizer/activations. Validate against a real run.
- [ ] **Lab D — FSDP:** wrap a model in FSDP (even on 1–2 GPUs / Colab) and observe the memory difference vs plain DP.
- [ ] **Lab E (analysis):** given a compute budget in FLOPs, compute the Chinchilla-optimal (N, D).

---

## ⚠️ Common pitfalls & gotchas

- Forgetting optimizer state is **the** memory hog (12 bytes/param with AdamW), not the weights.
- Believing "bigger model = better" — Chinchilla says a smaller, better-trained model can win.
- No warmup → early loss explosion.
- Training on contaminated data → inflated benchmark scores, dishonest results.
- Ignoring dedup → memorization and wasted compute.
- Confusing tensor vs pipeline vs data parallelism (know the comms pattern of each).

---

## 🔥 Mastery checks (answer without notes)

- [ ] Compute total VRAM to full-fine-tune a 7B model in bf16 + AdamW. Break down every term.
- [ ] Given compute budget `C`, derive compute-optimal params & tokens using `C ≈ 6ND` + Chinchilla.
- [ ] Explain ZeRO stages 1/2/3 — what each sustains and what it shards.
- [ ] Why is bf16 preferred over fp16 for training stability?
- [ ] Explain the tradeoff gradient checkpointing makes and when it's worth it.
- [ ] Contrast data / tensor / pipeline parallelism by communication pattern and where each is used.
- [ ] Why might you deliberately "over-train" a smaller-than-Chinchilla-optimal model in practice?
- [ ] What does deduplication buy you, concretely?

---

## ✅ Stage 2 checklist

- [ ] Read Chinchilla, LLaMA, ZeRO
- [ ] Labs A–C complete (D/E if you have GPU access)
- [ ] Can do the 7B memory math from memory
- [ ] All mastery checks passable
- [ ] Notes written in your own words

**When complete → proceed to [Stage 3](../stage-3-adaptation-alignment/README.md).**
