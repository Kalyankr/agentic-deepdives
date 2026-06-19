# Stage 2 — Answer Key (Pretraining at Scale)

> Full worked answers to [interview-questions.md](interview-questions.md). Memory/FLOPs arithmetic is done live with units — that's the skill being tested. Notation: $N$ params, $D$ training tokens, $C$ compute (FLOPs), $L$ layers, $d$ width.

---

## 🟢 Fundamentals

**1. Objective of a base LLM.**
Self-supervised **next-token prediction** (autoregressive language modeling): minimize cross-entropy of the true next token given all previous tokens, $\mathcal{L}=-\frac1T\sum_t\log p_\theta(x_t\mid x_{<t})$. No labels needed — the text is its own supervision.

**2. Perplexity, intuitively.**
$\text{PPL}=e^{\mathcal{L}}$ where $\mathcal{L}$ is per-token cross-entropy (nats). It's the model's **effective branching factor** — the average number of equally-likely choices it's deciding among at each token. PPL 20 ≈ "as uncertain as a uniform choice over 20 tokens."

**3. Three levers of pretraining outcome.**
**Parameters ($N$), data ($D$), and compute ($C$)** — tied by $C\approx 6ND$. (Data *quality* and the data *mixture* are the fourth, often-decisive lever.)

**4. Gradient accumulation and why.**
Run several micro-batches forward/backward, summing gradients, and step the optimizer only every $k$ micro-batches — simulating a $k\times$ larger batch without the memory of one. Used when the target (large) batch doesn't fit in VRAM; trades step latency for an effective large batch.

**5. Why warm up the learning rate?**
At init, parameters and Adam's moment estimates are uninformative; a full LR immediately can blow up activations/gradients (especially in deep pre-norm stacks). Linearly ramping LR from 0 over the first hundreds–thousands of steps lets statistics stabilize, preventing early divergence, before a cosine decay.

**6. What does gradient clipping protect against?**
Occasional **exploding gradients** — a bad batch or rare token can produce a huge gradient norm that takes a destructive step. Clipping the global grad norm to a threshold (e.g. 1.0) bounds the step size, preventing loss spikes/NaNs.

**7. Mixed-precision training.**
Do the heavy matmuls in low precision (bf16/fp16) for speed and memory, while keeping a **fp32 master copy** of weights and accumulating sensitive ops (softmax, loss, optimizer update) in fp32. Gives ~2× throughput and memory savings with full-precision stability.

**8. What deduplication buys you.**
Removes repeated/near-duplicate documents so the model doesn't waste capacity memorizing duplicates, **reduces train/eval contamination and memorization/regurgitation**, and improves effective data diversity per token — often a larger quality win than architecture tweaks.

**9. Data parallelism in one sentence.**
Replicate the full model on each device, give each a different shard of the batch, and **all-reduce** the gradients so every replica applies the same averaged update.

**10. Why data quality > architecture, often.**
Above a competent baseline architecture, returns to architecture tweaks are small, while garbage/duplicated/low-diversity data directly caps achievable loss and downstream quality. Curation (filtering, dedup, mixture) repeatedly beats clever layers in controlled studies — "data is the program."

---

## 🟡 Core (L4–L5)

**11. Chinchilla result; GPT-3's mistake.**
For a fixed compute budget, loss is minimized when **params and tokens scale together** — roughly **~20 tokens per parameter** at the optimum. GPT-3 (175B, ~300B tokens ≈ 1.7 tok/param) was badly **under-trained / over-parameterized**: a smaller model trained on more tokens (Chinchilla 70B on 1.4T) beat it at equal compute. Lesson: don't spend all compute on size.

**12. `C ≈ 6ND` relation.**
Compute = (FLOPs/token/param) × params × tokens. Forward is ~2 FLOPs/param/token, backward ~4, so ~6 total → $C\approx 6ND$. It says the three levers aren't free: at fixed $C$, increasing $N$ forces decreasing $D$ and vice-versa, which is exactly the tradeoff scaling laws optimize.

**13. Why bf16 over fp16.**
**bf16 has the same 8-bit exponent as fp32**, so its dynamic range is huge — gradients/activations rarely overflow/underflow, and you usually need *no loss scaling*. fp16 has only 5 exponent bits (max ≈ 65504), so it overflows easily and needs dynamic loss scaling. bf16 trades mantissa precision (fewer significand bits) for range, which training tolerates well — hence the default on A100/H100.

**14. Four GPU memory components in training.**
(1) **Parameters**, (2) **gradients**, (3) **optimizer state** (Adam's fp32 master + $m$ + $v$), and (4) **activations** (stored for backward, scales with batch×seq×layers). The first three are fixed by $N$; activations are the knob you control via batch size and gradient checkpointing.

**15. Data vs tensor vs pipeline parallelism (comms pattern).**
- **Data parallel:** replicate model, split batch; comms = **all-reduce of gradients** once per step (large, but overlappable).
- **Tensor parallel:** split *individual matmuls* across devices; comms = **all-reduce of activations inside every layer** (frequent, latency-sensitive) → keep within a node over NVLink.
- **Pipeline parallel:** split *layers* into stages across devices; comms = **point-to-point activations** between adjacent stages, plus a "bubble" of idle time; micro-batching hides the bubble.

**16. Gradient checkpointing trade.**
It **discards most activations in the forward pass and recomputes them during backward**, trading ~33% extra compute for a large (often $\sqrt L$) reduction in activation memory. Worth it when you're activation-memory bound (long sequences, big batch, large models) and have compute headroom.

**17. Why AdamW state is memory-hungry.**
AdamW stores, per parameter, a first moment $m$ and second moment $v$, *plus* a fp32 master weight in mixed precision — that's $4{+}4{+}4=12$ bytes/param of optimizer state, **6× the 2 bytes/param of the bf16 weights themselves**. This is what dominates training memory and motivates ZeRO/FSDP sharding.

**18. Data mixture as a tuned hyperparameter.**
Pretraining data is a weighted blend of sources (web, code, books, math, multilingual). The **ratios** change capabilities: more code → better reasoning/coding, more multilingual → broader languages but possible English regression. Because it directly trades off skills, the mixture is tuned (sometimes via small proxy models / DoReMi) like any hyperparameter.

**19. Benchmark contamination at pretraining.**
If eval-set text leaks into pretraining data, the model has *memorized* the answers, so benchmark scores overstate real ability. It matters because it invalidates evaluation and comparisons; you must **decontaminate** (n-gram/substring match against eval sets) before training and report overlap.

**20. Kaplan vs Chinchilla.**
Kaplan (2020) concluded you should spend most extra compute on **bigger models** (under-weighting data). Chinchilla (2022) fixed methodology (proper LR schedule per budget) and found $N$ and $D$ should grow **equally** (~20 tok/param). Practically: Kaplan → giant under-trained models (GPT-3); Chinchilla → smaller, longer-trained, compute-optimal models.

---

## 🔴 Senior / Staff deep dives

**21. Fixed compute $C$ → choose $N$ and $D$.**
Start from $C\approx 6ND$. Chinchilla-optimal sets $D\approx 20N$, so $C\approx 6N(20N)=120N^2 \Rightarrow N\approx\sqrt{C/120}$, then $D=20N$. **But** add the inference caveat: if you'll serve the model a lot, deliberately pick a *smaller* $N$ and *larger* $D$ (over-train) so per-query serving cost drops — you pay more training once to save inference forever (Llama philosophy). Final answer balances training-optimal vs total-cost-of-ownership-optimal.

**22. VRAM to full-fine-tune a 7B in bf16 + AdamW.**
Per parameter: weights (bf16) 2B, gradients (bf16) 2B, optimizer = fp32 master 4B + $m$ 4B + $v$ 4B = 12B → **16 bytes/param** of model/optimizer state.
- Params: $2\times 7\text{B}=14$ GB
- Grads: $2\times 7\text{B}=14$ GB
- Optimizer: $12\times 7\text{B}=84$ GB
- Subtotal ≈ **112 GB** *before activations.*
Activations add tens of GB depending on batch/seq. → won't fit on one 80GB GPU; needs **ZeRO/FSDP sharding, or PEFT/LoRA** (which slashes optimizer state to the adapter only).

**23. ZeRO stages 1/2/3.**
All keep data-parallel semantics but shard state across DP ranks, gathering on demand:
- **Stage 1:** shard **optimizer state** (the 12B/param) → biggest, cheapest win.
- **Stage 2:** also shard **gradients**.
- **Stage 3 (≈ FSDP):** also shard **parameters** — each rank holds $1/P$ of everything and all-gathers a layer's params just-in-time for compute, then frees them.
Tradeoff: more sharding → less memory per GPU but more communication (extra all-gathers). Stage 3 lets memory scale with #GPUs, enabling models that don't fit on one device.

**24. Parallelism for 70B on N×8×A100 nodes.**
Compose 3D parallelism:
- **Tensor parallel within a node** (8 GPUs over NVLink) — TP's frequent activation all-reduces need the fast intra-node link.
- **Pipeline parallel across nodes** — split layers into stages; point-to-point activations tolerate slower inter-node links; use micro-batching to shrink the **pipeline bubble** ($\propto (\text{stages}-1)/\text{microbatches}$).
- **Data parallel / ZeRO across the remaining replicas** for throughput.
Watch **activation memory** (use checkpointing), balance stage compute, and place TP groups to never cross node boundaries.

**25. Intermittent loss spikes on a long run — diagnose & recover.**
Likely causes: a **bad data shard** (corrupt/degenerate batch), **LR too high / warmup too short**, **fp16 overflow** (move to bf16), **missing or too-loose grad clipping**, or optimizer-state corruption. Diagnose with per-step grad-norm and loss logging to localize the offending step/shard. Recover by **clipping**, **skipping the bad batch**, **rolling back to the last good checkpoint** and resuming past the shard, and lowering LR if spikes persist. Keep frequent checkpoints so rollback is cheap.

**26. Why train past Chinchilla-optimal.**
**Inference economics.** Chinchilla minimizes *training* loss per FLOP, ignoring that you serve the model billions of times. A smaller model over-trained on far more tokens reaches the same quality while being **cheaper and faster to serve** — lower latency, less memory, smaller KV cache. At scale, serving cost dwarfs training cost, so you over-train (Llama-2/3 trained 7B on ~2T+ tokens, far past ~140B "optimal").

**27. Pretraining data pipeline from raw crawl — stages.**
1. **Text extraction** from HTML/WARC (strip boilerplate).
2. **Language ID** and routing.
3. **Quality filtering:** heuristics (length, symbol ratios) + a trained quality classifier; remove spam/SEO.
4. **Deduplication:** exact + near-dup via **MinHash/LSH** at document and sometimes line level.
5. **Decontamination:** remove overlaps with eval benchmarks.
6. **PII / safety handling:** scrub or down-weight sensitive content.
7. **Mixture weighting:** blend sources to target ratios.
8. **Tokenization** and **sharding** into streamable files.

**28. Detect & prevent eval contamination.**
*Prevent:* before training, run n-gram/substring (e.g. 13-gram) or MinHash matching of every benchmark example against the corpus and remove hits; keep a held-out, recently-created eval set. *Detect after the fact:* measure performance gap on contaminated vs clean subsets, compare memorized-completion rates, or use canary strings. Always **report overlap statistics** so scores are trustworthy.

---

## 🧮 Math & derivations

**29. Four memory terms for AdamW mixed precision (bytes/param).**
- Weights (bf16): **2**
- Gradients (bf16): **2**
- Optimizer: fp32 master **4** + first moment $m$ (fp32) **4** + second moment $v$ (fp32) **4** = **12**
Total = **16 bytes/param** of fixed state (plus activations separately). So a model of $N$ params needs $\approx 16N$ bytes before activations.

**30. Why factor ~6 in `C ≈ 6ND`.**
Per token, each parameter does a multiply-add in the forward matmul = **2 FLOPs**. Backprop computes gradients w.r.t. both inputs and weights — about **2× the forward work = 4 FLOPs**. Total ≈ **6 FLOPs/param/token**. Multiply by $N$ params and $D$ tokens → $C\approx 6ND$.

**31. Largest Chinchilla-optimal model on 256×A100-80GB for 1 week (order of magnitude).**
A100 bf16 peak ≈ 312 TFLOP/s; realistic MFU ~40% → ~125 TFLOP/s each. 256 GPUs → ~3.2×10¹⁶ FLOP/s. One week ≈ 6.05×10⁵ s → $C\approx 1.9\times10^{22}$ FLOPs. Chinchilla: $C\approx120N^2\Rightarrow N\approx\sqrt{C/120}\approx\sqrt{1.6\times10^{20}}\approx 1.3\times10^{10}$ → **~10–13B parameters**, trained on $D\approx20N\approx$ **~250B tokens**. (Order-of-magnitude: ~10B.)

**32. Perplexity ↔ cross-entropy; PPL=20 operationally.**
$\text{PPL}=e^{\mathcal{L}}$ with $\mathcal{L}$ in nats (or $2^{\mathcal{L}}$ in bits). **PPL = 20** means the model is, on average, as uncertain as choosing uniformly among **20 tokens** at each step — i.e. average per-token loss $\mathcal{L}=\ln 20\approx 3.0$ nats. Lower PPL = more confident/accurate next-token prediction.

---

## 💻 Coding / implementation

**33. VRAM estimator.**
```python
def vram_estimate(N, dtype_bytes=2, optimizer="adamw",
                  batch=1, seq=2048, layers=32, d_model=4096, hidden_mult=4):
    opt_bytes = {"adamw": 12, "sgd": 4, "adamw_8bit": 6}[optimizer]
    params = N * dtype_bytes
    grads  = N * dtype_bytes
    optim  = N * opt_bytes
    # rough activation memory (per-layer, dominated by MLP + attention buffers)
    act_per_tok = layers * d_model * (hidden_mult + 8) * dtype_bytes
    activations = batch * seq * act_per_tok
    total = params + grads + optim + activations
    return {k: round(v/1e9, 2) for k, v in dict(
        params=params, grads=grads, optimizer=optim,
        activations=activations, total=total).items()}  # GB

# vram_estimate(7e9) -> params 14, grads 14, optimizer 84, ...
```

**34. Correct gradient accumulation.**
```python
optimizer.zero_grad()
for i, (x, y) in enumerate(loader):
    loss = model(x, y) / ACCUM_STEPS          # scale so summed grads = mean
    loss.backward()                            # grads accumulate
    if (i + 1) % ACCUM_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```
The `/ ACCUM_STEPS` is essential — otherwise you sum (not average) and your effective LR is `ACCUM_STEPS×` too large.

**35. bf16 autocast + clip + cosine-with-warmup.**
```python
import math, torch
def lr_lambda(step, warmup, total):
    if step < warmup: return step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))      # cosine decay to 0

sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, 2000, 100_000))
for x, y in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad()
```
(bf16 needs no `GradScaler`; fp16 would.)

**36. Streaming sharded data loader.**
```python
import glob, torch
from torch.utils.data import IterableDataset

class ShardedTokens(IterableDataset):
    def __init__(self, pattern, block=2048):
        self.files, self.block = sorted(glob.glob(pattern)), block
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        files = self.files[info.id::info.num_workers] if info else self.files
        for f in files:                                  # one shard at a time
            toks = torch.load(f, mmap=True)              # memory-mapped, not in RAM
            for i in range(0, len(toks) - self.block, self.block):
                chunk = toks[i:i + self.block + 1]
                yield chunk[:-1], chunk[1:]
```
Key ideas: `IterableDataset` + memory-mapping + per-worker shard sharding so the full corpus never sits in RAM and workers don't duplicate data.

**37. Wrap in FSDP / ZeRO; flags.**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # = ZeRO-3 (params+grads+optim)
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                   reduce_dtype=torch.bfloat16),
    auto_wrap_policy=transformer_wrap_policy,         # shard per transformer block
    limit_all_gathers=True,                           # cap memory spikes
    use_orig_params=True,                             # cleaner optimizer/param mapping
)
```
- `FULL_SHARD` = ZeRO-3 (max memory savings, more comms); `SHARD_GRAD_OP` = ZeRO-2.
- `auto_wrap_policy` decides the granularity of all-gather (per block keeps peak memory low).
- `mixed_precision` controls compute/reduce dtypes; `limit_all_gathers` prevents prefetch OOM.

---

## 🏗️ System design / applied

**38. Full training infra for a 13B model.**
- **Data:** offline pipeline (extract→filter→dedup→decontaminate→mix→tokenize→shard) producing streamable token shards on fast storage; a held-out eval set.
- **Compute/parallelism:** multi-node A100/H100; FSDP (ZeRO-3) + TP within node if needed; bf16; gradient checkpointing.
- **Checkpointing:** sharded, asynchronous checkpoints every N minutes to durable storage; save RNG + optimizer + dataloader position for exact resume.
- **Fault tolerance:** automatic restart from last checkpoint on node failure; elastic/redundant workers; idempotent steps.
- **Monitoring:** loss, grad-norm, LR, throughput (tokens/s), **MFU**, GPU util/mem, per-shard loss to catch bad data; alerting on spikes/NaNs.
- **Cost:** track $/token and MFU; use spot where resumable; right-size batch for utilization.

**39. Resilient run under spot preemptions.**
**Checkpoint frequently** (every few minutes), asynchronously, including dataloader position + optimizer + RNG so resume is exact and loses minimal work. Use an **elastic launcher** (torchrun/elastic) that re-forms the process group when nodes vanish/return; keep a **redundant pool** so the job continues at reduced size; make steps **idempotent** and writes atomic (write-temp-then-rename). Optionally a small on-demand "anchor" set of nodes plus spot for the rest.

**40. Cut training cost 40% with minimal quality loss — order of levers.**
1. **Raise MFU first** (free quality): fix data-loader stalls, use FlashAttention/fused kernels, larger batch, better parallelism placement — often recovers 20–30%.
2. **bf16 + gradient checkpointing** tuned so you're compute- not memory-bound.
3. **Better data** (dedup/quality filter) → reach target loss in fewer tokens.
4. **Chinchilla-size correctly** / mild under-train if quality budget allows.
5. **Spot instances** for the resumable portion.
Pull efficiency levers (no quality cost) before reducing tokens/params (quality cost).

---

## 🐞 Debugging

**41. Tokens/sec far below roofline.**
Check, roughly in order: **data-loader stalls** (GPU starved — profile, add workers/prefetch/mmap), **batch too small** (low arithmetic intensity), **no fused/Flash kernels**, **comms-bound parallelism** (TP over slow inter-node links, or DP all-reduce not overlapped), **gradient-checkpointing recompute overhead**, and generally **low MFU** from poor kernel/precision choices. Use a profiler and the MFU number to localize compute vs IO vs comms.

**42. Model memorizes/regurgitates training data.**
Cause: **insufficient deduplication** (and too many epochs over duplicated data) → verbatim memorization. Fix: stronger exact+near **dedup** (MinHash), **fewer epochs**/more unique tokens, quality filtering, and for sensitive data **differential privacy** or removal. Measure with regurgitation/canary tests.

**43. Multi-node 3× slower than expected.**
Almost always **interconnect-bound**: tensor parallelism placed *across* nodes (TP needs NVLink, not Ethernet/slow IB), DP all-reduce not overlapping with compute, wrong process/rank-to-GPU placement, or no topology-aware grouping. Fix by keeping TP intra-node, PP/DP inter-node, enabling comms/compute overlap, and checking NCCL is using the fast fabric.

**44. Loss decreases then plateaus far above expected.**
Suspects: **LR too low / decayed too fast** (cosine bottomed out), **model too small / under-capacity** for the data, **data quality or mixture** capping achievable loss, **too-aggressive regularization** (dropout/weight decay), a **tokenization/preprocessing bug** inflating loss, or **optimizer misconfig**. Compare against a scaling-law-predicted loss for your $N,D$; if far above, suspect data or LR schedule first.

---

## What strong answers share
Live **memory/FLOPs arithmetic** with correct units; treating **data curation as the primary lever**; knowing **when to follow scaling laws and when to break them** for inference economics; and mapping a model size to a concrete **parallelism + infrastructure** plan.

---
Back to [questions](interview-questions.md) · [Stage README](README.md) · [Index](../README.md)
