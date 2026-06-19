# LLM Training & Fine-Tuning Platform — Answer Key

> Full worked answers to [questions.md](questions.md). The bar: **lead with the two estimates** ($C\approx6ND$ and ~16 bytes/param), **separate the control and execution planes**, treat **parallelism as a toolkit** (match comms to the network, pick the cheapest split that fits), and make **reliability** (checkpoint + auto-resume + elasticity) first-class. Reference design: [README.md](README.md).
>
> Notation: $N$ = parameters, $D$ = training tokens, $C$ = FLOPs, MFU = model-FLOPs utilization, DP/TP/PP/EP = data/tensor/pipeline/expert parallelism, $P$ = data-parallel degree.

---

## 🟢 Fundamentals

**1. What does the platform do, and why is it separate from serving?**
It turns a **declarative job spec** into a trained, evaluated, registered model — handling distributed execution across thousands of GPUs, data pipelines, checkpointing, fault tolerance, and experiment tracking. It's separate from [serving](../llm-inference/README.md) because the objectives are opposite: training is **throughput- and durability-oriented** (run for days/weeks, minimize total cost to a target loss, survive hardware failure), while serving is **latency-oriented** (respond in milliseconds). Same GPUs, mirror-image system.

**2. The training-compute rule of thumb.**
$$C \approx 6ND$$
$C$ = total FLOPs, $N$ = parameters, $D$ = training tokens. The **6** ≈ 2 FLOPs per multiply-accumulate × 3 effective passes (one forward + roughly two for the backward, which computes grads w.r.t. both inputs and weights). It's the single most useful number in training design: it sets compute cost, GPU-count, and wall-clock. (Inference by contrast is ~$2N$ per token — no backward.)

**3. Why a large model can't train on one GPU.**
**Memory.** AdamW mixed-precision keeps **~16 bytes/param** (fp16 weight 2 + fp16 grad 2 + fp32 master 4 + Adam $m$ 4 + $v$ 4). A 70B model = ~1.12 TB of model states alone, vs 80 GB on an H100 — ~14 GPUs *before a single activation*. **Activations** (scaling with batch × seq × layers) often dominate on top. So you must **shard** states (ZeRO/FSDP), split the model (TP/PP), and **recompute** activations. Compute time is the second reason, but memory is what makes one GPU impossible.

**4. What each parallelism splits.**
- **Data parallel (DP):** splits the **batch**; each group holds a full model replica and **all-reduces gradients** each step.
- **Tensor parallel (TP):** splits **individual matmuls within a layer** across GPUs; communicates **inside** every layer (chatty) → keep it **intra-node** on NVLink.
- **Pipeline parallel (PP):** splits the **layers** into stages on different devices; passes **activations** across stage boundaries → cheap, works inter-node, but creates **bubbles**.

**5. ZeRO / FSDP.**
ZeRO (DeepSpeed) / FSDP (PyTorch) **shard the 16 bytes/param across the data-parallel ranks** instead of replicating. **Stage 1** shards optimizer states, **stage 2** adds gradients, **stage 3 / FSDP** adds parameters — gathering each layer's weights **just-in-time** for its forward/backward, then discarding them. Per-GPU model-state memory drops ~$16/P$ bytes/param, trading extra all-gather communication (mostly hidden by overlap) for the ability to fit huge models on commodity 80 GB GPUs.

**6. Activation checkpointing.**
Instead of storing every layer's activations for the backward pass, store only a few **and recompute the rest** during backward. It cuts activation memory roughly from $O(L)$ to $O(\sqrt L)$ for the cost of **~one extra forward pass (~33% more compute)**. It's the standard lever to fit longer sequences or bigger micro-batches — trading compute (cheap) for memory (scarce).

**7. Why checkpointing is essential.**
Large jobs run for **days to weeks on thousands of GPUs**, where hardware *will* fail and jobs *will* be preempted. A checkpoint (weights + optimizer states + data-loader position + RNG + step) lets the job **resume from the last save instead of from scratch** — turning a catastrophic loss of days of compute into a few minutes of replay. It's also what makes **preemption safe** (so the scheduler can keep utilization high) and enables divergence rollback.

**8. What is MFU?**
**Model-FLOPs Utilization** = useful model FLOPs achieved ÷ hardware peak FLOPs. It's the headline metric because it **directly determines cost and time-to-train**: at fixed compute $C=6ND$, doubling MFU halves both. Healthy large-model runs sit ~**40–55%**; below ~30% something is wrong (exposed communication, data-loader stall, pipeline bubbles, fragmented placement). Every MFU point at scale is worth a lot of money.

**9. SFT vs LoRA vs RLHF vs DPO.**
- **SFT** — supervised next-token on curated (prompt, response) pairs; full-model training.
- **LoRA/PEFT** — freeze the base, train small **low-rank adapters**; tiny memory/cost, many tenants share one base.
- **RLHF (PPO)** — optimize against a **reward model** with RL rollouts; **heaviest** (up to 4 models in memory + generation).
- **DPO** — preference optimization **without** a separate reward model or rollouts; ~2 models, far simpler — the pragmatic default for alignment.

**10. Gang scheduling.**
Synchronous training needs **all** of a job's GPUs running together (every step does a collective across all ranks). **Gang scheduling** allocates them **all-or-nothing** — never a partial set. Without it, two big jobs could each grab half the GPUs they need and **deadlock**, both waiting forever. Gang scheduling + topology-aware placement is what makes large synchronous jobs actually run.

---

## 🟡 Core design

**11. Lifecycle of a training job.**
`Submit spec → validate → QUEUED` → scheduler **gang-allocates** topology-aligned GPUs → orchestrator launches workers in the parallelism layout → **RUNNING** loop (load batch → forward → backward → all-reduce → optimizer step), **checkpointing** every *N* steps and **evaluating** every *M* → on failure, **auto-resume** from last checkpoint → at the end, run final eval, and if it **passes the gate**, promote the artifact (weights + tokenizer + config + eval card) to the **registry** for the inference service to consume.

**12. Data pipeline design.**
Offline: **clean/normalize → quality-filter (heuristics + classifier) → dedup (exact + near-dup MinHash/LSH) → decontaminate (remove eval leakage) → tokenize → pack to `seq_len` → write sharded, shuffled token store.** Online: a **streaming loader** samples shards by **mix weights** (e.g. 60/30/10 web/code/domain), prefetches, and overlaps I/O with compute, with a **seeded resumable sampler** for deterministic order. Dedup and decontamination are the highest-leverage steps; tokenization is precomputed so the GPU loop only reads token IDs.

**13. Combining 3D parallelism.**
Map each split to the right network level: **TP intra-node** on NVLink (chatty, inside each layer), **PP across a few nodes** (cheap activation passes at stage boundaries), **DP across the rest** (one all-reduce/step). Example for 2048 GPUs: TP=8 × PP=4 × DP=64. Then add **ZeRO/FSDP** to shard states within the DP dimension and **activation checkpointing** to fit memory. Pick the **cheapest split that fits**: DP first → ZeRO → activation checkpointing → TP → PP → offload. Overlap all collectives with compute.

**14. Checkpointing at scale.**
**Sharded** (each rank writes its own shard in parallel — no single-writer bottleneck) + **asynchronous** (snapshot tensors to a CPU buffer, flush to the parallel FS in the background so the GPU step resumes in seconds). Save weights + optimizer states + data-loader position + RNG + step. **Cadence:** often enough that work-lost-per-failure ≪ checkpoint cost (more GPUs/bigger model → checkpoint more often). **Retention:** keep last *k* + milestone checkpoints for rollback/eval. Write to a high-bandwidth parallel filesystem.

**15. Surviving a hardware failure.**
A **health monitor** (heartbeats, NCCL timeouts, ECC errors) detects the dead/slow rank → orchestrator **halts the job, swaps in a hot spare, and relaunches from the last checkpoint** automatically. With **elastic training** it can instead **resize the world** (e.g. 2048→1920) and continue. Net effect: a node failure costs minutes of replay, not days. A small **spare pool** makes replacement near-instant.

**16. Multi-tenant scheduler.**
**Gang scheduling** + **topology-aware placement** (keep a job on one InfiniBand island so collectives stay fast). **Quotas + hierarchical fair-share** per team; **priority tiers** (production > experiments); **preemption** (safe because jobs checkpoint, so preempted work resumes later); **backfill** (slot small jobs into gaps while a big gang waits). Match jobs to GPU types. The goal: high utilization without starving anyone — and preemption-via-checkpointing is what unlocks it.

**17. RLHF (PPO) resource needs vs supervised.**
PPO holds up to **four models** at once — **actor** (policy being trained), **critic** (value head), **reward model**, and a **frozen reference** (for the KL penalty) — *and* runs **generation rollouts** (the policy samples completions every step). So it needs far more memory and couples **training with inference machinery** (a rollout/generation engine), plus it's less stable (RL + reward hacking + KL tuning). Supervised SFT is just forward/backward on fixed targets. This is why **DPO** (no reward model, no rollouts) is often preferred.

**18. Keeping GPUs fed.**
Precompute tokenization offline so the loop reads only token IDs; **shard** the token store across many readers; **prefetch** several batches ahead and **overlap I/O with compute**; **pack** sequences to avoid padding waste; cache hot shards on local NVMe. Monitor data-loader wait time per step — if GPUs idle on input, MFU craters. The loader must sustain aggregate throughput matching thousands of GPUs consuming millions of tokens/step.

**19. Reproducibility.**
Pin and record **everything**: code SHA, container image, library/CUDA versions, dataset **versions + mix**, full resolved config, seeds, parallelism layout, and a **deterministic resumable data order**. Caveat: bitwise determinism across **different GPU counts** is hard (collective reduction order changes), so reproduce on the **same topology** and treat small run-to-run variance as expected. Track the run in the experiment tracker with full lineage (base → data → job → artifact → evals).

**20. Evaluation integration.**
**In-loop:** held-out loss/perplexity every *N* steps (primary convergence signal, on the critical path but cheap). **Periodic benchmarks:** spin up **eval workers** on a milestone checkpoint (often served by the inference service) to run MMLU/HumanEval/domain suites **off the critical path**. **Gate:** promote to the registry only if it **beats the incumbent** with no regressions; attach an eval card. For aligned models, add preference/safety/red-team evals. Decontamination (§12) is what makes these numbers trustworthy.

---

## 🔴 Senior / Staff deep dives

**21. Loss spike / divergence.**
First **don't panic-kill** — roll back to the **last good checkpoint** and resume with a **lower LR and/or skip the offending batch**. Diagnose causes: LR too high or warmup too short, a bad/corrupt data shard, fp16 overflow (switch to **bf16**), missing/loose **gradient clipping**, or a bad mixing change. Defenses going forward: grad clipping, bf16, warmup + cosine decay, grad-norm monitoring with auto-halt, and milestone checkpoints to roll back to. Persistent spikes at a given step often point to **data**, not optimization.

**22. Raising MFU from 25%.**
Profile to find the stall, then attack in order: **data loader** (are GPUs idle waiting for input? → prefetch/shard/pack); **exposed communication** (overlap all-reduce/all-gather with backward, keep **TP intra-node**, avoid cross-island placement); **pipeline bubbles** (more micro-batches, interleaved 1F1B); **kernels** (FlashAttention, fused ops); **precision** (bf16/fp8); **batch size** (bigger amortizes comms); **right-size parallelism** (drop unnecessary TP/PP boundaries). 25% almost always means exposed comms, a loader stall, or fragmented placement.

**23. Stragglers.**
Synchronous training moves at the **slowest rank**, so one degraded GPU/NIC stalls every collective. Detect via **per-rank step-time outliers** (and ECC/thermal/NVLink telemetry). Fix: **fence and blocklist** the bad host, replace it from the spare pool (or **elastically resize** without it), and relaunch from checkpoint. Long-term: topology-aware placement, health-based node draining, and avoiding known-bad hardware in future placements.

**24. 70B vs 7B parallelism plan.**
**7B** (~112 GB states) fits with simple **DP + ZeRO** (maybe ZeRO-2/3) — no need for TP/PP; add activation checkpointing for long sequences. **70B** (~1.1 TB states) **cannot** live on one GPU, so you need **model splitting**: **TP=8 intra-node** + **PP across nodes** + **DP + ZeRO** on top, plus activation checkpointing. The principle: **add parallelism complexity only as memory forces you to** — TP/PP add communication and lower MFU, so the small model stays simple and the big model layers them in.

**25. Elastic / fault-tolerant training.**
Make the **world size dynamic**: a coordinator tracks live workers; on node loss it **reconfigures the process group, rescales the DP degree, and resumes from the last checkpoint** without a full re-queue; on capacity return it can scale back up. Requires **sharded checkpoints** that can be re-loaded under a different layout, a **rendezvous/coordination** service (e.g. etcd-style), **hot spares**, and care that effective batch/LR are rescaled when DP changes. Net: the job rides through node churn and even spot reclaims.

**26. Checkpoint stalls without checkpointing less.**
Make the write **off the critical path**: (1) **snapshot to a pinned CPU buffer** quickly, then **flush to storage asynchronously** in the background; (2) **shard** the write so every rank writes its slice in parallel (no single writer); (3) write to a **high-bandwidth parallel FS / local NVMe then drain**; (4) optionally **overlap** the flush with the next steps. The GPU step pauses only for the fast device→host snapshot (seconds), not the slow storage flush.

**27. Scheduling big pretrains + many small fine-tunes.**
**Reserve** capacity (or a partition) for the big gang jobs so they can actually be placed; **backfill** the small short fine-tunes/sweeps into the gaps; use **priority + preemption** (small low-priority jobs yield to production pretrains, resuming later because they checkpoint); apply **hierarchical fair-share** so teams get their quota; route small jobs to **cheaper/older GPUs**. The combination keeps utilization high while neither workload starves.

**28. Spot/preemptible GPUs safely.**
Training is **checkpointed and restartable**, which makes it a natural fit for spot. Run **fault-tolerant/elastic** jobs there with **frequent async checkpoints**, **auto-resume**, and **graceful-shutdown handlers** that flush a checkpoint on the reclaim warning. Keep **production/critical pretrains on reserved** capacity (or mirror progress), and use spot for fine-tunes, sweeps, and lower-priority runs. Net: big cost savings with reclaim handled by the same fault-tolerance machinery you already built.

---

## 🧮 Math & estimation

**29. GPU-days to pretrain 70B on 15T tokens.**
$$C = 6ND = 6 \times 70\text{e}9 \times 15\text{e}12 \approx 6.3\times10^{24}\ \text{FLOPs}$$
At ~440 TFLOP/s effective per H100 (~45% MFU of ~990 TFLOP/s bf16):
$$\text{GPU-seconds} = \frac{6.3\text{e}24}{4.4\text{e}14} \approx 1.43\times10^{10}\ \Rightarrow\ \approx \mathbf{166\text{K GPU-days}}.$$
So ~**21 days on 8,192 GPUs** (or ~166 days on 1,024). The GPU-days are fixed by compute; more GPUs only buy wall-clock — and only if MFU holds.

**30. Memory for AdamW mixed-precision training of 7B.**
Model states at **16 bytes/param**: $7\text{e}9 \times 16 = \mathbf{112\ GB}$ (fp16 weight 2 + grad 2 + fp32 master 4 + $m$ 4 + $v$ 4). That alone exceeds one 80 GB GPU — before activations — so you need **ZeRO/FSDP sharding** (and/or activation checkpointing). With ZeRO-3 across $P$=8 GPUs, model states drop to ~14 GB/GPU, leaving room for activations.

**31. Checkpoint size for 70B.**
A **full resumable** checkpoint stores weights **and optimizer states** (≈16 bytes/param): $70\text{e}9 \times 16 \approx \mathbf{1.1\ TB}$. (A weights-only inference checkpoint is just ~2 bytes/param ≈ 140 GB in bf16; the optimizer states are what make the resumable one ~8× bigger.) This size — written every ~30–60 min — is why checkpointing must be **sharded + async** to a high-bandwidth parallel filesystem.

**32. Chinchilla compute-optimal split.**
For a fixed budget $C=6ND$, loss is minimized by scaling **params and tokens together** — roughly **$D \approx 20N$ tokens per parameter** (compute-optimal). So given $C$, solve for $N$ and $D$ jointly rather than making $N$ huge and undertraining it. Practical caveat: if you'll **serve** the model a lot, you deliberately **over-train a smaller model** past Chinchilla-optimal (more tokens, fewer params) to cut inference cost — training-optimal ≠ deployment-optimal.

**33. All-reduce volume per DP step.**
Gradient all-reduce moves **~2× the gradient size per step** (ring all-reduce ≈ $2(P-1)/P \times$ size ≈ 2× for large $P$). Gradients are ~2 bytes/param (fp16), so for a model (or shard) of $g$ params it's ~$4g$ bytes/step on the wire. For a 7B model that's ~28 GB of all-reduce traffic per step — which is why you **overlap it with backward** (bucketed) and keep DP on fast interconnect; with ZeRO it becomes reduce-scatter + all-gather of similar total volume.

**34. Activation-checkpointing savings.**
It reduces stored activations roughly from $O(L)$ to $O(\sqrt{L})$ layers' worth (store every ~$\sqrt L$-th, recompute the rest), often a **large multiplier** of activation-memory savings for deep models, at the cost of **~one extra forward pass (~33% more compute)** in the backward. Net: trade ~33% throughput for the ability to fit much longer sequences / bigger micro-batches — usually worth it because memory, not compute, is the binding constraint.

---

## 🏗️ Design variations

**35. Multi-tenant LoRA/QLoRA service.**
One **shared frozen base** (optionally 4-bit quantized for QLoRA) loaded once; each tenant trains only small **low-rank adapters** — tiny optimizer state, fast, cheap. Schedule many such jobs on shared GPUs (they pack well — low memory each); store adapters as small versioned artifacts in the registry. This mirrors **multi-LoRA serving** in the [inference service](../llm-inference/README.md): hot-swap adapters over a common base. Default most tenant fine-tuning here — a fraction of full-fine-tune cost for most adaptation needs.

**36. RLHF pipeline end to end.**
(1) **SFT** a base model on demonstrations. (2) Collect **preference data** (pairwise human rankings) and train a **reward model**. (3) **PPO loop:** the policy **generates rollouts** (needs a generation engine), the reward model scores them, and PPO updates the policy with a **KL penalty to a frozen reference** to prevent drift/reward-hacking. Hold **4 models** (actor, critic, reward, reference) + rollouts in memory. Add safety/preference evals as gates. Because this is heavy and unstable, offer **DPO** as a lighter alternative.

**37. Hyperparameter-sweep system.**
A **parent spec + search space** fans out child jobs sharing the scheduler; a **search controller** (grid/random/Bayesian/**ASHA**) proposes configs and **early-stops** weak trials to save compute. The experiment tracker logs all runs for comparison; results feed a leaderboard. Use **small proxy runs** (fewer tokens/smaller model) to triage, then promote winners to full scale. Respect quotas/priority so a sweep can't starve production. Early-stopping + proxy runs are the cost-control levers.

**38. Continued pretraining / domain adaptation.**
Start from a **registry base model** (load its weights, fresh or warm optimizer state), then continue pretraining on a **domain-weighted mix** (e.g. 10–30% domain + general data to avoid catastrophic forgetting), usually with a **lower LR** and a short re-warmup. Decontaminate against target evals, watch for forgetting on general benchmarks, and gate on **both** domain gains and no general regressions. Cheaper than from-scratch and a common platform workload; for lighter needs, prefer LoRA.

---

## 🐞 Debugging & ops

**39. Throughput drop 512→2048 GPUs.**
Classic **scaling inefficiency**: 4× the GPUs but communication grew faster than compute. Likely culprits — **cross-island placement** (collectives now traverse slower fabric → topology-aware scheduling), **exposed all-reduce** no longer hidden by compute (the DP all-reduce is larger / less overlapped), **pipeline bubbles** if PP grew, **data loader** can't feed 4× the GPUs (I/O bound), or **global batch** held constant so per-GPU work (and arithmetic intensity) dropped. Profile comm-vs-compute and loader wait; fix placement, overlap, batch, and loader throughput.

**40. OOM only at long sequence lengths.**
Activations scale with **batch × seq (× seq for attention)**, so long sequences blow up memory. Fixes, cheapest first: **activation checkpointing**, **smaller micro-batch** + grad accumulation, **FlashAttention** (no full $O(\text{seq}^2)$ attention matrix in memory), **sequence/context parallelism** to split the sequence across GPUs, and shard states with **ZeRO-3**. If only the longest bucket OOMs, also check **sequence packing/bucketing** so one outlier sample doesn't set the memory ceiling.

**41. Same config, different final loss.**
Expected to a small degree (non-determinism), but large gaps mean a **reproducibility gap**: unseeded or non-resumable **data order**, non-deterministic kernels/reductions, **different GPU count** (changes reduction order and effective batch), library/CUDA version drift, or dropout/RNG not seeded per-rank. Fix: pin seeds (per rank), **deterministic resumable data sampler**, pin versions/image, reproduce on the **same topology**, and log everything. Then quantify residual variance with a couple of controlled repeats.

**42. NaN after hours of training.**
**Halt and roll back** to the last good checkpoint immediately. Triage causes: **fp16 overflow** (→ bf16 or fix loss scaling), a **bad/corrupt data batch** (inspect the batch at that step — a resumable data order makes this reproducible), **exploding gradients** (→ tighter grad clipping, lower LR), division/log of zero in a custom op, or a **learning-rate spike**. Add **NaN/Inf guards** that auto-halt + alert, grad-norm monitoring, and resume with the offending batch skipped and/or a lower LR.

---

## What strong answers share
- **Both estimates up front:** $C\approx6ND$ (→ thousands of GPUs, ~170K GPU-days, MFU = money) and **~16 bytes/param** (→ can't fit one GPU → must shard). Reason everything from these.
- **Planes split:** cheap **control plane** (scheduler/orchestrator/tracker/registry) over an expensive **execution plane** (GPU mesh + data + checkpoints); train = throughput/durability, the mirror of latency-oriented serving.
- **Parallelism as a toolkit:** know what DP/TP/PP/EP split and cost, **match comms to the network**, **overlap comms with compute**, and add complexity **only as memory forces it** (DP → ZeRO/FSDP → activation checkpointing → TP → PP → offload).
- **Reliability is the hard part:** sharded + async **checkpoints**, **auto-resume**, **elasticity**, straggler/divergence handling — multi-week jobs on flaky hardware; **preemption is safe because of checkpoints**.
- **Fine-tuning ladder:** SFT vs **LoRA/QLoRA** (cheap, multi-tenant) vs **RLHF/PPO** (4 models + rollouts) vs **DPO** (no reward model — pragmatic default).
- **MFU + scheduling are the economics:** name the MFU levers and the "<30% = something's wrong" heuristic; gang + topology-aware placement + preemption + spot keep the cluster cheap.

---

[← Back to training platform HLD](README.md) · [Questions](questions.md) · [Cheat-sheet](cheat-sheet.md) · [Index](../../README.md)
