# LLM Mastery — Printable Study Sheet

> All 399 flashcards as Q -> A, grouped by stage and section. Generated from [llm-mastery-flashcards-full.csv](llm-mastery-flashcards-full.csv).
>
> **How to use:** fold the page or cover the **A** line, read the **Q**, answer aloud, then reveal. Star anything you miss and re-drill.

**Total: 399 cards across 8 stages + a System Design track.**

## Contents

- [Stage 1 — Transformer Internals](#stage-1--transformer-internals)
- [Stage 2 — Pretraining at Scale](#stage-2--pretraining-at-scale)
- [Stage 3 — Adaptation & Alignment](#stage-3--adaptation--alignment)
- [Stage 4 — Evaluation](#stage-4--evaluation)
- [Stage 5 — Inference Optimization](#stage-5--inference-optimization)
- [Stage 6 — Production / LLMOps](#stage-6--production--llmops)
- [Stage 7 — Advanced Specialization](#stage-7--advanced-specialization)
- [Stage 8 — Safety & Security](#stage-8--safety--security)
- [System Design — ChatGPT, RAG & Inference](#system-design--chatgpt-rag--inference-hld)

---

## Stage 1 — Transformer Internals

### Fundamentals

**Q1.** Explain self-attention in one sentence.  
**A.** Each token's query is compared (dot product) to every key to get softmax weights, then takes that weighted sum of values — a content-based mixture of all positions.

**Q2.** Why do transformers need positional encodings at all?  
**A.** Attention is permutation-equivariant (a sum over a set), so without position info word order is invisible; encodings inject order.

**Q3.** Encoder-only vs decoder-only vs encoder-decoder?  
**A.** Encoder-only (BERT) bidirectional, understanding/embeddings; decoder-only (GPT) causal, generation; encoder-decoder (T5) seq-to-seq like translation.

**Q4.** What does causal masking do, and where?  
**A.** Sets future-position scores to -inf before softmax so token t attends only to <=t; applied on the T x T score matrix in each decoder self-attention.

**Q5.** Why multiple attention heads instead of one big head?  
**A.** One softmax expresses one relationship per token; multiple heads attend to different things in different subspaces at once.

**Q6.** Job of the feed-forward (MLP) sub-layer?  
**A.** Attention moves information between positions; the MLP processes it per-token and holds most of the model's knowledge/computation.

**Q7.** Why a residual connection around each sub-layer?  
**A.** Gives gradients an identity path to every layer and lets each sub-layer learn a delta to the running representation — the residual stream.

**Q8.** LayerNorm vs BatchNorm — why LayerNorm?  
**A.** LayerNorm normalizes per-token across features, independent of batch/length — works with variable length and single-token decoding.

**Q9.** What is BPE and why byte-level?  
**A.** Greedily merges frequent adjacent symbol pairs into subwords; byte-level base means any Unicode string is representable with no UNK.

**Q10.** Time and memory complexity of self-attention in T?  
**A.** Compute O(T^2 d); memory O(T^2) for the score matrix — quadratic in sequence length.

### Core

**Q11.** Tensor shapes from token ids to logits?  
**A.** ids (B,T) -> embed (B,T,d) -> per block QKV (B,h,T,dk), scores (B,h,T,T), out (B,T,d) -> final LN -> logits (B,T,V).

**Q12.** Why divide attention scores by sqrt(d_k)?  
**A.** q.k of d_k unit-variance terms has variance d_k (std sqrt(d_k)); dividing keeps logits O(1) so softmax isn't saturated and gradients survive.

**Q13.** Pre-norm vs post-norm — which and why?  
**A.** Modern LLMs use pre-norm (x + Sublayer(LN(x))) — clean identity residual so gradients reach depth; post-norm needs careful warmup.

**Q14.** Absolute vs learned vs RoPE vs ALiBi — why RoPE won?  
**A.** RoPE encodes relative position in the QK dot product, adds no params, and extends to longer context (interpolation/YaRN).

**Q15.** Why tie input embedding and output projection?  
**A.** Both are d x V maps between token and model space; tying saves ~Vd params, regularizes, and improves perplexity.

**Q16.** Explain the residual-stream mental model.  
**A.** A per-token width-d bus; each sub-layer reads via LN, computes, and writes back by addition; blocks communicate through what they write.

**Q17.** GELU vs ReLU vs SwiGLU?  
**A.** ReLU hard gate; GELU smooth probabilistic gate; SwiGLU gated unit (Swish(xW) elementwise xV) — better quality/param, used in Llama/PaLM.

**Q18.** Where do most params live: attention or MLP?  
**A.** The MLP: 8 d^2 per block vs attention 4 d^2 — about two-thirds of block params.

**Q19.** Why is attention permutation-invariant without positions?  
**A.** Output for token i is a sum over j of softmax(qi.kj) vj; permuting keys/values permutes summands but not the sum.

**Q20.** How does the KV cache change attention at inference vs training?  
**A.** Training attends over the whole sequence in parallel; decoding caches past K/V so each step attends the new query over the cache — O(T) not O(T^2).

### Senior / Staff

**Q21.** Derive the sqrt(d) scaling.  
**A.** Sum of d_k independent zero-mean unit-variance products has variance d_k; dividing by sqrt(d_k) rescales variance to 1, keeping softmax gradients healthy.

**Q22.** 128K-token context: what breaks and options?  
**A.** O(T^2) compute + linear KV growth + lost-in-middle + out-of-range positions; use FlashAttention, sparse/sliding, GQA/MQA, RoPE interpolation, retrieval.

**Q23.** Design an attention variant to shrink the KV cache.  
**A.** Reduce KV heads: MHA (full) -> GQA (grouped, near-MHA quality) -> MQA (one head, max savings, quality drop); GQA is the default.

**Q24.** Why is post-norm unstable at depth; what does pre-norm do?  
**A.** Post-norm renormalizes the residual path, shrinking gradients to early layers; pre-norm leaves an identity highway so gradients reach depth.

**Q25.** Softmax numerical stability — the max trick.  
**A.** Subtract max(x) before exp: shift-invariant, prevents fp16 overflow (max~65504), largest exponent becomes 1; FlashAttention does this streaming.

**Q26.** Many dumb failures trace to one choice — which?  
**A.** Tokenization: the token (not the character) is the atom — hence bad spelling, digit arithmetic, and whitespace/glitch tokens.

**Q27.** Redesign attention for streaming / infinite generation.  
**A.** Sliding window over the last W tokens plus attention sinks (keep first few tokens) and cache eviction — O(W) memory (StreamingLLM).

**Q28.** CNN/RNN vs transformer inductive biases — why transformers scaled?  
**A.** CNN/RNN bake in strong priors (locality, recurrence) that cap capacity and (RNN) block parallelism; transformers' weak prior + full parallelism scale better.

### Math

**Q29.** Write the full attention equation with shapes.  
**A.** softmax(QK^T/sqrt(dk) + M) V; Q,K,V are (T,dk), QK^T and M are (T,T), softmax over keys, output (T,dk).

**Q30.** Show a block has ~12 d^2 params; where is the 8 d^2?  
**A.** Attention 4 d^2 (Q,K,V,O); MLP up d x 4d + down 4d x d = 8 d^2; total 12 d^2 per block.

**Q31.** Justify forward ~2N FLOPs per token.  
**A.** Each parameter does one multiply-add (2 FLOPs) per token; with N params that's ~2N; backward ~2x so training ~6N.

**Q32.** Gradient of softmax cross-entropy wrt logits?  
**A.** softmax(z) - onehot(y) — the softmax Jacobian telescopes against cross-entropy to give predicted minus target.

**Q33.** Derive O(T^2 d) attention cost; which matmul dominates at long T?  
**A.** QK^T and score.V are each T^2 dk; projections are T d^2; the T^2 d attention term dominates once T > d.

### Coding

**Q34.** Implement scaled dot-product attention with a causal mask.  
**A.** scores = Q K^T / sqrt(dk); masked_fill future with -inf using a lower-triangular mask; softmax over keys; multiply by V.

**Q35.** Reshape (B,T,C) to (B,h,T,dk) and back.  
**A.** view(B,T,h,C/h).transpose(1,2) to split; transpose back then .contiguous().view(B,T,C) to merge (contiguous needed after transpose).

**Q36.** Implement one BPE merge step.  
**A.** get_stats: count adjacent pairs; pick the most frequent; merge: scan and replace that pair with a new id, minting it in the vocab.

**Q37.** Implement a KV cache for single-token decoding.  
**A.** Keep per-layer K,V; each step project only the new token, concat to cache along the time axis, attend the new query over the full cache.

**Q38.** Implement RoPE and apply to Q and K.  
**A.** Build per-position angles theta = base^(-2i/d); rotate even/odd dims by (cos,sin) and apply to Q and K before the dot product.

**Q39.** Vectorize a naive triple-nested attention loop.  
**A.** Replace i,j loops with scores = Q K^T / sqrt(dk), one softmax(dim=-1), then scores @ V; batch and head are leading axes.

### System Design

**Q40.** Design a tokenizer for NL + code + multilingual.  
**A.** Byte-level BPE (no UNK), vocab ~50-130k tuned by fertility across languages, preserve whitespace/indentation, split digits, cover code tokens.

**Q41.** Architecture for long legal docs under a latency budget.  
**A.** RoPE+interpolation or ALiBi, GQA for KV, FlashAttention, possibly sliding-window+global; deeper-over-wider within SLA; consider retrieval over full context.

### Debugging

**Q42.** Loss near zero but generation is gibberish — cause?  
**A.** Causal-mask leak: the model sees future tokens during teacher-forced training (trivial loss) but has none at inference.

**Q43.** Loss becomes NaN after a few hundred steps — suspects?  
**A.** No/short LR warmup, LR too high, fp16 overflow (use bf16), missing grad clip, bad init, or a corrupt batch.

**Q44.** Model insensitive to word order — where to look?  
**A.** Positional info missing or wrong — embeddings not added, wrong axis, zeroed, or RoPE not applied to Q/K.

**Q45.** Generations loop and repeat — diagnose.  
**A.** Decoding: greedy/low-temp loops -> use sampling, repetition/no-repeat-ngram penalties; training: undertrained or degenerate data.

**Q46.** Val loss much worse than train within epoch 1 — suspicious?  
**A.** Not classic overfitting yet; suspect train/val mismatch or leakage, preprocessing/tokenization difference, bad val set, or dropout/eval-mode bug.

---

## Stage 2 — Pretraining at Scale

### Fundamentals

**Q1.** What objective is a base LLM trained on?  
**A.** Self-supervised next-token prediction: minimize cross-entropy of the true next token given previous tokens.

**Q2.** What is perplexity, intuitively?  
**A.** e^(cross-entropy); the effective branching factor — average number of equally-likely next tokens the model is choosing among.

**Q3.** Three levers of pretraining outcome?  
**A.** Parameters N, data D, and compute C, tied by C ~ 6ND (data quality/mixture is the decisive fourth).

**Q4.** What is gradient accumulation and why?  
**A.** Sum gradients over k micro-batches and step once — simulates a k-times larger batch without its memory.

**Q5.** Why warm up the learning rate?  
**A.** Early params and Adam moments are uninformative; ramping LR from 0 avoids early blow-up before cosine decay.

**Q6.** What does gradient clipping protect against?  
**A.** Occasional exploding gradients — clipping the global norm bounds the step and prevents loss spikes/NaNs.

**Q7.** What is mixed-precision training?  
**A.** Heavy matmuls in bf16/fp16 for speed, with an fp32 master copy and fp32-sensitive ops (loss, optimizer) for stability.

**Q8.** What does deduplication buy you?  
**A.** Less wasted capacity memorizing duplicates, less contamination/regurgitation, and more effective diversity per token.

**Q9.** Data parallelism in one sentence?  
**A.** Replicate the model on each device, split the batch, and all-reduce gradients so every replica applies the same update.

**Q10.** Why is data quality often more important than architecture?  
**A.** Above a competent baseline, architecture gains are small while bad/duplicated data directly caps achievable loss and downstream quality.

### Core

**Q11.** State the Chinchilla result; GPT-3's mistake?  
**A.** Scale N and D together (~20 tokens/param); GPT-3 (175B, 300B tokens) was badly under-trained vs a smaller model on more tokens.

**Q12.** Use C ~ 6ND to explain the relationship.  
**A.** ~6 FLOPs/param/token; at fixed C, raising N forces lowering D and vice versa — the tradeoff scaling laws optimize.

**Q13.** Why bf16 over fp16 for stability?  
**A.** bf16 keeps fp32's 8-bit exponent (huge range), rarely overflows, needs no loss scaling; fp16's 5 exponent bits overflow easily.

**Q14.** Four GPU memory components in training?  
**A.** Parameters, gradients, optimizer state (fp32 master + m + v), and activations.

**Q15.** Data vs tensor vs pipeline parallelism comms?  
**A.** DP: all-reduce gradients once/step; TP: all-reduce activations every layer (intra-node); PP: point-to-point between stages plus a bubble.

**Q16.** What does gradient checkpointing trade?  
**A.** Discards activations in forward and recomputes in backward — ~33% more compute for large activation-memory savings.

**Q17.** Why is AdamW's optimizer state memory-hungry?  
**A.** Per param it stores fp32 master + m + v = 12 bytes — 6x the 2-byte bf16 weight, dominating training memory.

**Q18.** What is a data mixture and why tuned?  
**A.** Weighted blend of sources (web, code, math, multilingual); ratios trade off capabilities, so they're tuned like hyperparameters.

**Q19.** Benchmark contamination at pretraining — why it matters?  
**A.** Eval text leaking into training means memorized answers, inflating scores and invalidating comparisons; decontaminate first.

**Q20.** Kaplan vs Chinchilla scaling laws?  
**A.** Kaplan over-weighted model size (giant under-trained models); Chinchilla fixed methodology — grow N and D equally (~20 tok/param).

### Senior / Staff

**Q21.** Fixed compute C: choose N and D.  
**A.** Chinchilla D ~ 20N gives C ~ 120 N^2 so N ~ sqrt(C/120); but over-train a smaller model if serving cost dominates.

**Q22.** VRAM to full-fine-tune a 7B in bf16 + AdamW?  
**A.** params 14 + grads 14 + optimizer 84 = ~112 GB before activations; motivates ZeRO/FSDP/PEFT.

**Q23.** Explain ZeRO stages 1/2/3.  
**A.** Stage 1 shards optimizer state; stage 2 also gradients; stage 3 (~FSDP) also parameters, all-gathered just-in-time.

**Q24.** Parallelism to train 70B on N x 8 x A100?  
**A.** TP within a node (NVLink), PP across nodes (point-to-point, mind the bubble), DP/ZeRO across replicas; checkpoint activations.

**Q25.** Intermittent loss spikes on a long run — diagnose/recover.  
**A.** Suspect bad data shard, high LR/short warmup, fp16 overflow, loose clipping; log grad-norm, clip, skip the batch, roll back to a checkpoint.

**Q26.** Why train past Chinchilla-optimal?  
**A.** Inference economics — a smaller over-trained model is cheaper/faster to serve, and serving cost dwarfs training at scale.

**Q27.** Pretraining data pipeline from raw crawl — stages?  
**A.** Extract text, language ID, quality filter, dedup (MinHash), decontaminate, PII handling, mixture weighting, tokenize, shard.

**Q28.** Detect and prevent eval contamination?  
**A.** N-gram/MinHash overlap of benchmarks vs corpus, remove hits, hold out fresh sets, seed canaries, and report overlap.

### Math

**Q29.** Four AdamW mixed-precision memory terms (bytes/param)?  
**A.** weights 2 + grads 2 + optimizer 12 (fp32 master 4 + m 4 + v 4) = 16 bytes/param.

**Q30.** Why the factor ~6 in C ~ 6ND?  
**A.** Forward ~2 FLOPs/param/token (multiply-add); backward ~4 (grads wrt inputs and weights); total ~6.

**Q31.** Largest Chinchilla-optimal model on 256 A100 for 1 week?  
**A.** ~1.9e22 FLOPs at ~40% MFU; N ~ sqrt(C/120) ~ 10-13B params on ~250B tokens (order of magnitude ~10B).

**Q32.** Perplexity vs cross-entropy; PPL 20 operationally?  
**A.** PPL = e^L; PPL 20 means average per-token loss ln(20) ~ 3.0 nats — as uncertain as choosing among 20 tokens.

### Coding

**Q33.** Write a VRAM estimator (inputs to breakdown).  
**A.** params N x dtype + grads N x dtype + optimizer N x opt_bytes + activations ~ batch x seq x layers x d x const; sum in GB.

**Q34.** Implement gradient accumulation correctly.  
**A.** Divide loss by ACCUM_STEPS, backward each micro-batch, and step+zero_grad every ACCUM_STEPS (so summed grads equal the mean).

**Q35.** Add bf16 autocast + clip + cosine-with-warmup.  
**A.** autocast(bf16) around forward; clip_grad_norm_ to 1.0; LambdaLR linear warmup then cosine decay; bf16 needs no GradScaler.

**Q36.** Implement a streaming sharded data loader.  
**A.** IterableDataset that memory-maps one shard at a time and shards files per worker, so the full corpus never sits in RAM.

**Q37.** Wrap a model in FSDP / ZeRO — key flags.  
**A.** FULL_SHARD (=ZeRO-3), mixed_precision bf16, auto_wrap_policy per transformer block, limit_all_gathers, use_orig_params.

### System Design

**Q38.** Design training infra for a 13B model.  
**A.** Offline data pipeline, FSDP + bf16 + checkpointing, sharded async checkpoints with exact-resume state, auto-restart, monitoring (loss/grad/MFU), cost tracking.

**Q39.** Make a 3-week run resilient to spot preemptions.  
**A.** Frequent async checkpoints (incl. dataloader/optimizer/RNG), elastic launcher that re-forms the group, redundant pool, idempotent atomic writes.

**Q40.** Cut training cost 40% with minimal quality loss — order?  
**A.** Raise MFU first (loader, FlashAttention, batch), then bf16+checkpointing, better data, correct sizing/mild under-train, spot instances.

### Debugging

**Q41.** Throughput far below the GPU roofline — where to look?  
**A.** Data-loader stalls, tiny batch, no fused/Flash kernels, comms-bound parallelism, recompute overhead — generally low MFU; profile.

**Q42.** Model memorizes/regurgitates training data — cause/fix?  
**A.** Insufficient dedup and too many epochs over duplicates; fix with stronger dedup, fewer epochs, filtering, and DP for sensitive data.

**Q43.** Multi-node run 3x slower than expected — diagnose.  
**A.** Interconnect-bound: TP placed across nodes (needs NVLink), DP all-reduce not overlapped, or wrong rank-to-GPU placement.

**Q44.** Loss plateaus far above expected — suspects?  
**A.** LR too low/decayed too fast, model under-capacity, data quality/mixture, over-regularization, or a tokenization bug; compare to scaling-law prediction.

---

## Stage 3 — Adaptation & Alignment

### Fundamentals

**Q1.** Base model vs instruct model?  
**A.** Base completes text and follows instructions poorly; instruct is further trained (SFT + alignment) to follow instructions in an assistant style.

**Q2.** What is supervised fine-tuning (SFT)?  
**A.** Continue training on curated (prompt -> ideal response) pairs with next-token loss computed only on the response.

**Q3.** Why mask prompt tokens in the SFT loss?  
**A.** So the model learns to generate the response, not predict the user's prompt — prevents parroting.

**Q4.** What is LoRA in one sentence?  
**A.** Freeze pretrained weights and learn a low-rank update dW = BA added to chosen matrices, training under 1% of params.

**Q5.** What problem does QLoRA solve?  
**A.** Fine-tuning very large models on a single GPU by 4-bit quantizing the frozen base and training small bf16 adapters on top.

**Q6.** What is RLHF at a high level?  
**A.** SFT, then train a reward model from human preferences, then optimize the policy (PPO) to maximize reward with a KL penalty.

**Q7.** What is a reward model?  
**A.** A model (LLM + scalar head) mapping (prompt, response) to a quality score, trained on human preference pairs.

**Q8.** What is DPO and why popular?  
**A.** Direct Preference Optimization trains the policy directly on preference pairs with a simple loss — no separate RM or RL loop; simpler and stable.

**Q9.** What is catastrophic forgetting?  
**A.** Fine-tuning on narrow data overwrites broad pretrained capabilities; mitigated by PEFT, low LR, fewer epochs, and replay.

**Q10.** What is a chat template and why does it matter?  
**A.** The exact role/special-token format the model was trained on; a mismatch at inference silently degrades or garbles outputs.

### Core

**Q11.** LoRA decomposition dW = BA — which train, which freeze?  
**A.** h = W0 x + (alpha/r) B A x; A and B train (A random, B zero-init), W0 stays frozen.

**Q12.** Three-stage RLHF pipeline?  
**A.** SFT on demonstrations; reward modeling from preferences (Bradley-Terry); PPO to maximize reward minus KL to the SFT reference.

**Q13.** Role of the KL penalty in RLHF/PPO?  
**A.** Anchors the policy to the reference; without it the policy drifts off-distribution and reward-hacks the imperfect RM.

**Q14.** QLoRA's three tricks?  
**A.** NF4 4-bit quantization (optimal for normal weights), double quantization (quantize the constants), paged optimizers (page state to CPU).

**Q15.** How does DPO avoid RM + RL?  
**A.** It uses the closed-form RLHF optimum: implicit reward = beta log(pi/pi_ref); substitute into Bradley-Terry to get a direct loss on logprobs.

**Q16.** When choose full fine-tuning over PEFT?  
**A.** Lots of data/compute, deep changes (new language/modality), or maximum quality; PEFT can underfit very large adaptations.

**Q17.** What is reward hacking and what guards against it?  
**A.** Policy exploits RM flaws for high score without quality; guard with KL penalty, early stopping, stronger/ensembled RM, fresh data.

**Q18.** DPO vs IPO vs KTO vs ORPO?  
**A.** DPO needs paired prefs; IPO regularizes near-deterministic prefs; KTO needs only binary good/bad; ORPO merges SFT+pref in one stage (no ref).

**Q19.** How do you pick LoRA r and alpha?  
**A.** r sets capacity (8 light, 16-64+ for bigger shifts); alpha scales update (alpha/r), often alpha=2r; tune on validation and pick which matrices.

**Q20.** Why can a few thousand clean SFT examples beat millions of noisy?  
**A.** The base already knows language/facts; SFT mainly teaches format/behavior, which clean diverse data conveys; noise injects contradictory targets.

### Senior / Staff

**Q21.** 1 GPU, preference data, safer assistant — method?  
**A.** QLoRA SFT then DPO (not PPO): 4-bit base fits one GPU, DPO consumes pairs directly and is stable; eval win-rate + safety + capability tax.

**Q22.** DPO loss intuition and link to RLHF?  
**A.** RLHF optimum gives implicit reward beta log(pi/pi_ref); plug into Bradley-Terry to raise preferred over rejected, anchored to the reference.

**Q23.** Why does LoRA work (low rank)?  
**A.** Fine-tuning has low intrinsic dimension — adaptation moves in a small subspace, so dW is approximately low-rank.

**Q24.** Why can QLoRA fine-tune a 7B on one GPU?  
**A.** 4-bit frozen base ~3.5 GB + small bf16 adapters + paged optimizer = a few GB vs ~112 GB for full bf16+AdamW FT.

**Q25.** RLHF policy produces high-reward gibberish — diagnose/fix.  
**A.** Reward hacking off the RM's distribution; raise KL coefficient, early stop, retrain/ensemble the RM with on-policy data, validate with humans.

**Q26.** Alignment pipeline with safety, no RL infra?  
**A.** SFT (QLoRA) -> DPO/ORPO on preferences (KTO if only binary), scale feedback with RLAIF/Constitutional, red-team eval, iterate.

**Q27.** What is the alignment tax and how to measure/minimize?  
**A.** Capability regression from alignment; measure aligned vs pre-aligned on a fixed suite; minimize with PEFT/low LR, data replay, smaller KL drift.

**Q28.** How to build a preference dataset; biases?  
**A.** Pairwise human rankings with clear guidelines and agreement checks; control length, position, sycophancy bias; augment with calibrated RLAIF.

### Math

**Q29.** LoRA trainable params for a d x d matrix at rank r?  
**A.** 2 d r vs d^2; ratio 2r/d (e.g. d=4096, r=8 -> ~0.4%).

**Q30.** Reward-model ranking loss (Bradley-Terry)?  
**A.** -log sigmoid(r(x, y_w) - r(x, y_l)) — maximize probability the preferred response scores higher.

**Q31.** DPO loss; role of beta and pi_ref?  
**A.** -log sigmoid(beta[(log pi/pi_ref)_w - (log pi/pi_ref)_l]); pi_ref anchors the shift, beta controls preference strength vs drift.

**Q32.** Memory: full FT vs LoRA vs QLoRA for 7B?  
**A.** Full ~112 GB; LoRA ~15-16 GB (bf16 base, tiny adapter state); QLoRA ~5 GB (4-bit base). ~20x smaller than full.

### Coding

**Q33.** Implement SFT loss masking.  
**A.** Set prompt and padding label positions to -100 so cross_entropy ignores them; only response tokens contribute gradient.

**Q34.** Implement a LoRA layer over a frozen nn.Linear.  
**A.** Freeze base; add A (r x in, random) and B (out x r, zero); forward = base(x) + (alpha/r) (x A^T) B^T.

**Q35.** Implement the DPO loss from logprobs.  
**A.** -logsigmoid(beta[(pol_chosen-pol_rej) - (ref_chosen-ref_rej)]).mean() using summed sequence logprobs.

**Q36.** Tokenize a multi-turn chat with role masking.  
**A.** apply_chat_template, then set labels to -100 everywhere except assistant spans so only assistant tokens are supervised.

**Q37.** Merge LoRA back into base at correct scale.  
**A.** W += (alpha/r) B @ A in no_grad; the alpha/r scale must match training, then serve a plain Linear.

### System Design

**Q38.** Design a fine-tuning service.  
**A.** Validate/dedup/PII-scan data, run managed QLoRA jobs, auto eval-gate vs base, store adapters and hot-swap onto a shared frozen base with versioning.

**Q39.** Support 50 customer models cheaply — architecture?  
**A.** One shared frozen base + 50 small LoRA adapters loaded dynamically (S-LoRA/punica style), cutting memory/cost ~50x.

**Q40.** Build the human-feedback data flywheel.  
**A.** Log prompts/responses + implicit signals (thumbs, edits, regens); mine preference pairs; SFT/DPO; eval-gate; ship; measure online; repeat.

### Debugging

**Q41.** After SFT the model parrots the question — bug?  
**A.** Prompt tokens weren't masked in the loss, so it learned to reproduce the input; mask them with -100.

**Q42.** After FT, lost general knowledge / repetitive — cause?  
**A.** Catastrophic forgetting from high LR/too many epochs/full FT on narrow data; use PEFT, lower LR, fewer epochs, replay general data.

**Q43.** Garbled only in production — suspect?  
**A.** Chat-template/special-token mismatch between train and serve; diff the rendered strings and use the exact same template/tokenizer.

**Q44.** DPO not improving win-rate — what to check?  
**A.** beta tuning, correct reference model/logprobs, noisy or low-margin preference pairs, too few steps, length bias, or swapped chosen/rejected.

---

## Stage 4 — Evaluation

### Fundamentals

**Q1.** Intrinsic vs extrinsic evaluation?  
**A.** Intrinsic measures a property in isolation (perplexity, accuracy); extrinsic measures real downstream task success — they often disagree.

**Q2.** What does perplexity measure and its limits?  
**A.** Average uncertainty on held-out text; only measures LM fit, isn't comparable across tokenizers, and ignores helpfulness/correctness/safety.

**Q3.** What do BLEU/ROUGE measure and when mislead?  
**A.** N-gram overlap with references; mislead on open-ended tasks where correct paraphrases score low and fluent-wrong text scores high.

**Q4.** What is pass@k and why trustworthy?  
**A.** Generate k samples, count solved if any passes the unit tests; trustworthy because the signal is objective and executable.

**Q5.** Three benchmarks and what each tests?  
**A.** MMLU (broad knowledge MCQ), HumanEval/MBPP (code via tests), GSM8K/MATH (multi-step math reasoning).

**Q6.** What is benchmark contamination?  
**A.** Test items (or near-duplicates) appear in training data, so the model memorized answers and scores overstate ability.

**Q7.** What is LLM-as-a-judge?  
**A.** Using a strong LLM to score/compare outputs against a rubric or pairwise; scalable but biased (position/length/self-preference).

**Q8.** Why is human evaluation still needed?  
**A.** For subjective/open-ended quality and safety, and as the ground truth that validates whether automatic metrics/judges are trustworthy.

**Q9.** What is a hallucination as an eval problem?  
**A.** Fluent, confident, unsupported output; hard to detect with surface metrics — needs faithfulness checks against sources.

**Q10.** In RAG, what two things to evaluate separately?  
**A.** Retrieval (recall@k, MRR, nDCG) and generation (faithfulness + answer relevance) — to localize the failing sub-system.

### Core

**Q11.** Top MMLU yet fails in production — why?  
**A.** Contamination, distribution/format shift, metric-goal mismatch (no multi-turn/tools/safety), and brittleness to messy phrasing.

**Q12.** LLM-judge biases and a fix for each?  
**A.** Position (swap+average), length (length-control), self-preference (different/ensemble judge), style (rubric scoring); validate vs humans.

**Q13.** Detect train/benchmark contamination?  
**A.** N-gram/MinHash overlap between items and corpus, plus behavioral tests (original vs perturbed equivalents) and canary strings.

**Q14.** What is HELM's philosophy?  
**A.** Holistic evaluation across many scenarios and dimensions (accuracy, robustness, bias, toxicity, efficiency, calibration), not one number.

**Q15.** Why pairwise over absolute scoring?  
**A.** Raters/judges are noisy on absolute scales but consistent at relative judgments; pairwise has higher agreement and aggregates via Elo.

**Q16.** Retrieval vs generation quality in RAG?  
**A.** Retrieval with gold labels (recall@k/MRR/nDCG); generation as faithfulness + answer-relevance given the retrieved context.

**Q17.** What is calibration and why it matters?  
**A.** Stated confidence matches empirical accuracy; matters because users/systems act on confidence; measured with ECE.

**Q18.** Goodhart's law in eval?  
**A.** When a measure becomes a target it stops being good — optimizing a benchmark/judge games it without real improvement.

**Q19.** How report results honestly?  
**A.** Confidence intervals, multiple seeds/decoding, significance tests, test-set size/provenance, contamination checks, and prompt details.

**Q20.** Offline benchmarks vs online evals?  
**A.** Offline: fast, reproducible, gate-able, but static/contamination-prone; online: real A/B product metrics, ground truth but slow/confounded.

### Senior / Staff

**Q21.** Eval a task with no ground-truth labels?  
**A.** Rubric LLM-judge (order-swapped) + pairwise preference + reference-free metrics + human spot-checks; report judge-human agreement and CIs.

**Q22.** End-to-end eval before launching an assistant?  
**A.** Capability benchmarks, faithfulness/hallucination, safety/red-team, robustness, latency/cost, A/B + online metrics, and a regression suite.

**Q23.** Offline up but users unhappy — what's wrong with eval?  
**A.** Contamination, Goodhart/metric mismatch, distribution shift, prompt-format gap, stale/tiny test set, or judge bias; align eval to production + add online.

**Q24.** Validate and de-bias an LLM-judge for a leaderboard?  
**A.** Swap order, length-control, ensemble/different family, calibrate vs human labels (kappa), report CIs, and audit fixed prompts.

**Q25.** Contamination-resistant eval program?  
**A.** Private held-out and time-split/fresh sets, rotating benchmarks, canaries, n-gram/MinHash scans, and separating eval curators from data curators.

**Q26.** Evaluate RAG to localize failures?  
**A.** Score retrieval and generation independently and build an attribution matrix: retrieval miss vs unfaithful generation vs both.

**Q27.** Statistically decide A beats B?  
**A.** Paired test (McNemar/paired bootstrap), report CI and effect size, check power/sample size, and correct for multiple comparisons.

**Q28.** Critique a popular benchmark (e.g. MMLU)?  
**A.** MCQ rewards recognition, contamination-prone, some mislabeled items, static knowledge, format-sensitive — says little about real-task success.

### Stats

**Q29.** precision@k, recall@k, MRR, nDCG — when?  
**A.** precision@k clean top; recall@k found-all (RAG); MRR rank of first relevant (one answer); nDCG graded relevance with order.

**Q30.** Compute a bootstrap CI for accuracy?  
**A.** Resample per-item correctness with replacement n times, recompute accuracy, repeat ~10k times, take 2.5/97.5 percentiles.

**Q31.** 71% vs 73% on 500 items — significant?  
**A.** SE ~ sqrt(p(1-p)/n) ~ 2%, so 2 points is ~1 SE — not significant; use a paired test (McNemar/bootstrap).

**Q32.** Unbiased pass@k from n>k samples?  
**A.** pass@k = 1 - C(n-c,k)/C(n,k): one minus the chance a random size-k subset contains no passing sample (hypergeometric).

### Coding

**Q33.** Implement a pairwise judge controlling position bias.  
**A.** Ask both orders (A,B) and (B,A), average; count a win only if consistent across both, else tie.

**Q34.** Implement pass@k estimation.  
**A.** pass@k = 1 - comb(n-c,k)/comb(n,k); score each problem by counting samples that pass all unit tests.

**Q35.** Implement an n-gram overlap contamination scanner.  
**A.** Build the set of training n-grams; for each benchmark item, flag items whose n-grams intersect (hash/MinHash at scale).

**Q36.** Implement recall@k and MRR.  
**A.** recall@k = |top-k ∩ gold| / |gold|; MRR = 1/rank of the first retrieved gold doc (0 if none).

**Q37.** Implement bootstrap resampling for a CI.  
**A.** Resample indices with replacement, recompute the metric B times, return the metric plus its percentile interval.

### System Design

**Q38.** Design continuous eval on live traffic with drift alerts.  
**A.** Sample privacy-safe traffic, run automatic evals on a schedule, track metrics with drift detection, alert on degradation, human-validate judges.

**Q39.** Eval harness a 50-person team can extend?  
**A.** Plugin registry for tasks/datasets/metrics/judges, standard interfaces, versioned data + seeds, CI regression suite, ownership per task.

**Q40.** Produce a trustworthy internal leaderboard?  
**A.** Mixed private/rotating datasets, validated/calibrated judges + human eval, contamination policy, report CIs, refresh cadence, published methodology.

### Debugging

**Q41.** Same model, different scores on equivalent prompts?  
**A.** Prompt-format sensitivity; standardize prompts, report across formats (mean±variance), test few-shot stability, treat swings as a robustness issue.

**Q42.** Judge always prefers the longer answer — fix?  
**A.** Verbosity bias; instruct it to ignore length, length-control/normalize, or match length in pairs; validate vs humans.

**Q43.** Suspiciously high scores after a data refresh?  
**A.** First hypothesis is contamination; run overlap scans against eval sets and compare to fresh equivalents before trusting gains.

**Q44.** A/B shows no difference but you know it's better?  
**A.** Underpowered test, wrong/insensitive metric, segment/heterogeneous effects, or confounds (novelty, latency); add power, better metric, segment analysis.

---

## Stage 5 — Inference Optimization

### Fundamentals

**Q1.** Two phases of autoregressive generation?  
**A.** Prefill (process the whole prompt in parallel, fills the KV cache) then decode (one token at a time using the cache).

**Q2.** What is the KV cache and why?  
**A.** Stores past keys/values so each decode step reuses them instead of recomputing — turns per-step cost from O(T^2) to O(T).

**Q3.** Greedy vs sampling vs beam search?  
**A.** Greedy takes the argmax (repetitive); sampling draws from the distribution (diverse); beam keeps top-b sequences (good for MT, weak for open-ended).

**Q4.** What do temperature, top-k, top-p control?  
**A.** Temperature sharpens/flattens logits; top-k keeps the k most likely; top-p keeps the smallest set summing to p (nucleus).

**Q5.** What is quantization and why faster?  
**A.** Store weights/activations in fewer bits (int8/int4); decode is memory-bound, so moving fewer bytes from HBM speeds it up.

**Q6.** What is FlashAttention solving?  
**A.** Standard attention writes the T x T scores to HBM; FlashAttention tiles and computes softmax online in SRAM — same math, far fewer memory reads/writes.

**Q7.** What is continuous (in-flight) batching?  
**A.** Schedule at token granularity: finished sequences leave and new ones join each step, keeping the GPU busy instead of waiting for the slowest.

**Q8.** What is speculative decoding?  
**A.** A small draft model proposes several tokens; the big model verifies them in one pass and accepts a prefix — fewer big-model steps, identical distribution.

**Q9.** Three serving frameworks and a use case?  
**A.** vLLM (high-throughput GPU serving), TensorRT-LLM (max NVIDIA latency), llama.cpp/Ollama (local/edge CPU-GPU).

**Q10.** Define TTFT, TPOT, throughput.  
**A.** TTFT time to first token (prefill latency); TPOT time per output token (decode speed); throughput total tokens/sec across all requests.

### Core

**Q11.** Why is decode bandwidth-bound, prefill compute-bound?  
**A.** Decode is one token x all weights (low arithmetic intensity, HBM-limited); prefill multiplies many tokens through weights (compute-limited).

**Q12.** Derive the KV-cache size.  
**A.** 2 (K,V) x layers x kv_heads x head_dim x seq x batch x bytes — linear in sequence length and batch.

**Q13.** How do MQA and GQA cut the KV cache; tradeoff?  
**A.** Share K/V across query heads — MQA one KV head (max savings, some quality loss), GQA a few groups (near-MHA quality); shrink cache by the head ratio.

**Q14.** Why does FlashAttention speed up without changing math?  
**A.** It's IO-aware: tiling + online softmax keep the scores in SRAM and never materialize the T x T matrix in HBM, cutting memory traffic.

**Q15.** Compare GPTQ, AWQ, GGUF.  
**A.** GPTQ post-training int4 via Hessian-aware rounding (GPU); AWQ protects salient weights by activation scale; GGUF the llama.cpp file format with k-quants for CPU/edge.

**Q16.** How does speculative decoding preserve the distribution?  
**A.** Accept draft token with prob min(1, p_target/p_draft); on reject, resample from the normalized residual — provably equals sampling from the target.

**Q17.** What does PagedAttention do?  
**A.** Stores the KV cache in fixed non-contiguous pages (like virtual memory), eliminating fragmentation and enabling prefix sharing — big throughput gains.

**Q18.** Latency vs throughput tradeoff and batch size?  
**A.** Bigger batches raise throughput (amortize weight loads) but increase per-request latency; tune batch to the SLA.

**Q19.** Weight-only vs weight+activation quantization?  
**A.** Weight-only (int4) cuts memory/bandwidth, great for decode; weight+activation (int8 both) also speeds compute via int8 matmul but is harder due to activation outliers.

**Q20.** When distillation/pruning over quantization?  
**A.** When you want a permanently smaller/faster model or quantization stalls — distill into a small student or prune structurally; both need retraining.

### Senior / Staff

**Q21.** Serving stack for p99 < 300ms TTFT at 1000 RPS for 13B?  
**A.** vLLM/TRT-LLM with continuous batching + PagedAttention, GQA, int8/fp8, tensor parallel, prefill/decode disaggregation, autoscaling, admission control.

**Q22.** Tokens/sec below FLOPs roofline in decode — why expected?  
**A.** Decode is memory-bound (arithmetic intensity ~1): you re-read all weights per token, so HBM bandwidth, not FLOPs, sets the ceiling.

**Q23.** Cut cost-per-token 2x — levers and risk?  
**A.** Quantize (int8/int4 small quality risk), GQA/MQA, continuous batching + Paged, speculative decoding, distillation, right-size — biggest risk is quant/distill accuracy.

**Q24.** Speculative decoding accept/reject in detail?  
**A.** For each draft token accept with min(1, p_t/p_d); on first reject resample from max(0, p_t - p_d) normalized and stop; verify k drafts in one target pass.

**Q25.** Serve 100 LoRA adapters on one base — path?  
**A.** Keep the base resident; batch requests, apply per-request low-rank dW in the kernel (S-LoRA/punica), page adapters in/out — near base throughput.

**Q26.** 128K context blowing up memory — serving options?  
**A.** GQA/MQA, KV quantization, PagedAttention, sliding-window/sink + eviction, chunked prefill, offload, or retrieval instead of full context.

**Q27.** Explain MFU; why low in decode; how to raise?  
**A.** MFU = achieved/peak FLOPs; decode's low arithmetic intensity wastes FLOPs; raise with larger effective batch (continuous batching) and speculative decoding.

**Q28.** Disaggregated prefill/decode serving — why split?  
**A.** Prefill is compute-bound, decode memory-bound; running them on separate pools sized/optimized independently avoids interference and improves both TTFT and throughput.

### Math

**Q29.** Compute KV-cache bytes for a given config.  
**A.** bytes = 2 x layers x kv_heads x head_dim x seq x batch x dtype_bytes; e.g. Llama-2-13B fp16 at 4k is several GB per long request.

**Q30.** Estimate decode latency from size and HBM bandwidth.  
**A.** Per token ~ bytes of weights read / bandwidth; e.g. 13B int8 (~13 GB) on ~2 TB/s ~ 6-7 ms/token lower bound.

**Q31.** Arithmetic intensity of a decode step; why batching helps?  
**A.** ~1 FLOP/byte (read each weight once, one MAC) — memory-bound; batching reuses each loaded weight across B tokens, raising intensity ~B.

**Q32.** int4 vs fp16 weights — decode speed ratio?  
**A.** int4 moves ~4x fewer bytes than fp16; since decode is bandwidth-bound, expect up to ~4x faster (minus dequant overhead).

### Coding

**Q33.** Implement greedy + temperature + top-k + top-p sampling.  
**A.** Scale logits by 1/T, keep top-k, then keep the smallest cumulative-prob set >= p (nucleus), renormalize, and sample; T=0 means argmax.

**Q34.** Implement an incremental KV cache and single-step decode.  
**A.** Cache per-layer K,V; each step project only the new token, append to cache, attend its query over the cache, sample, and feed back.

**Q35.** Implement the speculative-decoding accept/reject loop.  
**A.** Draft k tokens; for each accept with min(1, p_t/p_d), else resample from the residual and break; append one bonus token from the target.

**Q36.** Benchmark harness for TTFT, TPOT, throughput.  
**A.** Time first token (TTFT), average inter-token gap (TPOT), and total output tokens / wall time under concurrency; report p50/p95/p99.

**Q37.** Implement naive int8 weight quantization + dequant.  
**A.** scale = max|W|/127; q = round(W/scale) clamped to int8; dequant = q x scale; per-channel scales reduce error.

### System Design

**Q38.** Design inference for a multi-tenant LLM API.  
**A.** Gateway (auth, rate limit, routing), per-model vLLM pools with continuous batching, quotas/fair scheduling, KV/semantic caching, autoscale, per-tenant metrics.

**Q39.** Edge deployment: 7B on a laptop/phone.  
**A.** 4-bit GGUF via llama.cpp, GQA, small context + KV quant, memory-mapped weights, Metal/NNAPI acceleration; trade speed/quality for footprint.

**Q40.** Pick hardware for three workloads (A100/H100/L4/CPU)?  
**A.** High-throughput training/serving -> H100; steady mid serving -> A100; small/cheap or bursty -> L4; offline/local low-QPS -> CPU + GGUF.

### Debugging

**Q41.** Throughput collapses when long-context requests arrive — why?  
**A.** A few long sequences hog KV memory and dominate batch compute, shrinking batch size; isolate by length, chunk prefill, cap context, separate pools.

**Q42.** int4 quantization tanked accuracy on one task — diagnose.  
**A.** Outlier-sensitive task hit by quant error; use AWQ/GPTQ with better calibration, per-channel/group scales, keep sensitive layers higher precision, or int8.

**Q43.** p50 fine but p99 terrible — causes?  
**A.** Head-of-line blocking from long requests, queueing, cold starts, GC/paging, batch stragglers; fix with chunked prefill, admission control, length-aware scheduling.

**Q44.** Speculative decoding gave no speedup — why?  
**A.** Low acceptance (draft too weak or mismatched) or draft too costly; pick an aligned, much smaller draft, tune k, and verify acceptance rate.

---

## Stage 6 — Production / LLMOps

### Fundamentals

**Q1.** What is prompt engineering? Three techniques.  
**A.** Designing inputs to steer the model: few-shot examples, chain-of-thought, and clear role/format instructions (plus output constraints).

**Q2.** What is chain-of-thought prompting?  
**A.** Asking the model to reason step by step before answering; allocates more compute to intermediate steps and improves multi-step accuracy.

**Q3.** What is RAG and why?  
**A.** Retrieve relevant documents and put them in context so the model answers from current, private, sourced data — reduces hallucination and staleness.

**Q4.** What is an embedding and a vector database?  
**A.** An embedding is a semantic vector for text; a vector DB indexes them (e.g. HNSW) for fast approximate nearest-neighbor similarity search.

**Q5.** What is reranking?  
**A.** A second-stage cross-encoder re-scores the top retrieved candidates jointly with the query for far better precision than the bi-encoder recall stage.

**Q6.** What is function/tool calling?  
**A.** The model emits a structured call (name + JSON args); your code runs the tool and returns the result, letting the LLM act on the world.

**Q7.** What is the ReAct pattern?  
**A.** Interleave Reasoning and Acting: Thought -> Action -> Observation loop until the model has enough to answer.

**Q8.** Why use structured output (JSON mode)?  
**A.** Makes outputs machine-parseable and reliable for downstream code via constrained/grammar decoding or a schema; avoids brittle free-text parsing.

**Q9.** What are guardrails?  
**A.** Input/output checks around the model (injection/PII/topic/format/safety filters) that block or transform unsafe or malformed content.

**Q10.** When fine-tune instead of RAG?  
**A.** Fine-tune to change behavior/style/format or teach a skill; RAG to inject changing/factual knowledge — often combine both.

### Core

**Q11.** Walk through a full RAG pipeline.  
**A.** Ingest -> chunk -> embed -> index; at query: (rewrite) -> retrieve (hybrid) -> rerank -> assemble context -> generate with citations -> post-check.

**Q12.** Explain chunking tradeoffs.  
**A.** Small chunks = precise but fragmented context; large = more context but noisy/diluted; tune size + overlap, respect structure, store metadata.

**Q13.** What is hybrid search and why combine dense + sparse?  
**A.** Mix dense (semantic) with BM25 (exact keyword/rare terms); fusion (RRF) covers each other's misses — robust recall.

**Q14.** How does reranking improve quality and what cost?  
**A.** A cross-encoder reads query+doc together for precise relevance, lifting answer quality; cost is extra latency per candidate, so rerank only the top-N.

**Q15.** Decision framework: prompt vs RAG vs fine-tune?  
**A.** Prompt for behavior the base has; RAG for external/changing knowledge + citations; fine-tune for durable style/skill/format; combine as needed.

**Q16.** How do agents fail and how contain it?  
**A.** Compounding errors, loops, wrong/over-privileged tool calls; contain with step limits, validation, least-privilege tools, human approval, observability.

**Q17.** What is semantic caching and when help?  
**A.** Cache answers keyed by embedding similarity of queries; helps with repetitive/FAQ traffic (cost/latency) but risks stale or wrong-match hits.

**Q18.** What is HyDE / query rewriting?  
**A.** Generate a hypothetical answer (or rewrite/expand the query) and embed that to retrieve — bridges the question-vs-answer vocabulary gap.

**Q19.** How ground a model and force citations?  
**A.** Instruct answer-only-from-context, require per-claim source ids, and post-verify that cited spans support the claims; refuse if unsupported.

**Q20.** What to log/monitor in a production LLM app?  
**A.** Prompts/responses, retrieved docs, tokens/cost, latency, tool calls, errors, user feedback, safety flags, and quality/drift metrics.

### Senior / Staff

**Q21.** Design production RAG over 10M docs with citations + access control.  
**A.** Scalable ingest/embeddings, ANN index + metadata, hybrid retrieve -> rerank, pre-retrieval permission filtering (ACLs), grounded generation with citations, eval + observability.

**Q22.** Localize a wrong RAG answer: retrieval or generation?  
**A.** Check if gold docs are in the retrieved set: missing -> retrieval (embeddings/chunking/rerank); present but ignored/contradicted -> generation (prompt/grounding).

**Q23.** Prompt vs RAG vs fine-tune for three cases?  
**A.** Changing facts/citations -> RAG; fixed tone/format/skill -> fine-tune; capability already present, just needs instruction -> prompt; mix when needed.

**Q24.** Design an agent that books travel via tools.  
**A.** Define typed tools (search/price/book), ReAct/planner loop, schema-validated args, confirmation before booking, step/budget limits, retries, full tracing.

**Q25.** RAG fine in demos but hallucinates in prod — plan?  
**A.** Measure retrieval vs generation separately; fix recall (chunking/hybrid/rerank), enforce grounding + citations + abstention, eval on real queries, add monitoring.

**Q26.** Build the eval + observability layer to ship RAG safely.  
**A.** Golden Q/A set, retrieval + faithfulness metrics, LLM-judge + human spot-checks, offline gate in CI, online A/B + feedback, tracing and drift alerts.

**Q27.** Cut LLM API cost 60% without hurting quality.  
**A.** Route easy queries to small models, semantic + prompt caching, shorten prompts/context, cap output, batch, distill/self-host hot paths; measure quality.

**Q28.** When is an agent the wrong choice?  
**A.** When the workflow is known and fixed — a deterministic pipeline/chain is cheaper, faster, and more reliable than open-ended planning.

### Coding

**Q29.** Implement a minimal RAG loop.  
**A.** embed(query) -> vector_store.search(k) -> build a context-grounded prompt with the chunks -> llm.generate, returning the answer plus sources.

**Q30.** Implement a reranking step with a cross-encoder.  
**A.** Score each (query, candidate) pair with a cross-encoder, sort by score, and keep the top-n to pass into the prompt.

**Q31.** Implement self-consistency.  
**A.** Sample N chain-of-thought answers at temperature>0, extract each final answer, and return the majority vote.

**Q32.** Implement a tool-calling loop with a step limit.  
**A.** While under max_steps: call the model, if it requests a tool run it and append the observation, else return the final answer; cap iterations.

**Q33.** Implement schema-validated JSON output with retry.  
**A.** Parse against a schema (e.g. Pydantic); on failure feed the validation error back and retry up to k times before failing closed.

**Q34.** Implement hybrid search (BM25 + dense).  
**A.** Run BM25 and vector search, normalize/fuse with Reciprocal Rank Fusion (sum of 1/(k+rank)), and return the merged ranking.

### System Design

**Q35.** Design an LLM gateway.  
**A.** Single API fronting many providers: auth, routing/fallback, rate limits/quotas, caching, retries, prompt versioning, logging, cost/latency metrics.

**Q36.** Design the feedback flywheel.  
**A.** Capture thumbs/edits/regens with context, mine SFT/preference data, curate + eval-gate, retrain/improve prompts, ship behind A/B, repeat.

**Q37.** Design a multi-tenant knowledge assistant with isolation + citations.  
**A.** Per-tenant namespaces/indexes, ACL-filtered retrieval, tenant-scoped keys/quotas, grounded answers with citations, per-tenant eval and audit logs.

**Q38.** Design CI/CD for prompts and RAG indexes.  
**A.** Version prompts/configs/index, run eval suite on PRs as a gate, canary/A-B rollout, monitor online metrics, and one-click rollback.

### Debugging

**Q39.** Retrieval recall@k high but answers miss the point — why?  
**A.** Generation-side: context too long/noisy (reorder, rerank, trim), weak grounding prompt, or lost-in-the-middle; fix prompt + reduce/curate context.

**Q40.** Latency spikes when context is long — diagnose.  
**A.** Prefill cost grows with tokens; trim/rerank context, cap k, cache, chunked prefill, and summarize history instead of stuffing it.

**Q41.** Agent loops forever or calls the wrong tool — fix.  
**A.** Add step/budget limits and loop detection, tighten tool descriptions + schemas, validate args, few-shot correct usage, and fall back to ask-the-user.

**Q42.** Quality silently degraded over a month with no code change — why?  
**A.** Data/model drift: stale or growing index, provider model update, changing query mix; add continuous eval, version pins, and drift alerts.

---

## Stage 7 — Advanced Specialization

### Cross-Track Fundamentals

**Q1.** What is a Mixture-of-Experts model and why help?  
**A.** Replace the MLP with many experts and route each token to a few; huge total capacity at low per-token FLOPs.

**Q2.** Why is long context hard (two reasons)?  
**A.** Attention compute/memory grows with sequence length, and quality drops (lost-in-the-middle, position extrapolation) even when it fits.

**Q3.** What is test-time compute and why improve reasoning?  
**A.** Spend more inference compute (longer CoT, sampling, search/verification) to explore and check more solution paths, raising accuracy.

**Q4.** How do multimodal models get images into an LLM?  
**A.** A vision encoder embeds the image into patch features, a projector maps them into the token space, and they're prepended as soft tokens.

**Q5.** What is mechanistic interpretability trying to do?  
**A.** Reverse-engineer the algorithms inside the network — features and circuits — to explain behavior, not just predict it.

**Q6.** What is synthetic data and model collapse?  
**A.** Model-generated training data; collapse is the degradation from training on too much self-generated data, losing tail diversity.

### Track A · Mixture-of-Experts

**Q7.** How does top-k routing work; why only k experts?  
**A.** A gate scores experts per token and picks the top-k (often 2); only k run, so FLOPs stay low while total capacity is large.

**Q8.** Why huge total params but low per-token FLOPs in MoE?  
**A.** Params scale with the number of experts (all stored), but each token activates only k, so compute scales with k not the total.

**Q9.** What does the load-balancing loss prevent?  
**A.** Expert collapse — an auxiliary loss spreads tokens across experts so a few don't dominate while others are starved.

**Q10.** Design an MoE layer.  
**A.** Linear router -> top-k softmax over E experts, capacity factor with token dropping/overflow, expert-parallel dispatch, gate-weighted combine, plus an auxiliary load-balance loss.

**Q11.** Why are MoEs harder to serve than dense?  
**A.** All experts must be in memory (high VRAM), routing is dynamic/imbalanced, and expert-parallel all-to-all adds communication.

**Q12.** Dense vs 8x7B MoE at matched active params — FLOPs/memory?  
**A.** Similar active FLOPs per token, but the MoE stores ~all experts (much more memory) for higher capacity at the same compute.

**Q13.** Implement top-2 gating + dispatch with balance loss.  
**A.** softmax gate -> top-2 experts, dispatch tokens, weight outputs by gate probs; add aux loss = N * sum(frac_tokens * mean_prob) per expert.

### Track B · Long Context

**Q14.** How does RoPE enable extrapolation and where break?  
**A.** Relative rotations generalize somewhat past training length, but rotation phases go out-of-distribution, so raw extrapolation degrades — needs interpolation/YaRN.

**Q15.** What is lost-in-the-middle?  
**A.** U-shaped recall: models use info at the start and end of a long context well but miss content in the middle.

**Q16.** What is sliding-window attention and what it sacrifices?  
**A.** Each token attends only to the last W tokens (linear cost) but loses direct access to distant context (recovered partly via depth/sinks).

**Q17.** Extend a 4K model to 128K without full retraining — plan?  
**A.** RoPE position interpolation/YaRN + short long-context fine-tune, efficient attention (Flash/sliding+global), KV strategy, needle-in-haystack eval.

**Q18.** KV-cache strategies for very long context?  
**A.** GQA/MQA, KV quantization, sliding window + attention sinks, eviction/H2O, paging/offload, or retrieval instead of full attention.

**Q19.** Show attention compute/memory growth with sequence length.  
**A.** Scores are T x T: compute O(T^2 d), memory O(T^2); doubling T quadruples both — the long-context bottleneck.

**Q20.** Implement RoPE position interpolation + needle eval.  
**A.** Scale positions by train_len/target_len before RoPE (or YaRN), then plant a fact at varying depths and measure retrieval accuracy.

### Track C · Reasoning & Test-Time Compute

**Q21.** Outcome RM vs process RM — difference, why PRMs help?  
**A.** ORM scores only the final answer; PRM scores each reasoning step, giving denser feedback and catching flawed paths that reach right answers.

**Q22.** How do best-of-n and self-consistency trade compute for accuracy?  
**A.** Generate many candidates and pick via verifier (best-of-n) or majority vote (self-consistency); more samples raise accuracy with diminishing returns.

**Q23.** What changed conceptually with o1/R1-style models?  
**A.** Trained (often via RL on verifiable rewards) to produce long internal reasoning chains, scaling accuracy with test-time thinking.

**Q24.** Design an RL pipeline for reasoning on verifiable math/code.  
**A.** Sample CoT solutions, score with automatic verifiers (tests/checker) as reward, optimize with PPO/GRPO, filter, iterate; guard against reward hacking.

**Q25.** Test-time scaling curve and its limits?  
**A.** Accuracy rises with log of samples/thinking then plateaus; limited by generator coverage and verifier quality (a bad verifier caps gains).

**Q26.** Expected best-of-n accuracy with a perfect verifier?  
**A.** 1 - (1-p)^n, where p is per-sample success — the chance at least one of n samples is correct.

**Q27.** Implement best-of-n with a verifier on GSM8K.  
**A.** Sample n CoT answers, run the verifier/checker on each, return any that passes (or the majority); report pass rate vs n.

### Track D · Multimodal

**Q28.** What does CLIP's contrastive objective learn?  
**A.** A shared image-text space: maximize similarity of matched pairs and minimize mismatched (InfoNCE), enabling zero-shot via text prompts.

**Q29.** Vision-encoder -> projector -> LLM architecture?  
**A.** A pretrained vision encoder yields patch features, an MLP/attention projector maps them into token space, and the frozen/tuned LLM consumes them as tokens.

**Q30.** How are images tokenized/patchified?  
**A.** Split into fixed patches (e.g. 16x16), linearly embed each into a vector + position; patches become the visual tokens.

**Q31.** Design a VLM and its two-stage training.  
**A.** Vision encoder + projector + LLM; stage 1 align the projector on image-caption pairs (freeze others), stage 2 instruction-tune on visual QA.

**Q32.** How evaluate VLMs and detect visual hallucination?  
**A.** Task benchmarks (VQA/captioning) plus grounding/POPE-style object-existence probes and human checks for objects described but not present.

**Q33.** Wire a frozen vision encoder + projector to an LLM for VQA.  
**A.** Encode the image, project patches to LLM token dim, prepend as soft tokens before the text prompt, and train only the projector first.

### Track E · Interpretability

**Q34.** What is the logit lens / activation patching?  
**A.** Logit lens decodes intermediate residuals via the unembedding to see evolving predictions; activation patching swaps activations between runs to find causal components.

**Q35.** What is an induction head?  
**A.** An attention head implementing copy-completion: find a prior occurrence of the current token and predict what followed it — the basis of in-context learning.

**Q36.** What is superposition and why do SAEs help?  
**A.** Networks pack more features than neurons by overlapping them, so neurons are polysemantic; sparse autoencoders unpack activations into monosemantic features.

**Q37.** Design an experiment to locate and validate a circuit.  
**A.** Form a hypothesis, localize with activation patching/path patching, validate by ablation (breaks behavior) and necessity/sufficiency tests across inputs.

**Q38.** How could interpretability improve safety/debugging?  
**A.** Detect deception/backdoors, audit reasoning, find failure causes, and enable targeted edits — moving from black-box guessing to mechanism-level fixes.

**Q39.** Find induction heads via activation patching.  
**A.** On a repeated random sequence, patch each head's activations and measure recovery of next-token copying; high-effect heads are induction heads.

### Track F · Synthetic Data

**Q40.** What is self-instruct and rejection-sampling fine-tuning (RFT)?  
**A.** Self-instruct bootstraps instructions/responses from the model itself; RFT samples many answers, keeps only verified-correct ones, and fine-tunes on them.

**Q41.** What is model collapse and what causes it?  
**A.** Iterative training on model output narrows the distribution and loses tails; caused by compounding sampling bias and missing real-data diversity.

**Q42.** How filter synthetic data for quality and diversity?  
**A.** Verify correctness (tests/checker/judge), dedup and embed-cluster for diversity, filter toxicity/PII, and balance the mixture; keep real data in the loop.

**Q43.** Design a synthetic-data flywheel without collapse.  
**A.** Generate -> verify/filter -> keep real anchor data -> fine-tune -> eval for diversity/quality -> iterate, monitoring distribution drift each round.

**Q44.** When does distillation from a teacher beat human data?  
**A.** When a stronger teacher is available and the task is verifiable/coverable — cheaper, scalable, consistent labels; human data wins for novel or teacher-weak domains.

**Q45.** Generate + filter self-instruct data, fine-tune, compare to baseline.  
**A.** Prompt the model for diverse instructions+answers, verify/dedup/filter, fine-tune, and A/B against a human-data model on a held-out eval.

---

## Stage 8 — Safety & Security

### Fundamentals

**Q1.** What is prompt injection?  
**A.** Malicious instructions in the input that override the system prompt, making the model ignore rules or follow the attacker (LLM01).

**Q2.** Direct vs indirect prompt injection?  
**A.** Direct: the user types it. Indirect: planted in third-party content (web, PDF, retrieved doc) that a normal user unknowingly triggers.

**Q3.** Why can't an LLM reliably separate instructions from data?  
**A.** Everything is one token stream with no privilege boundary; the model treats persuasive data as instructions — no hard code/data separation.

**Q4.** What is jailbreaking?  
**A.** Crafting prompts (role-play, obfuscation, many-shot) that bypass safety training to elicit disallowed content.

**Q5.** What is training-data poisoning?  
**A.** Injecting malicious/backdoored examples into training/fine-tuning data so the model misbehaves or carries a trigger (LLM03).

**Q6.** What is insecure output handling?  
**A.** Downstream code trusting LLM output and feeding it to eval/SQL/shell/HTML — enabling injection/XSS/RCE (LLM02).

**Q7.** What is excessive agency?  
**A.** Giving an LLM/agent too much permission/autonomy so a bad output causes real damage (delete data, send money) (LLM08).

**Q8.** What is PII and why care?  
**A.** Personally identifiable information; leaking it via memorization, logs, or context breaks privacy law and trust — must detect, minimize, redact.

**Q9.** What is the OWASP Top 10 for LLM Apps?  
**A.** A standard list of top LLM risks (prompt injection, insecure output, poisoning, data leakage, excessive agency, etc.) for threat modeling.

**Q10.** What is a model-extraction attack?  
**A.** Querying an API to clone the model or steal data/weights/behavior; mitigate with rate limits, monitoring, watermarking, output limits.

### Core

**Q11.** Why is indirect injection dangerous for RAG/agents?  
**A.** Retrieved/browsed content enters the prompt as trusted text and can hijack tools/actions with no user awareness — a wide attack surface.

**Q12.** Layered defenses against prompt injection?  
**A.** Separate/delimit untrusted data, instruction hierarchy, input/output filters, least-privilege tools, human approval, and injection red-teaming — no single fix.

**Q13.** How does least privilege apply to an agent's tools?  
**A.** Grant only the minimal scoped tools/permissions needed, read-only by default, per-action authz, so a hijack can't escalate to destructive actions.

**Q14.** How can dedup/filtering reduce privacy leakage?  
**A.** Deduping rare sequences and filtering PII cuts memorization of unique secrets, lowering regurgitation risk (with DP/eval as backups).

**Q15.** Exploit and fix for insecure output handling?  
**A.** Exploit: model output run as SQL/HTML/shell -> injection/XSS/RCE. Fix: treat output as untrusted — validate, escape, parameterize, sandbox.

**Q16.** Balance safety refusals vs helpfulness; measure both?  
**A.** Track over-refusal (false positives on benign) and attack-success-rate together; tune thresholds and report both, not one in isolation.

**Q17.** Supply-chain risks for models?  
**A.** Compromised base weights, poisoned datasets, malicious deps/pickles, trojaned adapters; mitigate with provenance, hashes, scanning, safe formats.

**Q18.** How red-team an LLM feature before launch?  
**A.** Systematic adversarial probing (injection, jailbreak, PII, harmful, tools) with measured ASR, automated + human attacks, fixes, and a regression suite.

**Q19.** What logging/redaction do you need?  
**A.** Capture prompts/outputs/tool calls for audit while redacting PII/secrets, with access controls, retention limits, and tamper-evident security logs.

**Q20.** Map an agentic RAG app to its top-3 OWASP-LLM risks.  
**A.** Indirect prompt injection (LLM01) via retrieved docs, insecure output handling (LLM02) into tools, and excessive agency (LLM08) on actions.

### Senior / Staff

**Q21.** Defense-in-depth for an agent that browses and acts?  
**A.** Treat web content as untrusted (delimit), least-privilege scoped tools, output validation, human approval for high-impact actions, sandboxing, monitoring, red-team.

**Q22.** Injection hidden in a PDF your RAG ingests — trace and mitigate.  
**A.** Trace ingestion -> retrieval -> prompt; sanitize/strip instructions, isolate retrieved text, require grounding+citations, constrain tools, detect anomalies.

**Q23.** Build a continuous red-teaming + safety-eval program.  
**A.** Maintained adversarial suite + automated attack generation, ASR/over-refusal dashboards, gate releases, incident loop feeding new tests, periodic human red teams.

**Q24.** Model regurgitates secrets — root cause and remediation.  
**A.** Memorized duplicated/sensitive training data; dedup + PII filtering, DP or unlearning, output PII scanning, and stronger pretraining data governance.

**Q25.** Design the trust & safety layer for a public chatbot.  
**A.** Input guardrails (injection/abuse), output moderation, rate limits/abuse detection, PII redaction, audit logging, escalation, and continuous safety eval.

**Q26.** Secure a third-party tool/plugin ecosystem?  
**A.** Review/sign plugins, sandbox + scoped permissions, validate schemas/outputs, rate-limit, monitor, and require user consent for sensitive scopes.

**Q27.** Safety classifier over-blocks legitimate queries — fix safely?  
**A.** Measure false-positive/over-refusal, add hard-negative benign data, calibrate thresholds, use graded responses/context, keep ASR flat while reducing false blocks.

**Q28.** Threat-model an LLM system end to end.  
**A.** Enumerate trust boundaries (input, retrieval, model, output, tools), map OWASP-LLM risks per boundary, rank by impact x likelihood, add layered controls + monitoring.

### Coding

**Q29.** Implement an input guardrail for injection/jailbreak.  
**A.** Flag known patterns (ignore previous, system-prompt override, role-play unlock) via rules + a classifier; block or sanitize and log before the model.

**Q30.** Implement a PII detector + redactor.  
**A.** Regex for emails/phones/SSN/cards + NER for names/addresses; replace matches with typed placeholders before logging or prompting.

**Q31.** Implement an output handler that refuses raw exec/SQL/shell.  
**A.** Never eval model text; validate against an allow-list/schema, parameterize queries, escape for the sink, and sandbox any execution.

**Q32.** Implement a tool allow-list + argument schema validator.  
**A.** Check the requested tool is permitted, validate args against its schema/types/ranges, deny unknown tools/args, and log the decision.

**Q33.** Implement an attack-success-rate harness.  
**A.** Run a labeled attack set through the system, auto-judge whether each elicited the bad behavior, and report ASR overall and per category.

### System Design

**Q34.** Design a guardrail service (input + output).  
**A.** Pipeline of injection/PII/topic/format/moderation checks pre- and post-model, fail-closed with policies, low-latency, configurable, logged and versioned.

**Q35.** Design data governance for a fine-tuning pipeline.  
**A.** Source provenance + consent, PII scanning/redaction, poisoning checks/dedup, access controls, dataset versioning/lineage, and approval gates.

**Q36.** Design monitoring/alerting for safety incidents.  
**A.** Real-time guardrail/abuse/PII/ASR metrics with thresholds and anomaly detection, alerts to on-call, audit trails, dashboards, and an incident-response runbook.

### Debugging

**Q37.** Role-play jailbreak discovered overnight — response plan?  
**A.** Contain (patch filter/system prompt), measure scope via ASR, add the exploit to red-team tests, deploy fix, monitor, and postmortem for root cause.

**Q38.** Agent with DB access ran a destructive query — what failed, prevent?  
**A.** Excessive agency + insecure output; enforce least privilege/read-only, parameterized queries, approval for writes, sandboxing, and audit logging.

**Q39.** RAG started citing attacker-planted documents — respond.  
**A.** Treat ingestion as untrusted: vet/sign sources, sanitize content, detect anomalous docs, require grounding, and purge poisoned entries from the index.

**Q40.** Logs containing user PII were exposed — remediate.  
**A.** Contain/rotate access, purge/redact, notify per policy; fix by redacting at capture, encrypting, access controls, and retention limits going forward.

---

## System Design — ChatGPT, RAG & Inference (HLD)

> Cross-cutting ML-systems design track. Full write-ups: [ChatGPT HLD](../system-design/chatgpt/README.md) · [RAG platform](../system-design/rag-platform/README.md) · [LLM inference service](../system-design/llm-inference/README.md).

### Design ChatGPT

**Q1.** Design ChatGPT: what is the unit of work, and why does it reshape the design?  
**A.** A stateful, multi-second, GPU-bound token stream (not a cheap stateless request) -> optimize GPU utilization (MFU) and tail latency (TTFT/TPOT).

**Q2.** TTFT vs TPOT -- what does each govern?  
**A.** TTFT (prefill+queue) governs responsiveness; TPOT (decode speed) governs how fast the answer flows and must beat reading ~15-40 tok/s.

**Q3.** Why stream tokens over SSE instead of returning the whole completion?  
**A.** A long answer takes seconds; streaming shows the first token in ~hundreds of ms, matches token-by-token decode, and enables incremental moderation + early cancel.

**Q4.** Why keep the chat data plane stateless?  
**A.** Stateless gateways/orchestrators scale horizontally and fail over trivially; conversation state lives in a replicated store, only the in-flight KV cache pins to a replica.

**Q5.** List the layers a chat request passes through, client to GPU and back.  
**A.** Client -> CDN/WAF -> gateway (auth/limit) -> orchestrator -> input safety -> context build (history+memory+RAG) -> router -> scheduler -> GPU (prefill->decode) -> output safety -> SSE back.

**Q6.** How do you assemble context when history exceeds the window?  
**A.** Recent turns verbatim + a rolling summary of older turns + retrieved memory/RAG chunks, budgeted by relevance and recency; prefix-cache the stable system prompt.

**Q7.** Why route between multiple model sizes instead of always the biggest?  
**A.** Most queries are easy; route them to a small cheap model and reserve the flagship for hard ones -- often the biggest cost lever -- plus A/B and load fallback.

**Q8.** Where does safety sit in the chat path (defense in depth)?  
**A.** Input moderation/injection/PII -> untrusted retrieval+tool boundary -> aligned generation with scoped tools -> output guard + stream-time moderation; measure over-refusal vs attack-success-rate.

**Q9.** What is the data flywheel for a chat product?  
**A.** log -> curate/PII-scrub -> label/preferences -> SFT -> RM -> RLHF/DPO -> offline eval+red-team gate -> canary -> A/B -> ramp, with a registry and 1-click rollback.

**Q10.** What storage backs each data type in a chat system?  
**A.** Conversations -> wide-column KV by user; vectors -> sharded ANN; docs -> object+search; accounts/billing -> Postgres; sessions/quotas -> Redis; events -> Kafka->lake; weights -> registry.

**Q11.** Make a streaming chat request survive a replica failure mid-generation.  
**A.** The stream is stateful (KV on that replica): fail cleanly + idempotent client retry, or checkpoint progress and resume elsewhere; drain for planned restarts.

**Q12.** Capacity: 100M DAU x 10 msgs/day -> peak request and token rates?  
**A.** 1B msgs/day ~12K req/s avg, ~40K req/s peak; x ~500 output tokens ~20M output tok/s peak -- the number that sizes the GPU fleet.

**Q13.** Cut chat serving cost per token by 2x without hurting quality -- in order.  
**A.** Route easy traffic to a small model -> quantize + MoE -> raise MFU (batching/paging/speculation) -> cache -> cap output/context -> reserved + spot capacity.

**Q14.** Under a GPU shortage, what beats hard-failing requests?  
**A.** Graceful degradation: fall back to a smaller/quantized model, shorten context, queue honestly with backpressure, and use a model fallback chain.

### RAG Platform

**Q15.** What problem does RAG solve over a plain LLM call?  
**A.** It fetches relevant evidence at query time and grounds the answer in it -- fixing stale knowledge, hallucination, and attribution (citations) without retraining.

**Q16.** Fine-tune vs retrieve -- when do you use each?  
**A.** Fine-tune to change behavior/format/skill; retrieve to supply knowledge (fresh, attributable, access-controlled). They compose.

**Q17.** Dense vs sparse retrieval -- what does each catch?  
**A.** Dense (embeddings) captures semantics/paraphrase; sparse (BM25) nails exact terms/IDs/rare words. Hybrid fuses both since each covers the other's blind spot.

**Q18.** What is reranking and why add it?  
**A.** A cross-encoder scores query+doc jointly on the top-N candidates (e.g. 100->8) for a precision lift the fast bi-encoder can't give; cost is GPU latency.

**Q19.** Why must RAG access control happen at retrieval, not generation?  
**A.** The LLM summarizes anything in its context; if an unauthorized doc reaches the prompt it can leak -- so filter to authorized docs during retrieval.

**Q20.** Two planes of a RAG platform, and why separate them?  
**A.** Ingestion (async, throughput: parse->chunk->embed->index) vs query (sync, latency: rewrite->retrieve->filter->rerank->generate); opposite SLOs, meet at the shared index.

**Q21.** Why is chunking the highest-leverage RAG knob?  
**A.** The chunk is both what you match and what you feed the model: too small fragments context, too big dilutes the embedding and wastes tokens. Tune against the eval set.

**Q22.** How do you combine dense and sparse results?  
**A.** Reciprocal Rank Fusion: sum 1/(k+rank) across retrievers -- rank-based so you don't calibrate incomparable scores; docs ranked high by both float up.

**Q23.** Pre-filter vs post-filter for ACLs/metadata in retrieval?  
**A.** Post-filter (retrieve then drop) wrecks recall and can leak existence; pre-filter inside the ANN keeps k valid results and is secure -- default to pre-filter for ACLs.

**Q24.** How do you keep a RAG index fresh and handle deletes?  
**A.** CDC streams changes -> re-embed only deltas, upsert by stable chunk id, version embeddings; deletes = tombstone now (exclude) + background purge across all stores.

**Q25.** How do you evaluate a RAG system?  
**A.** Separately: retrieval (recall@k, MRR, nDCG vs a golden set) and generation (faithfulness, answer relevance, citation correctness); offline CI gate + online thumbs/abstention.

**Q26.** Right doc is in the corpus but never appears in the answer -- diagnose.  
**A.** Walk the funnel: indexed? retrieved (raw candidates)? filtered out (over-broad ACL)? reranked down? or in-context but ignored (too long/poor order)? Fix the stage that drops it.

**Q27.** Upgrade the embedding model with zero downtime -- how?  
**A.** It invalidates every vector: dual-index -- keep serving old while a background job re-embeds the corpus into a new index, validate, then atomically swap (needs versioned vectors).

**Q28.** Scale a vector index to 10B+ chunks?  
**A.** Shard (scatter-gather + merge) and replicate; PQ-compress to fit memory; co-locate metadata for filtered search; dedicated shards for whale tenants; compact to fight fragmentation.

**Q29.** HNSW vs IVF-PQ at platform scale?  
**A.** HNSW: great recall/latency, high memory, fragments on updates. IVF-PQ: compressed/scalable, needs centroid training, lower recall. Pick by memory/scale vs quality/latency.

**Q30.** A grounded RAG system still hallucinates -- drive it down?  
**A.** Most hallucinations are retrieval misses -> fix retrieval first; then tighten the grounding prompt, verify citations, enable abstention, and measure faithfulness continuously.

**Q31.** Defend RAG against indirect prompt injection?  
**A.** Retrieved content is untrusted: delimit/label it as data, instruct the model to ignore instructions in context, sanitize, run least-privilege tools, and guard outputs.

**Q32.** Estimate storage for 1B chunks of 1024-dim fp16 vectors.  
**A.** 1024 x 2 B = 2 KB/vector -> 2 TB raw; HNSW overhead ~2-3x -> ~5 TB, sharded across ~50-100 nodes; PQ cuts vectors ~8-16x.

### LLM Inference Service

**Q33.** Why is decode memory-bandwidth-bound while prefill is compute-bound?  
**A.** Prefill runs all prompt tokens in one parallel pass (matmul-heavy, reuses weights); decode emits one token/step, re-reading all weights from HBM for little math.

**Q34.** Arithmetic intensity, and why it forces batching.  
**A.** Decode at batch 1 has intensity ~1 (move ~all weights per token) -> memory-bound, low MFU. Batching B reuses each weight load B times -> intensity ~B -> higher MFU.

**Q35.** What is continuous (in-flight) batching?  
**A.** Token-granular scheduling: each step finished sequences exit and queued ones join the running batch -- keeps the GPU busy despite varied lengths; biggest throughput lever.

**Q36.** Why does the KV cache cap concurrency?  
**A.** It grows linearly with context x batch x layers x KV-heads, so at long context/high concurrency it -- not the weights -- runs out of HBM first, bounding batch size.

**Q37.** What is MFU and why is it the cost budget?  
**A.** Model FLOPs utilization = fraction of peak FLOPs doing useful work; cost is GPU-seconds, so $/token scales as 1/MFU -- the engine exists to raise MFU at the SLO tail.

**Q38.** What does PagedAttention solve?  
**A.** Naive per-sequence contiguous KV fragments HBM; paging stores KV in fixed non-contiguous pages with a block table -> ~no fragmentation, higher batch, prefix sharing.

**Q39.** The inference scheduler's three jobs?  
**A.** Admission control (saturated -> queue/429, don't miss SLO), priority queues (interactive>batch, paid>free, fair), and KV-bounded batch formation with chunked prefill.

**Q40.** How do you serve thousands of fine-tunes cheaply?  
**A.** Multi-LoRA: keep one frozen base resident, apply per-request LoRA adapters (MBs) in-kernel, batch different adapters together -> thousands of fine-tunes at ~base-model cost.

**Q41.** Why an OpenAI-compatible API for an inference service?  
**A.** The ecosystem already speaks /v1/chat/completions, so it is drop-in: zero client rewrites, instant tooling/SDK support; you compete on price/latency/models, not protocol.

**Q42.** What signals should you autoscale a GPU inference fleet on?  
**A.** Queue depth, TTFT, and MFU/GPU saturation -- never CPU. GPUs start in minutes, so add predictive scaling + warm pools; reserved baseline + burst + spot for batch.

**Q43.** What does prefix-aware routing buy you?  
**A.** Routing requests that share a long prefix (system prompt / conversation) to the same replica hits its prefix cache, skipping re-prefill -> lower TTFT and prefill compute.

**Q44.** Why disaggregate prefill and decode onto separate pools?  
**A.** Opposite bottlenecks that interfere (a big prefill stalls decodes -> p99); separate pools scale/tune independently, at the cost of shipping the KV cache between them. Else chunked prefill.

**Q45.** Walk speculative decoding -- and when does it not help?  
**A.** Draft proposes k tokens, target verifies in one pass, accept w/ min(1, p_t/p_d), resample residual -> 2-3x fewer steps. Fails if acceptance low or the batch is already compute-saturated.

**Q46.** A few long prompts are tanking everyone's TPOT -- fix?  
**A.** Head-of-line blocking + KV hogging: chunked prefill, length-aware scheduling / separate pool, disaggregate prefill/decode, cap context, and admission-limit concurrent long requests.

**Q47.** Estimate single-stream decode latency for 70B fp8 on 8xH100.  
**A.** Bandwidth-bound: 70 GB weights / ~26 TB/s ~2.7 ms/token (~370 tok/s); batched ~10K tok/s/node -- why batching is mandatory.

**Q48.** KV bytes per sequence -- the formula and what it caps.  
**A.** 2 x L x h_kv x d_head x seq x dtype (2 = K and V). At long context it caps batch, so batch formation is KV-memory-bounded; shrink with GQA/MQA + KV-quant.

**Q49.** Speculative decoding made throughput worse -- why?  
**A.** It adds draft+verify cost and only pays off at high acceptance with spare compute; under large batches the GPU is already saturated -- disable speculation when the batch is big.

**Q50.** Scale-to-zero cold models without hurting warm-model SLOs?  
**A.** Pin hot models warm; cold models load on demand (accept a cold start); keep a shared warm GPU pool, predictively preload scheduled models, and isolate loading from warm replicas.

---

[← Back to flashcards](README.md) · [Index](../README.md)