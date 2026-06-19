# Stage 5 — Inference Optimization

> **Objective:** Serve LLMs **fast and cheap**, and reason precisely about latency, throughput, and memory tradeoffs. Training gets the headlines; inference is where the money is spent — usually 10× more than training over a model's lifetime.

[← Stage 4](../stage-4-evaluation/README.md) · [Index](../README.md) · Next: [Stage 6 — LLMOps](../stage-6-production-llmops/README.md)

📝 **Interview prep:** [interview-questions.md](interview-questions.md) · ✅ [answer key](answers.md)

---

## Why this stage matters

A model that's accurate but too slow or expensive is unshippable. Inference optimization is the highest-ROI engineering skill in applied LLM work, and it's where deep understanding of the architecture (Stage 1) pays off directly.

---

## Mental model

Generation has **two phases** with totally different performance profiles:

```
PREFILL  — process the whole prompt at once → compute-bound (big matmuls, GPU-saturating)
DECODE   — generate one token at a time     → memory-bandwidth-bound (tiny matmuls, waiting on memory)
```

Most user-facing latency is **decode**, and decode is bottlenecked by **memory bandwidth** (moving weights + KV-cache), not raw FLOPs. Almost every optimization below attacks memory movement.

---

## Concept-by-concept deep dive

### 5.1 Decoding strategies (how tokens are chosen)
- **Greedy:** always argmax. Deterministic, can be repetitive/bland.
- **Beam search:** keep top-b sequences. Good for translation; poor for open-ended chat (bland, costly).
- **Temperature:** scales logits before softmax. `T<1` sharpens (safer), `T>1` flattens (more creative).
- **Top-k:** sample from the k highest-prob tokens.
- **Top-p (nucleus):** sample from the smallest set whose cumulative prob ≥ p. Adapts to the distribution's shape — the common default.
- **min-p, repetition/frequency penalties:** further control quality and reduce loops.
- **Know:** these change *quality and determinism*, not model weights. Reproducibility needs fixed seed + greedy or fixed sampling params.

### 5.2 The KV cache (the central inference concept)
- During decode, attention needs the keys/values of **all previous tokens**. Recomputing them every step is wasteful, so we **cache** them.
- **KV-cache size** ≈ `2 (K&V) × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes`. It grows **linearly with context length and batch** and can exceed the model weights for long contexts.
- This cache is the main memory pressure during serving and the reason long context is expensive.

### 5.3 Attention efficiency: GQA / MQA / FlashAttention
- **MQA (multi-query):** all query heads share **one** K/V head → much smaller KV-cache, faster decode, small quality cost.
- **GQA (grouped-query):** middle ground — groups of query heads share K/V heads (LLaMA-2/3). The modern default.
- **FlashAttention:** an **IO-aware** exact-attention kernel. It avoids materializing the full `(T×T)` score matrix in slow HBM by tiling and recomputing in fast SRAM → big speed + memory wins. Same math, better memory choreography. (Connects to "decode is memory-bound.")

### 5.4 Quantization (smaller weights = faster, cheaper)
- **Idea:** store/compute weights in fewer bits (fp16 → int8 → int4).
- **Why it helps decode:** less memory to move per token → directly faster in the bandwidth-bound regime.
- **Methods:**
  - **GPTQ:** post-training, layer-wise error-minimizing quantization to 4-bit.
  - **AWQ:** activation-aware — protects the most salient weight channels; strong 4-bit quality.
  - **GGUF (llama.cpp):** quantized format for CPU/edge/local inference.
  - **Weight-only vs weight+activation (e.g., SmoothQuant):** activation quantization is harder (outliers).
- **Tradeoff:** lower bits → less memory/faster, but potential accuracy loss. 4-bit weight-only is often near-lossless; push lower carefully and **measure** (Stage 4).

### 5.5 Speculative decoding (break the sequential bottleneck)
- Decode is sequential (one token per forward pass). **Speculative decoding** uses a small fast **draft model** to propose several tokens, then the big model **verifies them in parallel** in one pass; accepted tokens are kept.
- **Crucially, it preserves the exact output distribution** of the big model — it's a speedup, not an approximation.
- Variants: **Medusa** (extra prediction heads instead of a separate draft model), EAGLE.

### 5.6 Serving systems & batching
- **Continuous (in-flight) batching:** instead of waiting to batch requests statically, add/remove sequences from the batch every step → massively higher throughput under concurrent load.
- **PagedAttention (vLLM):** manage the KV-cache like OS virtual memory — non-contiguous "pages" → near-zero fragmentation, higher batch sizes, prefix sharing.
- **Serving stacks to know:** **vLLM** (throughput king), **TGI** (HF), **TensorRT-LLM** (NVIDIA, max perf), **llama.cpp/Ollama** (local/edge), **SGLang** (structured + fast).

### 5.7 Distillation & pruning (make a smaller model)
- **Knowledge distillation:** train a small "student" to mimic a large "teacher" (logits/behavior). Cheaper to serve.
- **Pruning:** remove low-importance weights/heads/layers; structured pruning gives real speedups.
- Use when quantization isn't enough and you can afford a training step.

### 5.8 The metrics that matter
- **TTFT** (time to first token) — dominated by prefill.
- **TPOT / ITL** (time per output token / inter-token latency) — dominated by decode.
- **Throughput** (tokens/sec aggregate) — for batch/offline.
- **Latency vs throughput tradeoff:** bigger batches → more throughput but higher per-request latency. Tune to your SLA.
- **Cost per 1M tokens:** the business metric.

---

## Ordered learning path

1. Read **FlashAttention** (v1 then v2) — IO-aware thinking.
2. Read the **vLLM / PagedAttention** paper.
3. Read **GPTQ** and **AWQ**.
4. Read the **speculative decoding** paper.
5. Do the labs (benchmark everything).

---

## 🛠️ Hands-on labs

- [ ] **Lab A — Serve with vLLM:** stand up your model; measure **TTFT**, **TPOT**, and throughput under concurrency.
- [ ] **Lab B — Quantization study:** run fp16 vs int8 vs int4 (GPTQ/AWQ). Report speed, memory, and a quality delta (reuse Stage-4 evals).
- [ ] **Lab C — KV-cache math:** compute and then *measure* KV-cache growth vs context length; confirm your formula.
- [ ] **Lab D — Decoding sweep:** fix a prompt; vary temperature/top-p/top-k; observe quality vs diversity.
- [ ] **Lab E — Speculative decoding:** enable a draft model; measure speedup and confirm outputs match the target distribution.
- [ ] **Lab F — Throughput vs latency:** sweep batch size; plot the tradeoff curve and pick an SLA operating point.

---

## ⚠️ Common pitfalls & gotchas

- Optimizing FLOPs when decode is **memory-bandwidth-bound** — wrong bottleneck.
- Ignoring KV-cache memory → OOM at long context or high concurrency.
- Quantizing without an eval → silent quality regression.
- Reporting average latency only (report **p50 and p99**).
- Comparing serving stacks at different batch sizes / settings (apples-to-oranges).
- Assuming beam search helps chat (it usually hurts).
- Forgetting prefill vs decode have different bottlenecks when profiling.

---

## 🔥 Mastery checks (answer without notes)

- [ ] Why is decode memory-bandwidth-bound while prefill is compute-bound?
- [ ] Compute KV-cache size for given `layers, heads, head_dim, seq_len, batch, dtype`.
- [ ] How do MQA/GQA reduce KV-cache, and what's the tradeoff?
- [ ] Why does FlashAttention speed things up *without* changing the math?
- [ ] Explain how speculative decoding preserves the exact output distribution.
- [ ] How does PagedAttention raise achievable batch size?
- [ ] Choose a quantization method for (a) accuracy-critical and (b) latency-critical cases; justify.
- [ ] Explain the latency↔throughput tradeoff and how batch size moves it.

---

## ✅ Stage 5 checklist

- [ ] Read FlashAttention, vLLM, GPTQ/AWQ, speculative decoding
- [ ] Labs A–C complete (D–F for depth)
- [ ] Have a benchmark report (speed/memory/quality) for ≥2 quantization levels
- [ ] Can do KV-cache math from memory
- [ ] All mastery checks passable
- [ ] Notes in your own words

**When complete → proceed to [Stage 6](../stage-6-production-llmops/README.md).**
