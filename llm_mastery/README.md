# 🧠 LLM Mastery Study Plan — From Architecture to Production

> Goal: Go from "knows the prerequisites" to **elite-level LLM engineer/scientist** who can design, train, evaluate, optimize, and ship LLM systems — and reason about them from first principles.

**How to use this plan**
- Work Stages 1 → 5 **linearly** (each builds on the last). Start Stage 6 (RAG/prompting) **in parallel** early.
- Each stage has: 🎯 Objectives · 📚 Concepts (checkboxes) · 📄 Must-read papers · 🛠️ Build milestone · 🔥 Mastery checks.
- **Mastery checks are the bar for "cracked."** Don't move on until you can do them *without notes*.
- Rule of thumb: **build > read**. Every stage ends with code you wrote.
- Keep a `notes/` folder per stage and an experiments log.

**Legend:** `[ ]` todo · `[~]` in progress · `[x]` done

---

## �️ Folder Map (deep-dive modules)

Each stage folder contains: a deep-dive `README.md` (explanations, labs, pitfalls, mastery checks), a full **`interview-questions.md`** (screening → staff/principal level) with a worked **`answers.md`**, a `notes/` template, and an `experiment-log.md`. This README is the high-level index; click into a stage to go deep.

| # | Stage | Deep-dive module |
|---|-------|------------------|
| 1 | Transformer Internals | [stage-1-transformer-internals](stage-1-transformer-internals/README.md) |
| 2 | Pretraining at Scale | [stage-2-pretraining-at-scale](stage-2-pretraining-at-scale/README.md) |
| 3 | Adaptation & Alignment | [stage-3-adaptation-alignment](stage-3-adaptation-alignment/README.md) |
| 4 | Evaluation | [stage-4-evaluation](stage-4-evaluation/README.md) |
| 5 | Inference Optimization | [stage-5-inference-optimization](stage-5-inference-optimization/README.md) |
| 6 | Production / LLMOps | [stage-6-production-llmops](stage-6-production-llmops/README.md) |
| 7 | Advanced Specialization | [stage-7-advanced-specialization](stage-7-advanced-specialization/README.md) |
| 8 | Safety & Security | [stage-8-safety-security](stage-8-safety-security/README.md) |
| 🏆 | Capstone Projects | [capstones](capstones/README.md) |
| 📄 | Paper Reading List | [papers](papers/README.md) |
| 🏛️ | System Design — **7 full HLDs, each with Q&A + cheat-sheet**: [ChatGPT](system-design/chatgpt/README.md) · [RAG platform](system-design/rag-platform/README.md) · [LLM inference service](system-design/llm-inference/README.md) · [Training & fine-tuning platform](system-design/training-platform/README.md) · [Vector database](system-design/vector-database/README.md) · [Feature store](system-design/feature-store/README.md) · [Claude Code CLI](system-design/claude-code-cli/README.md) | [system-design](system-design/README.md) |
| 🗂️ | Flashcards (Anki: 85 curated · 399 full · 62 cloze · printable sheet) | [flashcards](flashcards/README.md) |
| 🧬 | Domain Applications — **LLMs in life sciences & biochemistry** (real-time apps, foundation models, buildable prototypes) | [llm-in-life-sciences](llm-in-life-sciences.md) |

> The sections below are the **summary** of each stage. Open the linked folder for the full deep-dive.

---

## �📅 Suggested Pacing (adjust to your schedule)

| Phase | Stages | Focus |
|------|--------|-------|
| Phase 1 | Stage 1–2 | Build & understand transformers + pretraining |
| Phase 2 | Stage 3–4 | Fine-tuning, alignment, evaluation |
| Phase 3 | Stage 5–6 | Inference optimization + production systems |
| Phase 4 | Stage 7 + Capstone | Specialize, ship a portfolio project |

---

## Stage 1 — Transformer Internals (the bedrock)

🎯 **Objective:** Implement a decoder-only transformer from scratch and explain every tensor shape on a whiteboard.

📚 **Concepts**
- [ ] Self-attention math: `softmax(QKᵀ/√d)·V` — derive shapes, explain the `√d` scaling
- [ ] Multi-head attention: why split heads, head dim, concat + output projection
- [ ] Causal masking & why it enables autoregressive training
- [ ] Residual connections + LayerNorm; **pre-norm vs post-norm** (and why pre-norm trains more stably)
- [ ] Feed-forward block (MLP), GELU/SwiGLU activations
- [ ] Positional encoding evolution: sinusoidal → learned → **RoPE** → ALiBi (know the tradeoffs)
- [ ] Tokenization deep dive: **BPE** algorithm, byte-level BPE, vocab size tradeoffs
- [ ] Weight tying, embedding scaling, logit computation

📄 **Papers / Resources**
- [ ] *Attention Is All You Need* (Vaswani et al., 2017)
- [ ] Karpathy — *Let's build GPT* + **nanoGPT** repo
- [ ] *The Illustrated Transformer* (Jay Alammar)
- [ ] RoPE paper (RoFormer)

🛠️ **Build milestone**
- [ ] Implement a decoder-only transformer **from scratch** (PyTorch, no `nn.Transformer`)
- [ ] Train on TinyShakespeare; generate coherent text
- [ ] Implement your own BPE tokenizer

🔥 **Mastery checks**
- [ ] Hand-derive attention output shape given batch/seq/heads/dim
- [ ] Explain why attention is O(n²) in sequence length and where the memory goes
- [ ] Explain what breaks without positional encoding
- [ ] Rebuild nanoGPT's forward pass from memory

---

## Stage 2 — Pretraining at Scale

🎯 **Objective:** Understand how frontier models are actually trained, and compute resource budgets by hand.

📚 **Concepts**
- [ ] Causal LM objective & cross-entropy loss; perplexity intuition
- [ ] Data pipeline: collection, cleaning, **dedup**, quality filtering, data mixtures
- [ ] **Scaling laws** — Kaplan vs **Chinchilla** (compute-optimal tokens-per-param ≈ 20)
- [ ] Optimizers: AdamW internals (momentum + variance), weight decay
- [ ] LR schedules: warmup + cosine decay; gradient clipping; batch size effects
- [ ] Mixed precision: **bf16 vs fp16**, loss scaling, numerical stability
- [ ] Memory math: params + gradients + optimizer states + activations
- [ ] **Distributed training**: data / tensor / pipeline parallelism
- [ ] **ZeRO** stages & **FSDP**; gradient checkpointing (compute-memory tradeoff)

📄 **Papers / Resources**
- [ ] GPT-3 paper (*Language Models are Few-Shot Learners*)
- [ ] Chinchilla (*Training Compute-Optimal LLMs*)
- [ ] ZeRO paper + PyTorch FSDP docs
- [ ] LLaMA / LLaMA-2 papers (data + architecture choices)

🛠️ **Build milestone**
- [ ] Train nanoGPT with bf16 + gradient accumulation + checkpointing
- [ ] Plot a mini scaling-law curve (loss vs params or vs tokens)
- [ ] Write a script that estimates VRAM for a given model + batch + optimizer

🔥 **Mastery checks**
- [ ] Compute memory for full fine-tuning a 7B model in bf16 with AdamW (params 14GB + optim 56GB + grads 14GB + activations…) and explain each term
- [ ] Given a compute budget (FLOPs), pick compute-optimal model size & token count
- [ ] Explain ZeRO-1/2/3 differences and what each sharding saves
- [ ] Explain why bf16 is preferred over fp16 for training stability

---

## Stage 3 — Adaptation & Alignment

🎯 **Objective:** Take a base model to an instruction-following, aligned assistant — and know which technique to use when.

📚 **Concepts**
- [ ] **SFT / instruction tuning**: dataset format, chat templates, loss masking on prompts
- [ ] Catastrophic forgetting & mitigation
- [ ] **PEFT — LoRA**: low-rank decomposition, which layers, rank/alpha, why it works
- [ ] **QLoRA**: 4-bit NF4 quantization + LoRA, double quantization, paged optimizers
- [ ] Other PEFT: adapters, prefix/prompt tuning (and why LoRA usually wins)
- [ ] **RLHF pipeline**: reward model training, **PPO** for LLMs, KL penalty
- [ ] **DPO**: derivation intuition, why it removes the reward model + RL loop
- [ ] DPO variants (IPO, KTO, ORPO), RLAIF, Constitutional AI
- [ ] Reward hacking, alignment tax

📄 **Papers / Resources**
- [ ] InstructGPT (*Training LMs to follow instructions with human feedback*)
- [ ] LoRA paper + QLoRA paper
- [ ] DPO (*Direct Preference Optimization*)
- [ ] HuggingFace TRL library docs

🛠️ **Build milestone**
- [ ] **QLoRA fine-tune** a 7B model (Llama/Mistral/Qwen) on an instruction dataset
- [ ] Run **DPO** on a preference dataset over your SFT model
- [ ] Side-by-side compare: base vs SFT vs DPO outputs on a fixed prompt set

🔥 **Mastery checks**
- [ ] Explain LoRA math: how `W + BA` works, parameter count saved at rank r
- [ ] Explain why QLoRA lets you fine-tune a 7B on a single 24GB GPU
- [ ] Explain the DPO loss and why it's equivalent in spirit to RLHF
- [ ] Decide: full FT vs LoRA vs QLoRA vs DPO for a given constraint
- [ ] Explain the KL term's role in RLHF (staying near the reference policy)

---

## Stage 4 — Evaluation (the underrated elite skill)

🎯 **Objective:** Design rigorous evals and distrust benchmarks intelligently.

📚 **Concepts**
- [ ] Intrinsic vs extrinsic metrics: perplexity vs task accuracy
- [ ] Generation metrics: BLEU, ROUGE, **pass@k**, exact match — and their failure modes
- [ ] Core benchmarks: MMLU, HellaSwag, GSM8K, HumanEval, BIG-bench, **HELM**
- [ ] **Benchmark contamination** & how to detect it
- [ ] **LLM-as-a-judge**: setup, position/verbosity/self-preference bias, mitigation
- [ ] Human eval design: rubrics, inter-annotator agreement
- [ ] Safety/robustness eval: red-teaming, toxicity, bias, **hallucination** detection
- [ ] Eval for RAG: retrieval (recall@k, MRR) vs generation faithfulness, **separately**

📄 **Papers / Resources**
- [ ] HELM (Holistic Evaluation of Language Models)
- [ ] MT-Bench / Chatbot Arena (LLM-as-judge)
- [ ] `lm-evaluation-harness` (EleutherAI)

🛠️ **Build milestone**
- [ ] Score your Stage-3 models on a public benchmark via lm-eval-harness
- [ ] Build a **custom eval set** for a narrow task you care about
- [ ] Implement an LLM-judge pipeline and **measure its bias** (swap answer order, etc.)

🔥 **Mastery checks**
- [ ] Explain why a model can ace MMLU yet fail in production
- [ ] Design an eval for a task with no ground-truth labels
- [ ] List 3 LLM-judge biases and how to control each
- [ ] Explain how to separate retrieval failures from generation failures in RAG

---

## Stage 5 — Inference Optimization

🎯 **Objective:** Serve models fast and cheap; reason about latency/throughput/memory tradeoffs.

📚 **Concepts**
- [ ] Decoding: greedy, beam, **temperature**, top-k, **top-p (nucleus)**, repetition penalty, min-p
- [ ] **KV cache**: what's cached, memory growth with context, GQA/MQA savings
- [ ] **Quantization**: int8/int4, **GPTQ / AWQ / GGUF**, weight-only vs activation, accuracy impact
- [ ] **FlashAttention**: IO-aware attention, why it's faster (memory hierarchy)
- [ ] **Speculative decoding** (draft model) & medusa-style heads
- [ ] **Continuous batching** & **PagedAttention** (vLLM)
- [ ] Prefill vs decode phases; compute-bound vs memory-bound
- [ ] Latency vs throughput; p50/p99; cost-per-token modeling
- [ ] Distillation & pruning for smaller deployable models

📄 **Papers / Resources**
- [ ] FlashAttention (v1/v2)
- [ ] vLLM / PagedAttention paper
- [ ] GPTQ + AWQ papers
- [ ] Speculative decoding paper

🛠️ **Build milestone**
- [ ] Serve your model with **vLLM**; benchmark tokens/sec + p50/p99 latency
- [ ] Compare **int4 vs fp16**: speed, memory, quality delta
- [ ] Measure KV-cache memory growth as context length increases

🔥 **Mastery checks**
- [ ] Explain why decoding is memory-bandwidth-bound and prefill is compute-bound
- [ ] Compute KV-cache size for a given model/context/batch
- [ ] Explain how speculative decoding preserves the output distribution
- [ ] Explain how PagedAttention reduces memory fragmentation
- [ ] Pick a quantization method for an accuracy-sensitive vs latency-sensitive case

---

## Stage 6 — Production Systems / LLMOps (start early, in parallel)

🎯 **Objective:** Build reliable, grounded, observable LLM applications and pick the right architecture.

📚 **Concepts**
- [ ] **Prompt engineering**: few-shot, **chain-of-thought**, ReAct, self-consistency, structured prompting
- [ ] **RAG**: chunking strategies, embedding models, vector DBs, similarity search, **reranking**, grounding/citations
- [ ] Advanced RAG: hybrid search, query rewriting, HyDE, parent-document retrieval
- [ ] **Agents**: tool/function calling, planning, multi-step orchestration, ReAct loops
- [ ] Structured output: JSON mode, schema/grammar-constrained decoding
- [ ] **Guardrails**: input/output validation, safety filters, fallbacks
- [ ] Caching (semantic + exact), cost optimization, model routing
- [ ] **Observability**: tracing, logging, prod evals, drift, A/B testing
- [ ] **Decision framework: prompt vs RAG vs fine-tune** (and combinations)

📄 **Papers / Resources**
- [ ] RAG original paper (Lewis et al.) + a modern RAG survey
- [ ] ReAct paper; Toolformer
- [ ] A framework's docs (LlamaIndex/LangChain) — **but learn the primitives, not just the API**

🛠️ **Build milestone**
- [ ] Build a **RAG pipeline** over your own docs with reranking + citations
- [ ] Add **retrieval eval** (recall@k) *separately* from generation eval
- [ ] Build a small **agent** with 2–3 tools and function calling

🔥 **Mastery checks**
- [ ] Given a use case, justify prompt vs RAG vs fine-tune with tradeoffs
- [ ] Diagnose a RAG failure: is it retrieval or generation? How do you prove it?
- [ ] Explain chunking tradeoffs (size vs context vs precision)
- [ ] Design observability for a prod LLM app (what to log & alert on)

---

## Stage 7 — Advanced / Specialization (pick your edges)

📚 **Tracks (choose 1–2 to go deep)**
- [ ] **Mixture of Experts (MoE)**: routing, load balancing, sparse compute (Mixtral, DeepSeek)
- [ ] **Long context**: RoPE scaling, YaRN, ring/streaming attention, context extension
- [ ] **Reasoning & test-time compute**: CoT scaling, o1-style RL, process reward models
- [ ] **Multimodal**: vision-language (CLIP, LLaVA), audio, any-to-any
- [ ] **Interpretability**: mechanistic interp, SAEs, probing, attention analysis
- [ ] **Synthetic data**: generation, filtering, self-improvement loops

## Stage 8 — Safety & Security (always-on)
- [ ] **Prompt injection** & indirect injection (esp. in RAG/agents)
- [ ] Jailbreaking & defenses
- [ ] Data poisoning, model extraction
- [ ] PII handling, privacy, data governance
- [ ] Output filtering & content moderation

---

## 🏆 Capstone Projects (portfolio = proof of "cracked")

Pick at least **two**. These are what get you hired/respected.

- [ ] **End-to-end model**: pretrain a small LM → SFT → DPO → eval → serve with vLLM. Document the whole pipeline.
- [ ] **Production RAG system**: grounded Q&A over a real corpus with reranking, eval harness, observability, and a cost/latency report.
- [ ] **Inference optimization study**: take one model, apply quantization + speculative decoding, publish a benchmark report (quality vs speed vs cost).
- [ ] **Agent system**: multi-tool agent with planning, guardrails, and failure-mode analysis.
- [ ] **Eval framework**: build a rigorous, contamination-aware eval suite for a domain + an LLM-judge with measured bias.

> For each capstone: write it up (README + results + what you'd do differently). Teaching/writing is the final mastery test.

---

## 📄 Essential Paper Reading List (the core ~18)

**Architecture & Pretraining**
- [ ] Attention Is All You Need (2017)
- [ ] GPT-2 / GPT-3
- [ ] Chinchilla — Compute-Optimal Training
- [ ] LLaMA & LLaMA-2
- [ ] RoFormer (RoPE)

**Alignment & Fine-tuning**
- [ ] InstructGPT
- [ ] LoRA
- [ ] QLoRA
- [ ] DPO

**Inference & Efficiency**
- [ ] FlashAttention (v1 & v2)
- [ ] vLLM / PagedAttention
- [ ] GPTQ / AWQ
- [ ] Speculative Decoding

**Applications & Reasoning**
- [ ] RAG (Lewis et al.)
- [ ] Chain-of-Thought Prompting
- [ ] ReAct
- [ ] Mixtral (MoE)
- [ ] Self-Consistency

---

## 🔁 Recurring Practices (do these throughout)

- [ ] **Paper-a-week**: read 1 paper deeply; write a 1-paragraph summary + 1 critique
- [ ] **Implement-to-understand**: re-implement one core mechanism per stage from scratch
- [ ] **Teach it**: write a short blog/notes explaining each stage in your own words
- [ ] **Experiment log**: keep a running log of runs, hyperparams, results, surprises
- [ ] **Follow the field**: track new releases, but always map them back to fundamentals

---

## ✅ Progress Tracker

| Stage | Status | Build milestone done? | Notes |
|-------|--------|----------------------|-------|
| 1. Transformer internals | [ ] | [ ] | |
| 2. Pretraining at scale | [ ] | [ ] | |
| 3. Adaptation & alignment | [ ] | [ ] | |
| 4. Evaluation | [ ] | [ ] | |
| 5. Inference optimization | [ ] | [ ] | |
| 6. Production / LLMOps | [ ] | [ ] | |
| 7. Advanced specialization | [ ] | [ ] | |
| 8. Safety & security | [ ] | [ ] | |
| Capstone #1 | [ ] | [ ] | |
| Capstone #2 | [ ] | [ ] | |

---

*You are "cracked" when you can: implement the core pieces from scratch, reason about tradeoffs from first principles, debug a failing pipeline at any stage, and ship a working system end-to-end. Build relentlessly.*
