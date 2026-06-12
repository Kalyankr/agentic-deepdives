# 09 · Key Papers — The Annotated Reading List

> Frontier labs (Anthropic especially) expect **paper fluency**: in a depth round or a research
> deep-dive you may be asked to explain a seminal result, critique a method, or place your own work
> in the literature. You don't need to have read every page — you need to **discuss each in ~2 minutes**:
> *what problem it solved, the key idea, the headline result, and what it changed.*

Each entry: **what it introduced → key result → likely interview question → the one-line takeaway.**

---

## If you only internalize eight

These eight give you 80% of the conversations. Master these cold first.

1. **Attention Is All You Need** — the Transformer.
2. **GPT-3 (Language Models are Few-Shot Learners)** — in-context learning at scale.
3. **Scaling Laws (Kaplan)** + **Chinchilla** — how to spend compute.
4. **InstructGPT** — RLHF for instruction following (the ChatGPT recipe).
5. **Constitutional AI** — RLAIF + a written constitution (Anthropic's core method).
6. **DPO** — alignment without a reward model or RL loop.
7. **FlashAttention** + **PagedAttention/vLLM** — the two systems ideas behind fast serving.
8. **Toy Models of Superposition / Scaling Monosemanticity** — Anthropic's interpretability program.

---

## Architecture & core models

- **Attention Is All You Need** (Vaswani et al., 2017). Introduced the **Transformer**: self-attention
  replaces recurrence/convolution, enabling full parallelism over sequence length.
  *Key result:* SOTA MT with far less training time; the architecture all LLMs use.
  *Likely Q:* "Why did attention beat RNNs?" → parallelism + direct long-range dependencies (`O(1)` path length) vs RNN's sequential `O(T)`.
  *Takeaway:* the substrate of everything since.

- **GPT / GPT-2 / GPT-3** (Radford 2018; Radford 2019; Brown et al., 2020). Decoder-only **generative
  pretraining**; GPT-2 showed zero-shot multitasking from scale; **GPT-3 (175B)** showed **in-context /
  few-shot learning** with no gradient updates.
  *Key result:* one pretrained model, many tasks via prompting.
  *Likely Q:* "What is in-context learning and why is it surprising?" → task adaptation purely from the prompt, no weight change.
  *Takeaway:* scale + next-token prediction → general capability.

- **BERT** (Devlin et al., 2018). Bidirectional **masked-LM** encoder for understanding tasks.
  *Likely Q:* "Encoder vs decoder — when each?" → BERT for classification/retrieval embeddings, GPT for generation.

- **RoFormer / RoPE** (Su et al., 2021). **Rotary** position embeddings encode *relative* position by
  rotating Q/K; extrapolate better and are the modern default (LLaMA, etc.).
  *Likely Q:* "How do RoPE/ALiBi help length generalization?"

- **(context)** *T5* (Raffel 2020, text-to-text framing) and *Switch Transformer* (Fedus 2021, **MoE**:
  sparse experts to grow params at ~constant FLOPs/token) round out the architecture map.

## Scaling laws

- **Scaling Laws for Neural LMs** (Kaplan et al., 2020). Loss falls as a **power law** in params `N`,
  data `D`, and compute `C`; smooth and predictable across orders of magnitude.
  *Takeaway:* you can *forecast* a bigger model's loss before training it.

- **Training Compute-Optimal LLMs / "Chinchilla"** (Hoffmann et al., 2022). For a fixed compute budget,
  scale **data and params together** — roughly **~20 tokens per parameter**. GPT-3 was undertrained.
  *Key result:* Chinchilla (70B) beat Gopher (280B) at equal compute.
  *Likely Q:* "How would you allocate a fixed FLOPs budget?" → toward Chinchilla-optimal, then *past* it (more tokens, smaller model) to cut **inference** cost (the LLaMA argument).

- **Emergent Abilities** (Wei et al., 2022) **vs "Are Emergent Abilities a Mirage?"** (Schaeffer et al.,
  2023). Some capabilities appear to jump at scale — but often that's a **discontinuous metric**
  (exact-match) artifact; smooth metrics show smooth improvement.
  *Likely Q:* "Are emergent abilities real?" → nuance: capability is continuous; the *metric* makes it look like a jump.

## Fine-tuning & PEFT

- **LoRA** (Hu et al., 2021). Freeze base weights; train low-rank adapters `ΔW = (α/r)·BA`.
  *Key result:* <1% trainable params, mergeable at inference (zero added latency).
  *Likely Q:* "Why does low rank suffice?" → fine-tuning updates are intrinsically low-rank.

- **QLoRA** (Dettmers et al., 2023). 4-bit **NF4** frozen base + bf16 LoRA → fine-tune a 65B model on a
  single 48GB GPU with ~no quality loss.
  *Takeaway:* democratized large-model fine-tuning.

## Alignment & preference learning

- **Deep RL from Human Preferences** (Christiano et al., 2017). The seed of RLHF: learn a reward model
  from human comparisons, optimize with RL.

- **InstructGPT** (Ouyang et al., 2022). The full **RLHF** recipe: SFT → reward model (Bradley–Terry) →
  PPO with a KL penalty. A 1.3B aligned model was preferred over 175B GPT-3.
  *Likely Q:* "Walk through RLHF and the role of the KL term." → KL keeps the policy near the reference to prevent reward hacking.

- **Constitutional AI** (Bai et al., 2022, **Anthropic**). Replace human harmlessness labels with **AI
  feedback (RLAIF)** guided by a written **constitution**; the model critiques and revises its own
  outputs.
  *Likely Q:* "Why RLAIF over RLHF for harmlessness?" → scalability, consistency, fewer humans exposed to harmful content. *Anthropic-critical.*

- **DPO — Direct Preference Optimization** (Rafailov et al., 2023). Shows the RLHF objective has a
  closed-form optimum, so you can train **directly on preference pairs** with a simple classification
  loss — **no reward model, no RL loop**.
  *Likely Q:* "Derive DPO / why is it stabler than PPO?" → logistic loss on the policy-vs-reference log-ratio margin; fewer moving parts. *Often asked.*
  *Takeaway:* know KTO/ORPO as variants (unpaired data; reference-free).

## Reasoning, prompting & agents

- **Chain-of-Thought** (Wei et al., 2022) + **Self-Consistency** (Wang et al., 2022). Eliciting
  intermediate reasoning ("let's think step by step") sharply improves multi-step tasks; sampling
  multiple chains and majority-voting improves it further.

- **ReAct** (Yao et al., 2022). Interleave **Reason + Act** (tool calls + observations) — the template
  behind most agent loops. (You built this in [lab07](../labs/lab07_agent/).)

- **Toolformer** (Schick et al., 2023). Models can **teach themselves** when/how to call tools via
  self-supervised API-call insertion.

- **RAG** (Lewis et al., 2020). **Retrieval-Augmented Generation**: condition generation on retrieved
  documents to inject fresh, attributable knowledge. (You built the retrieval core in [lab06](../labs/lab06_rag/).)
  *Likely Q:* "RAG vs fine-tuning for knowledge?" → RAG for facts/freshness/citations; fine-tuning for behavior/format.

## Systems & inference

- **FlashAttention** (Dao et al., 2022; v2 2023). **IO-aware exact** attention: tiling + online softmax
  keep the `T×T` matrix out of HBM.
  *Key result:* large speedups + linear memory in sequence length, **no approximation**.
  *Likely Q:* "If FLOPs are unchanged, why is it faster?" → attention is **memory-bandwidth** bound; it cuts HBM traffic.

- **vLLM / PagedAttention** (Kwon et al., 2023). Manage the KV cache in fixed **pages** (like OS virtual
  memory) → near-zero fragmentation, prefix sharing, much higher throughput.
  *Takeaway:* the systems idea behind modern serving + **continuous batching**.

- **Megatron-LM** (Shoeybi et al., 2019) and **ZeRO / DeepSpeed** (Rajbhandari et al., 2020).
  **Tensor parallelism** (split a matmul across GPUs) and **optimizer/grad/param sharding** (ZeRO 1/2/3 ≈ FSDP).
  *Likely Q:* "How do you train a model that doesn't fit on one GPU?" → TP within a node, PP across nodes, DP for replicas, ZeRO to shard states.

- **Speculative Decoding** (Leviathan et al.; Chen et al., 2023). A small **draft** model proposes `k`
  tokens; the big model **verifies** them in parallel → ~2–3× faster, **output-identical**.

## Safety & interpretability (Anthropic core)

- **A Mathematical Framework for Transformer Circuits** (Elhage et al., 2021, Anthropic). Reverse-engineer
  attention-only transformers; introduces **induction heads** (the mechanism behind in-context learning).

- **Toy Models of Superposition** (Elhage et al., 2022, Anthropic). Networks pack **more features than
  neurons** by representing them in superposition — why neurons look **polysemantic**.

- **Towards / Scaling Monosemanticity** (Anthropic, 2023–2024). **Sparse autoencoders (SAEs)** extract
  sparse, interpretable **monosemantic features** from activations — and scale to production models.
  *Likely Q:* "What is mechanistic interpretability trying to do?" → understand *internal computation*, not just behavior. *Anthropic-critical.*

- **Sleeper Agents** (Hubinger et al., 2024, Anthropic). Backdoored deceptive behavior can **survive**
  SFT, RLHF, and adversarial training — safety training may teach hiding, not removal.
  *Likely Q:* "Why is deceptive alignment scary?" → standard safety training didn't remove it.

- **(policy, not a paper)** Anthropic's **Responsible Scaling Policy (RSP)** ties safeguards to
  **AI Safety Levels (ASL)** capability thresholds; OpenAI's analog is the **Preparedness Framework**.
  See [05-safety-alignment](05-safety-alignment.md).

- **(worth knowing)** *Weak-to-strong generalization* (OpenAI 2023) and *Many-shot jailbreaking*
  (Anthropic 2024) are frequent discussion starters on scalable oversight and attack surface.

---

## How to discuss a paper (the 60-second template)

> **Problem → Idea → Result → Limitation → Significance.**
> "Before X, the problem was ___. X's key idea was ___. The headline result was ___ (with a number).
> Its main limitation is ___. It mattered because it changed ___."

Practice this on the eight core papers until it's automatic. Bonus signal: connect a paper to a
**trade-off** ("Chinchilla says train smaller-and-longer, but inference cost pushes you *past*
compute-optimal") and to something **you've run** (the [labs](../labs/README.md) and [notebooks](../notebooks/README.md)).

> Reading tip: read the **abstract, figures, and conclusion** of each, plus one good blog/explainer.
> You're optimizing for *being able to discuss it*, not reproducing it.
