# Resources Library

A curated reference set. Prefer **primary sources** (papers, official docs) and **build-first** courses. You don't need all of this — use it as a lookup keyed to the modules.

---

## Foundational courses
- **Karpathy — Neural Networks: Zero to Hero** (micrograd → makemore → nanoGPT → tokenizer → GPT-2). The best build-first start. [Modules 01–02]
- **fast.ai — Practical Deep Learning.** Top-down, practical. [Modules 01–02]
- **Stanford CS231n** (vision/CNNs, backprop), **CS224n** (NLP/transformers), **CS336 — Language Modeling from Scratch** (build an LLM end-to-end). [Modules 01–05]
- **d2l.ai — Dive into Deep Learning.** Free, code-first textbook. [Modules 01–02]
- **Hugging Face — LLM Course / NLP Course / Agents Course.** [Modules 02–08]
- **Full Stack Deep Learning / "LLM Bootcamp".** Productionizing. [Modules 08–10]

## Books
- *Mathematics for Machine Learning* — Deisenroth et al. (free) [00]
- *Deep Learning* — Goodfellow, Bengio, Courville (reference) [01]
- *Dive into Deep Learning* — Zhang et al. (free) [01–02]
- *Designing Data-Intensive Applications* — Kleppmann [05, 11]
- *Designing Machine Learning Systems* — Chip Huyen [09–11]
- *AI Engineering* — Chip Huyen [06–11]
- *Programming Massively Parallel Processors (PMPP)* — Kirk & Hwu [04]
- *Build a Large Language Model (From Scratch)* — Sebastian Raschka [02–03]
- *Hands-On Large Language Models* — Alammar & Grootendorst [02, 06–08]
- *Site Reliability Engineering* — Google (free) [10]

---

## Canonical papers by module

### Transformers & modeling [02]
- *Attention Is All You Need* — Vaswani et al., 2017
- *BERT* — Devlin et al., 2018; *GPT-2/3* — Radford/Brown et al.
- *RoFormer (RoPE)* — Su et al., 2021
- *GQA* — Ainslie et al., 2023
- *FlashAttention 1/2/3* — Dao et al.
- *Mixtral / Mixture-of-Experts* — Jiang et al., 2024
- *Chinchilla (Compute-Optimal)* — Hoffmann et al., 2022
- *Scaling Laws* — Kaplan et al., 2020
- *Llama 2 / Llama 3* reports

### Training, RLHF, DPO [03]
- *InstructGPT* — Ouyang et al., 2022
- *Constitutional AI* — Bai et al. (Anthropic), 2022
- *DPO* — Rafailov et al., 2023
- *PPO* — Schulman et al., 2017
- *LoRA* — Hu et al., 2021; *QLoRA* — Dettmers et al., 2023
- *Let's Verify Step by Step* — Lightman et al., 2023
- *DeepSeek-R1* — 2025

### GPU & inference [04]
- *PagedAttention / vLLM* — Kwon et al., 2023
- *Orca (continuous batching)* — Yu et al., 2022
- *GPTQ*, *AWQ*, *SmoothQuant*
- *Speculative Decoding* — Leviathan et al., 2023; *Medusa*; *EAGLE*
- "Making Deep Learning Go Brrrr From First Principles" — Horace He

### Distributed [05]
- *Megatron-LM* — Shoeybi et al., 2019
- *ZeRO* — Rajbhandari et al., 2020; PyTorch **FSDP**
- *GPipe* — Huang et al., 2019; *PipeDream*
- *Ring Attention* — Liu et al., 2023
- *DistServe* / *Mooncake* (prefill–decode disaggregation)
- "How to Scale Your Model" — DeepMind; *Ultra-Scale Playbook* — Hugging Face

### RAG & vector search [06]
- *RAG* — Lewis et al., 2020; *DPR* — Karpukhin et al., 2020
- *HNSW* — Malkov & Yashunin, 2016; *Product Quantization* — Jégou et al., 2011
- *Lost in the Middle* — Liu et al., 2023
- *Contextual Retrieval* — Anthropic, 2024

### Agents [07]
- *Building Effective Agents* — Anthropic, 2024
- *A Practical Guide to Building Agents* — OpenAI, 2025
- *ReAct* — Yao et al., 2022; *Reflexion* — Shinn et al., 2023
- *Toolformer* — Schick et al., 2023; *Tree of Thoughts* — Yao et al., 2023
- **Model Context Protocol** spec

### Prompting & orchestration [08]
- *Chain-of-Thought* — Wei et al., 2022; *Self-Consistency* — Wang et al., 2022
- *DSPy* — Khattab et al., 2023; *OPRO* — Yang et al., 2023

### Evaluation [09]
- *HELM* — Liang et al., 2022
- *MT-Bench / Chatbot Arena (LLM-as-judge)* — Zheng et al., 2023
- *SWE-bench* — Jimenez et al.; *τ-bench* — Yao et al.; *GAIA* — Mialon et al.
- *RAGAS* — Es et al., 2023

### Production & safety [10]
- OWASP **Top 10 for LLM Applications**
- Anthropic **Responsible Scaling Policy**; OpenAI **Preparedness Framework**
- NIST **AI Risk Management Framework**
- OpenTelemetry **GenAI** semantic conventions

---

## Tooling cheat sheet
- **Train/fine-tune:** PyTorch, Hugging Face `transformers` / `peft` / `trl` / `accelerate`, DeepSpeed, FSDP, Axolotl, Unsloth
- **Serve:** vLLM, TensorRT-LLM + Triton, SGLang, TGI, llama.cpp, Ollama
- **Kernels:** CUDA, Triton, CUTLASS
- **RAG / vectors:** FAISS, Qdrant, Weaviate, Milvus, pgvector, LlamaIndex
- **Agents / orchestration:** LangGraph, OpenAI Agents SDK, Claude Agent SDK, DSPy, AutoGen, CrewAI, MCP
- **Evals:** `inspect_ai`, `lm-evaluation-harness`, OpenAI Evals, RAGAS, Braintrust
- **Observability:** LangSmith, Langfuse, Arize Phoenix, Helicone, Prometheus/Grafana, OpenTelemetry
- **Infra:** Docker, Kubernetes, Ray, Slurm, Terraform, W&B/MLflow
- **Compute:** Colab, Kaggle, Lambda, RunPod, Modal, vast.ai

## Staying current
- Follow lab blogs: Anthropic, OpenAI, DeepMind, Meta AI, Mistral, DeepSeek.
- Papers: arXiv cs.CL/cs.LG, "Papers with Code," weekly digests.
- Communities: relevant subreddits, EleutherAI, Hugging Face forums, key practitioners on X.
- Read model **system cards** and **technical reports** as they drop — they're free senior-level curriculum.
