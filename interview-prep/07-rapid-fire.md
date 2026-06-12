# 07 · Rapid-Fire Flashcards

> 130+ one-line recall checks. Cover the answer, read the question, answer out loud, reveal. If you
> hesitate, send it to the corresponding deep-dive file. Great for daily warm-ups in the final weeks.

---

## Transformers & architecture

1. **Q:** Why scale attention scores by `1/√d_k`? **A:** Dot-product variance grows with `d_k`; scaling keeps softmax out of saturation so gradients survive.
2. **Q:** Self-attention time/space complexity? **A:** `O(T²·d)` time, `O(T²)` memory in `T`.
3. **Q:** What does FlashAttention optimize? **A:** Memory access — tiles + online softmax, no `T×T` matrix in HBM; exact, not approximate.
4. **Q:** MQA vs GQA vs MHA? **A:** MQA = 1 KV head (smallest cache), MHA = KV head per query head, GQA = grouped middle ground.
5. **Q:** What does GQA shrink? **A:** The KV cache (fewer KV heads) → bigger batch/throughput.
6. **Q:** RoPE encodes what? **A:** **Relative** position, by rotating Q/K — extrapolates better than absolute.
7. **Q:** ALiBi mechanism? **A:** Adds a linear distance penalty to attention scores; good length extrapolation.
8. **Q:** Params per decoder layer (rule)? **A:** ~`12·d²` (4d² attention + 8d² FFN).
9. **Q:** FFN inner dimension (typical)? **A:** `4·d_model`.
10. **Q:** Pre-norm vs post-norm? **A:** Pre-norm gives a clean residual identity path → stabler deep training; modern default.
11. **Q:** Why RMSNorm over LayerNorm? **A:** No mean-subtraction/bias → cheaper, ~same quality.
12. **Q:** SwiGLU is? **A:** A gated GLU FFN activation used in modern LLMs (better than ReLU/GELU at scale).
13. **Q:** Encoder-only vs decoder-only? **A:** Encoder (BERT) = bidirectional/understanding; decoder (GPT) = causal/generation.
14. **Q:** Why multiple heads? **A:** Attend in different subspaces (syntax, coreference…) at ~constant total cost.
15. **Q:** Where is RoPE applied? **A:** To Q and K, before the dot product.

## Training & scaling

16. **Q:** Pretraining objective? **A:** Next-token prediction (cross-entropy), self-supervised.
17. **Q:** Training FLOPs rule? **A:** `C ≈ 6·N·D` (params × tokens).
18. **Q:** Inference FLOPs/token? **A:** `≈ 2·N`.
19. **Q:** Chinchilla compute-optimal ratio? **A:** ~**20 tokens per parameter**.
20. **Q:** Why train past Chinchilla (LLaMA)? **A:** To cut **inference** cost — smaller model, more tokens.
21. **Q:** bf16 vs fp16? **A:** bf16 = fp32 range, fewer mantissa bits, usually no loss scaling; fp16 needs loss scaling.
22. **Q:** Why warmup? **A:** Avoid early divergence from large LR on a fresh model.
23. **Q:** Gradient clipping fixes? **A:** Loss spikes from occasional huge gradients.
24. **Q:** Gradient checkpointing trades? **A:** Recompute activations in backward → less memory, more compute.
25. **Q:** AdamW state memory? **A:** Two moments (`m`,`v`) in fp32 = 8 bytes/param.
26. **Q:** Training memory/param (mixed Adam)? **A:** ~**16–20 bytes** (weights+grads+master+moments).
27. **Q:** Emergent abilities caveat? **A:** Often a discontinuous-**metric** artifact; smooth metrics → smoother curves.
28. **Q:** Most important pretraining data lever? **A:** Quality + **dedup** (and mixture) over raw quantity.
29. **Q:** Scaling-law functional form? **A:** Loss falls as a **power law** in N, D, C.
30. **Q:** MFU is? **A:** Model FLOPs Utilization — fraction of peak FLOPs used; ~40–55% is good.

## Fine-tuning & PEFT

31. **Q:** SFT loss-masking rule? **A:** Train only on **assistant** tokens; mask the prompt.
32. **Q:** LoRA update formula? **A:** `W' = W + (α/r)·B·A`, low-rank, base frozen.
33. **Q:** Why can LoRA merge with zero latency? **A:** `BA` adds into `W` at inference — no extra ops.
34. **Q:** QLoRA key idea? **A:** 4-bit (NF4) frozen base + bf16 LoRA adapters → single-GPU big-model fine-tune.
35. **Q:** Multi-LoRA serving? **A:** One shared base + many small adapters routed per request.
36. **Q:** Typical LoRA trainable %? **A:** Often <1% of params.
37. **Q:** Instruction tuning is? **A:** SFT on diverse (instruction, response) pairs to follow tasks.

## Alignment

38. **Q:** RLHF stages? **A:** SFT → preference data → reward model → PPO (with KL).
39. **Q:** Reward model loss? **A:** Bradley–Terry: maximize `log σ(r_chosen − r_rejected)`.
40. **Q:** Why the KL penalty in PPO? **A:** Prevent reward hacking / drift from the reference policy.
41. **Q:** DPO removes what? **A:** The separate reward model **and** the RL loop.
42. **Q:** DPO loss in words? **A:** Logistic loss on the policy-vs-reference log-ratio margin between chosen/rejected.
43. **Q:** DPO vs PPO? **A:** DPO simpler/stabler on fixed data; PPO/online can reach a higher ceiling.
44. **Q:** KTO needs paired data? **A:** No — learns from unpaired good/bad labels.
45. **Q:** ORPO special property? **A:** Folds preference into SFT; **no reference model**.
46. **Q:** Constitutional AI replaces? **A:** Human harmlessness labels with AI feedback (RLAIF) guided by a written constitution.
47. **Q:** Sycophancy is? **A:** Telling users what they want to hear (raters rewarded agreement).
48. **Q:** Reward hacking is? **A:** Optimizing the proxy reward, not the true objective.

## Inference & serving

49. **Q:** Prefill vs decode bound by? **A:** Prefill = compute-bound; decode = memory-bandwidth-bound.
50. **Q:** TTFT set by? **A:** Prefill (prompt processing).
51. **Q:** TPOT set by? **A:** Decode (per-token generation).
52. **Q:** Why batch decode? **A:** Amortize weight reads (bandwidth) across requests → more tok/s.
53. **Q:** Continuous batching? **A:** Swap finished/new sequences each step instead of waiting for the batch.
54. **Q:** PagedAttention? **A:** KV cache in fixed pages (like virtual memory) → no fragmentation, sharing.
55. **Q:** KV-cache bytes/token? **A:** `2·layers·kv_heads·d_head·bytes`.
56. **Q:** Speculative decoding? **A:** Small draft proposes k tokens, big model verifies in parallel; lossless ~2–3×.
57. **Q:** Weight-only INT8 quality? **A:** ~Lossless, ~½ memory/bandwidth.
58. **Q:** INT4 (GPTQ/AWQ)? **A:** ~4× smaller, small quality loss — eval after.
59. **Q:** Hardest thing to quantize? **A:** Activations (outliers).
60. **Q:** Decode throughput rule? **A:** `≈ HBM_bandwidth / bytes_read_per_token`.
61. **Q:** Prefix caching saves? **A:** Re-using the KV of a shared prompt prefix → lower TTFT/cost.
62. **Q:** Disaggregated serving? **A:** Separate prefill and decode pools (different bottlenecks).

## Distributed

63. **Q:** DP/DDP communicates? **A:** All-reduce of gradients.
64. **Q:** Tensor parallel keep within? **A:** A node (NVLink) — heavy communication.
65. **Q:** Pipeline parallel issue? **A:** The "bubble"; hide it with micro-batches.
66. **Q:** ZeRO stage 3 ≈? **A:** FSDP — shards params too; per-GPU memory ≈ 1/N of states.
67. **Q:** ZeRO stage 1/2/3 shard? **A:** 1: optimizer states; 2: + grads; 3: + params.
68. **Q:** Ring all-reduce bytes/GPU? **A:** ~2× param bytes, independent of N.
69. **Q:** all-gather vs reduce-scatter? **A:** Gather shards / reduce-then-split — FSDP fwd vs bwd.
70. **Q:** 3D parallelism? **A:** TP (in node) × PP (across nodes) × DP (replicas).
71. **Q:** all-to-all used by? **A:** MoE expert routing.
72. **Q:** Why checkpoint often at scale? **A:** Hardware failures are constant on 1000s of GPUs.

## RAG & retrieval

73. **Q:** RAG vs fine-tune for fresh facts? **A:** RAG (fine-tune is for behavior, not knowledge).
74. **Q:** Biggest RAG quality levers? **A:** Chunking + reranking.
75. **Q:** Why hybrid retrieval? **A:** Dense misses exact terms; BM25 catches them — fuse (RRF).
76. **Q:** Cross-encoder reranker vs bi-encoder? **A:** Joint query-doc scoring, far more accurate, used on top-k.
77. **Q:** Retrieval metrics? **A:** recall@k, MRR, nDCG.
78. **Q:** Generation/RAG metric? **A:** Faithfulness/groundedness + citation accuracy.
79. **Q:** HNSW vs IVF-PQ? **A:** HNSW = high recall, more memory; IVF-PQ = compressed, disk-friendly.
80. **Q:** PQ does what? **A:** Product quantization compresses vectors ~4–32× (approximate distances).
81. **Q:** ANN tuning dial in IVF? **A:** `nprobe` — more probes = higher recall, more work.
82. **Q:** First RAG debugging step? **A:** Check retrieval (is the right chunk in top-k?) before blaming generation.
83. **Q:** Lost-in-the-middle? **A:** Long-context models under-use middle content; rerank/position matters.

## Agents

84. **Q:** Agent vs workflow? **A:** Agent = LLM controls flow (flexible, costly); workflow = you control flow (predictable).
85. **Q:** ReAct loop? **A:** Thought → Action (tool) → Observation → … → Answer.
86. **Q:** Compounding error formula? **A:** Success ≈ `pⁿ` over n steps.
87. **Q:** #1 agent security threat? **A:** (Indirect) prompt injection via tool/retrieved content.
88. **Q:** Injection defense principle? **A:** Treat tool output as **data, not instructions** + least privilege + approval.
89. **Q:** MCP is? **A:** Model Context Protocol — open standard to expose tools/data to models.
90. **Q:** Multi-agent when worth it? **A:** Parallelizable subtasks or strong separation of concerns.
91. **Q:** Key agent guardrails? **A:** Max steps, spend cap, sandboxing, human approval for high-impact tools.
92. **Q:** Best tool-design lever? **A:** Clear description + forgiving error messages (the agent–computer interface).
93. **Q:** Orchestrator-workers pattern? **A:** Planner decomposes → workers execute → synthesize.

## Evaluations

94. **Q:** Why a confidence interval on eval scores? **A:** Finite samples — a 1–2 pt "win" can be noise.
95. **Q:** pass@k unbiased estimator? **A:** `1 − C(n−c,k)/C(n,k)`.
96. **Q:** LLM-as-judge top biases? **A:** Position, verbosity, self-preference.
97. **Q:** Fix position bias? **A:** Average over both A/B orderings.
98. **Q:** Validate a judge against? **A:** Human labels (measure agreement).
99. **Q:** Elo/Bradley-Terry used for? **A:** Ranking models from pairwise preferences (Chatbot Arena).
100. **Q:** Eval flywheel? **A:** Prod logs → error analysis → new eval cases → fix → re-eval.
101. **Q:** Code-gen benchmark? **A:** HumanEval / SWE-bench.
102. **Q:** Agent/tool benchmark? **A:** τ-bench, GAIA, WebArena.
103. **Q:** ECE measures? **A:** Calibration — does confidence match accuracy.
104. **Q:** Why not BLEU/ROUGE for quality? **A:** Weak correlation with human judgment; prefer task metrics/judges.

## Safety & alignment

105. **Q:** Misuse vs misalignment? **A:** Bad human use of a working model vs the model pursuing unintended goals.
106. **Q:** Outer vs inner alignment? **A:** Specifying the right objective vs the model internalizing it.
107. **Q:** Anthropic RSP ties safeguards to? **A:** Capability thresholds (ASL levels).
108. **Q:** OpenAI equivalent? **A:** Preparedness Framework (CBRN, cyber, persuasion, autonomy).
109. **Q:** Scalable oversight problem? **A:** Supervising tasks humans can't directly evaluate.
110. **Q:** Debate (safety) is? **A:** Two models argue; a judge decides — a scalable-oversight approach.
111. **Q:** Weak-to-strong generalization? **A:** Can a weak supervisor elicit a stronger model's full ability?
112. **Q:** Dangerous-capability eval examples? **A:** CBRN uplift, cyber-offense, autonomous replication.
113. **Q:** Over-refusal is? **A:** Rejecting benign requests — a real failure to measure too.
114. **Q:** Superposition (interp)? **A:** Models pack more features than neurons by overlapping them.
115. **Q:** SAEs (interp) do? **A:** Extract sparse, monosemantic features from activations.
116. **Q:** Product safety strategy? **A:** Defense-in-depth — layered filters, sandboxing, approval, monitoring.
117. **Q:** Jailbreak response strategy? **A:** Continuous red-team, train on found attacks, classifiers, fast patch.

## Numbers to know cold

118. **Q:** bf16 bytes/param? **A:** 2.
119. **Q:** fp32 bytes/param? **A:** 4.
120. **Q:** 7B model weights in bf16? **A:** ~14 GB.
121. **Q:** 70B model weights in bf16? **A:** ~140 GB.
122. **Q:** Why 70B won't train on one 80GB GPU? **A:** States ≈ 16–20 B/param ⇒ ~1.2–1.4 TB.
123. **Q:** Avg QPS from daily reqs? **A:** `daily / 86,400`.
124. **Q:** Peak factor (typical)? **A:** 2–5× average.
125. **Q:** H100 HBM size / bandwidth (order)? **A:** ~80 GB, ~3.3 TB/s.
126. **Q:** Perplexity from loss? **A:** `ppl = exp(CE_loss)`.
127. **Q:** Random-init LM loss ≈? **A:** `ln(vocab_size)`.

## One-liners on trade-offs

128. **Q:** Bigger batch effect on latency/throughput? **A:** ↑ throughput, ↑ TPOT (latency).
129. **Q:** Quantization trade-off? **A:** ↓ memory/bandwidth (↑ speed) vs possible ↓ accuracy — eval it.
130. **Q:** Long context cost driver? **A:** KV cache (linear in batch × length) + `O(T²)` attention.
131. **Q:** Model cascade benefit? **A:** Most traffic on a cheap model; escalate only hard cases.
132. **Q:** Streaming benefit? **A:** Lower **perceived** latency (user sees tokens at TTFT).

---

> Mastery check: answer any 20 random cards in under 5 minutes without hesitation. Gaps → revisit the
> matching deep-dive file ([01](01-coding.md)–[06](06-behavioral-mission.md)) and the [notebooks](../notebooks/README.md).
