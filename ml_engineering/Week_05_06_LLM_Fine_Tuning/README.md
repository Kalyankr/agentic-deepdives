# Week 5-6: LLM Fine-Tuning

This folder covers the Week 5-6 roadmap goal: understand LLM fine-tuning deeply enough to fine-tune with LoRA/QLoRA, evaluate the result, and prepare artifacts for serving.

The focus is not only running a trainer. The focus is the full fine-tuning loop: data, formatting, loss masking, adaptation method, memory strategy, evaluation, merging, and serving.

## Learning Goals

By the end of this section, you should be able to:

- Explain continued pretraining, supervised fine-tuning, instruction tuning, LoRA, and QLoRA.
- Decide when fine-tuning is appropriate versus prompt engineering, RAG, or tool use.
- Prepare raw domain text for continued pretraining and explain when CPT is worth doing.
- Prepare instruction and chat datasets for fine-tuning.
- Explain chat templates, `input_ids`, `attention_mask`, labels, and assistant-only loss masking.
- Compare full fine-tuning, feature extraction, partial fine-tuning, adapters, LoRA, and QLoRA.
- Implement LoRA conceptually from scratch.
- Use Hugging Face PEFT to attach LoRA adapters to a causal language model.
- Explain QLoRA, NF4, double quantization, and k-bit training preparation.
- Explain preference tuning with chosen/rejected pairs, DPO, and reward modeling.
- Save adapter artifacts and understand when to merge adapters.
- Evaluate a fine-tuned model beyond training loss.
- Prepare a fine-tuned model or adapter for serving with Transformers or vLLM.

## Fine-Tuning Concept Map

| Type | What Changes | Data Needed | Use When |
|---|---|---|---|
| Continued pretraining | Base model weights | Raw domain text | You need domain language or terminology adaptation |
| Supervised fine-tuning | Model behavior on examples | Prompt-response pairs | You want the model to imitate target answers |
| Instruction tuning | Instruction-following behavior | Diverse instruction-response examples | You want broad task-following behavior |
| Full fine-tuning | All model weights | High-quality task/domain data | You have compute and need maximum flexibility |
| Partial fine-tuning | Selected layers or heads | Task data | You want a cheaper baseline than full tuning |
| Adapter tuning | Small inserted modules | Task data | You want modular parameter-efficient updates |
| LoRA | Low-rank adapter matrices | Task or instruction data | You want cheap, strong fine-tuning |
| QLoRA | LoRA adapters on 4-bit base | Task or instruction data | You want to fine-tune larger models on limited GPU memory |
| Preference tuning | Preference behavior | Chosen/rejected pairs | You want outputs that better match human preferences |
| Distillation | Student model behavior | Teacher outputs | You want a smaller or cheaper model |

## Recommended Order

| Step | Notebook | Focus |
|------|----------|-------|
| 1 | [Fine-Tuning Foundations and Data](01_finetuning_foundations_and_data.ipynb) | Fine-tuning types, dataset schemas, quality checks, JSONL splits |
| 2 | [Tokenization, Chat Templates, and Loss Masks](02_tokenization_chat_templates_and_loss_masks.ipynb) | Chat formatting, tokenization, assistant-only labels, padding, truncation |
| 3 | [LoRA From Scratch](03_lora_from_scratch.ipynb) | LoRA math, parameter savings, PyTorch implementation, merging |
| 4 | [Hugging Face PEFT LoRA SFT](04_huggingface_peft_lora_sft.ipynb) | PEFT adapters, SFT dataset, trainer setup, adapter saving |
| 5 | [QLoRA 4-bit Fine-Tuning](05_qlora_4bit_finetuning.ipynb) | NF4, bitsandbytes, k-bit prep, QLoRA config, memory knobs |
| 6 | [Evaluation, Merging, and Serving](06_evaluation_merging_and_serving.ipynb) | Regression evals, adapter loading, merging, vLLM serving plan |
| 7 | [Continued Pretraining for Domain Adaptation](07_continued_pretraining_domain_adaptation.ipynb) | Raw domain text, causal LM labels, sequence packing, CPT risks |
| 8 | [Full, Partial, Linear Probe, and Adapter Tuning](08_full_partial_and_adapter_tuning.ipynb) | Full tuning, frozen probes, partial tuning, bottleneck adapters, parameter trade-offs |
| 9 | [Preference Tuning, DPO, and Reward Modeling](09_preference_tuning_dpo_and_reward_modeling.ipynb) | Chosen/rejected data, DPO loss, reward models, RLHF/PPO/GRPO map |

## Week 5-6 Deliverable

A clean deliverable for this week is a small fine-tuning package with:

- a documented dataset schema,
- optional raw domain text checks for continued pretraining,
- train/eval JSONL files,
- a formatting and loss-masking explanation,
- a LoRA or QLoRA configuration,
- a clear choice among full, partial, adapter, LoRA, or QLoRA tuning,
- saved adapter artifacts,
- a base-vs-tuned evaluation table,
- optional preference-pair evaluation if doing DPO,
- a merge or non-merge decision,
- and a serving plan.

## Completion Checklist

- [ ] Explain when fine-tuning is better than prompting, RAG, or tool use.
- [ ] Explain when continued pretraining is useful and when it is wasteful.
- [ ] Create a small instruction or chat dataset.
- [ ] Validate dataset quality, duplicates, missing fields, and length distribution.
- [ ] Format examples with a chat template or clear instruction template.
- [ ] Explain which tokens receive loss and which tokens are masked.
- [ ] Compare full fine-tuning, linear probing, partial fine-tuning, and adapters.
- [ ] Implement LoRA on a small PyTorch layer from scratch.
- [ ] Attach LoRA adapters with Hugging Face PEFT.
- [ ] Explain target modules, rank, alpha, and dropout.
- [ ] Explain how QLoRA uses 4-bit base weights plus LoRA adapters.
- [ ] Explain DPO using policy log probabilities, reference log probabilities, and chosen/rejected pairs.
- [ ] Explain how reward modeling fits into RLHF-style pipelines.
- [ ] Save adapter artifacts with base model metadata.
- [ ] Evaluate base vs fine-tuned behavior on a held-out set.
- [ ] Decide whether to merge adapters for serving.
- [ ] Write a short fine-tuning report.

## Optional Advanced Extensions

Add these after the expanded Week 5-6 path is comfortable:

- ORPO, KTO, IPO, and deeper DPO variants.
- Full RLHF/PPO or GRPO implementation.
- Distillation from a larger teacher model.
- Multi-adapter routing for multiple domains.
- Safety regression testing and refusal behavior evaluation.
- Synthetic data generation and filtering pipelines.

## Practical Dependency Notes

The concept notebooks use base packages already common in this workspace. The real PEFT/QLoRA notebooks require optional fine-tuning packages:

```bash
pip install transformers datasets accelerate peft
pip install bitsandbytes
```

`bitsandbytes` is hardware and CUDA sensitive. If it does not install cleanly on the local machine, run the QLoRA notebook in a compatible Linux GPU environment or container.
