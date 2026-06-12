# Module 01 · Deep Learning Foundations

> **Goal:** Understand neural networks deeply enough to implement autograd, training loops, and the core building blocks **from scratch** — then reproduce them in PyTorch.

**Duration:** ~4 weeks. **Prereqs:** [Module 00](00-prerequisites.md).

---

## 1.1 Neural network fundamentals

- The neuron: linear layer + nonlinearity
- Activation functions: sigmoid, tanh, ReLU, GELU, SiLU/Swish — and *why* GELU/SiLU dominate transformers
- Multilayer perceptrons (MLPs) as universal approximators
- Forward pass as a computation graph

## 1.2 Backpropagation & autograd

- The chain rule on computation graphs
- Reverse-mode automatic differentiation
- Why we cache activations (memory cost of training)
- Vanishing/exploding gradients

> **Build (the keystone exercise):** Implement a tiny autograd engine (à la Karpathy's `micrograd`) — a `Value`/`Tensor` class with `+`, `*`, `matmul`, activations, and `.backward()`. Train an MLP on a toy dataset using *only* your engine. This single exercise demystifies all of deep learning.

## 1.3 Optimization

- Loss functions: MSE, cross-entropy, NLL
- SGD, momentum, Nesterov
- Adaptive methods: RMSProp, **Adam**, **AdamW** (decoupled weight decay — the LLM default)
- Learning-rate schedules: warmup, cosine decay, linear decay; why warmup matters for transformers
- Gradient clipping, gradient accumulation
- Batch size effects, the linear scaling rule

## 1.4 Regularization & generalization

- Overfitting vs. underfitting, bias–variance
- L2/weight decay, dropout, label smoothing
- Data augmentation (concept), early stopping
- Normalization: BatchNorm vs. LayerNorm vs. **RMSNorm** (and why transformers use Layer/RMSNorm)
- Residual/skip connections — why deep nets need them

## 1.5 Training in practice

- The canonical training loop: forward → loss → backward → step → zero_grad
- Mixed precision (AMP), `bfloat16`, loss scaling
- Checkpointing, resuming, reproducibility (seeds, deterministic ops)
- Gradient checkpointing (activation recomputation) to save memory
- Debugging: overfit a single batch first; monitor grad norms; sanity-check shapes
- Experiment tracking: Weights & Biases / TensorBoard

## 1.6 The PyTorch you must know cold

- `Tensor`, `autograd`, `nn.Module`, `Parameter`, `optim`
- `Dataset` / `DataLoader`, collate functions, samplers
- `device` management, `.to()`, pinned memory, `num_workers`
- `torch.no_grad()` / `inference_mode()`, `model.eval()` vs `train()`
- `state_dict` save/load
- `torch.compile` (intro; deep dive in Module 04)

> **Build:** Re-implement your micrograd MLP in PyTorch. Train a CNN on CIFAR-10 and an MLP on MNIST. Hit a target accuracy, then write up every hyperparameter choice and ablate three of them.

---

## 1.7 Sequence modeling primer (bridge to transformers)

- Word/token embeddings, the idea of distributed representations
- RNNs, the long-range dependency problem, vanishing gradients over time
- LSTMs/GRUs (conceptual) — gating to preserve gradient flow
- seq2seq with encoder–decoder
- **The attention mechanism** as a fix for the fixed-context bottleneck → motivates [Module 02](02-transformer-internals.md)

> **Build:** A character-level RNN/LSTM language model trained on a small text corpus (e.g., Shakespeare). Generate samples. You'll replace it with a transformer next module and compare.

---

## Module 01 capstone

1. A working from-scratch autograd engine + MLP trained with it (the centerpiece).
2. CIFAR-10 CNN reaching a credible accuracy, with an ablation write-up.
3. A char-level LSTM language model with generated samples and a loss/perplexity report.

## Exit criteria
- [ ] You implemented `.backward()` yourself and trust it.
- [ ] You can explain AdamW, warmup+cosine schedules, and weight decay.
- [ ] You can debug a training run (NaNs, no learning, overfitting) systematically.
- [ ] You're fluent in the PyTorch training loop without copying boilerplate.

## Core resources
- Karpathy — *Neural Networks: Zero to Hero* (micrograd → makemore → GPT)
- *Deep Learning* — Goodfellow, Bengio, Courville (reference)
- *Dive into Deep Learning* (d2l.ai) — code-first, free
- CS231n (CNNs), CS224n (sequence models) lecture notes
- PyTorch official tutorials
