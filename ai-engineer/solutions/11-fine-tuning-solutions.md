# Chapter 11 — Fine-tuning: LoRA, QLoRA, PEFT · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-3-llm-stack/11-fine-tuning.md)

---

## Interview answers

### Q: "Why is full fine-tuning so memory-expensive?"

The **optimizer state**, not the weights, dominates. With AdamW in mixed precision you store, per parameter: the fp16 weight, the fp16 gradient, the fp32 master copy, and Adam's two moments ($m$, $v$) in fp32 — roughly **16 bytes per parameter** before activations. For a 7B model that's ~**112 GB** just for parameters+optimizer state, which won't fit on a single 80 GB GPU. So full fine-tuning of large models needs multi-GPU sharding (FSDP/ZeRO). This memory wall is the entire motivation for parameter-efficient fine-tuning.

### Q: "Explain LoRA and why it works."

LoRA (Low-Rank Adaptation) **freezes** the pretrained weight $W$ and learns a small **low-rank update**: $W' = W + BA$, where $B\in\mathbb{R}^{d\times r}$ and $A\in\mathbb{R}^{r\times d}$ with rank $r \ll d$. Only $A,B$ get gradients and optimizer state — often <1% of parameters. It works because the **fine-tuning update $\Delta W$ is empirically low-rank**: adapting a model to a task doesn't require full-rank changes, just a small subspace (the SVD/low-rank insight from Chapter 2). $B$ is initialized to **zero**, so training starts as an exact no-op ($BA = 0$, $W' = W$) and the model departs smoothly from the pretrained point.

### Q: "Does LoRA add inference latency?"

**No — if you merge.** Since $W' = W + BA$ is just another weight matrix, you can fold the adapter into the base weights once and serve at exactly the base model's speed (zero overhead). You keep the adapter **separate** only when you want to **hot-swap** adapters (serve many tasks from one base) — then you pay a tiny extra matmul ($xA$ then $\cdot B$) per layer, which is negligible.

### Q: "What does QLoRA add?"

QLoRA makes LoRA work on a **4-bit frozen base**, so you can fine-tune huge models on a single GPU. Three pieces: (1) **NF4** — a 4-bit "NormalFloat" data type information-theoretically suited to normally-distributed weights; (2) **double quantization** — quantize the quantization constants too, saving more memory; (3) **paged optimizers** — page optimizer state to CPU to survive memory spikes. The base stays frozen in 4-bit while the LoRA adapters train in 16-bit — fine-tuning a 65B model on one 48 GB GPU became possible.

### Q: "Fine-tune vs RAG vs prompt?"

A decision ladder — try the cheapest first:

- **Prompt engineering** first — fastest, no training; often enough.
- **RAG** when you need to change **what the model knows** — fresh, private, or factual knowledge, with citations and easy updates (just change the data).
- **Fine-tuning** when you need to change **how the model behaves** — style, format, tone, a skill, or a domain it can't be prompted into.

They **compose**: a common production setup is a fine-tuned model (behavior) **plus** RAG (knowledge). Don't fine-tune to inject facts (brittle, expensive to update) or RAG to fix formatting.

### Q: "How do you serve many custom models cheaply?"

**Multi-LoRA serving.** Host **one** copy of the base model and thousands of small, swappable **adapters** on shared hardware; route each request to its adapter and batch across adapters (e.g., S-LoRA, vLLM multi-LoRA). Because adapters are tiny (MBs) and the expensive base is shared, you serve thousands of "custom models" at roughly the cost of one — the economics that make per-customer fine-tunes viable.

---

## Exercise solutions

### Exercise 1 — `LoRALayer`: no-op at init, only A/B train

```python
import torch, torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, alpha=16):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W.weight.requires_grad = False          # FREEZE the base
        self.A = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_dim))  # B = 0 -> starts as no-op
        self.scale = alpha / r
    def forward(self, x):
        return self.W(x) + (x @ self.A @ self.B) * self.scale

torch.manual_seed(0)
layer = LoRALayer(64, 64, r=8)
x = torch.randn(4, 64)

# (1) no-op at init: output equals the frozen base alone
print("no-op at init:", torch.allclose(layer(x), layer.W(x)))      # True (B=0)

# (2) only A and B receive gradients
layer(x).sum().backward()
print("W grad is None     :", layer.W.weight.grad is None)         # True (frozen)
print("A grad present     :", layer.A.grad is not None)            # True
print("B grad present     :", layer.B.grad is not None)            # True
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f"trainable: {trainable}/{total} = {100*trainable/total:.1f}%")
```

**Result:** at initialization the layer is an exact no-op (because $B=0$), so fine-tuning starts from the pretrained behavior and departs smoothly. Only $A$ and $B$ carry gradients; the base $W$ is frozen (no grad, no optimizer state) — which is the entire memory saving.

### Exercise 2 — Trainable-parameter percentage vs rank

For a $d\times d$ layer, LoRA adds $2 \cdot d \cdot r$ parameters vs $d^2$ full, so the ratio is $2r/d$.

```python
d = 4096
full = d * d
for r in (4, 16, 64):
    lora = 2 * d * r
    print(f"rank {r:2d}: {lora:>10,} / {full:,} = {100*lora/full:.3f}% trainable")
```

**Result:**

| Rank | LoRA params | % of full $4096^2$ |
|---|---|---|
| 4 | 32,768 | 0.195% |
| 16 | 131,072 | 0.781% |
| 64 | 524,288 | 3.125% |

Even rank 64 trains ~3% of one layer's parameters; rank 4–16 is <1%. Multiply across a whole model and LoRA turns billions of trainable parameters into millions — the reason it fits on small hardware.

### Exercise 3 — Fine-tune a small model with HF `peft` (style transfer)

```python
# pip install transformers peft datasets accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

model_name = "gpt2"
tok = AutoTokenizer.from_pretrained(model_name); tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(model_name)

lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"],
                      lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()      # e.g. "trainable: 0.24% of all params"

# Toy "pirate style" dataset (replace with your own style pairs):
pairs = [{"text": "Hello there. -> Ahoy there, matey!"},
         {"text": "How are you? -> How be ye farin', sailor?"}] * 64
ds = Dataset.from_list(pairs).map(lambda e: tok(e["text"], truncation=True,
                                                padding="max_length", max_length=32),
                                  remove_columns=["text"])
ds = ds.map(lambda e: {"labels": e["input_ids"]})

args = TrainingArguments(output_dir="lora-out", per_device_train_batch_size=8,
                         num_train_epochs=3, learning_rate=2e-4, logging_steps=10)
Trainer(model=model, args=args, train_dataset=ds).train()

model.save_pretrained("lora-out/adapter")     # tiny adapter (a few MB)
```

**Result:** `print_trainable_parameters()` confirms you're training <1% of GPT-2, yet after a few epochs the model's *style* shifts toward the target (here, "pirate") while leaving its core knowledge intact. The saved **adapter is a few MB**, not a full model copy — that's the deliverable you hot-swap or share.

### Exercise 4 — Merge the adapter (W + BA) and verify identical outputs

```python
import torch

torch.manual_seed(0)
layer = LoRALayer(64, 64, r=8)
# simulate a trained adapter by giving B nonzero values
with torch.no_grad():
    layer.B.copy_(torch.randn_like(layer.B) * 0.1)

x = torch.randn(4, 64)
unmerged = layer(x)

# Merge: W' = W + scale * (A @ B)^T  (note Linear stores weight as [out, in])
with torch.no_grad():
    delta = (layer.A @ layer.B).T * layer.scale      # [out, in]
    merged_W = layer.W.weight + delta

merged = x @ merged_W.T
print("merged == unmerged:", torch.allclose(unmerged, merged, atol=1e-5))   # True
```

**Result:** the merged single matrix produces outputs **identical** to the separate base+adapter computation — confirming that merging is lossless and **adds zero inference latency**. You merge for max-speed single-task serving; keep separate only to hot-swap adapters.

### Exercise 5 — Prompt vs RAG vs LoRA on the same task (write-up)

A worked decision, the kind that makes a great portfolio post. Suppose the task is **"answer customer questions about our product in our brand voice."**

| Approach | What it changes | Result on this task | Cost / update |
|---|---|---|---|
| **Prompt-only** | In-context instructions/examples | Decent voice; **wrong/outdated facts** (model doesn't know your product) | ~free; edit the prompt |
| **RAG** | Injects **knowledge** | **Accurate, current facts with citations**; voice only as good as the prompt | cheap; just update the doc store |
| **LoRA** | Changes **behavior/voice** | **Perfect brand voice & format**; still doesn't *know* private facts | train once; retrain to change behavior |

**Verdict:** for this task, **RAG + LoRA together** wins — LoRA nails the *voice/format* (behavior) and RAG supplies the *facts* (knowledge), each doing what it's best at. RAG alone gives right facts in a generic voice; LoRA alone gives perfect voice but hallucinated facts; prompt alone gives neither reliably. **The principle:** match the tool to whether you're changing *what the model knows* (RAG) or *how it behaves* (fine-tune) — and always try prompting first.

---

[← Chapter 10 solutions](10-inference-optimization-solutions.md) · [Solutions index](README.md) · [Next: Chapter 12 solutions →](12-rag-and-agents-solutions.md)
