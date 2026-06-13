# Chapter 17 — Serving & MLOps · Solutions

[← Solutions index](README.md) · [Read the chapter](../part-4-systems/17-serving-mlops.md)

---

## Interview answers

### Q: "How would you deploy an LLM in production?"

A complete answer touches the whole stack:

1. **Serving engine** — a purpose-built engine (**vLLM**, TGI, TensorRT-LLM) for **continuous batching** + **paged KV cache**, not naive Flask.
2. **Packaging** — **Docker** (CUDA base image, pinned deps), orchestrated by **Kubernetes** with **GPU-aware autoscaling** (scale on queue depth/utilization, keep warm pools for cold-start).
3. **UX** — **stream tokens** (SSE/WebSocket) so perceived latency is the TTFT, not the full response.
4. **Release safety** — **gate on evals** (Chapter 13) in CI/CD, **progressive rollout** (canary), fast **rollback**.
5. **Observability** — monitor latency (TTFT/TPOT), throughput, **quality/drift**, and **cost** per request.

The theme: production = reliability + scale + cost + observability + security, far beyond "it runs in a notebook."

### Q: "API vs self-hosting?"

A real cost/control/privacy tradeoff:

- **API** (OpenAI/Anthropic): ship **fast**, no infra, pay per token. Best for low/moderate volume, prototyping, or when you want frontier quality now.
- **Self-host** (open model on your GPUs): better **unit economics at scale**, full **customization** (fine-tunes, quantization), and **data privacy/residency**. Costs infra + expertise.

There's a **computable crossover** (Exercise 6): below some QPS the API is cheaper; above it, self-hosting wins. Decide with the dollar math plus privacy/customization requirements, not vibes.

### Q: "How do you monitor an LLM in production?"

Four layers:

1. **System metrics** — latency (TTFT/TPOT), throughput, GPU utilization, errors.
2. **Quality** — online evals on sampled traffic, user feedback (thumbs, edits, regenerations), LLM-judge on a sample.
3. **Drift** — data drift (input distribution shifts) and concept drift (the right answer changes over time).
4. **Cost** — tokens and dollars per request, per feature, per user.

Plus **request tracing** for RAG/agents (which tools/docs were used). The danger with LLMs is **silent degradation** — outputs get worse without errors — so you specifically watch quality/drift, not just uptime.

### Q: "How do you cut inference cost?"

The highest-leverage levers, framed in **dollars saved**:

- **Right-size / route** — send easy queries to a small/cheap model, hard ones to the big model (Exercise 4).
- **Quantization** — int8/int4 → cheaper, faster (Chapter 10).
- **Continuous batching** — maximize GPU utilization (the biggest serving win).
- **Caching** — exact + **semantic** caching of repeated/similar queries (Exercise 3).
- **Spot/preemptible instances** for batch workloads.

Routing + caching + quantization together often cut the majority of inference spend.

### Q: "What is concept/data drift?"

- **Data drift** — the **input distribution** changes (users start asking about new topics, new slang, new languages) while the model is fixed → it sees inputs unlike its training/eval data.
- **Concept drift** — the **correct mapping** from input to output changes over time (e.g., "current best model" or a policy answer changes), so even unchanged inputs now have different right answers.

Both cause **silent quality decay**. You detect them by monitoring input distributions and quality metrics over time and **trigger retraining / data refresh** when they cross thresholds.

### Q: "Why not just use Flask for serving?"

Because a naive Flask endpoint processes **one request at a time** with no **batching** and no **KV-cache management**, so GPU utilization is terrible — you pay for an expensive GPU that sits ~idle. Purpose-built engines (vLLM/TGI/TRT-LLM) implement **continuous batching**, **paged attention**, and optimized kernels, delivering **10×+ throughput** on the same hardware. Flask is fine as a thin HTTP layer *in front of* such an engine, but never as the inference engine itself.

---

## Exercise solutions

### Exercise 1 — Serve with vLLM, load-test vs a Flask baseline

```python
# pip install vllm
# Start an OpenAI-compatible server:
#   python -m vllm.entrypoints.openai.api_server --model facebook/opt-1.3b

import asyncio, aiohttp, time

async def one(session, url, payload):
    async with session.post(url, json=payload) as r:
        return await r.json()

async def load_test(url, n=100, concurrency=20):
    payload = {"model": "facebook/opt-1.3b",
               "prompt": "The future of AI is", "max_tokens": 64}
    sem = asyncio.Semaphore(concurrency)
    async def guarded(session):
        async with sem:
            return await one(session, url, payload)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as s:
        await asyncio.gather(*(guarded(s) for _ in range(n)))
    dt = time.perf_counter() - t0
    return n / dt          # requests/sec

# throughput = asyncio.run(load_test("http://localhost:8000/v1/completions"))
# print(f"vLLM throughput: {throughput:.1f} req/s")
```

**Result:** under concurrent load, vLLM **continuously batches** the requests, keeping the GPU busy and delivering **many×** the throughput of a single-request Flask loop (which serializes and leaves the GPU idle between requests). The gap **widens with concurrency** — exactly the scenario production faces — which is the empirical case for purpose-built engines.

### Exercise 2 — Containerize a model API (with pinned dependencies)

```dockerfile
# Dockerfile — pin EVERYTHING for reproducibility
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Pin exact versions (a lockfile is even better: uv.lock / requirements.txt w/ hashes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .
EXPOSE 8000
CMD ["python3", "serve.py"]
```

```text
# requirements.txt — exact, hash-pinned versions
vllm==0.6.3
fastapi==0.115.0
uvicorn==0.30.6
torch==2.4.0
```

```bash
docker build -t my-llm:1.0 .
docker run --gpus all -p 8000:8000 my-llm:1.0
```

**Result:** the image bundles a **specific CUDA runtime + pinned Python deps**, so it runs identically on any GPU host — no "works on my machine." **Dependency pinning** (exact versions, ideally a hash-locked file) is the non-negotiable MLOps practice: ML stacks are fragile to version drift (CUDA/torch/vLLM compatibility), and an unpinned build that worked today can break tomorrow. The `--gpus all` flag exposes the GPU to the container.

### Exercise 3 — Semantic cache (hit rate + cost savings)

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, threshold=0.92):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.keys, self.vals, self.embs = [], [], []
        self.threshold = threshold
        self.hits = self.misses = 0
    def get(self, query):
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        if self.embs:
            sims = np.array(self.embs) @ q
            i = int(sims.argmax())
            if sims[i] >= self.threshold:            # semantically close enough
                self.hits += 1; return self.vals[i]
        self.misses += 1; return None
    def put(self, query, value):
        self.keys.append(query); self.vals.append(value)
        self.embs.append(self.embedder.encode([query], normalize_embeddings=True)[0])

cache = SemanticCache()
def answer(query, llm_cost=0.01):
    cached = cache.get(query)
    if cached is not None: return cached, 0.0        # cache hit -> $0
    resp = f"answer to: {query}"; cache.put(query, resp)
    return resp, llm_cost                            # miss -> pay the model

queries = ["What is your refund policy?", "How do refunds work?",   # semantically same
           "What's the return policy?", "What is your refund policy?"]  # near/exact dups
total = sum(answer(q)[1] for q in queries)
print(f"hits={cache.hits} misses={cache.misses} "
      f"hit rate={cache.hits/(cache.hits+cache.misses)*100:.0f}%  cost=${total:.2f}")
```

**Result:** semantically equivalent phrasings ("refund policy" ≈ "how do refunds work") hit the cache even though the **exact strings differ** — an exact-match cache would miss them. Each hit costs **$0** instead of a model call, so on repetitive traffic (FAQs, common prompts) a semantic cache cuts a large fraction of spend and latency. Tune the **similarity threshold** to trade hit rate against the risk of returning a slightly-off cached answer.

### Exercise 4 — Model router (cost vs quality)

```python
def classify_difficulty(query):
    """Cheap heuristic (in practice: a small classifier model)."""
    hard_signals = ["prove", "derive", "step by step", "code", "analyze", "why"]
    long = len(query.split()) > 30
    return "hard" if (long or any(s in query.lower() for s in hard_signals)) else "easy"

PRICES = {"small": 0.0005, "big": 0.03}     # $ per request (illustrative)

def route(query):
    tier = "big" if classify_difficulty(query) == "hard" else "small"
    return tier, PRICES[tier]

queries = ["What's the capital of France?",                       # easy -> small
           "Derive the gradient of softmax step by step.",        # hard -> big
           "Hi!",                                                 # easy -> small
           "Analyze why this distributed training run diverged."] # hard -> big
routed = [route(q) for q in queries]
cost_routed = sum(c for _, c in routed)
cost_all_big = len(queries) * PRICES["big"]
print("routing decisions:", [(t) for t, _ in routed])
print(f"routed cost=${cost_routed:.4f}  vs all-big=${cost_all_big:.4f}  "
      f"saved {(1-cost_routed/cost_all_big)*100:.0f}%")
```

**Result:** easy queries (most traffic) go to the **cheap small model**; only genuinely hard ones hit the **expensive big model** — cutting cost dramatically (often 50–90%) while preserving quality where it matters. The risk is **misrouting** a hard query to the small model, so you validate the router against an eval set and tune the threshold. This is one of the single biggest cost levers in production.

### Exercise 5 — Observability: log TTFT/TPOT/tokens/cost + drift alert

```python
import time, numpy as np
from collections import deque

class Observability:
    def __init__(self, window=100):
        self.recent_lengths = deque(maxlen=window)   # for drift detection
        self.baseline_mean = None
    def trace(self, prompt, generate_fn, price_per_1k=0.03):
        t0 = time.perf_counter()
        first = generate_fn(prompt, n=1); ttft = time.perf_counter() - t0  # first token
        t1 = time.perf_counter()
        rest = generate_fn(prompt, n=63); tpot = (time.perf_counter() - t1) / 63
        n_tokens = 64; cost = n_tokens / 1000 * price_per_1k
        log = dict(ttft_ms=round(ttft*1e3, 1), tpot_ms=round(tpot*1e3, 2),
                   tokens=n_tokens, cost=round(cost, 4))
        self.recent_lengths.append(len(prompt.split()))
        log["drift_alert"] = self._check_drift()
        return log
    def _check_drift(self):
        if len(self.recent_lengths) < self.recent_lengths.maxlen: return False
        mean = np.mean(self.recent_lengths)
        if self.baseline_mean is None: self.baseline_mean = mean; return False
        # alert if input distribution shifts >50% from baseline
        return abs(mean - self.baseline_mean) / self.baseline_mean > 0.5

obs = Observability(window=5)
gen = lambda p, n: "x" * n
for prompt in ["short q"] * 5 + ["a much longer query " * 10] * 5:
    log = obs.trace(prompt, gen)
print("last log:", log)        # shows ttft/tpot/tokens/cost and drift_alert=True after the shift
```

**Result:** every request logs **TTFT** (responsiveness), **TPOT** (generation speed), **tokens**, and **cost** — the four numbers you dashboard. The drift detector watches the **input distribution** (here, prompt length) and **alerts** when it shifts past a threshold, catching the **silent** failure mode where inputs change and quality quietly drops. In production you'd ship these to Prometheus/Grafana and add quality sampling + tracing.

### Exercise 6 — API vs self-host cost crossover

```python
def monthly_cost_api(qps, tokens_per_req, price_per_1k):
    reqs = qps * 60 * 60 * 24 * 30
    return reqs * tokens_per_req / 1000 * price_per_1k

def monthly_cost_selfhost(n_gpus, gpu_hourly):
    return n_gpus * gpu_hourly * 24 * 30        # fixed infra cost (utilization-independent)

tokens_per_req, api_price = 500, 0.01           # $/1k tokens
gpu_hourly, gpus_needed = 2.0, 2                # 2×A100 @ $2/hr

print(f"{'QPS':>6} {'API $/mo':>12} {'Self-host $/mo':>16} {'cheaper':>10}")
for qps in (1, 5, 10, 25, 50, 100):
    api = monthly_cost_api(qps, tokens_per_req, api_price)
    self_ = monthly_cost_selfhost(gpus_needed, gpu_hourly)
    print(f"{qps:6d} {api:12,.0f} {self_:16,.0f} {'API' if api < self_ else 'self-host':>10}")
```

**Result:** API cost scales **linearly with traffic** (pay per token), while self-hosting is a **fixed** infra cost (you pay for the GPUs whether busy or not). So there's a **crossover QPS**: below it the API is cheaper (you'd waste idle GPUs); above it, self-hosting wins because the fixed cost amortizes over huge volume. The decision is this dollar math **plus** privacy/customization needs — exactly the framework from the interview answer. At low volume, ship on the API; at sustained high volume, self-host.

---

[← Chapter 16 solutions](16-frameworks-solutions.md) · [Solutions index](README.md) · [Next: Part V career solutions →](18-19-career-solutions.md)
