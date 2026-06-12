# Lab 04 — Inference Benchmark

> Module: [04 · GPU Architecture & Inference](../../modules/04-gpu-architecture-and-inference.md)

A load-testing harness for any **OpenAI-compatible** LLM server. Unlike the other labs,
this one **runs as-is** — your job is to use it to *characterize* a real server and then
extend it.

## What it measures

| Metric | Meaning | Bound by |
|--------|---------|----------|
| **TTFT** | time to first token | prefill (compute-bound) |
| **TPOT** | time per output token | decode (memory-bandwidth-bound) |
| **E2E** | end-to-end latency | both |
| **Throughput** | output tokens/sec, requests/sec | batching + GPU utilization |

## Run it

Start a server (any of these expose the OpenAI API):

```bash
# vLLM (needs a GPU)
uvx vllm serve facebook/opt-125m --port 8000
# or llama.cpp server, TGI, SGLang, Ollama (OpenAI shim), ...
```

Then benchmark:

```bash
uv run python -m lab04_inference_bench.benchmark \
    --url http://localhost:8000 --model facebook/opt-125m \
    --num-requests 64 --concurrency 8 --max-tokens 128
```

## The exercise: find the knee of the curve

Sweep concurrency and record the metrics:

```bash
for c in 1 2 4 8 16 32 64; do
  uv run python -m lab04_inference_bench.benchmark --url http://localhost:8000 \
    --model facebook/opt-125m --num-requests 128 --concurrency $c --max-tokens 128
done
```

Plot **throughput vs. concurrency** and **p95 latency vs. concurrency**. You'll see
throughput rise then plateau while latency climbs — that knee is your SLA operating point.
Explain *why* using prefill/decode and continuous batching (Module 04).

## Stretch goals

- Add a `--plot` flag that saves latency histograms / the throughput curve (install `viz`).
- Separate prefill vs. decode cost by sweeping prompt length and `--max-tokens`.
- Compare a model at FP16 vs. quantized (INT8/INT4); tabulate quality/latency/memory.
- Add prefix caching to the prompts and measure the TTFT drop on cache hits.
