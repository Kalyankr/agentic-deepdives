# Week 3-4: Model Serving & Optimization

This folder covers the Week 3-4 roadmap goal: move from a trained model to a production-minded inference service. The focus is latency, throughput, batching, export formats, Triton model repositories, vLLM-based LLM serving, load testing, and operational thinking.

## Learning Goals

By the end of this section, you should be able to:

- Explain the difference between offline evaluation and online inference.
- Measure latency with warmup, percentiles, and batch-size sweeps.
- Distinguish latency, throughput, concurrency, and utilization.
- Explain why batching improves throughput but can hurt tail latency.
- Export a PyTorch model toward deployable formats such as TorchScript and ONNX.
- Validate an ONNX graph and run it with ONNX Runtime.
- Build a Triton Inference Server model repository layout.
- Write a basic Triton client and understand dynamic batching configuration.
- Explain why vLLM is commonly used for LLM text-generation serving.
- Call an OpenAI-compatible vLLM endpoint and measure LLM-specific serving metrics.
- Design a load test and read p50/p95/p99 latency results.
- Package the serving work into a small capstone report with production notes.
- Name the production metrics an inference service should expose.

## Recommended Order

| Step | Notebook / Guide | Focus |
|------|------------------|-------|
| 1 | [Inference Latency and Batching](01_inference_latency_and_batching.ipynb) | Benchmarking, warmup, latency percentiles, batch-size trade-offs |
| 2 | [ONNX Export and ONNX Runtime](02_onnx_export_and_runtime.ipynb) | PyTorch export, dynamic axes, ONNX Runtime, parity checks, latency comparison |
| 3 | [Triton Model Repository and Client](03_triton_model_repository_and_client.ipynb) | Model repository layout, config.pbtxt, dynamic batching, HTTP client flow |
| 4 | [vLLM for LLM Serving](04_vllm_llm_serving.ipynb) | OpenAI-compatible LLM serving, continuous batching, KV cache, TTFT |
| 5 | [Serving Capstone Checklist](05_serving_capstone_checklist.md) | Deliverable template, benchmark report, client proof, production notes |

## Week 3-4 Deliverable

A clean deliverable for this week is a small inference package with:

- a model artifact,
- an ONNX export with a parity and latency check,
- a reproducible benchmark,
- a Triton model repository,
- a vLLM serving plan for LLM workloads,
- a client script,
- a short latency/throughput report,
- and notes on what changed after optimization.

Use [Serving Capstone Checklist](05_serving_capstone_checklist.md) as the final template.

## Completion Checklist

- [ ] Run a warmup-aware local inference benchmark.
- [ ] Compare single-item inference vs batched inference.
- [ ] Measure p50, p95, and p99 latency.
- [ ] Export a PyTorch model to ONNX with dynamic batch axes.
- [ ] Validate the ONNX graph and compare ONNX Runtime outputs to PyTorch.
- [ ] Explain why dynamic batching is useful.
- [ ] Create a Triton model repository folder structure.
- [ ] Write or review `config.pbtxt` for a model.
- [ ] Run or dry-run a Triton HTTP client request.
- [ ] Explain prefill, decode, KV cache, and continuous batching in vLLM.
- [ ] Start or dry-run a vLLM OpenAI-compatible server command.
- [ ] Measure request latency, tokens/sec, and time to first token for an LLM endpoint.
- [ ] Complete the serving capstone checklist and final report template.
- [ ] Document what you would monitor in production.
