# Week 3-4 Serving Capstone Checklist

Use this checklist after the Week 3-4 notebooks. The goal is to turn the concepts into a small, reviewable serving package: one model artifact, one benchmark, one serving option, one client request, and one short production-readiness note.

You do not need to use every serving tool. Choose the path that matches the workload.

| Workload | Best first serving path |
|---|---|
| Tabular classifier, vision model, embedding model, or general PyTorch model | ONNX Runtime or Triton |
| General multi-model inference service | Triton |
| Decoder-only LLM text generation | vLLM |
| Local experiment or sanity check | PyTorch inference benchmark |

## Suggested Folder Layout

```text
serving_capstone/
  README.md
  artifacts/
    model.onnx
    model.pt
  triton_model_repository/
    model_name/
      config.pbtxt
      1/
        model.onnx
  clients/
    request.py
    load_test.py
  reports/
    latency_report.md
    production_notes.md
```

Adjust this layout to the serving path you choose. For vLLM, you may not have a model artifact in the repo because the model is usually loaded from a Hugging Face model ID or local model directory.

## 1. Workload Decision

Before building the serving package, write down the target workload.

- [ ] Workload type: classification, regression, embedding, reranking, chat, completion, or batch generation.
- [ ] Expected input shape or prompt format is documented.
- [ ] Expected output shape or response format is documented.
- [ ] Latency target is defined.
- [ ] Throughput or concurrency target is defined.
- [ ] Serving choice is justified: PyTorch, ONNX Runtime, Triton, vLLM, or a combination.

Short decision template:

```md
## Serving Decision

Workload:
Serving choice:
Reason:
Rejected alternatives:
Main risk:
```

## 2. Model Artifact

For non-LLM models, create a reproducible artifact.

- [ ] Source model checkpoint or training run is identified.
- [ ] Export script or notebook cell is reproducible.
- [ ] Artifact path is documented.
- [ ] Input names and output names are documented.
- [ ] Input dtypes and output dtypes are documented.
- [ ] Static and dynamic dimensions are documented.
- [ ] Model version is recorded.

Useful artifact options:

| Artifact | Use when |
|---|---|
| PyTorch `.pt` or TorchScript | You want a direct PyTorch serving path or Triton PyTorch backend |
| ONNX `.onnx` | You want portability across runtimes or Triton ONNX backend |
| Hugging Face model ID or local model path | You are serving an LLM with vLLM |

## 3. ONNX Checks

Complete this section if you export to ONNX.

- [ ] Export uses clear input names and output names.
- [ ] Dynamic batch axes are configured when batch size can vary.
- [ ] `onnx.checker.check_model` passes.
- [ ] ONNX Runtime session loads successfully.
- [ ] ONNX Runtime output is compared against PyTorch output.
- [ ] Max absolute difference is recorded.
- [ ] Batch sizes 1, 8, 32, and 128 are tested if relevant.
- [ ] Optional quantization is measured before being accepted.

Parity table template:

| Runtime | Batch size | Max abs diff | Pass tolerance |
|---|---:|---:|---|
| PyTorch vs ONNX Runtime | 8 |  |  |

## 4. Benchmark

Every serving capstone should include a small benchmark.

- [ ] Warmup iterations are included.
- [ ] p50, p95, and p99 latency are reported.
- [ ] Throughput is reported.
- [ ] Batch size or concurrency settings are recorded.
- [ ] Hardware is recorded: CPU, GPU, memory, and runtime provider.
- [ ] Results are saved in a short report.

Benchmark table template:

| Runtime | Batch size | Concurrency | p50 ms | p95 ms | p99 ms | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch |  |  |  |  |  |  |
| ONNX Runtime |  |  |  |  |  |  |
| Triton |  |  |  |  |  |  |
| vLLM |  |  |  |  |  |  |

For LLM serving, add token metrics:

| Model | Concurrency | TTFT ms | Tokens/sec | p95 request latency ms | Error rate |
|---|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |

## 5. Triton Checks

Complete this section if you use Triton.

- [ ] Model repository folder has the correct version directory.
- [ ] Model artifact is placed in the version directory.
- [ ] `config.pbtxt` matches model input and output names.
- [ ] `config.pbtxt` matches model dtypes and dimensions.
- [ ] Dynamic batching settings are documented.
- [ ] Triton starts without model load errors.
- [ ] Health endpoint responds.
- [ ] Metadata endpoint is checked.
- [ ] One client request succeeds.
- [ ] One small load test is run or dry-run.

Minimum Triton client proof:

```md
Triton model name:
Health status:
Input payload shape:
Output shape:
Observed latency:
```

## 6. vLLM Checks

Complete this section if you serve an LLM.

- [ ] Model ID or local model path is documented.
- [ ] Context length is documented.
- [ ] GPU memory utilization setting is documented.
- [ ] Tensor parallel size is documented when using multiple GPUs.
- [ ] vLLM server command is saved.
- [ ] `/v1/models` responds.
- [ ] One OpenAI-compatible chat or completion request succeeds.
- [ ] Streaming is tested if the product needs streaming.
- [ ] Time to first token is measured.
- [ ] Tokens/sec is measured.
- [ ] Prompt and output logging policy is documented.

Minimum vLLM proof:

```md
Model:
Server command:
/v1/models status:
Sample prompt:
TTFT:
Tokens/sec:
Main bottleneck:
```

## 7. Client and Load Test

A serving package is incomplete until a client can call it.

- [ ] Client script has a clear base URL.
- [ ] Request payload is shown.
- [ ] Response parsing is shown.
- [ ] Timeout is configured.
- [ ] Errors are handled clearly.
- [ ] Load test script or notebook cell records concurrency.
- [ ] Load test reports failures, not only successful latency.

## 8. Production Notes

Write a short production-readiness note. Keep it practical.

- [ ] What should be monitored?
- [ ] What is the rollback plan?
- [ ] What happens if the model fails to load?
- [ ] What happens on out-of-memory errors?
- [ ] How are model versions promoted?
- [ ] How are secrets or model access tokens handled?
- [ ] What logs are safe to keep?
- [ ] What data should not be logged?

Monitoring checklist:

| Area | Metrics |
|---|---|
| Request health | request count, error rate, timeout count |
| Latency | p50, p95, p99, queue time |
| Throughput | requests/sec, items/sec, tokens/sec |
| Resource use | CPU, GPU utilization, GPU memory, RAM |
| Batching | batch size, queue delay, active sequences |
| LLM experience | TTFT, time per output token, context length |

## 9. Final Report Template

Create a short report using this structure.

```md
# Serving Capstone Report

## Goal

## Model

## Serving Choice

## Export or Artifact

## Benchmark Setup

## Results

## Client Proof

## Production Risks

## Next Improvements
```

## Pass Criteria

You are done when all of these are true:

- [ ] Another person can understand what is being served.
- [ ] Another person can find or reproduce the model artifact.
- [ ] A benchmark result exists with p50, p95, and p99 latency.
- [ ] A serving path is chosen and justified.
- [ ] A sample client request exists.
- [ ] Production monitoring notes exist.
- [ ] Known limitations are written down.

The point is not to make a perfect production system. The point is to practice the thinking loop: artifact, runtime, benchmark, client, monitoring, trade-offs.
