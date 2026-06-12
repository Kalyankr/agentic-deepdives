"""A latency/throughput benchmark for OpenAI-compatible LLM servers.

Point it at any server exposing `/v1/chat/completions` with streaming — e.g. **vLLM**,
TGI, SGLang, llama.cpp's server, or a local Ollama with the OpenAI shim — and it
measures the metrics that matter for serving (Module 04):

  * TTFT  — time to first token        (dominated by *prefill*)
  * TPOT  — time per output token      (dominated by *decode*, memory-bandwidth bound)
  * E2E   — end-to-end latency
  * Throughput — output tokens/sec and requests/sec under a given concurrency

Example:

    # start a server first, e.g.:  vllm serve <model> --port 8000
    uv run python -m lab04_inference_bench.benchmark \\
        --url http://localhost:8000 --model <model> \\
        --num-requests 64 --concurrency 8 --max-tokens 128

Sweep `--concurrency` and plot TTFT/TPOT/throughput to find the knee of the
latency–throughput curve (Module 04 capstone).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass

import httpx


@dataclass
class RequestResult:
    ok: bool
    e2e: float
    ttft: float | None = None
    tpot: float | None = None
    output_tokens: int = 0
    error: str | None = None


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100])."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def summarize(results: list[RequestResult], wall_time: float) -> dict[str, float]:
    """Aggregate per-request results into serving metrics. Pure + unit-testable."""
    ok = [r for r in results if r.ok]
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    tpots = [r.tpot for r in ok if r.tpot is not None]
    e2es = [r.e2e for r in ok]
    total_out = sum(r.output_tokens for r in ok)

    return {
        "requests": float(len(results)),
        "ok": float(len(ok)),
        "failed": float(len(results) - len(ok)),
        "wall_time_s": wall_time,
        "request_throughput_rps": len(ok) / wall_time if wall_time > 0 else float("nan"),
        "output_token_throughput_tps": total_out / wall_time if wall_time > 0 else float("nan"),
        "ttft_p50_ms": _percentile(ttfts, 50) * 1e3,
        "ttft_p95_ms": _percentile(ttfts, 95) * 1e3,
        "ttft_p99_ms": _percentile(ttfts, 99) * 1e3,
        "tpot_p50_ms": _percentile(tpots, 50) * 1e3,
        "tpot_p95_ms": _percentile(tpots, 95) * 1e3,
        "e2e_p50_ms": _percentile(e2es, 50) * 1e3,
        "e2e_p95_ms": _percentile(e2es, 95) * 1e3,
        "e2e_p99_ms": _percentile(e2es, 99) * 1e3,
    }


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict,
) -> RequestResult:
    start = time.perf_counter()
    first_token_t: float | None = None
    last_token_t: float = start
    n_tokens = 0
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                delta = obj["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if first_token_t is None:
                        first_token_t = now
                    last_token_t = now
                    n_tokens += 1
    except Exception as exc:  # noqa: BLE001 - report any failure as a failed request
        return RequestResult(ok=False, e2e=time.perf_counter() - start, error=repr(exc))

    e2e = time.perf_counter() - start
    ttft = (first_token_t - start) if first_token_t is not None else None
    tpot = None
    if n_tokens > 1 and first_token_t is not None:
        tpot = (last_token_t - first_token_t) / (n_tokens - 1)
    return RequestResult(ok=True, e2e=e2e, ttft=ttft, tpot=tpot, output_tokens=n_tokens)


async def run_benchmark(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    num_requests: int,
    concurrency: int,
    api_key: str | None,
    timeout: float,
) -> tuple[list[RequestResult], float]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def guarded() -> RequestResult:
            async with sem:
                return await _one_request(client, url, payload, headers)

        wall_start = time.perf_counter()
        results = await asyncio.gather(*(guarded() for _ in range(num_requests)))
        wall_time = time.perf_counter() - wall_start
    return list(results), wall_time


def _print_report(metrics: dict[str, float], concurrency: int) -> None:
    print("\n=== inference benchmark ===")
    print(f"concurrency            : {concurrency}")
    print(f"requests (ok/failed)   : {metrics['ok']:.0f}/{metrics['failed']:.0f}")
    print(f"wall time              : {metrics['wall_time_s']:.2f} s")
    print(f"request throughput     : {metrics['request_throughput_rps']:.2f} req/s")
    print(f"output token throughput: {metrics['output_token_throughput_tps']:.1f} tok/s")
    print(
        f"TTFT  p50/p95/p99 (ms) : {metrics['ttft_p50_ms']:.1f} / "
        f"{metrics['ttft_p95_ms']:.1f} / {metrics['ttft_p99_ms']:.1f}"
    )
    print(f"TPOT  p50/p95     (ms) : {metrics['tpot_p50_ms']:.1f} / {metrics['tpot_p95_ms']:.1f}")
    print(
        f"E2E   p50/p95/p99 (ms) : {metrics['e2e_p50_ms']:.1f} / "
        f"{metrics['e2e_p95_ms']:.1f} / {metrics['e2e_p99_ms']:.1f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000", help="server base URL")
    p.add_argument("--model", required=True, help="model name the server expects")
    p.add_argument("--prompt", default="Explain the KV cache in transformer inference.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--api-key", default=None)
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    results, wall_time = asyncio.run(
        run_benchmark(
            base_url=args.url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            api_key=args.api_key,
            timeout=args.timeout,
        )
    )
    metrics = summarize(results, wall_time)
    _print_report(metrics, args.concurrency)

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\nfirst error: {failed[0].error}")


if __name__ == "__main__":
    main()
