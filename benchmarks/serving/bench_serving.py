"""Self-contained online-serving benchmark for the UniServe OpenAI endpoint.

Fires a ShareGPT (or random) request stream at a target arrival rate against an
OpenAI-compatible ``/v1/chat/completions`` server with streaming enabled, and
reports the standard online-serving latency/throughput metrics (TTFT, TPOT, ITL,
end-to-end latency, input/output throughput).

Example:
    python -m benchmarks.serving.bench_serving \
        --base-url http://127.0.0.1:18081 --served-model-name Qwen3-32B \
        --num-prompts 120 --request-rate 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np

_DEFAULT_SHAREGPT = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/"
    "snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)


@dataclass
class RequestResult:
    success: bool = False
    prompt_len: int = 0
    output_tokens: int = 0
    ttft: float = 0.0
    latency: float = 0.0
    itls: list[float] = field(default_factory=list)
    error: str = ""


def load_sharegpt_prompts(path: str, num_prompts: int, seed: int) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    convs = [
        c["conversations"][0]["value"]
        for c in data
        if c.get("conversations") and c["conversations"][0].get("from") == "human"
        and len(c["conversations"][0].get("value", "")) > 8
    ]
    rng = random.Random(seed)
    rng.shuffle(convs)
    return convs[:num_prompts]


def random_prompts(num_prompts: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    return [
        "Tell me a short story using these words: "
        + " ".join(rng.choice(words) for _ in range(32))
        for _ in range(num_prompts)
    ]


async def one_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = RequestResult(prompt_len=len(prompt.split()))
    start = time.perf_counter()
    last = start
    first = True
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                result.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return result
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    continue
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                delta = choices[0].get("delta") if choices else {}
                piece = (delta or {}).get("content") or (delta or {}).get("reasoning_content")
                if piece:
                    now = time.perf_counter()
                    if first:
                        result.ttft = now - start
                        first = False
                    else:
                        result.itls.append(now - last)
                    last = now
                    result.output_tokens += 1
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens"):
                    result.output_tokens = int(usage["completion_tokens"])
        result.latency = time.perf_counter() - start
        result.success = result.output_tokens > 0
        if not result.success:
            result.error = "no output tokens"
    except Exception as exc:  # noqa: BLE001 - record the failure, keep the run going.
        result.error = repr(exc)
    return result


async def run(args: argparse.Namespace) -> dict:
    if args.dataset_name == "sharegpt" and os.path.exists(args.dataset_path):
        prompts = load_sharegpt_prompts(args.dataset_path, args.num_prompts, args.seed)
    else:
        prompts = random_prompts(args.num_prompts, args.seed)
    rng = random.Random(args.seed)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks: list[asyncio.Task] = []
        bench_start = time.perf_counter()
        for prompt in prompts:
            tasks.append(
                asyncio.create_task(
                    one_request(session, args.base_url, args.served_model_name, prompt, args.max_tokens)
                )
            )
            if args.request_rate != float("inf"):
                # Poisson arrivals at the target rate.
                await asyncio.sleep(rng.expovariate(args.request_rate))
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - bench_start
    return summarize(results, duration, args)


def _pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def summarize(results: list[RequestResult], duration: float, args: argparse.Namespace) -> dict:
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    ttfts = [r.ttft * 1000 for r in ok]
    itls = [v * 1000 for r in ok for v in r.itls]
    tpots = [
        (r.latency - r.ttft) / max(1, r.output_tokens - 1) * 1000
        for r in ok
        if r.output_tokens > 1
    ]
    e2es = [r.latency * 1000 for r in ok]
    total_out = sum(r.output_tokens for r in ok)
    total_in = sum(r.prompt_len for r in ok)
    summary = {
        "dataset_name": args.dataset_name,
        "num_prompts": args.num_prompts,
        "request_rate": args.request_rate if args.request_rate != float("inf") else "inf",
        "base_url": args.base_url,
        "completed": len(ok),
        "failed": len(failed),
        "duration_s": round(duration, 3),
        "output_throughput_tok_s": round(total_out / duration, 2) if duration else 0.0,
        "input_throughput_tok_s": round(total_in / duration, 2) if duration else 0.0,
        "total_output_tokens": total_out,
        "mean_ttft_ms": round(float(np.mean(ttfts)), 2) if ttfts else 0.0,
        "median_ttft_ms": round(_pct(ttfts, 50), 2),
        "p99_ttft_ms": round(_pct(ttfts, 99), 2),
        "mean_tpot_ms": round(float(np.mean(tpots)), 2) if tpots else 0.0,
        "median_tpot_ms": round(_pct(tpots, 50), 2),
        "mean_itl_ms": round(float(np.mean(itls)), 2) if itls else 0.0,
        "median_e2e_latency_ms": round(_pct(e2es, 50), 2),
        "p99_e2e_latency_ms": round(_pct(e2es, 99), 2),
    }
    if failed:
        summary["first_errors"] = [r.error for r in failed[:3]]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model-name", default="default")
    parser.add_argument("--dataset-name", default="sharegpt", choices=["sharegpt", "random"])
    parser.add_argument("--dataset-path", default=_DEFAULT_SHAREGPT)
    parser.add_argument("--num-prompts", type=int, default=120)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    summary = asyncio.run(run(args))
    print(json.dumps(summary, indent=2))
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
