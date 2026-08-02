#!/usr/bin/env python3
"""Summarize a UniServe scheduler bench trace for ShareGPT-style text serving.

Separates warmup from measured events using the harness ``run.json``
``started_at`` timestamp and reports:

  1. prefill residual distributions (``pos_range[0]``, ``token_cost``) and the
     measured prefix-cache reuse fraction;
  2. decode batches split by adjacency to prefill steps (host roundtrip,
     worker exec, and ``component_us`` attribution when the server was run
     with ``UNISERVE_FORWARD_METRICS=1``);
  3. per-request inter-decode-result gaps (server-side ITL).

Usage:
  scripts/parse_sched_trace.py <scheduler_trace.jsonl> [started_at]
  scripts/parse_sched_trace.py <scheduler_trace.jsonl> --run-json <run.json>

With neither ``started_at`` nor ``--run-json``, all events are treated as
measured.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))
    return s[idx]


def _dist(name: str, vals: list[float], unit: str = "") -> None:
    if not vals:
        print(f"  {name}: (empty)")
        return
    print(
        f"  {name}: n={len(vals)} p50={_pct(vals, 50):.1f} p90={_pct(vals, 90):.1f} "
        f"p95={_pct(vals, 95):.1f} p99={_pct(vals, 99):.1f} max={max(vals):.1f}{unit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="scheduler_trace.jsonl path")
    parser.add_argument("started_at", nargs="?", type=float, default=None)
    parser.add_argument("--run-json", help="harness run.json to read started_at from")
    args = parser.parse_args()

    started_at = args.started_at
    if args.run_json:
        started_at = float(json.load(open(args.run_json))["started_at"])
    if started_at is None:
        started_at = float("-inf")

    submitted: dict[int, dict] = {}
    resolved: dict[int, dict] = {}
    queued: dict[int, int] = {}
    for line in open(args.trace):
        d = json.loads(line)
        if d["event"] == "batch_submitted":
            submitted[d["step_id"]] = d
        elif d["event"] == "batch_resolved":
            resolved[d["step_id"]] = d
        elif d["event"] == "request_queued":
            queued[d["request_id"]] = d["prompt_tokens"]

    steps = sorted(s for s in submitted if submitted[s]["at_s"] >= started_at and s in resolved)

    def kind_of(step: int) -> str:
        operation_types = set(submitted[step]["operation_types"])
        if operation_types == {"token_decode"}:
            return "decode"
        if operation_types == {"token_extend"}:
            return "prefill"
        if operation_types == {"token_decode", "token_extend"}:
            return "mixed"
        return "+".join(sorted(operation_types))

    kinds = {s: kind_of(s) for s in steps}
    prefills = [s for s in steps if kinds[s] == "prefill"]
    decodes = [s for s in steps if kinds[s] == "decode"]
    others = [s for s in steps if kinds[s] not in ("prefill", "decode")]
    print(
        f"measured batches: total={len(steps)} prefill={len(prefills)} "
        f"decode={len(decodes)} other={len(others)} {sorted({kinds[s] for s in others})}"
    )

    print("\n== PREFILL residuals (measured) ==")
    start_pos: list[float] = []
    token_costs: list[float] = []
    batch_sizes: list[float] = []
    exec_ms: list[float] = []
    host_ms: list[float] = []
    n_pos0 = 0
    total_cost = 0
    total_prompt = 0
    for s in prefills:
        sub, res = submitted[s], resolved[s]
        batch_sizes.append(sub["batch_size"])
        exec_ms.append(res["worker_exec_us"] / 1000)
        host_ms.append(res["host_roundtrip_us"] / 1000)
        for op in sub["ops"]:
            token_cost = op["token_cost"]
            kv_target = op["resources"]["kv_target_tokens"]
            start = max(0, kv_target - token_cost)
            start_pos.append(start)
            token_costs.append(token_cost)
            total_cost += token_cost
            total_prompt += queued.get(op["request_id"]) or 0
            if start == 0:
                n_pos0 += 1
    _dist("batch_size", batch_sizes)
    _dist("op token_cost", token_costs, " tok")
    _dist("op pos_range[0]", start_pos, " tok")
    _dist("worker_exec", exec_ms, " ms")
    _dist("host_roundtrip", host_ms, " ms")
    if total_prompt:
        print(
            f"  prefill ops: {len(token_costs)}, pos0: {n_pos0}; residual tokens "
            f"{total_cost}/{total_prompt} = {total_cost / total_prompt * 100:.2f}% of prompt tokens"
        )

    print("\n== DECODE adjacency (measured) ==")
    idx_of = {s: i for i, s in enumerate(steps)}
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in decodes:
        i = idx_of[s]
        prev_k = kinds[steps[i - 1]] if i > 0 else "none"
        next_k = kinds[steps[i + 1]] if i + 1 < len(steps) else "none"
        grp = "near_prefill" if "prefill" in (prev_k, next_k) else "decode_only"
        res = resolved[s]
        comp: dict[str, int] = defaultdict(int)
        for partition_stats in res["forward_stats"]:
            for key, value in partition_stats["component_us"].items():
                comp[key] += value
        g = groups[grp]
        g["batch_size"].append(res["batch_size"])
        g["host_ms"].append(res["host_roundtrip_us"] / 1000)
        g["exec_ms"].append(res["worker_exec_us"] / 1000)
        for name, key in (
            ("deferred_wait_ms", "worker_deferred_wait"),
            ("cuda_ready_ms", "worker_cuda_ready_elapsed"),
            ("model_forward_ms", "text_model_forward"),
            ("build_batch_ms", "text_build_batch"),
            ("sample_ms", "text_sample"),
            ("finalize_ms", "worker_result_finalize"),
        ):
            if key in comp:
                g[name].append(comp[key] / 1000)
    for grp in ("decode_only", "near_prefill"):
        g = groups[grp]
        print(f" [{grp}] n={len(g['host_ms'])}")
        for k in (
            "batch_size", "host_ms", "exec_ms", "deferred_wait_ms", "cuda_ready_ms",
            "model_forward_ms", "build_batch_ms", "sample_ms", "finalize_ms",
        ):
            if g[k]:
                _dist(k, g[k])

    print("\n== inter-decode-result gaps per request (measured) ==")
    last_seen: dict[int, float] = {}
    gaps: list[float] = []
    for s in steps:
        res = resolved[s]
        t = res["at_s"]
        for op in res["ops"]:
            rid = op["request_id"]
            if op["operation_type"] == "token_decode":
                if rid in last_seen:
                    gaps.append((t - last_seen[rid]) * 1000)
                last_seen[rid] = t
            elif op["operation_type"] == "token_extend":
                last_seen[rid] = t
    _dist("inter-decode gap", gaps, " ms")


if __name__ == "__main__":
    main()
