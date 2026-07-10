# Benchmark Protocol

This document defines the benchmark matrix and diagnostic protocol for UniServe, SGLang, and vllm-omni comparisons. The runnable spec lives in `uniserve_eval/profiles.json` under `benchmarks.main`; the runner is `scripts/run_benchmarks.py`.

## Principles

- Benchmark definitions are profile data, not local script constants.
- Tracked files must not contain machine-local model, dataset, or virtualenv paths. Use environment variables documented below.
- Load is expressed as open-loop request arrival rate unless a workload is explicitly labeled as a correctness gate or a diagnostic.
- Server command lines are part of the audited spec.
- Backend comparisons use the declared production-facing server profiles.
- Every UniServe point records the sanitized `ExecutionPlan` compiled from the measured public request shape; reference points record the fixed reference protocol contract.

## Required Environment

Set these before launching the benchmark runner:

| variable | meaning |
|---|---|
| `UNISERVE_QWEN3_MODEL` | Qwen3-32B model path |
| `UNISERVE_SENSENOVA_MODEL` | SenseNova-U1 model path |
| `UNISERVE_BAGEL_MODEL` | BAGEL-7B-MoT model path |
| `UNISERVE_SHAREGPT_PATH` | ShareGPT JSON dataset path |
| `UNISERVE_SGLANG_PYTHON` | Python executable for `refs/sglang` |
| `UNISERVE_OMNI_VLLM` | `vllm` executable for vllm-omni |
| `UNISERVE_BENCH_CUDA_VISIBLE_DEVICES` | optional: overrides every profile's `cuda_visible_devices` for this run |

If any variable is missing, launch paths must fail before starting a server. Dry-run output may keep `${VAR}` placeholders for audit.

## Matrix

All official benchmark points use arrival rates:

```text
1, 2, 4, 8, 16 requests/s
```

### Qwen3 ShareGPT

| field | value |
|---|---|
| systems | UniServe, SGLang |
| model | Qwen3-32B |
| dataset | ShareGPT |
| prompts per point | 200 |
| load axis | arrival rate 1,2,4,8,16 |
| metrics | TTFT, TPOT, E2E latency, output tokens/s |

SGLang runs through the declared benchmark server profile. Cache-mode evidence for the Qwen3 text-serving comparison is tracked in `specs/tasks.md`.

### MJHQ T2I

| field | SenseNova | BAGEL |
|---|---|---|
| systems | UniServe, vllm-omni | UniServe, vllm-omni |
| dataset | MJHQ | MJHQ |
| prompts per point | 200 | 200 |
| load axis | arrival rate 1,2,4,8,16 | arrival rate 1,2,4,8,16 |
| image size | 2048x1152 | 1024x1024 |
| steps | 50 | 50 |
| metrics | images/s, image latency | images/s, image latency |

### Beans I2T

| field | SenseNova | BAGEL |
|---|---|---|
| systems | UniServe, vllm-omni | UniServe, vllm-omni |
| dataset | beans train split, seed 42, 200 JPEGs | same |
| prompts per point | 200 | 200 |
| load axis | arrival rate 1,2,4,8,16 | arrival rate 1,2,4,8,16 |
| max tokens | 256 | 256 |
| temperature | 0 | 0 |
| UniServe wire | openai_chat (streamed) | openai_chat (streamed) |
| vllm-omni wire | openai_chat_json (non-streamed chat JSON) | openai_chat_json (non-streamed chat JSON) |
| metrics | request throughput, output tokens/s, TTFT, TPOT, E2E | same |

### SenseNova UEval Default

| field | value |
|---|---|
| system | UniServe |
| dataset | UEval |
| prompts per point | 32 |
| load axis | arrival rate 1,2,4,8,16 |
| max tokens | 8192 |
| image cap | 4 |
| image size | 2048x1152 |
| steps | 50 |
| wire | openai_chat (/v1/chat/completions SSE with modalities ["text","image"] + image_config) |
| metrics | request throughput, output tokens/s, images/s, TTFT, TPOT, time-to-first-image |

Text ITL excludes generated-image spans: the inter-token gap that straddles an image delta is dropped so image generation time never inflates text ITL.

## Runner Contract

Use:

```bash
.venv/bin/python scripts/run_benchmarks.py --benchmark main
```

Useful non-executing audit mode:

```bash
.venv/bin/python scripts/run_benchmarks.py --benchmark main --dry-run --no-build
```

The runner must:

- Load servers, benchmark groups, load axes, and point templates from `uniserve_eval/profiles.json`.
- Run at most one server and one harness process at a time.
- Write `COMMANDS.md` from the same resolved specs used for execution.
- Write `command.txt`, `run.log`, `summary.json`, `preflight.txt`, and `postflight.txt` per point, plus harness-owned `run.json`, `requests.jsonl`, `artifact_manifest.json`, and `summary.md`.
- Probe `/v1/chat/completions/plan` before UniServe measurements and mark the artifact invalid when runtime plan inspection fails.
- Bind each matrix or profile artifact to sanitized execution provenance covering the complete inherited process environment, content hashes for invoked executables, and Git revisions plus tracked and untracked source-state digests for participating checkouts; environment values remain fingerprint-only in artifacts.
- Mark artifacts canonical only when every acceptance, plan-evidence, exact-request-count, and contract-fingerprint check passes; resume and comparison paths require `canonical-valid-v2` and an exact match with the current resolved protocol and selected workload rows.
- Refuse to start if another benchmark/server process is active.
- Require zero visible GPU memory only when `--require-clean-gpu` is passed.
- Preserve `--resume`, `--only`, `--only-bench`, and `--skip-bench` semantics.

## Profile Organization

Profiles use namespace-style keys:

- `benchmark/server/...` for servers used by the official benchmark matrix.
- `gate/server/...` for correctness and small tripwire runs.
- `gate/...` for chat-completions correctness workloads.
- `perf/tripwire/...` for small regression sentinels.
- `benchmarks.main` for the official benchmark matrix.

Official benchmark points live under `benchmarks.main`. Correctness and regression sentinels live under `workloads`.

## Required Investigations

### SenseNova UEval TTFT Jump

The SenseNova UEval diagnostic compares request rates 4 and 8 and attributes p90 TTFT to request queueing, first-forward work, image spans, stream flushing, and client-side accounting.

Minimum artifacts:

- Scheduler admission trace.
- Per-request TTFT decomposition.
- Queue-depth timeline.
- Request start/first-token/end timestamps.
- Per-request image count and image step counts.
- GPU utilization and memory timeline.
- Server and worker logs.

The investigation should distinguish actual compute saturation from admission queueing, head-of-line blocking, image-span effects, and SSE/client accounting.

### UniServe Radix/Prefix Cache

The Qwen3 ShareGPT diagnostic records SGLang results with the declared server profile, records the matching UniServe results when available, and determines whether UniServe needs prefix/radix-cache support before final text-serving claims.
