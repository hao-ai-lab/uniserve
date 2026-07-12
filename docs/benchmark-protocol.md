# Benchmark protocol

The benchmark matrix is defined in [`uniserve_eval/profiles.json`](../uniserve_eval/profiles.json) and executed by [`scripts/run_benchmarks.py`](../scripts/run_benchmarks.py). The profile fixes workload selection, request semantics, load cases, model and dataset identity, server profiles, hardware requirements, comparison metrics, and artifact validation before measurement.

## Running the matrix

```bash
.venv/bin/python scripts/run_benchmarks.py \
  --benchmark main \
  --output-root artifacts/benchmark
```

Use `--only` to select complete backend groups, `--dry-run --no-build` to inspect commands without launching servers, and `--formal` to require the canonical clean-source, pinned-revision, fresh-build, accelerator, NUMA, process, GPU, and execution-identity checks.

`--repeat N` executes complete selected matrices serially beneath numbered child roots and produces an aggregate containing every valid per-run ratio and its geometric mean. `--text-canary` and `--image-smoke` enable optional diagnostics; neither changes point validity or the primary metric.

## Execution model

A host-wide lock spans the command. The runner permits at most one benchmark server and one harness process at a time. Every point starts from a fresh server, and the server is stopped and its artifact validated before the next backend, workload, load case, or repeated matrix begins.

A failed point stops the run and preserves its partial artifacts and logs. The failure is diagnosed before artifact-producing execution continues.

## Workload matrix

| Workload | Systems | Requests per point | Load cases | Primary metric |
| --- | --- | ---: | --- | --- |
| Qwen3-32B ShareGPT | UniServe, SGLang | 200 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s, maximize |
| SenseNova-U1 MJHQ T2I | UniServe, vLLM-Omni | 32 | `c1`, `c32` | `c1`: mean image latency, minimize; `c32`: images/s, maximize |
| BAGEL MJHQ T2I | UniServe, vLLM-Omni | 32 | `c1`, `c32` | `c1`: mean image latency, minimize; `c32`: images/s, maximize |
| SenseNova-U1 Beans I2T | UniServe, vLLM-Omni | 32 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s |
| BAGEL Beans I2T | UniServe, vLLM-Omni | 32 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s, maximize |
| SenseNova-U1 UEval | UniServe | 32 | `r1`, `r2`, `r4`, `r8`, `r16` | Mixed-output service metrics |

The matrix contains 43 serial points. `rN` is an open-loop seed-42 Poisson trace with an offered rate of N requests/s and no client concurrency semaphore. `c1` submits the fixed prompt set immediately with client concurrency one. `c32` submits the same set immediately with client concurrency 32.

Timing covers the complete measured arrival and completion region. TTFT begins at client send and ends at the first non-empty content or reasoning delta; TPOT uses server completion-token accounting; output throughput is server-reported completion tokens divided by the complete timed region. Image latency is measured from request send through decoded output receipt, and image throughput is the number of successfully decoded images divided by the complete timed region.

## ShareGPT semantics

The Qwen workload follows the pinned SGLang ShareGPT behavior implemented in [`refs/sglang/python/sglang/benchmark/datasets/sharegpt.py`](../refs/sglang/python/sglang/benchmark/datasets/sharegpt.py) and [`refs/sglang/python/sglang/benchmark/serving.py`](../refs/sglang/python/sglang/benchmark/serving.py):

1. Keep conversations with at least two turns and use the first turn as the prompt and the second turn as the trace completion.
2. Shuffle with the configured seed and retain the first requested rows that pass the length filters.
3. Tokenize the prompt and trace completion with the selected model tokenizer.
4. Set each request's output-token budget from the trace completion token count.
5. Ignore EOS so each successful request performs that token work.

UniServe and SGLang receive the same selected rows, tokenizer/model content, request rate, prompt construction, sampling controls, and per-row token budget. The fixed-work check requires matching request identity and server-accounted prompt and output token work. Generated text equality is not part of the benchmark comparison.

The shared request uses greedy decoding with neutral penalties, seed 42, thinking enabled, streaming usage, and no early EOS termination. This is a serving workload and does not measure task accuracy or sampled-response quality.

## Multimodal semantics

SenseNova T2I fixes seed 42, non-thinking mode, `t_eps=0.02`, 50 denoising updates, text guidance 4.0, image guidance 1.0, no guidance renormalization, guidance interval `[0.0,1.0]`, and timestep shift 3.0.

BAGEL T2I fixes seed 42 in both autoregressive and diffusion stages, non-thinking mode, 50 effective denoising updates, text guidance 4.0, image guidance 1.5, global guidance renormalization, guidance interval `[0.4,1.0]`, and timestep shift 3.0. The vLLM-Omni profile supplies 51 schedule points because its pinned scheduler removes the terminal point before performing 50 updates.

Both sides receive the same selected inputs, seeds, requested image count and dimensions, semantic denoising work, load case, and accelerator allocation. Every generated image must decode successfully and match its declared format, dimensions, and count before a T2I comparison is valid.

## Optional diagnostics

`--text-canary` compares termination metadata and generated-content fingerprints without retaining response text. It is a numerical regression diagnostic because valid kernel and batching choices can change greedy decisions near numerical ties.

`--image-smoke` pairs generated images by request and image index and reports deterministic LPIPS plus pixel-space diagnostics. Its fixed LPIPS threshold is a same-model numerical regression canary, not an absolute image-quality metric and not a performance acceptance condition.

Dataset-level text or image quality requires a separately defined evaluation with an appropriate sample set, reference, metric, and threshold. Optional diagnostics do not substitute for that evaluation.

## Fairness and provenance

Candidate and reference points use the same selected inputs, semantic request work, load case, accelerator count and model, and harness measurement code. Backend-specific scheduling, batching, kernels, graph capture, cache representation, memory layout, and serialization remain implementation differences.

The profile records model and dataset revisions, backend launch commands, numerical and capacity settings, cache policy, source revisions, environment fingerprints, selected accelerator, host topology, and content digests. If a reference runtime cannot expose an equivalent feature or effective runtime field, the mismatch remains explicit in the artifact.

Each point retains `run.json`, `requests.jsonl`, `gpu_samples.jsonl`, `summary.json`, `summary.md`, `artifact_manifest.json`, command and log files, and pre/post snapshots. Generated image bytes are content-addressed and bound into the artifact. A comparison is emitted only for a complete candidate/reference pair whose point artifacts and fixed-work checks are valid.

## Interpreting results

`candidate_over_reference` is always the UniServe metric divided by the reference metric. The result records whether the metric is minimized or maximized: lower ratios are better for `c1` image latency, while higher ratios are better for throughput metrics. A performance cause is reported only when profiling localizes the changed time and a controlled serial ablation changes that effect without changing the fixed protocol.
