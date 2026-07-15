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

| Workload | Systems | Requests per point | Load cases | Comparison metrics |
| --- | --- | ---: | --- | --- |
| Qwen3-32B ShareGPT | UniServe, SGLang | 200 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s, maximize; mean TTFT and mean TPOT, minimize |
| SenseNova-U1 MJHQ T2I | UniServe, vLLM-Omni | 32 | `c1`, `c32` | `c1`: mean image latency, minimize; `c32`: images/s, maximize |
| BAGEL MJHQ T2I | UniServe, vLLM-Omni | 32 | `c1`, `c32` | `c1`: mean image latency, minimize; `c32`: images/s, maximize |
| SenseNova-U1 Beans I2T | UniServe, vLLM-Omni | 32 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s |
| BAGEL Beans I2T | UniServe, vLLM-Omni | 32 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s, maximize |
| SenseNova-U1 mixed MJHQ T2I and Beans I2T | UniServe, vLLM-Omni | 32 per ratio | `image_light` 8:24, `balanced` 16:16, `image_heavy` 24:8; concurrency 32 | Fixed-mix requests/s, maximize; per-task throughput and latency |
| BAGEL mixed MJHQ T2I and Beans I2T | UniServe, vLLM-Omni | 32 per ratio | `image_light` 8:24, `balanced` 16:16, `image_heavy` 24:8; concurrency 32 | Fixed-mix requests/s, maximize; per-task throughput and latency |
| SenseNova-U1 MJHQ interleaved generation | UniServe | 32 | `c1`, `c32` | Text throughput, TTFT, TPOT, image latency, images/s, modality transitions |

The matrix contains 52 serial points. `rN` is an open-loop seed-42 Poisson trace with an offered rate of N requests/s and no client concurrency semaphore. `c1` submits the fixed prompt set immediately with client concurrency one. `c32` submits the fixed workload immediately with client concurrency 32. Every mixed ratio uses `c32`; its rows repeat the smallest integral T2I:I2T block, giving `T2I,I2T,I2T,I2T` for image-light, `T2I,I2T` for balanced, and `T2I,T2I,T2I,I2T` for image-heavy.

Timing covers the complete measured arrival and completion region. TTFT begins at client send and ends at the first non-empty content or reasoning delta. TPOT is calculated per request as `(E2E - TTFT) / (server-reported completion tokens - 1)` and then averaged, matching the SGLang serving definition. Output throughput is server-reported completion tokens divided by the complete timed region. Image latency is measured from request send through decoded output receipt, and image throughput is the number of successfully decoded images divided by the complete timed region. ShareGPT reports output throughput, mean TTFT, and mean TPOT as separate comparison metrics; the protocol defines no composite score.

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

Beans I2T fixes the selected JPEG inputs, prompt, preprocessing, sampling controls, and 256-token completion limit. BAGEL ignores EOS, so the limit defines fixed output work. SenseNova respects EOS, so the same value is an upper bound and throughput uses the completion tokens reported by each server. Both runtimes must report prompt and output usage, but prompt token counts are not compared across runtimes because multimodal backends account for expanded image tokens differently; the shared parity contract binds the selected images, request order, prompt, preprocessing, and completion limit.

Each mixed point submits 32 measured requests at `c32`. Image-light contains 8 MJHQ T2I and 24 Beans I2T requests with a 1:3 warm-up; balanced contains 16 and 16 with a 2:2 warm-up; image-heavy contains 24 and 8 with a 3:1 warm-up. Four excluded warm-up requests are used in every case. Measured and warm-up rows are proportionally interleaved in their smallest integral repeating block. Request construction, T2I generation controls, I2T sampling controls, output limits, model inputs, and preprocessing are inherited unchanged from the corresponding homogeneous workloads.

The mixed primary metric is the 32-request completion rate over the complete timed region. Paired results also compare I2T output tokens/s, mean TTFT, mean TPOT, T2I images/s, and mean image latency, while point artifacts retain the complete latency distributions, per-task completion counts, and client-visible cross-task active interval. The comparison is valid only when both systems complete the declared ratio with no failed requests and all T2I outputs pass the image work checks. SenseNova retains natural-EOS I2T semantics, so its artifacts must report realized output-token counts and its mixed request-rate ratio is not interpreted as a fixed-token throughput ratio when those counts differ.

SenseNova interleaved generation uses the same seed-42 MJHQ selection as T2I and a deterministic prompt transform that requests one introductory sentence, one generated image, and one closing sentence. Each request uses natural EOS, a 256-token limit, one-image cap, 2048×1152 output, and the SenseNova 50-update image controls. A request is conformant only when it emits visible text and a decoded image with at least one transition between the two output modalities; image spans are excluded from text ITL. The main matrix records UniServe characterization points because the pinned vLLM-Omni SenseNova endpoint does not expose equivalent single-request alternating text/image generation semantics, so no cross-runtime ratio is emitted for this workload.

## Optional diagnostics

`--text-canary` compares termination metadata and generated-content fingerprints without retaining response text. It is a numerical regression diagnostic because valid kernel and batching choices can change greedy decisions near numerical ties.

`--image-smoke` pairs generated images by request and image index and reports deterministic LPIPS plus pixel-space diagnostics. Its fixed LPIPS threshold is a same-model numerical regression canary, not an absolute image-quality metric and not a performance acceptance condition.

Dataset-level text or image quality requires a separately defined evaluation with an appropriate sample set, reference, metric, and threshold. Optional diagnostics do not substitute for that evaluation.

## Fairness and provenance

Candidate and reference points use the same selected inputs, semantic request work, load case, accelerator count and model, and harness measurement code. Backend-specific scheduling, batching, kernels, graph capture, cache representation, memory layout, and serialization remain implementation differences.

The profile records model and dataset revisions, backend launch commands, numerical and capacity settings, cache policy, source revisions, environment fingerprints, selected accelerator, host topology, and content digests. If a reference runtime cannot expose an equivalent feature or effective runtime field, the mismatch remains explicit in the artifact.

Each point retains `run.json`, `requests.jsonl`, `gpu_samples.jsonl`, `summary.json`, `summary.md`, `artifact_manifest.json`, command and log files, and pre/post snapshots. Generated image bytes are content-addressed and bound into the artifact. A comparison is emitted only for a complete candidate/reference pair whose point artifacts, parity contracts, and task work checks are valid.

## Interpreting results

`candidate_over_reference` is always the UniServe metric divided by the reference metric. The result records whether the metric is minimized or maximized: lower ratios are better for `c1` image latency, while higher ratios are better for throughput metrics. A performance cause is reported only when profiling localizes the changed time and a controlled serial ablation changes that effect without changing the fixed protocol.
