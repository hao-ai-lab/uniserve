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

`--resume` continues a non-formal output root by skipping only points whose canonical artifact bundle, harness contract, matrix contract, and execution-bundle identity still match the selected profile. A missing, invalid, or stale point runs in place and the aggregate is rebuilt from the complete selected point set. Formal execution requires a fresh root and a complete matrix, so it does not permit `--resume` or point-level filters.

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

SenseNova interleaved generation uses the same seed-42 MJHQ selection as T2I and a deterministic prompt transform that requests one introductory sentence, one generated image, and one closing sentence. Each request uses natural EOS, the model serving interface's 4096-token generation limit, one-image cap, 2048×1152 output, and the SenseNova 50-update image controls. A request is conformant only when it emits visible text and a decoded image with at least one transition between the two output modalities; image spans are excluded from text ITL. The main matrix records UniServe characterization points because the pinned vLLM-Omni SenseNova endpoint does not expose equivalent single-request alternating text/image generation semantics, so no cross-runtime ratio is emitted for this workload.

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

## Recorded reference results

The following snapshot records the complete 52-point matrix under `artifacts/benchmark/runs/20260715T0855Z` on one NVIDIA GB200. Every point completed its declared request count with no failed requests and passed its point-artifact checks. Each point was measured once, so these values have no repeat-derived uncertainty interval. The sum of measured point durations was 8,285.32 seconds; server startup, shutdown, build, and validation time is excluded.

The snapshot is bound by point manifests rather than by one workspace source revision. Qwen3 points and UniServe SenseNova T2I/I2T points used `0be2b9279ecec8129218cb1c1e4312c5783d34a6`; vLLM-Omni SenseNova T2I/I2T points, SenseNova interleaved points, and the UniServe SenseNova image-light mixed point used `f9e95290e9c75735018dd98a6b56a482205987e3`; all BAGEL points and the remaining SenseNova mixed points used `4a451de7369fdd8b0b4e061f72e13fbe525790f2`. Reference runtime sources were pinned independently to SGLang `c6a7c98ae429760ed3b2df8d3a11600c3855d74a` and vLLM-Omni `b0f6b9de9691eabb202b80aa09040a6b0209d5bf`. It is therefore a per-point reference snapshot, not a single-revision formal measurement or evidence for the source revision containing this document.

Cells containing two values use `UniServe / reference (UniServe/reference ratio)`. The reference is SGLang for Qwen3 and vLLM-Omni for multimodal workloads.

### Qwen3-32B ShareGPT

| Load case | Output tokens/s | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: |
| `r1` | 227.45 / 227.48 (1.000×) | 59.63 / 59.48 (1.002×) | 12.57 / 12.43 (1.011×) |
| `r2` | 434.37 / 434.44 (1.000×) | 62.63 / 62.34 (1.005×) | 13.08 / 12.93 (1.012×) |
| `r4` | 796.53 / 796.99 (0.999×) | 67.77 / 67.87 (0.999×) | 14.41 / 14.08 (1.024×) |
| `r8` | 1,355.12 / 1,357.00 (0.999×) | 80.56 / 80.29 (1.003×) | 16.91 / 16.24 (1.041×) |
| `r16` | 1,857.14 / 1,907.33 (0.974×) | 104.19 / 107.32 (0.971×) | 23.99 / 21.39 (1.122×) |

The measured `r16` point has 2.6% lower UniServe output throughput and 12.2% higher mean TPOT than SGLang, while UniServe mean TTFT is 2.9% lower. The snapshot does not establish a UniServe ShareGPT performance advantage.

### Homogeneous T2I

| Model | Load case | Primary metric | UniServe / vLLM-Omni | Ratio |
| --- | --- | --- | ---: | ---: |
| SenseNova-U1 | `c1` | Mean image latency (ms) | 3,759.051 / 7,384.635 | 0.509× |
| SenseNova-U1 | `c32` | Images/s | 0.240 / 0.157 | 1.533× |
| BAGEL | `c1` | Mean image latency (ms) | 6,913.954 / 10,060.112 | 0.687× |
| BAGEL | `c32` | Images/s | 0.139 / 0.103 | 1.354× |

### Homogeneous I2T

| Model | Load case | Output tokens/s | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | --- | ---: | ---: | ---: |
| SenseNova-U1 | `r1` | 306.59 / 11.55 (26.56×) | n/a / n/a | n/a / n/a |
| SenseNova-U1 | `r2` | 582.10 / 10.06 (57.87×) | n/a / n/a | n/a / n/a |
| SenseNova-U1 | `r4` | 913.29 / 10.36 (88.17×) | n/a / n/a | n/a / n/a |
| SenseNova-U1 | `r8` | 1,141.38 / 10.83 (105.42×) | n/a / n/a | n/a / n/a |
| SenseNova-U1 | `r16` | 1,186.89 / 10.27 (115.60×) | n/a / n/a | n/a / n/a |
| BAGEL | `r1` | 309.23 / 204.51 (1.51×) | 443.77 / 5,069.62 | 6.42 / 12.20 |
| BAGEL | `r2` | 572.58 / 215.83 (2.65×) | 575.19 / 10,251.05 | 11.56 / 12.53 |
| BAGEL | `r4` | 869.20 / 209.41 (4.15×) | 1,390.04 / 14,389.80 | 13.71 / 13.43 |
| BAGEL | `r8` | 998.29 / 215.95 (4.62×) | 2,249.01 / 16,009.03 | 11.45 / 13.02 |
| BAGEL | `r16` | 1,132.09 / 216.27 (5.23×) | 2,783.77 / 17,097.92 | 7.27 / 13.01 |

BAGEL completed 8,192 output tokens on each backend at every load case, so its output-throughput ratios compare fixed output work. SenseNova completed 8,192 output tokens on UniServe and 5,378 on vLLM-Omni at every load case under the shared 256-token upper bound and natural-EOS policy; its rows report observed token rates but do not constitute fixed-output-work ratios. Neither SenseNova endpoint exposed homogeneous-I2T token timing through the measured interface.

### Mixed T2I and I2T at `c32`

| Model | Mix | Requests/s | I2T output tokens/s | I2T TTFT (ms) | I2T TPOT (ms) | T2I images/s | T2I image latency (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SenseNova-U1 | `image_light` | 0.736 / 0.079 (9.36×) | 140.87 / 10.01 (14.08×) | 13,299 / n/a | 26.84 / n/a | 0.184 / 0.020 (9.36×) | 30,578 / 183,412 (0.17×) |
| SenseNova-U1 | `balanced` | 0.437 / 0.090 (4.86×) | 55.97 / 7.66 (7.31×) | 21,559 / n/a | 25.93 / n/a | 0.219 / 0.045 (4.86×) | 47,265 / 168,861 (0.28×) |
| SenseNova-U1 | `image_heavy` | 0.298 / 0.113 (2.64×) | 19.08 / 4.81 (3.97×) | 34,321 / n/a | 24.04 / n/a | 0.224 / 0.085 (2.64×) | 65,504 / 134,629 (0.49×) |
| BAGEL | `image_light` | 0.539 / 0.326 (1.65×) | 103.42 / 62.63 (1.65×) | 4,093 / 20,847 | 16.61 / 19.36 | 0.135 / 0.082 (1.65×) | 39,267 / 62,094 (0.63×) |
| BAGEL | `balanced` | 0.281 / 0.188 (1.49×) | 35.99 / 24.09 (1.49×) | 3,998 / 14,734 | 16.20 / 19.34 | 0.141 / 0.094 (1.49×) | 66,439 / 96,933 (0.69×) |
| BAGEL | `image_heavy` | 0.190 / 0.133 (1.43×) | 12.19 / 8.52 (1.43×) | 3,547 / 8,915 | 15.85 / 18.67 | 0.143 / 0.100 (1.43×) | 93,212 / 129,103 (0.72×) |

All six mixed points on each backend completed the declared request composition and image checks. The three BAGEL paired comparisons contain every declared metric. The three SenseNova paired comparisons are incomplete because the vLLM-Omni response path did not expose I2T TTFT or TPOT; the available throughput and image measurements are retained, but no complete paired-comparison validity is claimed. SenseNova also produced different realized I2T output-token totals across backends under natural EOS, as allowed and disclosed by the protocol.

### SenseNova-U1 interleaved generation

| Load case | Output tokens/s | Mean TTFT (ms) | Mean TPOT (ms) | Images/s | Mean image latency (ms) | Conformant pattern |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `c1` | 154.69 | 66.53 | 6.80 | 0.068 | 8,079 | 32 / 32 `text→image→text` |
| `c32` | 287.69 | 78,427.09 | 39.54 | 0.127 | 126,713 | 32 / 32 `text→image→text` |

These are UniServe characterization points without a vLLM-Omni ratio. All 32 requests at each load case emitted the required text-image-text modality pattern.
