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
| SenseNova-U1 Beans I2T | UniServe, vLLM-Omni | 32 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s, maximize |
| BAGEL Beans I2T | UniServe, vLLM-Omni | 32 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s, maximize |
| SenseNova-U1 UEval interleaved generation | UniServe | 32 | `r0.1`, `r0.2`, `r0.4`, `r0.8`, `r1` | Text throughput, TTFT, TPOT, image latency, images/s, modality transitions |

The matrix contains 43 serial points. `rN` is an open-loop seed-42 Poisson trace with an offered rate of N requests/s and no client concurrency semaphore. `c1` submits the fixed prompt set immediately with client concurrency one. `c32` submits the fixed workload immediately with client concurrency 32.

Timing covers the complete measured arrival and completion region. TTFT begins at client send and ends at the first non-empty content or reasoning delta. TPOT is calculated per request as `(E2E - TTFT) / (server-reported completion tokens - 1)` and then averaged, matching the SGLang serving definition. Output throughput is server-reported completion tokens divided by the complete timed region. Image latency is measured from request send through decoded output receipt, and image throughput is the number of successfully decoded images divided by the complete timed region. ShareGPT reports output throughput, mean TTFT, and mean TPOT as separate comparison metrics; the protocol defines no composite score.

## Capacity parity

Capacity and scheduling settings are declared explicitly on both sides of every comparison so that the load case, not a server configuration difference, determines the offered concurrency:

| Setting | UniServe | SGLang | vLLM-Omni |
| --- | --- | --- | --- |
| Maximum concurrent running requests | `--max-running-requests 32` | `--max-running-requests 32` | `--max-num-seqs 32` (BAGEL: per stage in the deploy config) |
| Per-step batched token budget | `--max-num-batched-tokens`: 16384 text, 4096 multimodal | `--max-prefill-tokens 16384`, `--chunked-prefill-size 16384` | BAGEL: `max_num_batched_tokens: 4096` per stage |
| Static memory fraction | `--mem-fraction-static 0.70` | `--mem-fraction-static 0.70` | BAGEL: `gpu_memory_utilization: 0.45` on the Thinker stage |
| KV page size | `--page-size 64` | `--page-size 64` | Backend-selected |
| Model dtype | `--dtype bfloat16` | `--dtype bfloat16` | `dtype: bfloat16` |
| KV cache dtype | `--kv-cache-dtype bfloat16` | `--kv-cache-dtype bfloat16` | Not exposed for omni stages |
| Prefix cache | Enabled | Radix cache enabled | BAGEL: enabled on the Thinker stage |

32 is the matrix's maximum offered client concurrency and also the measured request count of every multimodal point, so no server admits fewer requests than its load case offers. The text budget of 16384 is the value SGLang derives automatically for this GPU class, so both text servers run at the reference's own operating point.

Both runtimes select their own attention backends from their own capability checks. UniServe supplies no backend override and neither reference server pins one, so each runtime uses the kernel it considers best for the model and device.

Where a reference runtime has no equivalent knob, the mismatch is explicit rather than silently absent. `vllm serve --omni` derives KV capacity from a memory fraction rather than a token count, its diffusion stages expose no memory-fraction field, and it accepts no KV-cache dtype for those stages. The SenseNova-U1 endpoint resolves to a single diffusion stage whose scheduler admits one request at a time unless `--max-num-seqs` is supplied, so that flag is a required part of the reference launch rather than an optional tuning choice.

## ShareGPT semantics

The Qwen workload follows the pinned SGLang ShareGPT behavior implemented in [`refs/sglang/python/sglang/benchmark/datasets/sharegpt.py`](../refs/sglang/python/sglang/benchmark/datasets/sharegpt.py) and [`refs/sglang/python/sglang/benchmark/serving.py`](../refs/sglang/python/sglang/benchmark/serving.py):

1. Keep conversations with at least two turns and use the first turn as the prompt and the second turn as the trace completion.
2. Shuffle with the configured seed and retain the first requested rows that pass the length filters.
3. Tokenize the prompt and trace completion with the selected model tokenizer.
4. Set each request's output-token budget from the trace completion token count.
5. Ignore EOS so each successful request performs that token work.

UniServe and SGLang receive the same selected rows, tokenizer/model content, request rate, prompt construction, sampling controls, and per-row token budget. The fixed-work check requires matching request identity and server-accounted prompt and output token work. Generated text equality is not part of the benchmark comparison.

The shared request uses greedy decoding with neutral penalties, seed 42, thinking enabled, streaming usage, and no early EOS termination. This is a serving workload and does not measure task accuracy or sampled-response quality.

SGLang requires an explicit `--reasoning-parser qwen3` to separate thinking output into `reasoning_content`; UniServe selects the same parser from the model. Token accounting is identical either way, because both runtimes count thinking tokens in the reported completion total.

## Multimodal semantics

SenseNova T2I fixes seed 42, non-thinking mode, `t_eps=0.02`, 50 denoising updates, text guidance 4.0, image guidance 1.0, no guidance renormalization, guidance interval `[0.0,1.0]`, and timestep shift 3.0.

BAGEL T2I fixes seed 42 in both autoregressive and diffusion stages, non-thinking mode, 50 effective denoising updates, text guidance 4.0, image guidance 1.5, global guidance renormalization, guidance interval `[0.4,1.0]`, and timestep shift 3.0. The vLLM-Omni profile supplies 51 schedule points because its pinned scheduler removes the terminal point before performing 50 updates.

Both sides receive the same selected inputs, seeds, requested image count and dimensions, semantic denoising work, load case, and accelerator allocation. Every generated image must decode successfully and match its declared format, dimensions, and count before a T2I comparison is valid.

Beans I2T fixes the selected JPEG inputs, prompt, preprocessing, and sampling controls. Both runtimes terminate on the model's own EOS and share a non-binding 8192-token output ceiling, so response length is a model property rather than a protocol constant, and both sides are measured under the same termination rule. Throughput uses the completion tokens each server reports. The paired work check therefore binds request identity, the selected images, request order, prompt, preprocessing, and the shared output ceiling, and treats realized output length as measured data. Prompt token counts are not compared across runtimes because multimodal backends account for expanded image tokens differently. Because realized output length varies with the response, an I2T throughput ratio is a service-rate ratio and is interpreted alongside the realized token totals recorded in each artifact.

SenseNova interleaved generation uses [UEval](https://huggingface.co/datasets/zlab-princeton/UEval), whose prompts are authored to require both text and images in a single answer, so the modality alternation under measurement comes from the request rather than from a harness-side instruction. Each request uses the same seed-42 selection procedure as the other workloads, natural EOS, an 8192-token generation ceiling, a one-image cap, 2048×1152 output, and the SenseNova 50-update image controls. A request is conformant only when it emits visible text and a decoded image with at least one transition between the two output modalities; image spans are excluded from text ITL. The matrix records UniServe characterization points because the pinned vLLM-Omni SenseNova endpoint does not expose equivalent single-request alternating text/image generation semantics, so no cross-runtime ratio is emitted for this workload.

## Optional diagnostics

`--text-canary` compares termination metadata and generated-content fingerprints without retaining response text. It is a numerical regression diagnostic because valid kernel and batching choices can change greedy decisions near numerical ties.

`--image-smoke` pairs generated images by request and image index and reports deterministic LPIPS plus pixel-space diagnostics. Its fixed LPIPS threshold is a same-model numerical regression canary, not an absolute image-quality metric and not a performance acceptance condition.

Dataset-level text or image quality requires a separately defined evaluation with an appropriate sample set, reference, metric, and threshold. Optional diagnostics do not substitute for that evaluation.

## Fairness and provenance

Candidate and reference points use the same selected inputs, semantic request work, load case, accelerator count and model, harness measurement code, and the capacity settings tabulated above. Backend-specific scheduling, batching, kernels, graph capture, cache representation, memory layout, and serialization remain implementation differences.

The profile records model and dataset revisions, backend launch commands, numerical and capacity settings, cache policy, source revisions, environment fingerprints, selected accelerator, host topology, and content digests. If a reference runtime cannot expose an equivalent feature or effective runtime field, the mismatch remains explicit in the artifact.

Each point retains `run.json`, `requests.jsonl`, `gpu_samples.jsonl`, `summary.json`, `summary.md`, `artifact_manifest.json`, command and log files, and pre/post snapshots. Generated image bytes are content-addressed and bound into the artifact. A comparison is emitted only for a complete candidate/reference pair whose point artifacts, parity contracts, and task work checks are valid.

## Interpreting results

`candidate_over_reference` is always the UniServe metric divided by the reference metric. The result records whether the metric is minimized or maximized: lower ratios are better for `c1` image latency, while higher ratios are better for throughput metrics. A performance cause is reported only when profiling localizes the changed time and a controlled serial ablation changes that effect without changing the fixed protocol.
