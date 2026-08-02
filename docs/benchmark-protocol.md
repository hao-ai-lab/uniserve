# Benchmark protocol

The benchmark matrix is defined in [`uniserve_eval/profiles.json`](../uniserve_eval/profiles.json) and executed by [`scripts/run_benchmarks.py`](../scripts/run_benchmarks.py). The profile fixes workload selection, request semantics, load cases, model and dataset identity, server profiles, hardware requirements, comparison metrics, and artifact validation before measurement.

## Running the matrix

```bash
.venv/bin/python scripts/run_benchmarks.py \
  --benchmark main \
  --output-root artifacts/benchmark
```

Use `--only` to select complete backend groups, `--only-bench` to select exact expanded points, `--dry-run --no-build` to inspect commands without launching servers, and `--formal` to require the canonical clean-source, pinned-revision, fresh-build, accelerator, NUMA, process, GPU, and execution-identity checks.

`--repeat N` executes complete selected matrices serially beneath numbered child roots and produces an aggregate containing every valid per-run ratio and its geometric mean. `--text-canary` and `--image-smoke` enable optional diagnostics; neither changes point validity or the primary metric.

## Execution model

A host-wide lock spans the command. The runner permits at most one benchmark server and one harness process at a time. Every point starts from a fresh server, and the server is stopped and its artifact validated before the next backend, workload, load case, or repeated matrix begins.

A failed point stops the run and preserves its partial artifacts and logs. The failure is diagnosed before artifact-producing execution continues.

`--resume` continues a non-formal output root by skipping only points whose canonical artifact bundle, harness contract, matrix contract, and execution-bundle identity still match the selected profile. A missing, invalid, or stale point runs in place and the aggregate is rebuilt from the complete selected point set. Formal execution requires a fresh root and a complete matrix, so it does not permit `--resume` or point-level filters.

## Generation-runtime qualification

The benchmark harness executes profile-selected requests and emits measurements and integrity evidence. It does not choose construction checks, comparator roots, regression thresholds, architecture claims, or candidate acceptance. Those experiment-control decisions are defined by [`specs/generation-runtime-qualification.md`](../specs/generation-runtime-qualification.md) and materialized by the immutable candidate manifest described in [`specs/decode-runtime-construction.md`](../specs/decode-runtime-construction.md).

Each formal qualification sequence uses one immutable candidate source state and one fresh artifact root, executes points serially, permits no automatic retry, and stops after the first invalid or failing point. A failed point is diagnosed before affected measurement continues. Successful evidence remains valid only while every bound source, executable, configuration, workload, model, hardware, dependency, and profile input remains unchanged.

## Workload matrix

| Workload | Systems | Requests per point | Load cases | Comparison metrics |
| --- | --- | ---: | --- | --- |
| Qwen3-32B ShareGPT | UniServe, SGLang | 200 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s, maximize; mean TTFT and mean TPOT, minimize |
| SenseNova-U1 MJHQ T2I | UniServe, vLLM-Omni | 32 | `c1`, `c32` | `c1`: mean image latency, minimize; `c32`: images/s, maximize |
| BAGEL MJHQ T2I | UniServe, vLLM-Omni | 32 | `c1`, `c32` | `c1`: mean image latency, minimize; `c32`: images/s, maximize |
| SenseNova-U1 Beans I2T | UniServe, vLLM-Omni | 32 | `r1`, `r2`, `r4`, `r8`, `r16`, `c32` | Output tokens/s, maximize |
| BAGEL Beans I2T | UniServe, vLLM-Omni | 32 | `r1`, `r2`, `r4`, `r8`, `r16` | Output tokens/s, maximize |
| SenseNova-U1 UEval interleaved generation | UniServe | 32 | `c1`, `c2`, `c4`, `c8`, `c16` | TTFT, TPOT, image latency, aggregate transition latency, directional transition latency |
| SenseNova-U1 stochastic UEval interleaved generation | UniServe | 32 | `c4` | Depth-one semantic parity, lifecycle and continuation evidence, forbidden synchronization, TTFT, TPOT, image latency, aggregate transition latency, directional transition latency |

The matrix contains 46 serial points. `rN` is an open-loop seed-42 Poisson trace with an offered rate of N requests/s and no client concurrency semaphore. `cN` submits the fixed prompt set immediately under a client concurrency of N, so the server sees exactly N requests in flight until the set is exhausted.

Timing covers the complete measured arrival and completion region. TTFT begins at client send and ends at the first non-empty content or reasoning delta. TPOT is calculated per request as `(E2E - TTFT) / (server-reported completion tokens - 1)` and then averaged, matching the SGLang serving definition. Output throughput is server-reported completion tokens divided by the complete timed region. Image latency is measured from request send through decoded output receipt, and image throughput is the number of successfully decoded images divided by the complete timed region. ShareGPT reports output throughput, mean TTFT, and mean TPOT as separate comparison metrics; the protocol defines no composite score.

Every visible UEval SSE event is timestamped with the benchmark client's monotonic clock before parsing. Consecutive visible events of one modality form a segment; transition latency is the destination segment's first event timestamp minus the source segment's last event timestamp. UEval reports complete distributions for TTFT, TPOT, decoded-image latency, aggregate transition latency, text-to-image transition latency, and image-to-text transition latency. A point is invalid unless all 32 requests conform, every realized visible event has a timestamp, every realized transition is nonnegative and unambiguous, every realized sample is present, and each request records its ordered modality-segment signature. Candidate and baseline may contain different realized output lengths, image counts, transition counts, and signatures while remaining comparable under an identical normalized request-specification digest, selected-row digest and count, benchmark-definition fingerprint, and load case. Dataset content is bound by its revision and selected rows; its artifact-local extraction path is provenance rather than request work.

## Capacity parity

Capacity and scheduling settings are declared explicitly on both sides of every comparison so that the load case, not a server configuration difference, determines the offered concurrency:

| Setting | UniServe | SGLang | vLLM-Omni |
| --- | --- | --- | --- |
| Maximum concurrent running requests | `--max-running-requests 128` | `--max-running-requests 128` | `--max-num-seqs 32` (BAGEL: per stage in the deploy config) |
| Per-step batched token budget | `--max-num-batched-tokens`: 16384 text, 8192 multimodal | `--max-prefill-tokens 16384`, `--chunked-prefill-size 16384` | BAGEL: `max_num_batched_tokens: 8192` per stage |
| Static memory fraction | `--mem-fraction-static 0.70` | `--mem-fraction-static 0.70` | BAGEL: `gpu_memory_utilization: 0.45` on the Thinker stage |
| KV page size | `--page-size 64` | `--page-size 64` | Backend-selected |
| Model dtype | `--dtype bfloat16` | `--dtype bfloat16` | `dtype: bfloat16` |
| KV cache dtype | `--kv-cache-dtype bfloat16` | `--kv-cache-dtype bfloat16` | Not exposed for omni stages |
| Prefix cache | Enabled | Radix cache enabled | BAGEL: enabled on the Thinker stage |

The text capacity of 128 preserves the complete open-loop arrival burst without imposing server admission backpressure, while 32 is the maximum offered client concurrency and measured request count of every multimodal point. The budget of 16384 is the value SGLang derives automatically for this GPU class, so both text servers run at the reference's own operating point.

The multimodal budget of 8192 is set by the vLLM-Omni BAGEL stages, which admit a multimodal item whole rather than across steps. The declared modality limits restrict those stages to the image-to-text input the matrix actually sends, so the binding item is the 4900-token vision encoding of one input image rather than a larger editing item the model also supports. UniServe applies the same budget as a per-step scheduling budget over planned token work across its separately declared encoder, decoder, and flow routes.

On both text servers the static memory fraction bounds long-lived device memory as a share of the device total and leaves the remainder for transient work. UniServe counts resident weights, every KV pool, and its captured graph executables inside that share; SGLang counts weights and its KV pool. Both therefore size KV capacity from the same declared share of the same device, and both land above the peak resident ShareGPT request count, so the accounting difference does not bind at any point in this matrix.

UniServe and SGLang select their own attention backends from their own capability checks, and the SenseNova reference does the same, so those servers use the kernel each runtime considers best for the model and device. The BAGEL reference pins Triton attention for its language model and torch SDPA for its vision encoder: the CUTLASS flash-attention kernels it would otherwise select fail to compile on this accelerator, so the pins are what make the reference runnable rather than a tuning choice.

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

SenseNova T2I fixes seed 42, non-thinking mode, `t_eps=0.02`, 50 denoising updates, text guidance 4.0, image guidance 1.0, no guidance renormalization, guidance interval `[0.0,1.0]`, and timestep shift 3.0. Each request states these controls under both the canonical `image_config` object and the request-root field names the reference chat implementation reads, so one declared operating point reaches either server.

`t_eps` is the clamp applied when converting an x-prediction into a velocity at the end of the denoising schedule. vLLM-Omni applies the declared value; UniServe applies its own numerical floor and exposes no request-level control, so the two runtimes differ in the terminal step of an otherwise identical schedule. The difference changes generated image content, not the amount of work per request, so latency and throughput stay comparable while image content is not expected to match across runtimes.

BAGEL T2I fixes seed 42 in both autoregressive and diffusion stages, non-thinking mode, 50 effective denoising updates, text guidance 4.0, image guidance 1.5, global guidance renormalization, guidance interval `[0.4,1.0]`, and timestep shift 3.0. The vLLM-Omni profile supplies 51 schedule points because its pinned scheduler removes the terminal point before performing 50 updates.

Both sides receive the same selected inputs, seeds, requested image count and dimensions, semantic denoising work, load case, and accelerator allocation. Every generated image must decode successfully and match its declared format, dimensions, and count before a T2I comparison is valid.

Beans I2T fixes the selected JPEG inputs, prompt, preprocessing, and sampling controls. Both runtimes terminate on the model's own EOS and share a non-binding 8192-token output ceiling, so response length is a model property rather than a protocol constant, and both sides are measured under the same termination rule. Throughput uses the completion tokens each server reports. The paired work check therefore binds request identity, the selected images, request order, prompt, preprocessing, and the shared output ceiling, and treats realized output length as measured data. Prompt token counts are not compared across runtimes because multimodal backends account for expanded image tokens differently. Because realized output length varies with the response, an I2T throughput ratio is a service-rate ratio and is interpreted alongside the realized token totals recorded in each artifact.

SenseNova interleaved generation uses [UEval](https://huggingface.co/datasets/zlab-princeton/UEval), whose prompts are authored to require both text and images in a single answer, so the modality alternation under measurement comes from the request rather than from a harness-side instruction. Each request uses the same seed-42 selection procedure as the other workloads, natural EOS, an 8192-token generation ceiling, 2048×1152 output, and the SenseNova 50-update image controls. The request states no image count, so a prompt produces as many images as its answer calls for and the measured image work is a model property rather than a protocol constant. A request is conformant only when it emits visible text and a decoded image with at least one transition between the two output modalities; image spans are excluded from text ITL. The matrix records UniServe characterization points because the pinned vLLM-Omni SenseNova endpoint does not expose equivalent single-request alternating text/image generation semantics, so no cross-runtime ratio is emitted for this workload.

The stochastic UEval point uses the same dataset selection, output ceiling, image controls, topology, and load semantics with the fixed non-greedy processor controls in the qualification specification. It is an architecture-claim profile: the runtime trace and depth-one oracle are required evidence, and greedy UEval or default travel cannot substitute for it.

## Optional diagnostics

`--text-canary` compares termination metadata and generated-content fingerprints without retaining response text. It is a numerical regression diagnostic because valid kernel and batching choices can change greedy decisions near numerical ties.

`--image-smoke` pairs generated images by request and image index and reports deterministic LPIPS plus pixel-space diagnostics. Its fixed LPIPS threshold is a same-model numerical regression canary, not an absolute image-quality metric and not a performance acceptance condition.

Dataset-level text or image quality requires a separately defined evaluation with an appropriate sample set, reference, metric, and threshold. Optional diagnostics do not substitute for that evaluation.

## Fairness and provenance

Candidate and reference points use the same selected inputs, request caps, sampling and quality controls, load case, accelerator count and model, harness measurement code, and the capacity settings tabulated above. Backend-specific scheduling, batching, kernels, graph capture, cache representation, memory layout, serialization, and realized generation trajectories remain implementation differences.

The profile records model and dataset revisions, backend launch commands, numerical and capacity settings, cache policy, source revision and change digest, environment fingerprints, selected accelerator, host topology, and content digests. If a reference runtime cannot expose an equivalent feature or effective runtime field, the mismatch remains explicit in the artifact.

Each point retains `run.json`, `requests.jsonl`, `gpu_samples.jsonl`, `summary.json`, `summary.md`, `artifact_manifest.json`, command and log files, and pre/post snapshots. Generated image bytes are content-addressed and bound into the artifact. A comparison is emitted only for a complete candidate/reference pair whose point artifacts, parity contracts, requested-workload contracts, and output-conformance checks are valid.

## Interpreting results

`candidate_over_reference` is always the UniServe metric divided by the reference metric. The result records whether the metric is minimized or maximized: lower ratios are better for `c1` image latency, while higher ratios are better for throughput metrics. A performance cause is reported only when profiling localizes the changed time and a controlled serial ablation changes that effect without changing the fixed protocol.
