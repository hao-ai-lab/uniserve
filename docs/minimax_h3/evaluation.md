# Evaluation

The evaluator runs fixed workloads against the deployments in the [guide](minimax_h3.md) and validates every MP4: canvas, frame count, frame rate and stereo soundtrack. Each run writes its resolved commands, request data, per-request latencies, media validation and GPU memory observations under `artifacts/`. Resolve a plan before running it, and run points serially, one server and one harness at a time, from the checkout whose `.venv` holds the environment:

```bash
uv sync --locked --python /usr/bin/python3.12 --extra gpu --extra bench
```

| Profile | Contents |
| --- | --- |
| `uniserve_eval/fast_h3.toml` | The [UniServe FastH3 post](https://hao-ai-lab.github.io/blogs/uniserve-fasth3/) measurements |
| `uniserve_eval/fast_h3_h200.toml` | The H200 measurements below |
| `uniserve_eval/minimax_h3.toml` | The W1–W5 workloads across checkpoints and engines |

## FastH3 post

The workload is 72 latency requests one at a time, 12 for each combination of a 5, 10 or 15 second clip and a 1000- or 10000-token prompt, and 32 throughput requests at each concurrency after a priming phase, all preceded by six unmeasured warmup requests. The suites are `gb200-4-bf16`, `gb200-4-nvfp4`, `gb200-8-bf16`, `gb200-8-nvfp4`, `rtx-pro-6000-8-bf16` and `rtx-pro-6000-8-nvfp4`:

```bash
export UNISERVE_FAST_H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
export UNISERVE_FAST_H3_NVFP4_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2-NVFP4
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml plan gb200-4-bf16
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml run gb200-4-bf16
```

The `gb200-8` suites start the head on `rank-0`; start `uniserve-host` on `rank-1` for each server the suite starts, as in [Two hosts](minimax_h3.md#two-hosts).

## H200

These use `FastVideo/FastVideo-FastH3-8-Step-V2` at revision `3da2ddfe1954d9cda4c05b643dc0f26007a655c5`, the evaluator's synthesized prompts with seed 1000, full graphs, `--mem-fraction-static 0.92` and `NCCL_CUMEM_HOST_ENABLE=1`. Latency is the mean ± SD of three requests at concurrency one after one warmup; the DP8 points use eight warmups and 16 requests at concurrency eight.

| Deployment | Precision | Workload | Result |
| --- | --- | --- | --- |
| Eight-H200 Ulysses | `quality` | 5 s, 1000 tokens | 8.194 ± 0.818 s |
| Eight-H200 Ulysses | `quality` | 5 s, 10000 tokens | 14.218 ± 0.880 s |
| Eight-H200 Ulysses | `quality` | 10 s, 1000 tokens | 18.539 ± 1.071 s |
| Eight-H200 Ulysses | `quality` | 10 s, 10000 tokens | 26.578 ± 0.600 s |
| Eight-H200 Ulysses | `quality` | 15 s, 1000 tokens | 31.069 ± 0.045 s |
| Eight-H200 Ulysses | `quality` | 15 s, 10000 tokens | 44.228 ± 0.878 s |
| Four-H200 Ulysses | `quality` | 5 s, 1000 tokens | 13.864 ± 0.013 s |
| Gather8 | `quality` | 5 s, 10000 tokens | 20.170 ± 0.698 s |
| Eight-H200 Ulysses | FP8 MLPs | 5 s, 1000 tokens | 7.448 ± 0.734 s |
| Eight-H200 Ulysses | FP8 MLPs | 15 s, 10000 tokens | 42.380 ± 0.807 s |
| DP8 + text TP8 | `quality` | 5 s, 1000 tokens, concurrency 8 | 0.1519 videos/s, 52.231 ± 0.358 s, 90,329 MiB peak per GPU |
| DP8 + text TP8 | FP8 MLPs | 5 s, 1000 tokens, concurrency 8 | 0.1676 videos/s, 47.314 ± 0.386 s, 81,531 MiB peak per GPU |

The `h200-ulysses8`, `h200-ulysses4` and `h200-fp8` suites and the `h200-gather8-5s-10k` and `h200-dp8-5s-1k` points reproduce them:

```bash
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3_h200.toml run h200-ulysses8 --reuse-deployment
```

## MiniMax-H3 workloads

`uniserve_eval/minimax_h3.toml` runs these workloads on UniServe, SGLang, vLLM-Omni and FastVideo:

| Workload | Request | Measured seeds |
| --- | --- | --- |
| W1 | t2va, official prompt, 1344×768, 5 seconds (124 frames) | 0–7 |
| W2 | t2va, official prompt, 1344×768, 15 seconds (362 frames) | 0–3 |
| W3 | fl2va, official first keyframe, automatic canvas, 8 seconds (192 frames) | 0–7 |
| W4 | ref2va, reference image and voice, 5 seconds | 0–7 |
| W5 | ref2va, reference video with its soundtrack and a reference voice, 5 seconds | 0–7 |

Each point sends one unmeasured warmup at seed 42, then its measured requests one at a time. The manifests in `uniserve_eval/workloads/minimax_h3/` hold every request; each row's `metadata.condition_media` gives every condition file's path under `UNISERVE_MINIMAX_H3_INPUTS`, its source URL and its SHA-256. Reference media decode with FFmpeg 8.1.2.

```bash
export UNISERVE_MINIMAX_H3_MODEL=/workspace/models/MiniMax-H3
export UNISERVE_MINIMAX_H3_INPUTS=/workspace/uniserve/artifacts/minimax_h3/inputs
export UNISERVE_MINIMAX_H3_FFMPEG=/workspace/tools/ffmpeg-8.1.2/bin
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml plan gb200-4-w3
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run gb200-4-w3
```

The `gb200-4`, `gb200-4-reference`, `gb200-4-omniref`, `fast-h3-gb200-4` and `gb200-8` suites group points by checkpoint and placement. OmniRef points also need `UNISERVE_MINIMAX_H3_OMNIREF` and its pinned base copy `UNISERVE_MINIMAX_H3_OMNIREF_BASE`; two-host points need `UNISERVE_MINIMAX_H3_HEAD_ADDRESS`.

Cross-engine results are not equal-work ratios: the engines differ in reference-image resizing, noise generators, media codecs and distributed attention. vLLM-Omni's W4 image processing keeps a smaller condition layout than the others, and its diffusion worker does not implement the two-host placement.
