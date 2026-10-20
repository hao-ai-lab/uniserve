# Evaluation

`uniserve-eval` starts a server, sends a fixed workload, validates every MP4 (canvas, frame count, frame rate and soundtrack), and writes each request's latency and stage timings under `artifacts/`. It installs with the `bench` extra:

```bash
uv sync --locked --python 3.12 --extra gpu --extra bench
.venv/bin/uniserve-eval --config uniserve_eval/fasth3.toml plan gb200-4-bf16
.venv/bin/uniserve-eval --config uniserve_eval/fasth3.toml run gb200-4-bf16 --reuse-deployment
```

| Profile | Measures |
| --- | --- |
| `uniserve_eval/fasth3.toml` | FastH3 text-to-video latency and throughput on GB200, H200 and RTX PRO 6000 |
| `uniserve_eval/fasth3_omniref.toml` | FastH3 OmniRef reference-to-video on GB200, H200 and RTX PRO 6000 |
| `uniserve_eval/minimax_h3.toml` | The MiniMax-H3 checkpoint on GB200 |

Each profile's header names the environment variables it reads. Suites are named `<gpu>-<count>[-<variant>]`; run them one at a time. For two-host suites, start `uniserve-host` on `rank-1` as in [Two hosts](minimax_h3.md#two-hosts).

## FastH3 OmniRef on GB200

16 reference requests (one to three images, up to one reference video and up to two audio references; 5, 10 and 15 s; 16:9 and 9:16), each at 768p and 480p, on one replica of one to eight GB200 with the [validated settings](minimax_h3.md#deployments). Each cell is the mean end-to-end seconds of three requests at concurrency one after a warmup, MP4 encoding included; no cell's standard deviation exceeds 0.39 s.

768p:

| # | Request | 1 x GB200 | 2 x GB200 | 4 x GB200 | 8 x GB200 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 5 s, 16:9, 1 image + 2 audio | 22.10 | 12.42 | 6.60 | 3.91 |
| 2 | 5 s, 16:9, 3 images | 24.41 | 13.55 | 7.12 | 4.12 |
| 3 | 5 s, 16:9, 2 images | 24.49 | 13.65 | 7.20 | 4.35 |
| 4 | 5 s, 9:16, 1 image + 1 video + 1 audio | 34.50 | 18.89 | 9.91 | 5.63 |
| 5 | 5 s, 9:16, 3 images + 1 video | 55.35 | 30.33 | 15.93 | 8.71 |
| 6 | 5 s, 16:9, 3 images + 1 audio | 25.52 | 14.20 | 7.52 | 4.28 |
| 7 | 5 s, 16:9, 2 images + 1 audio | 25.47 | 14.17 | 7.54 | 4.37 |
| 8 | 5 s, 9:16, 2 images + 1 video + 1 audio | 36.27 | 19.87 | 10.34 | 5.81 |
| 9 | 10 s, 16:9, 2 images | 67.94 | 36.97 | 18.99 | 10.13 |
| 10 | 10 s, 9:16, 1 image + 1 video + 1 audio | 70.16* | 38.27 | 19.65 | 10.60 |
| 11 | 10 s, 16:9, 3 images + 1 video | 97.03* | 52.89 | 27.13 | 14.51 |
| 12 | 10 s, 9:16, 2 images + 1 video + 1 audio | 74.67* | 40.76 | 20.85 | 11.30 |
| 13 | 15 s, 16:9, 1 image + 2 audio | 69.89* | 38.11 | 19.65 | 10.50 |
| 14 | 15 s, 16:9, 3 images | 73.99* | 40.36 | 20.74 | 10.98 |
| 15 | 15 s, 16:9, 2 images + 1 audio | 72.53* | 39.71 | 20.42 | 10.86 |
| 16 | 15 s, 16:9, 2 images + 1 audio | 79.88* | 43.77 | 22.33 | 11.87 |

480p:

| # | Request | 1 x GB200 | 2 x GB200 | 4 x GB200 | 8 x GB200 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 5 s, 16:9, 1 image + 2 audio | 10.47 | 6.10 | 3.44 | 2.24 |
| 2 | 5 s, 16:9, 3 images | 11.81 | 6.77 | 3.66 | 2.21 |
| 3 | 5 s, 16:9, 2 images | 11.87 | 6.83 | 3.71 | 2.23 |
| 4 | 5 s, 9:16, 1 image + 1 video + 1 audio | 21.26 | 11.89 | 6.33 | 3.71 |
| 5 | 5 s, 9:16, 3 images + 1 video | 39.67 | 22.15 | 11.78 | 6.68 |
| 6 | 5 s, 16:9, 3 images + 1 audio | 12.53 | 7.14 | 3.97 | 2.32 |
| 7 | 5 s, 16:9, 2 images + 1 audio | 12.59 | 7.14 | 3.93 | 2.45 |
| 8 | 5 s, 9:16, 2 images + 1 video + 1 audio | 22.62 | 12.49 | 6.61 | 3.85 |
| 9 | 10 s, 16:9, 2 images | 33.13 | 18.41 | 9.59 | 5.33 |
| 10 | 10 s, 9:16, 1 image + 1 video + 1 audio | 41.32 | 22.62 | 11.70 | 6.54 |
| 11 | 10 s, 16:9, 3 images + 1 video | 59.82 | 32.44 | 16.90 | 9.07 |
| 12 | 10 s, 9:16, 2 images + 1 video + 1 audio | 44.51 | 24.30 | 12.61 | 6.91 |
| 13 | 15 s, 16:9, 1 image + 2 audio | 28.43 | 15.77 | 8.32 | 4.74 |
| 14 | 15 s, 16:9, 3 images | 30.58 | 16.85 | 8.87 | 4.95 |
| 15 | 15 s, 16:9, 2 images + 1 audio | 29.98 | 16.62 | 8.72 | 5.22 |
| 16 | 15 s, 16:9, 2 images + 1 audio | 34.23 | 18.69 | 9.78 | 5.67 |

\* With `--graph-policy off`, which one GB200 needs for these requests.

## FastH3 on H200

`FastVideo/FastVideo-FastH3-8-Step-V2`, `--mem-fraction-static 0.92` and `NCCL_CUMEM_HOST_ENABLE=1`. Latency is the mean ± SD of three requests at concurrency one; the DP8 rows measure 16 requests at concurrency eight.

| Deployment | Precision | Workload | Result |
| --- | --- | --- | --- |
| 8 x H200, Ulysses8 | `quality` | 5 s, 1000 tokens | 8.194 ± 0.818 s |
| 8 x H200, Ulysses8 | `quality` | 5 s, 10000 tokens | 14.218 ± 0.880 s |
| 8 x H200, Ulysses8 | `quality` | 10 s, 1000 tokens | 18.539 ± 1.071 s |
| 8 x H200, Ulysses8 | `quality` | 10 s, 10000 tokens | 26.578 ± 0.600 s |
| 8 x H200, Ulysses8 | `quality` | 15 s, 1000 tokens | 31.069 ± 0.045 s |
| 8 x H200, Ulysses8 | `quality` | 15 s, 10000 tokens | 44.228 ± 0.878 s |
| 4 x H200, Ulysses4 | `quality` | 5 s, 1000 tokens | 13.864 ± 0.013 s |
| 8 x H200, all-gather | `quality` | 5 s, 10000 tokens | 20.170 ± 0.698 s |
| 8 x H200, Ulysses8 | FP8 MLPs | 5 s, 1000 tokens | 7.448 ± 0.734 s |
| 8 x H200, Ulysses8 | FP8 MLPs | 15 s, 10000 tokens | 42.380 ± 0.807 s |
| 8 x H200, `dp8-text-tp8.json` | `quality` | 5 s, 1000 tokens, concurrency 8 | 0.1519 videos/s, 52.231 ± 0.358 s, 90,329 MiB peak per GPU |
| 8 x H200, `dp8-text-tp8.json` | FP8 MLPs | 5 s, 1000 tokens, concurrency 8 | 0.1676 videos/s, 47.314 ± 0.386 s, 81,531 MiB peak per GPU |

```bash
.venv/bin/uniserve-eval --config uniserve_eval/fasth3.toml run h200-8-bf16 --reuse-deployment
```

## FastH3 on RTX PRO 6000

Eight RTX PRO 6000 Blackwell Server Edition GPUs on one PCIe host, `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4` unless noted, 32 requests per concurrency. Leave `NCCL_P2P_LEVEL` at its default on such a host: `SYS` cuts eight-GPU all-to-all bandwidth from 24.5 to 1.1 GB/s.

| Deployment | Point | Concurrency | Videos/s | Latency p50 / p95 | Peak device memory |
| --- | --- | --- | --- | --- | --- |
| `ulysses4x2.json` | `rtx-pro-6000-8-nvfp4-throughput-c4` | 4 | 0.0415 | 89.9 / 149.6 s | 74,408 MiB |
| `dp8-text-tp8.json` | `rtx-pro-6000-8-nvfp4-dp8-throughput-c8` | 8 | 0.0426 | 127.9 / 318.5 s | 97,156 MiB |
| `dp8-text-tp8-host-weights.json`, BF16 checkpoint | `rtx-pro-6000-8-bf16-dp8-throughput-c8` | 8 | 0.0346 | 167.2 / 366.5 s | 79,692 MiB |

```bash
.venv/bin/uniserve-eval --config uniserve_eval/fasth3.toml run rtx-pro-6000-8-nvfp4-dp8 --reuse-deployment
```

## Other suites

`fasth3_omniref.toml` also measures the OmniRef reference matrix, nine reference compositions at 5, 10 and 15 seconds (`gb200-8-refs`, `h200-8-refs`, `rtx-pro-6000-8-refs`, and `-enlarged` suites that enlarge every reference image to its encoding size). `minimax_h3.toml` measures the MiniMax-H3 checkpoint on its official text, keyframe and reference requests. The workload files are in `uniserve_eval/workloads/minimax_h3/`; each row names its condition media with their source URLs and SHA-256.
