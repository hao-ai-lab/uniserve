# Evaluation

`uniserve-eval` starts a server, sends a fixed workload, checks every MP4 (canvas, frame count, frame rate and soundtrack), and writes each request's latency and stage timings under `artifacts/`. It installs with the `bench` extra and runs a suite or a single point of a profile:

```bash
uv sync --locked --python 3.12 --extra gpu --extra bench
uniserve-eval --config uniserve_eval/fasth3.toml plan gb200-4-bf16
uniserve-eval --config uniserve_eval/fasth3.toml run gb200-4-bf16 --reuse-deployment
```

`uniserve_eval/fasth3.toml` measures FastH3 and `uniserve_eval/fasth3_omni.toml` FastH3 Omni. Each profile's header names the environment variables it reads, its workloads and its suites. Run one suite at a time; for a two-host suite, start `uniserve-host` on `rank-1` as in [Two hosts](minimax_h3.md#two-hosts). `plan` prints the server command of every point it selects.

## FastH3 Omni on GB200

16 reference requests, each at 768p and 480p, on one replica of one to eight GB200 with the [validated settings](minimax_h3.md#validated-settings). Each cell is the mean end-to-end seconds of three requests at concurrency one after a warmup, MP4 encoding included; no cell's standard deviation exceeds 0.39 s. The requests' media are not distributed with UniServe.

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
| 15 | 15 s, 16:9, 2 images (1024×1024) + 1 audio | 72.53* | 39.71 | 20.42 | 10.86 |
| 16 | 15 s, 16:9, 2 images (1306×1204, 1774×887) + 1 audio | 79.88* | 43.77 | 22.33 | 11.87 |

\* With `--graph-policy off`, which one GB200 needs for these requests.

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
| 15 | 15 s, 16:9, 2 images (1024×1024) + 1 audio | 29.98 | 16.62 | 8.72 | 5.22 |
| 16 | 15 s, 16:9, 2 images (1306×1204, 1774×887) + 1 audio | 34.23 | 18.69 | 9.78 | 5.67 |

## FastH3 on H200

`FastVideo/FastVideo-FastH3-8-Step-V2` on 8 H200 unless noted. Latency is the mean ± SD of three requests at concurrency one; the DP8 rows measure 16 requests at concurrency eight.

| Deployment | Precision | Workload | Result | Point |
| --- | --- | --- | --- | --- |
| `ulysses8.json` | `quality` | 5 s, 1000 tokens | 8.194 ± 0.818 s | `h200-8-bf16-5s-1k` |
| `ulysses8.json` | `quality` | 5 s, 10000 tokens | 14.218 ± 0.880 s | `h200-8-bf16-5s-10k` |
| `ulysses8.json` | `quality` | 10 s, 1000 tokens | 18.539 ± 1.071 s | `h200-8-bf16-10s-1k` |
| `ulysses8.json` | `quality` | 10 s, 10000 tokens | 26.578 ± 0.600 s | `h200-8-bf16-10s-10k` |
| `ulysses8.json` | `quality` | 15 s, 1000 tokens | 31.069 ± 0.045 s | `h200-8-bf16-15s-1k` |
| `ulysses8.json` | `quality` | 15 s, 10000 tokens | 44.228 ± 0.878 s | `h200-8-bf16-15s-10k` |
| `ulysses4.json`, 4 H200 | `quality` | 5 s, 1000 tokens | 13.864 ± 0.013 s | `h200-4-bf16-5s-1k` |
| `gather8.json` | `quality` | 5 s, 10000 tokens | 20.170 ± 0.698 s | `h200-8-bf16-gather8-5s-10k` |
| `ulysses8.json` | FP8 MLPs | 5 s, 1000 tokens | 7.448 ± 0.734 s | `h200-8-fp8-5s-1k` |
| `ulysses8.json` | FP8 MLPs | 15 s, 10000 tokens | 42.380 ± 0.807 s | `h200-8-fp8-15s-10k` |
| `dp8-text-tp8.json` | `quality` | 5 s, 1000 tokens, concurrency 8 | 0.1519 videos/s, 52.231 ± 0.358 s, 90,329 MiB peak per GPU | `h200-8-bf16-dp8-5s-1k` |
| `dp8-text-tp8.json` | FP8 MLPs | 5 s, 1000 tokens, concurrency 8 | 0.1676 videos/s, 47.314 ± 0.386 s, 81,531 MiB peak per GPU | `h200-8-fp8-dp8-5s-1k` |

## FastH3 on RTX PRO 6000

Eight RTX PRO 6000 Blackwell Server Edition GPUs on one PCIe host, `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4` unless noted, 32 requests per concurrency.

| Deployment | Concurrency | Videos/s | Latency p50 / p95 | Peak device memory | Point |
| --- | --- | --- | --- | --- | --- |
| `ulysses4x2.json` | 4 | 0.0415 | 89.9 / 149.6 s | 74,408 MiB | `rtx-pro-6000-8-nvfp4-throughput-c4` |
| `dp8-text-tp8.json` | 8 | 0.0426 | 127.9 / 318.5 s | 97,156 MiB | `rtx-pro-6000-8-nvfp4-dp8-throughput-c8` |
| `dp8-text-tp8-host-weights.json`, BF16 checkpoint | 8 | 0.0346 | 167.2 / 366.5 s | 79,692 MiB | `rtx-pro-6000-8-bf16-dp8-throughput-c8` |

Steady state, issue #5's whole-node metric. Closed-loop clients, two per replica, cycle the 5 s workload `fasth3-throughput-5s.jsonl` (six 1K- and six 10K-token prompts). Throughput counts the accepted videos that complete inside a 30-minute window opened 10 minutes after the load starts. Run `uniserve-eval --config uniserve_eval/fasth3.toml run rtx-pro-6000-8-nvfp4-steady`. The TP2x4 and DP7 rows were measured with the same clients and window outside `uniserve-eval`; measured that way, the DP8 point gave 352 videos per node-hour (176 in the window), p50 165.4 s and p95 200.9 s.

| Deployment | Clients | Accepted videos / node-hour | Latency p50 / p95 | Peak device memory | Point |
| --- | --- | --- | --- | --- | --- |
| `dp8-text-tp8.json` | 16 | 348 (174 in the window) | 160.9 / 201.1 s | 96,114 MiB; the warmup also sends 10 and 15 s requests | `rtx-pro-6000-8-nvfp4-dp8-steady-5s` |
| `dp8-text-tp2x4.json` | 16 | 392 (196) | 147.0 / 180.6 s | 97,292 MiB | `rtx-pro-6000-8-nvfp4-dp8-tp2x4-steady-5s` |
| `dp7-text-tp1.json` | 14 | 318 (159) | 152.2 / 225.9 s | 68,421 MiB, 61,109 MiB on the text GPU | `rtx-pro-6000-8-nvfp4-dp7-steady-5s` |
