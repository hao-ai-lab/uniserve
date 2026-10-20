# Evaluation

`uniserve-eval` runs fixed workloads against a server it starts, validates every MP4 (canvas, frame count, frame rate and stereo soundtrack), and writes each run's commands, requests, per-request latencies and server stage timings, media validation and GPU memory samples under `artifacts/`. Install it with the GPU environment:

```bash
uv sync --locked --python 3.12 --extra gpu --extra bench
```

A profile defines servers, benchmark points and suites. `plan` resolves a suite without starting a server and `run` measures it; `--reuse-deployment` keeps one server across the points that share it:

```bash
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml plan gb200-4-bf16
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml run gb200-4-bf16 --reuse-deployment
```

Run points serially, one server at a time. A two-host suite starts the head on `rank-0`; when it logs `awaiting a launcher`, start `uniserve-host` on `rank-1` as in [Two hosts](minimax_h3.md#two-hosts).

| Profile | Contents |
| --- | --- |
| `uniserve_eval/fast_h3.toml` | FastH3 8-Step V2 text-to-video on GB200 and RTX PRO 6000: 72 latency requests one at a time (5, 10 and 15 s clips with 1000- and 10000-token prompts) and 32 throughput requests per concurrency |
| `uniserve_eval/fast_h3_h200.toml` | FastH3 8-Step V2 on H200 |
| `uniserve_eval/minimax_h3.toml` | The base checkpoint on W1 to W5, FastH3 OmniRef on W4 and W5, and the OmniRef reference matrix on GB200, H200 and RTX PRO 6000 |

`UNISERVE_FAST_H3_MODEL` and `UNISERVE_FAST_H3_NVFP4_MODEL` name local copies of `FastVideo/FastVideo-FastH3-8-Step-V2` and its `-NVFP4` variant; `UNISERVE_MINIMAX_H3_MODEL` names `MiniMaxAI/MiniMax-H3`, `UNISERVE_MINIMAX_H3_OMNIREF` names `FastVideo/FastVideo-FastH3-Omni-8-Step-V1`, `UNISERVE_MINIMAX_H3_INPUTS` a directory whose `media/` holds the condition files the workload rows name, and `UNISERVE_MINIMAX_H3_FFMPEG` the `bin` directory of FFmpeg 8.1.2.

## FastH3 OmniRef on GB200

FastH3 OmniRef (`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`) on one replica of one, two, four and eight GB200, with the deployments and options of the [validated settings](minimax_h3.md#deployments). The workload is 16 reference-conditioned requests: one to three images, up to one reference video and up to two audio references, at 5, 10 and 15 seconds, 16:9 and 9:16, each at 768p and at 480p. Each cell is the mean end-to-end seconds of three requests (seeds 0 to 2) at concurrency one after one warmup, through `/v1/videos/sync` including MP4 encoding; no cell's standard deviation exceeds 0.39 s. Eight GB200 span two four-GPU hosts.

768p:

| # | Request | 1 GB200 | 2 GB200 | 4 GB200 | 8 GB200 |
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

| # | Request | 1 GB200 | 2 GB200 | 4 GB200 | 8 GB200 |
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

\* Served with `--graph-policy off`: on one GB200 the captured graphs leave too little device memory for a 768p request longer than 10 seconds or one with a 10-second reference video. Every other cell runs with CUDA graphs.

Every output is a 24 fps H.264 MP4 of the requested canvas and frame count (124, 243 or 362 frames) with stereo 32-kHz AAC. Two runs of the eight-GPU suite produced byte-identical MP4s for all 96 requests.

## FastH3 on H200

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

## FastH3 on RTX PRO 6000

One host with eight RTX PRO 6000 Blackwell Server Edition GPUs, `FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4` unless noted, six unmeasured warmups, the priming requests of the concurrency, then 32 throughput requests. Peak device memory is the highest per-GPU use sampled once a second.

| Deployment | Point | Concurrency | Videos/s | Latency p50 / p95 | Peak device memory |
| --- | --- | --- | --- | --- | --- |
| `fasth3-ulysses4x2.json` | `rtx-pro-6000-8-nvfp4-throughput-c4` | 4 | 0.0415 | 89.9 / 149.6 s | 74,408 MiB |
| `fasth3-dp8-text-tp8.json` | `rtx-pro-6000-8-dp8-nvfp4-throughput-c8` | 8 | 0.0426 | 127.9 / 318.5 s | 97,156 MiB |
| `fasth3-dp8-text-tp8-host-weights.json`, BF16 checkpoint | `rtx-pro-6000-8-dp8-bf16-throughput-c8` | 8 | 0.0346 | 167.2 / 366.5 s | 79,692 MiB |

FastH3 OmniRef runs there as one Ulysses8 reference replica (`rtx-pro-6000-8-omniref-refs` of `uniserve_eval/minimax_h3.toml`, graphs off, at most 56,192 condition rows), each cell the mean ± SD of three requests after its warmup:

| Composition | 5 s | 10 s | 15 s |
| --- | ---: | ---: | ---: |
| `image` | 21.15 ± 0.13 s | 33.52 ± 0.46 s | 49.00 ± 0.36 s |
| `image-voice` | 21.39 ± 0.16 s | 34.56 ± 0.27 s | 50.33 ± 0.46 s |
| `video` | 36.80 ± 0.10 s | 56.06 ± 0.47 s | 74.04 ± 0.34 s |
| `mixed` | 46.49 ± 0.26 s | 69.26 ± 0.25 s | 93.07 ± 0.87 s |

Leave `NCCL_P2P_LEVEL` at its default on this PCIe host: `SYS` routes GPU pairs that share no PCIe switch peer to peer, which cuts eight-GPU all-to-all bandwidth from 24.5 to 1.1 GB/s.

```bash
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml run rtx-pro-6000-8-dp8-nvfp4 --reuse-deployment
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run rtx-pro-6000-8-omniref-refs --reuse-deployment
```

## MiniMax-H3 workloads

| Workload | Request | Measured seeds |
| --- | --- | --- |
| W1 | t2va, official prompt, 1344×768, 5 seconds (124 frames) | 0–7 |
| W2 | t2va, official prompt, 1344×768, 15 seconds (362 frames) | 0–3 |
| W3 | fl2va, official first keyframe, automatic canvas, 8 seconds (192 frames) | 0–7 |
| W4 | ref2va, reference image and voice, 5 seconds | 0–7 |
| W5 | ref2va, reference video with its soundtrack and a reference voice, 5 seconds | 0–7 |

Each point sends one unmeasured warmup at seed 42, then its measured requests one at a time. The manifests in `uniserve_eval/workloads/minimax_h3/` hold every request; each row's `metadata.condition_media` gives every condition file's path under `UNISERVE_MINIMAX_H3_INPUTS`, its source URL and its SHA-256.

```bash
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run gb200-4 --reuse-deployment
```

## OmniRef reference matrix

The matrix runs FastH3 OmniRef on nine reference compositions of the ref2va interface, each at 5, 10 and 15 seconds, every target 16:9 (1344×768). A workload is `ref-<composition>-<seconds>s`:

| Composition | References | Prompt tokens with vision | Condition rows | Enlarged: prompt tokens | Enlarged: condition rows |
| --- | --- | --- | --- | --- | --- |
| `image` | One 16:9 subject image | about 3,900 | 3,712 | about 7,600 | 7,296 |
| `image-voice` | One 16:9 subject image and a reference voice | about 3,950 | 4,224 to 4,480 | about 7,650 | 7,808 to 8,064 |
| `images3` | Three 16:9 images: a subject, a product sheet and a scene | about 8,900 | 8,576 | about 22,400 | 22,016 |
| `images9` | Nine square images | 14,415 | 13,952 | 37,701 | 36,864 |
| `clip` | One 2-second reference clip with its soundtrack | about 3,500 | 14,080 | about 3,500 | 14,080 |
| `video` | One reference video (6 s, with its soundtrack) | 6,458 to 7,630 | 46,592 to 51,200 | 6,458 to 7,630 | 46,592 to 51,200 |
| `clips3` | Three 2-second reference clips, each with its soundtrack | 9,718 | 42,240 | 9,718 | 42,240 |
| `mixed` | One 16:9 image, one reference video with its soundtrack and a reference voice | 10,215 to 11,422 | 50,816 to 55,680 | 13,911 to 15,118 | 54,400 to 59,264 |
| `full` | Nine square images and three 2-second clips with their soundtracks | 23,874 | 56,192 | 47,160 | 79,104 |

A request's packed sequence is its prompt, which carries every visual reference's vision tokens, its condition rows and its generated rows: 37,710 at 5 s, 73,386 at 10 s and 109,062 at 15 s. The `-enlarged` suites enlarge every reference image to the 2048-pixel short edge (`enlarge_references`); the others keep each reference at its own size unless it exceeds its encoding size, the server's default. A point sends one warmup at seed 42 and three measured requests, seeds 0 to 2.

```bash
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run gb200-8-omniref-refs --reuse-deployment
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run h200-8-omniref-refs --reuse-deployment
```

On one eight-H200 host as one Ulysses8 replica (`ulysses8-reference.json`, `--mem-fraction-static 0.92`), mean ± SD of three requests, default reference sizing:

| Composition | 5 s | 10 s | 15 s |
| --- | ---: | ---: | ---: |
| `image` | 10.17 ± 0.10 s | 17.55 ± 0.16 s | 26.16 ± 0.39 s |
| `image-voice` | 10.56 ± 0.46 s | 17.74 ± 0.26 s | 26.41 ± 0.10 s |
| `images3` | 13.80 ± 0.26 s | 23.13 ± 0.85 s | 33.94 ± 0.06 s |
| `images9` | 18.71 ± 0.81 s | 30.91 ± 0.36 s | 43.71 ± 0.98 s |
| `clip` | 11.48 ± 0.55 s | 18.14 ± 0.22 s | 26.46 ± 0.34 s |
| `video` | 18.03 ± 0.29 s | 29.47 ± 0.18 s | 39.44 ± 0.28 s |
| `clips3` | 20.98 ± 0.54 s | 29.85 ± 0.33 s | 40.09 ± 0.20 s |
| `mixed` | 23.70 ± 0.95 s | 36.58 ± 1.11 s | 48.48 ± 0.98 s |
| `full` | 38.93 ± 1.11 s | 53.34 ± 1.05 s | 70.09 ± 1.04 s |

Every reference enlarged to its encoding size (`h200-8-omniref-refs-enlarged`):

| Composition | 5 s | 10 s | 15 s |
| --- | ---: | ---: | ---: |
| `image` | 13.33 ± 0.78 s | 22.43 ± 0.89 s | 33.64 ± 0.10 s |
| `image-voice` | 13.77 ± 0.73 s | 22.94 ± 0.88 s | 33.06 ± 1.31 s |
| `images3` | 27.93 ± 0.24 s | 41.07 ± 0.38 s | 57.98 ± 0.54 s |
| `images9` | 48.56 ± 0.89 s | 68.99 ± 1.07 s | 94.15 ± 0.58 s |
| `clip` | 12.15 ± 0.36 s | 18.30 ± 0.33 s | 27.33 ± 0.53 s |
| `video` | 19.14 ± 1.01 s | 28.88 ± 0.30 s | 39.90 ± 1.87 s |
| `clips3` | 21.38 ± 0.59 s | 29.06 ± 0.06 s | 40.40 ± 0.86 s |
| `mixed` | 30.04 ± 0.18 s | 44.48 ± 0.25 s | 58.34 ± 0.84 s |
| `full` | 89.32 ± 0.84 s | 125.78 ± 9.24 s | 148.41 ± 0.97 s |
