# Evaluation

The evaluator runs fixed workloads against the deployments in the [guide](minimax_h3.md) and validates every MP4: canvas, frame count, frame rate and stereo soundtrack. Each run writes its resolved commands, request data, per-request latencies, media validation and GPU memory observations under `artifacts/`. Resolve a plan before running it, and run points serially, one server and one harness at a time, from the checkout whose `.venv` holds the environment:

```bash
uv sync --locked --python /usr/bin/python3.12 --extra gpu --extra bench
```

| Profile | Contents |
| --- | --- |
| `uniserve_eval/fast_h3.toml` | The [UniServe FastH3 post](https://hao-ai-lab.github.io/blogs/uniserve-fasth3/) measurements |
| `uniserve_eval/fast_h3_h200.toml` | The H200 measurements below |
| `uniserve_eval/minimax_h3.toml` | The W1–W5 workloads across checkpoints and engines, and the OmniRef reference matrix |

## FastH3 post

The workload is 72 latency requests one at a time, 12 for each combination of a 5, 10 or 15 second clip and a 1000- or 10000-token prompt, and 32 throughput requests at each concurrency after a priming phase, all preceded by six unmeasured warmup requests. The suites are `gb200-4-bf16`, `gb200-4-nvfp4`, `gb200-8-bf16`, `gb200-8-nvfp4`, `rtx-pro-6000-8-bf16` and `rtx-pro-6000-8-nvfp4`:

```bash
export UNISERVE_FAST_H3_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2
export UNISERVE_FAST_H3_NVFP4_MODEL=/workspace/models/FastVideo-FastH3-8-Step-V2-NVFP4
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml plan gb200-4-bf16
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml run gb200-4-bf16
```

The `gb200-8` suites start the head on `rank-0`; start `uniserve-host` on `rank-1` for each server the suite starts, as in [Two hosts](minimax_h3.md#two-hosts).

The `gb200-4-bf16-w1-w2` suite of the same profile runs the W1 and W2 workloads of [MiniMax-H3 workloads](#minimax-h3-workloads) on FastH3 8-Step V2, on the `gb200-4-bf16` deployment.

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

## RTX PRO 6000

These run on one host with eight RTX PRO 6000 Blackwell Server Edition GPUs (PCIe, GPUs 0-3 and 4-7 on separate NUMA nodes), with FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4 under the throughput points of the [FastH3 post](#fasth3-post): six unmeasured warmups, the priming requests of the concurrency, then the 32 throughput requests. Every output validated as 1344×768 H.264 with stereo 32-kHz AAC. A one-GPU replica holds a whole 15-second sequence, so the DP8 servers bound prompts at 10240 tokens (the workload's longest is 10000) and prepare only the 16:9 canvas.

| Deployment | Concurrency | Videos/s | Latency p50 / p95 | Peak device memory |
| --- | --- | --- | --- | --- |
| `fasth3-ulysses4x2.json` (`rtx-pro-6000-8-nvfp4-throughput-c4`) | 4 | 0.0415 | 89.9 / 149.6 s | 74,408 MiB |
| `fasth3-dp8-text-tp8.json` (`rtx-pro-6000-8-dp8-nvfp4`) | 8 | 0.0426 | 127.9 / 318.5 s | 97,156 MiB |
| `fasth3-dp8-text-tp8.json` (`rtx-pro-6000-8-dp8-nvfp4`) | 16 | 0.0404* | 310.4 / 489.7 s | 96,706 MiB |
| `fasth3-dp8-text-tp4x2.json` (`rtx-pro-6000-8-dp8-tp4x2-nvfp4`) | 8 | 0.0423* | 126.1 / 312.5 s | 97,342 MiB |
| `fasth3-dp8-text-tp8-host-weights.json` (`rtx-pro-6000-8-dp8-bf16`, BF16 checkpoint) | 8 | 0.0346 | 167.2 / 366.5 s | 79,692 MiB |

\* Measured on an earlier revision without the payload-sized feed-forward ranges, the interval output exchange and the threshold tile selection; the concurrency-16 row also predates the calibrated NVFP4 encodings and the in-place SM120 sparse-attention launch. Peak device memory is the highest per-GPU use sampled once a second during the run; the DP8 servers run within 1 GiB of the device's 97,887 MiB, and in the NVFP4 TP8 run PyTorch's expandable-segment allocator retried four mappings after freeing cached blocks, without a failed request. The GPUs throttle under sustained load (power limit 600 W), so a run's busy SM clock moves its speed: the Ulysses4x2, NVFP4 DP8 and BF16 DP8 rows at concurrency eight and four averaged 2197, 2070 and 2033 MHz while busy. Throughput also depends on the run's tail: 14–19% of DP8 replica time is idle while the last 15-second requests finish, and when the last one starts varies between runs by tens of seconds, so the summed server time of the 32 requests is the steadier comparison. Running the TP8 deployment's processes as clients of one MPS daemon lowered it to 0.0353 videos/s with 20% more replica time per request.

Each Ulysses4 replica of `fasth3-ulysses4x2.json` gathers every layer's normalized rows for its head-parallel projection and returns the attended rows to their token owners in intervals of about 48 MiB sent per member (4,608 rows), each exchanged as soon as it is composed, so the output projection, the gated residual and the feed-forward of an early interval run while the later intervals are in flight. One request at a time takes 15.9 s at 5 seconds, 30.1 s at 10 seconds and 55.6 s at 15 seconds with 1000-token prompts, and 48.6 s at 10 seconds with 10000 tokens; with each shard returned as one interval, the same requests take 16.3, 31.7, 58.5 and 50.8 s. The intervals also split the output projection and feed-forward GEMMs into other row ranges, which changes their last bits and with them the sampled videos: returned as one interval, the four videos are byte-identical to the revision before the interval exchange. At concurrency four the 32 requests sum to 1362.4 s of server time. The persistent NVFP4 GEMM slows beside the transfers: it assigns its tiles to one CTA per SM, so the SMs a transfer's NCCL kernel holds delay their CTAs until the transfer ends, and beside a four-GPU all-gather it takes 2.30 instead of 1.34 ms, still 2.24 ms with NCCL limited to two channels, while a cuBLAS BF16 GEMM of the same shape slows by 1.7%. FlashInfer's cuDNN FP4 GEMM keeps its speed beside the transfer (+2.1%), but in the 14336-deep feed-forward down projection one to three of its 33 million products per call differ from CUTLASS's in the last BF16 bit, so the deployments keep CUTLASS and their outputs.

A one-GPU replica serves one request at a time; in the concurrency-eight run a request's server time averages 76.4 s at 5 seconds, 154.3 s at 10 seconds and 261.2 s at 15 seconds. The DP8 replicas keep every denoiser collective off PCIe, so their throughput is bounded by single-GPU compute: in a 10-second request profiled before the feed-forward ranges, the device is busy 99.9% of the time, with sparse attention 48% of each denoising step's kernel time, the BF16 attention projections 32% and the NVFP4 feed-forward GEMMs 11%; decoding takes 434 ms of device time per 25-frame clip. The feed-forward runs over ranges of one projection payload (6144 rows), which keeps a 15-second shard's intermediates near 0.5 GiB; its GEMMs of a 10-second sequence take 93.2 ms per layer in BF16 and 27.3 ms in NVFP4, against 93.1 and 26.5 ms over 16384-row ranges. With the BF16 checkpoint (`quality`) and resident weights, a DP8 replica beside the TP8 text encoder serves 10 seconds with a 10000-token prompt (96,075 MiB peak for one request) and runs out of device memory at 15 seconds. Shrinking the text encoder's grant does not close the gap: capped at three request slots it still holds 12.4 GiB of each device under load, which leaves a replica 82.6 GiB, and a 15-second BF16 step needs more (its 6.21 GiB Q/K/V/gate projection buffer arrives with 5.9 GiB free). Keeping the transformer's layer projections in host memory (`fasth3-dp8-text-tp8-host-weights.json`) releases 35.5 GiB of each device and leaves every output byte unchanged (the 10-second request gives the resident server's MP4); in the concurrency-eight run a request's server time averages 98.7 s at 5 seconds, 194.2 s at 10 seconds and 315.4 s at 15 seconds. A layer's 0.79 GiB copy finishes while earlier layers compute, so streaming adds no measurable time.

OmniRef runs as one Ulysses8 reference replica (`rtx-pro-6000-8-omniref-refs` of `uniserve_eval/minimax_h3.toml`, the eight-H200 capacities with graphs off: at most 56,192 condition rows and 24,576 model rows) with peak device memory 91,537 MiB and no allocator mapping failures across all twelve points; each cell is the mean ± SD of three requests after its warmup, with the GPUs at 2284 MHz while busy.

| Composition | 5 s | 10 s | 15 s |
| --- | ---: | ---: | ---: |
| `image` | 21.15 ± 0.13 s | 33.52 ± 0.46 s | 49.00 ± 0.36 s |
| `image-voice` | 21.39 ± 0.16 s | 34.56 ± 0.27 s | 50.33 ± 0.46 s |
| `video` | 36.80 ± 0.10 s | 56.06 ± 0.47 s | 74.04 ± 0.34 s |
| `mixed` | 46.49 ± 0.26 s | 69.26 ± 0.25 s | 93.07 ± 0.87 s |

Every collective in these runs uses NCCL's default transport. On this host `NCCL_P2P_LEVEL=SYS` makes NCCL use PCIe peer-to-peer between GPUs that share no PCIe switch, which its default connects through host memory: eight-GPU all-to-all bus bandwidth falls from 24.5 to 1.1 GB/s and all-reduce from 30 to 13 GB/s, and the Ulysses8 replica runs 2.5 times slower.

```bash
.venv/bin/uniserve-eval --config uniserve_eval/fast_h3.toml run rtx-pro-6000-8-dp8-nvfp4 --reuse-deployment
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run rtx-pro-6000-8-omniref-refs --reuse-deployment --keep-going
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

The `gb200-4`, `gb200-4-reference`, `gb200-4-omniref` and `gb200-8` suites group points by checkpoint and placement. OmniRef points also need `UNISERVE_MINIMAX_H3_OMNIREF`, and the Hugging Face cache supplies the base revision it pins. The `gb200-4-reference` and `gb200-4-omniref` servers enlarge references to their encoding size (`--mm-processor-kwargs '{"enlarge_references":true}'`), the sizing of the released pipeline and of the SGLang and FastVideo baselines W4 and W5 are compared with. The two-host SGLang suites, `sglang-gb200-8` and `sglang-gb200-8-reference`, need `UNISERVE_MINIMAX_H3_HEAD_ADDRESS`, the address of `rank-0`, and the reference suite also `UNISERVE_MINIMAX_H3_SHARED_SCRATCH`, a directory both hosts mount.

## OmniRef reference compositions

The reference matrix measures FastH3 OmniRef (`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`) on nine reference compositions of MiniMax-H3's ref2va interface, each at 5, 10 and 15 seconds, every target 16:9 (1344×768). A workload is `ref-<composition>-<seconds>s`:

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

A request's packed sequence is its prompt, which carries every visual reference's vision tokens, its condition rows and its generated rows: 37,710 at 5 s, 73,386 at 10 s and 109,062 at 15 s. A reference video or voice longer than the target contributes the target's duration, so its rows grow with the target up to its own length. The square images are centred crops and the clips the leading two seconds of their sources, cut without re-encoding; each row's `metadata.condition_media` names the source and the derivation.

The matrix is measured under both reference sizings (`enlarge_references` of `--mm-processor-kwargs`). The `gb200-8-omniref-refs` suite serves the default: a reference keeps its own size unless it exceeds its encoding size. Every image of the matrix has a short edge below 2048 pixels, so each is as many prompt tokens and condition rows as its own 32-pixel patches, 3,600 for the 2560×1440 subject and 576 to 2,025 for a square, and every reference video is larger than its canvas and scales down to it; the largest cell, `ref-full-15s`, packs 189,830 rows. The `gb200-8-omniref-refs-enlarged` suite enlarges each image to its 2048-pixel short edge, 7,296 prompt tokens and condition rows at 16:9 and 4,096 when square, as the released pipeline and SGLang do; its largest cell packs 236,294 rows.

A point sends one warmup at seed 42 and three measured requests one at a time, seeds 0 to 2. One deployment across two four-GB200 hosts (`ulysses8-reference-two-node.json`) serves every cell of a suite, with a text capacity for each cell's prompt and the condition rows of `full`: `--max-video-seconds 15 --max-model-len 24576 --video-text-capacities 4096,7168,8192,9216,10240,12288,15360 --max-condition-rows 56192` by default, and `--max-model-len 48128 --video-text-capacities 4096,7168,8192,10240,14336,15360,22528,37888 --max-condition-rows 79104 --mm-processor-kwargs '{"enlarge_references":true}'` for the enlarged suite.

```bash
export UNISERVE_MINIMAX_H3_OMNIREF=/workspace/models/FastVideo-FastH3-Omni-8-Step-V1
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run gb200-8-omniref-refs --reuse-deployment
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run gb200-8-omniref-refs-enlarged --reuse-deployment
```

Start `uniserve-host` on `rank-1` for the two-host server, as in [Two hosts](minimax_h3.md#two-hosts). The `sglang-gb200-8-reference` suite measures SGLang's base Ref2VA on the same cells and hosts with the base schedule (50 sigma points), one measured request per cell after its warmup; start SGLang's node rank 1 on `rank-1` beside the head, as for `sglang-gb200-8`. SGLang enlarges every reference image, so its results compare with the enlarged suite.

Cross-engine results are not equal-work ratios: the engines differ in reference-image resizing, noise generators, media codecs and distributed attention. vLLM-Omni's W4 image processing keeps a smaller condition layout than the others, and its diffusion worker does not implement the two-host placement.

## OmniRef on H200

FastH3 OmniRef runs on H200 with the SM90 kernel of its 128-row sparse-attention tiles. The `h200-8-omniref-refs` and `h200-8-omniref-refs-enlarged` suites of `uniserve_eval/minimax_h3.toml` run the reference matrix on one eight-H200 host as one Ulysses8 replica (`ulysses8-reference.json`), with the capacities, reference sizings, warmups and seeds of the `gb200-8-omniref-refs` suites and `--mem-fraction-static 0.92`. Each cell is the mean ± SD of its three measured requests.

Default reference sizing (`h200-8-omniref-refs`):

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

```bash
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run h200-8-omniref-refs --reuse-deployment
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run h200-8-omniref-refs-enlarged --reuse-deployment
```
