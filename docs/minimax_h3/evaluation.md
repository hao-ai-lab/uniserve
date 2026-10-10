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

The `gb200-4`, `gb200-4-reference`, `gb200-4-omniref`, `fast-h3-gb200-4` and `gb200-8` suites group points by checkpoint and placement. OmniRef points also need `UNISERVE_MINIMAX_H3_OMNIREF`, and the Hugging Face cache supplies the base revision it pins; two-host points need `UNISERVE_MINIMAX_H3_HEAD_ADDRESS`.

## OmniRef reference compositions

The reference matrix measures FastH3 OmniRef (`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`) on nine reference compositions of MiniMax-H3's ref2va interface, each at 5, 10 and 15 seconds, every target 16:9 (1344×768). A workload is `ref-<composition>-<seconds>s`:

| Composition | References | Prompt tokens with vision | Condition rows |
| --- | --- | --- | --- |
| `image` | One 16:9 subject image | about 7,600 | 7,296 |
| `image-voice` | One 16:9 subject image and a reference voice | about 7,650 | 7,808 to 8,064 |
| `images3` | Three 16:9 images: a subject, a product sheet and a scene | about 22,400 | 22,016 |
| `images9` | Nine square images | 37,701 | 36,864 |
| `clip` | One 2-second reference clip with its soundtrack | about 3,500 | 14,080 |
| `video` | One reference video (6 s, with its soundtrack) | 6,458 to 7,630 | 46,592 to 51,200 |
| `clips3` | Three 2-second reference clips, each with its soundtrack | 9,718 | 42,240 |
| `mixed` | One 16:9 image, one reference video with its soundtrack and a reference voice | 13,911 to 15,118 | 54,400 to 59,264 |
| `full` | Nine square images and three 2-second clips with their soundtracks | 47,160 | 79,104 |

A request's packed sequence is its prompt, which carries every visual reference's vision tokens, its condition rows and its generated rows: 37,710 at 5 s, 73,386 at 10 s and 109,062 at 15 s. A reference image, resized to a 2048-pixel short edge, is 7,296 prompt tokens and 7,296 condition rows at 16:9 and 4,096 of each when square. A reference video or voice longer than the target contributes the target's duration, so its rows grow with the target up to its own length. The largest cell, `ref-full-15s`, packs 236,294 rows. The square images are centred crops and the clips the leading two seconds of their sources, cut without re-encoding; each row's `metadata.condition_media` names the source and the derivation.

A point sends one warmup at seed 42 and three measured requests one at a time, seeds 0 to 2. One deployment across two four-GB200 hosts (`ulysses8-reference-two-node.json`) serves every cell: `--max-video-seconds 15 --max-model-len 48128 --video-text-capacities 4096,7168,8192,10240,14336,15360,22528,37888,48128 --max-condition-rows 79104`, a text capacity for each cell's prompt and the condition rows of `full`.

```bash
export UNISERVE_MINIMAX_H3_OMNIREF=/workspace/models/FastH3-OmniRef-v5-DMPDD8-w03-cfg2-step3500
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run gb200-8-omniref-refs --reuse-deployment
```

Start `uniserve-host` on `rank-1` for the two-host server, as in [Two hosts](minimax_h3.md#two-hosts). The `sglang-gb200-8-reference` suite measures SGLang's base Ref2VA on the same cells and hosts with the base schedule (50 sigma points), one measured request per cell after its warmup; start SGLang's node rank 1 on `rank-1` beside the head, as for `sglang-gb200-8`.

Cross-engine results are not equal-work ratios: the engines differ in reference-image resizing, noise generators, media codecs and distributed attention. vLLM-Omni's W4 image processing keeps a smaller condition layout than the others, and its diffusion worker does not implement the two-host placement.
