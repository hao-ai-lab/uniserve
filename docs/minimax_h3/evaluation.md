# MiniMax-H3 evaluation

`uniserve_eval/minimax_h3.toml` defines conditioned video workloads for UniServe, SGLang, vLLM-Omni and FastVideo. Each point fixes the checkpoint, deployment, precision, request schedule, media, seeds and validation. Resolve the plan before execution and run points serially, with one server and one harness active.

| Workload | Request | Measured seeds |
| --- | --- | --- |
| W1 | t2va, official prompt, 1344×768, 5 seconds (124 frames) | 0–7 |
| W2 | t2va, official prompt, 1344×768, 15 seconds (362 frames) | 0–3 |
| W3 | fl2va, official first keyframe, automatic canvas, 8 seconds (192 frames) | 0–7 |
| W4 | ref2va, reference image and voice, 5 seconds | 0–7 |
| W5 | ref2va, reference video including its soundtrack and a reference voice, 5 seconds | 0–7 |

Each point first performs one unmeasured warmup at seed 42, then sends measured requests one at a time. Base models use 50 sigma points (49 denoiser calls), video flow shift 12 and audio flow shift 3. Component exports use their published schedules. Every response must decode as a complete MP4 with the expected canvas, frame count, frame rate and stereo soundtrack.

Provision the checkpoint revisions named in the profile and the condition media the workload manifests name: each row's `metadata.condition_media` gives every file's path under `UNISERVE_MINIMAX_H3_INPUTS`, its source URL and its SHA-256. Install the repository's locked `bench` and `gpu` dependencies and the FFmpeg 8.1.2 build used to decode reference media.

```bash
export UNISERVE_MINIMAX_H3_MODEL=/workspace/models/MiniMax-H3
export UNISERVE_MINIMAX_H3_INPUTS=/workspace/uniserve/artifacts/minimax_h3/inputs
export UNISERVE_MINIMAX_H3_FFMPEG=/workspace/tools/ffmpeg-8.1.2/bin
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml plan gb200-4-w3
.venv/bin/uniserve-eval --config uniserve_eval/minimax_h3.toml run gb200-4-w3
```

The `gb200-4`, `gb200-4-reference`, `gb200-4-omniref`, `fast-h3-gb200-4` and `gb200-8` suites group points by checkpoint and placement. OmniRef also requires `UNISERVE_MINIMAX_H3_OMNIREF` and its pinned `UNISERVE_MINIMAX_H3_OMNIREF_BASE`. For two-host points, set `UNISERVE_MINIMAX_H3_HEAD_ADDRESS` and start the remote launcher with the head's advertised port as described in [Two hosts](minimax_h3.md#two-hosts).

## Comparability

Cross-engine comparisons must disclose differences in reference-image resizing, noise generators, media codecs and distributed attention. In particular, vLLM-Omni's W4 image processing retains a smaller condition layout than the other configured engines, and its diffusion worker does not implement the two-host placement. These points do not establish equal-work performance ratios.
