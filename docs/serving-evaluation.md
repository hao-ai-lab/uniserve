# Serving Evaluation and Benchmarking (`uniserve-eval`)

`uniserve_eval` is the profile-driven package that launches serving backends, runs correctness gates against them over public HTTP APIs, runs performance points through its measurement harness — including vLLM-Omni comparison servers under byte-for-byte the same workloads — and compares measured points side by side. Everything it does is declared in `uniserve_eval/profiles.json`; the code owns only mechanics. The CLI is installed as `uniserve-eval` (equivalently `python -m uniserve_eval`).

```
uniserve-eval list                                    # servers / workloads / suites, with descriptions
uniserve-eval launch sensenova-u1-single              # start one server (pid + log under artifacts/)
uniserve-eval verify bagel-t2i-seed42                 # one correctness workload against it
uniserve-eval perf bagel-i2t-perf                     # one perf point (gated)
uniserve-eval perf sensenova-t2i-perf-omni            # the same point against vLLM-Omni (measured, ungated)
uniserve-eval compare sensenova-t2i-perf sensenova-t2i-perf-omni   # side-by-side table from their artifacts
uniserve-eval run verify-all --manage-servers         # a suite, launching/cleaning servers per workload
uniserve-eval clean --all                             # stop everything it started
```

## Concepts

**server** — how to launch one serving backend: model path, served name, host/port, `CUDA_VISIBLE_DEVICES`, `env`, and either UniServe `serve_args` or an explicit `command` (any backend — this is how the vLLM-Omni comparison servers are declared). Specs support `extends` with deep-merge so variants stay single-source: `sensenova-u1-tp4` extends `sensenova-u1-single`; `sensenova-u1-perf` adds only the decode-burst env; `sensenova-u1-teacache` (extends `sensenova-u1-perf`) opts into the denoise residual cache via `UNISERVE_DENOISE_RESIDUAL_CACHE=1`; `sensenova-u1-tp4-graph` (extends `sensenova-u1-tp4`) adds `UNISERVE_DENOISE_STEP_GRAPH=1` + `UNISERVE_DECODE_TOKEN_BURST=8`.

**workload** — one verification or measurement unit bound to a compatible server (overridable with `--server`). Every workload carries a `description` in the config; `list` prints them.

**suite** — an ordered set of workloads, either a plain list of names or the dict form `{"workloads": [...], "compare": [[baseline, candidate, ...], ...]}`. With `--manage-servers` the driver cleans, launches (waiting for readiness), runs, and cleans per workload, so a suite reproduces from a cold checkout plus a built binary. After the workloads, each `compare` group is evaluated (first name = baseline). Artifacts land under `artifacts/servers/<name>/` (pid, log), `artifacts/workloads/<name>/` (summary.json, decoded PNGs, harness output), and `artifacts/comparisons/<name>/` (compare.md, compare.json).

## Workload types

### `verify` — correctness gates over the native SSE stream

POSTs the declared `payload` to `/generate` and validates the event stream. Gates: `expect_images` + `expect_image_width/height` (exact count and pixel dimensions of decoded PNGs, saved for inspection), `expect_image_steps` (exact per-image `image_step` event count — catches silently shortened denoise schedules), `expect_min_text_tokens` and `expect_finished` (text floors, clean termination), and any `error` event fails the workload. For understanding-mode inputs, `input_image_synthetic: {seed, width, height}` renders a deterministic geometric scene (blue sky, green ground, red house with brown roof, yellow sun upper-right) and injects it as `input_image_b64`, so i2t gates run hermetically with no datasets or checked-in image bytes.

These gates are UniServe-native-API-only by design; cross-backend comparison happens through the perf type below, which speaks each backend's public wire format.

### `perf` — measurement points (`perf` subcommand)

Runs a point through the measurement harness (`uniserve_eval/harness`, invoked as `python -m uniserve_eval.harness.cli`) against the workload's server: `task` (t2i / i2t / text / interleave) plus raw harness `args` (dataset, sizes, steps, seed, concurrency, warmup). Optional `expect_metrics` gates dotted metric paths with min/max bounds; any failed request also fails the workload.

Two flavors coexist:
- **Tripwires** (UniServe servers, with bounds): deliberately loose floors/ceilings that trip only when a path regresses to a known-slow regime — e.g. BAGEL i2t falling from the graphed/burst decode (~160 tok/s) back to eager per-token (~16 tok/s). They are not headline numbers and are insensitive to run-to-run noise.
- **Comparison points** (`*-omni`, no bounds): byte-for-byte the same workload — same prompt file (`uniserve_eval/data/t2i_prompts.jsonl`), sizes, steps, seed, concurrency, warmup — against a vLLM-Omni server declared with an explicit launch `command`. Same harness, same metrics pipeline, so numbers are directly comparable. The i2t mirrors switch only the wire (`--i2t-wire openai_chat`, since vLLM-Omni's understanding path is its chat API and does not stream).

### `script` — generic escape hatch

Runs `python <script> <args>` from the repo root with the workload's `env`, plus an optional `expect_file` existence check. No assumptions about what the script is.

## `compare` — side-by-side tables from measured artifacts

`uniserve-eval compare <workload> <workload> [...]` reads each workload's `artifacts/workloads/<name>/summary.json` (run the perf points first — it errors otherwise) and prints a markdown table with rows = metrics and one column per workload, plus a ratio column per candidate against the first workload (the baseline). Higher-is-better metrics (throughput, images/min) are reported as candidate/baseline; latency metrics as baseline/candidate — so >1.0 always reads "candidate is better". The metric set follows the summaries' `metric_family` (image: mean/p50 image latency + images/min; stream: output throughput, TTFT, ITL, E2E), plus request/ok counts and harness-sampled peak GPU memory when present; mixed families are refused. The table is also written to `artifacts/comparisons/<name>/compare.md` + `compare.json`. Suites with `compare` groups run these automatically after their workloads.

## Workload catalog and expected behavior

### `sensenova-travel-interleave-4x` — the guarded tour-guide case

One native `interleave` request: *"Generate a travel guide covering Sonoma, Sequoia, Tahoe, and the Golden Gate."*, temperature 0, max 4 images at 2048×1152, 50 denoise steps each.

Expected behavior: the model **autonomously decides** where images belong — it writes narrative text, emits an image trigger when it reaches a destination, the engine runs the denoise sub-program, commits the image back into the text KV, and generation continues conditioned on the committed image. A correct run streams exactly 4× (`image_begin` → 50×`image_step` → `image_done`) interleaved with text, finishes with 0 errors, and produces ~2800 text tokens at TP1 (a deterministic greedy trajectory; TP4 legitimately differs in token count but not in gate outcomes). Quality bar (from the original task definition): every image readable, coherent, non-collapsed, non-duplicated — the four PNGs are saved to the workload dir for inspection. This is the workload no other benchmarked system can serve; it exercises text decode, denoise, commit-writeback, and their interleaving in one request.

### `sensenova-t2i-seed42` / `bagel-t2i-seed42` — pure text-to-image gates

Native `image` mode, seed-42 bicycle prompt ("A red bicycle leaning against a weathered brick wall, golden hour light"), 50 steps; SenseNova at 2048×1152 (a profile resolution bucket), BAGEL at 1024×1024. Expected: exactly one PNG of exactly the requested size, exactly 50 `image_step` events, clean finish. Guards the prefill→denoise→commit chain per model — for BAGEL that means the paged-extend prefill, the batched CFG denoise rows, and the VAE-decode commit on the unified runtime. On an unchanged build the SenseNova image is a stable identity (the known warm/cold hash pair); after kernel-level changes, bitwise identity is only expected from layout/mechanics-only changes — otherwise judge the saved PNG by inspection.

### `sensenova-i2t-house` / `bagel-i2t-house` — image-understanding gates

Native `understand` mode over the injected synthetic scene, "Describe this image in detail.", temperature 0, max 128 tokens. Expected: ≥32 streamed text tokens, clean finish, no errors; a healthy answer names the scene's content (red house, brown roof, sun, sky/ground split — models think aloud first, which is fine). Guards the encode→prefill→decode chain: SenseNova's marker-in-prompt patch injection and BAGEL's dual (VAE+ViT) encode, plus the graphed/burst decode both models now share.

### `*-perf` tripwires and `*-perf-omni` comparison points

Same shape per model pair: t2i runs 4 prompts from the committed prompt set (closed-loop, concurrency 1, warmup 1, seed 42, 50 steps, model-native resolution); i2t runs 4 synthetic-image descriptions (greedy, 256 tokens). Healthy references at the time bounds were set (1× GB200): SenseNova t2i ~4.8–5.6 s/image and i2t ~160–170 tok/s (ITL ~5 ms); BAGEL t2i ~6.6 s/image and i2t ~155–165 tok/s. Bounds: t2i mean ceilings 8 s (SenseNova) / 10 s (BAGEL); i2t floors 120 / 100 tok/s with ITL ceiling 9 ms. The `-omni` mirrors measure vLLM-Omni under identical workloads with no bounds (references: SenseNova t2i ~6.9 s, i2t ~10 tok/s; BAGEL t2i ~10.8 s, i2t ~144 tok/s).

### Real-dataset workloads: `sensenova-t2i-mjhq`, `bagel-t2i-mjhq`, `sensenova-interleave-ueval` (suite `realdata-perf`)

The hermetic workloads above use committed prompts and synthetic images so gates run from a clean checkout with zero downloads. These points complement them with real datasets, auto-downloaded from the Hugging Face hub on first use (install the `eval-datasets` extra: `pip install -e '.[eval-datasets]'`; every loader also accepts `--dataset-path` for a local file instead).

- **MJHQ t2i** (`--dataset mjhq`, `playgroundai/MJHQ-30K`): 8 real photography captions per point, same sizes/steps/seed/bounds as the trace-based tripwires — realistic prompt length and content diversity.
- **UEval interleave** (`--dataset ueval`, `zlab-princeton/UEval`, 1,000 real interleaved-generation tasks such as "How to reset a forgotten password on Windows? Show each step both visually and textually"): 3 prompts, up to 2 images each at 2048×1152/50 steps. Expected behavior: the model decides image count and placement per prompt, so image totals legitimately vary run to run; the harness reports the image sub-block (per-image generation time, TTFI) and gates only text ITL, which excludes image spans by construction. This is the realistic companion to the deterministic tour-guide gate. Healthy reference: 3/3 ok, ~5.3 s per generated image, text ITL ~5 ms.
- **ShareGPT text** is already real-dataset (`qwen3-sharegpt-stress` below). **PIE-Bench** (i2i editing) has a working loader, but the server-side input-conditioned image mode is not yet functional, so no workload is declared for it — add one when i2i lands.

### `qwen3-sharegpt-stress`

Text-serving stress through the shared harness: 200 ShareGPT prompts at 16 req/s against Qwen3-32B (`--max-tokens 4096`, seed 42); gated on all 200 requests completing (`expect_metrics: completed ≥ 200`), and any failed request fails the point.

## Suites

| suite | contents | purpose |
|---|---|---|
| `sensenova-verify` | t2i seed-42, i2t house, tour-guide interleave | SenseNova correctness across all three task shapes |
| `bagel-verify` | t2i seed-42, i2t house | BAGEL correctness on the unified runtime |
| `verify-all` | both of the above | pre-merge correctness pass |
| `sensenova-perf`, `bagel-perf`, `perf-all` | the four gated tripwires | regression detection to pre-unification paths |
| `sensenova-vs-omni`, `bagel-vs-omni`, `matrix-vs-omni` | tripwires + omni mirrors, with `compare` groups pairing each UniServe point (baseline) with its omni mirror | the cross-backend comparison matrix under identical workloads, with the tables emitted automatically |

Note on `--manage-servers` with omni suites: the vLLM-Omni servers cold-start in minutes (engine + pipeline init); the default launch timeout (1800 s) covers it, but running the UniServe and omni halves against separately-launched servers is faster when iterating.

## Conventions

- Gates run against the **default** serving configuration. Opt-in feature envs live on server specs, not workloads: the perf profiles pin `UNISERVE_DECODE_TOKEN_BURST=8`; TeaCache (`sensenova-u1-teacache`) and TP4 denoise-step graphs (`sensenova-u1-tp4-graph`) stay off in gates because default-path outputs are the guarded identities — a gated feature that needs its own suite gets its own server variant.
- The vLLM-Omni comparison environment is a dedicated venv (default `/home/hal-ysun/omni-bench-venv`) with vllm 0.24 (the refs checkout's target: version stamp `0.24.0rc2`) plus refs' `requirements/common.txt`, and `refs/vllm-omni` installed into it non-editably: `VLLM_OMNI_TARGET_DEVICE=cuda uv pip install --no-deps refs/vllm-omni`. Installing the refs checkout (refs stays pristine) registers the omni CLI plugin so the standard `vllm serve --omni` path works — including `--cache-backend tea_cache` for the cached comparison — and the served code is exactly the refs revision. The `sensenova-omni` / `bagel-omni` server specs carry the launch `command`s; `bagel-omni` adds the attention-backend overrides that dodge its broken FlashAttention-CuTe build on this aarch64 stack, which costs it a few percent — noted for fairness.
- Adding a model: (1) a server spec; (2) a t2i `verify` gate with exact size/step bounds and a fixed seed, plus an i2t gate with `input_image_synthetic` and a token floor; (3) perf tripwires with bounds set from a measured healthy run (≈1.5× ceilings, ≈0.6× floors); (4) optional `-omni` mirrors if the model exists there, paired in a `<model>-vs-omni` suite with `compare` groups; (5) a `<model>-verify` suite appended to `verify-all`.
- The driver never writes outside `artifacts/`; `clean --all` reaps by pid file (process-group SIGTERM, then SIGKILL), so killed runs never leak servers.
