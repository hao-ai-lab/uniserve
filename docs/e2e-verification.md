# End-to-end verification and benchmarking (`scripts/e2e.py`)

`scripts/e2e.py` is the profile-driven driver that launches serving backends, runs correctness gates against them over public HTTP APIs, and runs performance points through the shared benchmark harness — including vLLM-Omni comparison servers under byte-for-byte the same workloads. Everything it does is declared in `scripts/e2e_config.json`; the script owns only mechanics.

```
scripts/e2e.py list                              # servers / workloads / suites, with descriptions
scripts/e2e.py launch sensenova-u1-single        # start one server (pid + log under artifacts/)
scripts/e2e.py generate bagel-t2i-seed42         # one correctness workload against it
scripts/e2e.py perf bagel-i2t-perf               # one perf point (gated)
scripts/e2e.py perf sensenova-t2i-perf-omni      # the same point against vLLM-Omni (measured, ungated)
scripts/e2e.py run verify-all --manage-servers   # a suite, launching/cleaning servers per workload
scripts/e2e.py clean --all                       # stop everything it started
```

## Concepts

**server** — how to launch one serving backend: model path, served name, host/port, `CUDA_VISIBLE_DEVICES`, `env`, and either UniServe `serve_args` or an explicit `command` (any backend — this is how the vLLM-Omni comparison servers are declared). Specs support `extends` so variants stay single-source (`sensenova-u1-tp4` extends `sensenova-u1-single`; `sensenova-u1-perf` adds only the decode-burst env).

**workload** — one verification or measurement unit bound to a compatible server (overridable with `--server`). Every workload carries a `description` in the config; `list` prints them.

**suite** — an ordered list of workloads. With `--manage-servers` the driver cleans, launches (waiting for readiness), runs, and cleans per workload, so a suite reproduces from a cold checkout plus a built binary. Artifacts land under `artifacts/servers/<name>/` (pid, log) and `artifacts/workloads/<name>/` (summary.json, decoded PNGs, harness output).

## Workload types

### `generate` — correctness gates over the native SSE stream

POSTs the declared `payload` to `/generate` and validates the event stream. Gates: `expect_images` + `expect_image_width/height` (exact count and pixel dimensions of decoded PNGs, saved for inspection), `expect_image_steps` (exact per-image `image_step` event count — catches silently shortened denoise schedules), `expect_min_text_tokens` and `expect_finished` (text floors, clean termination), and any `error` event fails the workload. For understanding-mode inputs, `input_image_synthetic: {seed, width, height}` renders a deterministic geometric scene (blue sky, green ground, red house with brown roof, yellow sun upper-right) and injects it as `input_image_b64`, so i2t gates run hermetically with no datasets or checked-in image bytes.

These gates are UniServe-native-API-only by design; cross-backend comparison happens through the perf type below, which speaks each backend's public wire format.

### `uniserve_bench` — perf points (`perf` subcommand)

Runs a point through the shared benchmark harness (`benchmarks/serving/uniserve_bench`) against the workload's server: `task` (t2i / i2t / text / interleave) plus raw harness `args` (dataset, sizes, steps, seed, concurrency, warmup). Optional `expect_metrics` gates dotted metric paths with min/max bounds; any failed request also fails the workload.

Two flavors coexist:
- **Tripwires** (UniServe servers, with bounds): deliberately loose floors/ceilings that trip only when a path regresses to a known-slow regime — e.g. BAGEL i2t falling from the graphed/burst decode (~160 tok/s) back to eager per-token (~16 tok/s). They are not headline numbers and are insensitive to run-to-run noise.
- **Comparison points** (`*-omni`, no bounds): byte-for-byte the same workload — same prompt file, sizes, steps, seed, concurrency, warmup — against a vLLM-Omni server declared with an explicit launch `command`. Same harness, same metrics pipeline, so numbers are directly comparable. The i2t mirrors switch only the wire (`--i2t-wire openai_chat`, since vLLM-Omni's understanding path is its chat API and does not stream). Headline benchmarking methodology and the committed prompt set live in `benchmarks/serving/sensenova/`.

### `bench` / `script` — legacy runners

`bench` drives `benchmarks/serving/bench_serving.py` (OpenAI-compatible text stress); `script` runs an arbitrary repo script with an optional `expect_file` check.

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

### `qwen3-sharegpt-stress`

Legacy text-serving stress: 200 ShareGPT prompts at 16 req/s against Qwen3-32B; expects 200 completed, 0 failed.

## Suites

| suite | contents | purpose |
|---|---|---|
| `sensenova-verify` | t2i seed-42, i2t house, tour-guide interleave | SenseNova correctness across all three task shapes |
| `bagel-verify` | t2i seed-42, i2t house | BAGEL correctness on the unified runtime |
| `verify-all` | both of the above | pre-merge correctness pass |
| `sensenova-perf`, `bagel-perf`, `perf-all` | the four gated tripwires | regression detection to pre-unification paths |
| `sensenova-vs-omni`, `bagel-vs-omni`, `matrix-vs-omni` | tripwires + omni mirrors | the cross-backend comparison matrix under identical workloads |

Note on `--manage-servers` with omni suites: the vLLM-Omni servers cold-start in minutes (engine + pipeline init); the default launch timeout (1800 s) covers it, but running the UniServe and omni halves against separately-launched servers is faster when iterating.

## Conventions

- Gates run against the **default** serving configuration. Opt-in feature envs live on server specs, not workloads: the perf profiles pin `UNISERVE_DECODE_TOKEN_BURST=8`; TeaCache (`UNISERVE_DENOISE_RESIDUAL_CACHE=1`) and TP4 denoise-step graphs (`UNISERVE_DENOISE_STEP_GRAPH=1`) stay off in gates because default-path outputs are the guarded identities — a gated feature that needs its own suite gets its own server variant.
- The vLLM-Omni comparison environment is documented in `benchmarks/serving/sensenova/serve_vllm_omni.sh` (a dedicated venv with vllm 0.24 and the refs checkout installed non-editably); the `bagel-omni` spec carries the attention-backend overrides that dodge its broken FlashAttention-CuTe build on this aarch64 stack, which costs it a few percent — noted for fairness.
- Adding a model: (1) a server spec; (2) a t2i `generate` gate with exact size/step bounds and a fixed seed, plus an i2t gate with `input_image_synthetic` and a token floor; (3) perf tripwires with bounds set from a measured healthy run (≈1.5× ceilings, ≈0.6× floors); (4) optional `-omni` mirrors if the model exists there; (5) a `<model>-verify` suite appended to `verify-all`.
- The driver never writes outside `artifacts/`; `clean --all` reaps by pid file (process-group SIGTERM, then SIGKILL), so killed runs never leak servers.
