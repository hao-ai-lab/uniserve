# End-to-end verification and perf gating (`scripts/e2e.py`)

`scripts/e2e.py` is the profile-driven driver that launches real UniServe
servers, runs correctness gates against them over the public HTTP APIs, and
runs perf tripwires through the shared benchmark harness. Everything it does
is declared in `scripts/e2e_config.json`; the script owns only mechanics.

```
scripts/e2e.py list                         # show servers / workloads / suites
scripts/e2e.py launch sensenova-u1-single   # start one server (pid+log under artifacts)
scripts/e2e.py generate bagel-t2i-seed42    # one correctness workload against it
scripts/e2e.py perf bagel-i2t-perf          # one perf tripwire
scripts/e2e.py run verify-all --manage-servers   # suites, launching/cleaning per workload
scripts/e2e.py clean --all                  # stop everything it started
```

## Concepts

- **server** — how to launch one serving topology: model path, served name,
  port, `CUDA_VISIBLE_DEVICES`, extra `serve_args`, and `env` (e.g.
  `UNISERVE_DECODE_TOKEN_BURST=8` on the perf profiles). Specs support
  `extends` for single-point-of-truth variants (`sensenova-u1-tp4` extends
  `sensenova-u1-single`).
- **workload** — one verification or benchmark unit, bound to a compatible
  server (overridable with `--server`).
- **suite** — an ordered list of workload names. With `--manage-servers` the
  driver cleans, launches (waiting for readiness), runs, and cleans again per
  workload, so a suite is reproducible from a cold checkout + built binary.

Artifacts land under `<artifact_root>/servers/<name>/` (pid, log) and
`<artifact_root>/workloads/<name>/` (summary.json, decoded images, benchmark
outputs).

## Workload types

### `generate` — correctness gates over the native SSE stream

POSTs the declared `payload` to `/generate` and validates the event stream:

- `expect_images`, `expect_image_width/height` — exact count and dimensions
  of decoded PNGs (validated from the PNG header, saved to the workload dir
  for visual inspection).
- `expect_image_steps` — exact per-image `image_step` event count (catches
  silently shortened denoise schedules).
- `expect_min_text_tokens`, `expect_finished` — text-output floors and clean
  stream termination; any `error` event fails the workload.
- `input_image_synthetic: {seed, width, height}` — understanding-mode inputs:
  the driver renders a deterministic geometric scene (sky/ground/red
  house/sun) and injects it as `input_image_b64`, so i2t gates run hermetically
  with no dataset or checked-in image bytes.

### `uniserve_bench` — perf tripwires (`perf` subcommand)

Runs a point through the shared benchmark harness
(`benchmarks/serving/uniserve_bench`) against the workload's server and gates
the resulting `summary.json`:

- `task` (`t2i` / `i2t` / `text` / `interleave`) plus raw `args` passed to the
  harness CLI (dataset, sizes, steps, seed, concurrency, warmup).
- `expect_metrics` — dotted metric paths with `min`/`max` bounds, e.g.
  `{"image_latency_ms.mean": {"max": 10000}, "output_throughput": {"min": 100}}`.
  Any failed request also fails the workload.

The bounds are deliberately **loose regression tripwires**, not headline
numbers: they are set to trip when a path falls back to a known-slow regime
(e.g. BAGEL i2t dropping from the graphed/burst decode (~160 tok/s) back to
eager per-token (~16 tok/s)), while staying insensitive to run-to-run noise.
Headline benchmarking methodology — including the vLLM-Omni comparison —
lives in `benchmarks/serving/sensenova/`.

### `bench` / `script` — legacy runners

`bench` drives `benchmarks/serving/bench_serving.py` (OpenAI-compatible text
stress, e.g. `qwen3-sharegpt-stress`); `script` runs an arbitrary repo script
with an optional `expect_file` artifact check.

## Current suites

| suite | contents | what it protects |
|---|---|---|
| `sensenova-verify` | t2i seed-42 (2048×1152×50, exact size/steps), i2t over the synthetic scene, `sensenova-travel-interleave-4x` (4 images, 200 steps, 0 errors) | SenseNova correctness across all three task shapes, including the interleave path no other system serves |
| `bagel-verify` | t2i seed-42 (1024²×50), i2t over the synthetic scene | BAGEL correctness on the unified runtime (paged-extend prefill, graphed decode, batched CFG denoise) |
| `verify-all` | both of the above | pre-merge correctness pass |
| `sensenova-perf` / `bagel-perf` / `perf-all` | the four `*-perf` tripwires | regressions to pre-unification eager paths |

Opt-in feature envs are part of the server spec, not the workload: the perf
profiles pin `UNISERVE_DECODE_TOKEN_BURST=8`; TeaCache
(`UNISERVE_DENOISE_RESIDUAL_CACHE=1`) and TP4 denoise graphs
(`UNISERVE_DENOISE_STEP_GRAPH=1`) stay off in gates because default-path
outputs are the guarded identities — add dedicated server variants when a
gated feature needs its own suite.

## Adding a model

1. Add a server spec (model path, port, GPUs, envs, serve_args).
2. Add a `generate` t2i workload with exact size/step gates and a seed, and an
   i2t workload with `input_image_synthetic` + `expect_min_text_tokens`.
3. Add `uniserve_bench` perf workloads; set bounds from a measured healthy run
   with generous margin (≈1.5× on ceilings, ≈0.6× on floors).
4. Register a `<model>-verify` suite and append it to `verify-all`.

## Conventions

- Gates run against the **default** serving configuration; a gate that needs a
  feature env gets its own server variant so the guarded baseline stays pinned.
- `generate` image workloads keep their PNGs in the workload dir — image
  *quality* is judged there by inspection when a gate's hash/size checks pass
  but the substrate changed (kernel-level rewrites are only expected to be
  bitwise-stable when the change is layout/mechanics-only).
- The driver never edits state outside `artifact_root`; killing it never leaks
  servers (`clean --all` reaps by pid file, process-group SIGTERM then
  SIGKILL).
