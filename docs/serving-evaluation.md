# Serving evaluation

`uniserve_eval` provides profile-driven correctness checks, serving workloads, and cross-runtime benchmark comparisons. [`uniserve_eval/profiles.json`](../uniserve_eval/profiles.json) is the executable configuration, and [`benchmark-protocol.md`](benchmark-protocol.md) defines the benchmark semantics.

## Benchmark entry

Run the configured 38-point matrix through one command:

```bash
.venv/bin/python scripts/run_benchmarks.py \
  --benchmark main \
  --output-root artifacts/benchmark
```

Select complete backend groups with `--only`, inspect generated commands with `--dry-run --no-build`, or continue a partially completed root with `--resume`. `--repeat N` wraps the selected matrix in serial child runs. `--text-canary` and `--image-smoke` add optional numerical diagnostics.

`--formal` enables the repository, hardware, revision, process, GPU, build, and execution-identity checks defined by the canonical profile. It requires an explicit empty output root and complete comparison pairs.

## Execution model

The runner holds a host-wide lock. Each matrix point starts a fresh server, launches one harness against it, stops the server, validates the resulting artifact, and only then advances to the next point. Backend groups, workload points, load cases, and repeated matrices are never run concurrently.

The runner writes `COMMANDS.md`, point-local artifacts, server logs and snapshots, `results.json`, and `results.md` beneath the selected root. Candidate/reference ratios are reported only when both point artifacts exist and the fixed-work comparison passes.

## Configuration

| Section | Contents |
| --- | --- |
| `shared` | Environment-variable descriptions |
| `servers` | UniServe and reference launch specifications |
| `workloads` | Correctness and small regression workloads |
| `suites` | Ordered workload collections |
| `benchmarks` | Workload points, backend roles, named load cases, datasets, and hardware requirements |

Server profiles either declare a UniServe model plus `serve_args` or an explicit reference command. Environment variables expand at launch, unresolved required values fail before execution, and inherited server lists represent complete effective values.

The main matrix uses these environment variables where applicable:

- `UNISERVE_QWEN3_MODEL`
- `UNISERVE_SENSENOVA_MODEL`
- `UNISERVE_BAGEL_MODEL`
- `UNISERVE_SGLANG_PYTHON`
- `UNISERVE_OMNI_VLLM`
- `UNISERVE_BENCH_CUDA_VISIBLE_DEVICES`

## Artifact integrity

Each harness point retains normalized request records, GPU samples, summary, manifest, command, log, and process snapshots. Generated images use content-addressed files bound to request metadata. Matrix artifacts also bind the active profile definition, named load case, selected rows, normalized request semantics, server and harness execution identity, model content, source revision where declared, selected accelerator, and support files.

Only complete canonical artifacts are included in `results.json`. Interrupted, failed, or structurally inconsistent points remain in their point directories for diagnosis and are not incorporated into comparisons.

## Focused checks

The `uniserve-eval` command runs focused correctness and regression workloads:

```bash
uniserve-eval list
uniserve-eval launch gate/server/sensenova
uniserve-eval verify gate/sensenova/default-travel
uniserve-eval perf perf/tripwire/bagel/i2t
uniserve-eval run gate/all --manage-servers
```

These checks provide implementation feedback and are separate from the benchmark matrix.
