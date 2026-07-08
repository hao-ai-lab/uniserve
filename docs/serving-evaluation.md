# Serving Evaluation (`uniserve-eval`)

`uniserve_eval` owns profile-driven serving checks and small perf tripwires.
The official benchmark matrix is defined in the same profile file and executed
by `scripts/run_benchmarks.py`; see `docs/benchmark-protocol.md` for benchmark
policy.

Everything tracked is declared in `uniserve_eval/profiles.json`. Local model,
dataset, SGLang, and vllm-omni paths come from environment variables, not from
tracked profile values.

## Commands

```bash
uniserve-eval list
uniserve-eval launch gate/server/sensenova
uniserve-eval verify gate/sensenova/default-travel
uniserve-eval perf perf/tripwire/bagel/i2t
uniserve-eval run gate/all --manage-servers
uniserve-eval clean --all

.venv/bin/python scripts/run_benchmarks.py --benchmark main --dry-run --no-build
```

## Profile Sections

| section | purpose |
|---|---|
| `shared` | Documented environment variables, shared load axes, and common server knobs. |
| `servers` | Launch specs. Namespace prefixes separate `benchmark/server/...` from `gate/server/...`. |
| `workloads` | Correctness gates and small perf tripwires. |
| `suites` | Ordered workload groups for `uniserve-eval run`. |
| `benchmarks` | Official benchmark matrices consumed by `scripts/run_benchmarks.py`. |

Server specs may either declare a UniServe model plus `serve_args`, or an
explicit command for another backend. Environment variables in profile values
are expanded at launch; unresolved variables fail before a server starts.

## Namespaces

- `benchmark/server/...`: servers used by the official benchmark matrix.
- `gate/server/...`: servers for correctness gates and small tripwires.
- `gate/...`: chat-completions correctness workloads.
- `perf/tripwire/...`: small regression sentinels, not headline numbers.
- `benchmarks.main`: the current benchmark matrix.

The namespace is part of the contract. Official benchmark workloads are defined
once in `benchmarks.main`.

## Workload Types

### `verify`

Chat-completions correctness gates. The verifier posts OpenAI-compatible chat payloads to `/v1/chat/completions` and checks status, OpenAI error chunks, finish reasons, terminal `[DONE]` for streamed requests, generated text, generated-image count, decoded PNG dimensions, and response shape. I2T gates can inject a deterministic synthetic image through `input_image_synthetic`.

### `perf`

Small harness points run through `python -m uniserve_eval.harness.cli`.
Tripwires use loose metric bounds to catch large regressions; they are not the
official benchmark matrix. Current tripwire load is arrival-rate based.

### `script`

Generic repo-local Python escape hatch with an optional expected output file.

## Suites

| suite | contents |
|---|---|
| `gate/sensenova` | SenseNova T2I, I2T, and default-generation correctness gates. |
| `gate/bagel` | BAGEL T2I and I2T correctness gates. |
| `gate/all` | All correctness gates. |
| `perf/tripwire` | UniServe SenseNova/BAGEL T2I/I2T regression sentinels. |

## Benchmark Matrix

Run the official benchmark matrix with:

```bash
.venv/bin/python scripts/run_benchmarks.py --benchmark main
```

The matrix is under `benchmarks.main` in `profiles.json`. It uses open-loop
arrival rates `1,2,4,8,16`; server launch commands come from the benchmark
profiles.

Dry-run mode writes the audited command set without starting servers:

```bash
.venv/bin/python scripts/run_benchmarks.py --benchmark main --dry-run --no-build
```

## Required Environment

See `docs/benchmark-protocol.md` for the full table. The short version:

- `UNISERVE_QWEN3_MODEL`
- `UNISERVE_SENSENOVA_MODEL`
- `UNISERVE_BAGEL_MODEL`
- `UNISERVE_SHAREGPT_PATH`
- `UNISERVE_SGLANG_PYTHON`
- `UNISERVE_OMNI_VLLM`

## Artifacts

`uniserve-eval` stores suite artifacts under the configured `artifact_root`.
The benchmark runner writes to `benchmarks.main.artifact_root` unless
`--output-root` overrides it. Each benchmark point gets:

- `command.txt`
- `run.log`
- `summary.json`
- `preflight.txt`
- `postflight.txt`
- `samples/` when the harness saves request samples
