# Audit dismantling: scope, protocol and acceptance

This document records the removal work driven by the architecture audit, the fixed measurement protocol governing its acceptance, and the delivered result.

## Delivered removals

Each change removed a mechanism that was unreachable, duplicated elsewhere, or confined to tests while occupying a production interface. Every entry states the evidence that established removability.

| Change | Evidence |
| --- | --- |
| Peer reduction | `allocate_peer_reductions` and `collective_scope` had no callers, so `_ACTIVE_REDUCTIONS` was permanently empty and `try_sum_reduction` could only return `False`. |
| `retryable` error field | Carried from the worker taxonomy through PyO3, FlatBuffers, the codec and the executor; the scheduler's only use was a tracing field. Nothing retried. |
| Dead helpers and test-only API | Sixteen symbols with no production caller. `video_segment_rgb` duplicated `VideoPostprocessor.forward`'s RGB24 reconstruction; `build_worker_info` and `make_transport` were one-line wrappers used only by tests. |
| Sparse-video dispatchers | Five operations with one provider each, whose `can_run` checks restated the Triton guard the kernel wrappers already perform. |
| `EngineSnapshot` and the fixed version | Built once and destructured immediately with three of six fields read; `uniserve_version()` returned the constant `"uniserve"` beside two other version numbers. |
| Checkpoint inspector | Model identity was declared three times, the server already read `model_type`, and `build_state` already asserted `engine.denoise_steps()` matched the inspector's contract. |
| Simulated backend | `--sim` made a control-plane test double a deployable mode and forced an `EngineBackendKind` branch and a fabricated EOS id through the server. |
| FlashInfer fast decode plan | Called private FlashInfer interfaces and hardcoded an internal positional ABI while the public `wrapper.plan()` path was already present with the same arguments. |
| Shared processing contract | Nine `uniserve_worker` modules imported it from `uniserve_models`, the concrete-model package. |
| Unused dependencies | Five Python runtime dependencies and five test tools with no import and no invoking configuration; four crate dependencies with no use. |

## Measurement protocol

The protocol is the one fixed in [model-library-performance.md](model-library-performance.md), reused without modification because the question is identical: whether a structural change altered serving performance.

Both roles run from `/workspace/UniServe` so the worker interpreter's editable installation resolves to the checkout under measurement. Each role builds its own release server and CPython 3.12 PyO3 extension with `PYO3_PYTHON=/workspace/UniServe/.venv/bin/python cargo build --locked --release -p uniserve -p uniserve-ipc-py --features pyo3/extension-module`, and installs the resulting `lib_uniserve_ipc.so` into the checkout before any point runs.

Workloads, sample counts, dataset revisions, arrival rates, concurrency, sampling, geometry, precision, cache behavior, warmup and resource limits come from the committed benchmark profiles. The model environment is fixed:

```bash
export UNISERVE_QWEN3_MODEL=/workspace/models/Qwen3-32B
export UNISERVE_BAGEL_MODEL=/workspace/models/BAGEL-7B-MoT
export UNISERVE_SENSENOVA_MODEL=/workspace/models/SenseNova-U1-8B-MoT-Interleaved
export UNISERVE_MINIMAX_H3_MODEL=/workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree
export UNISERVE_H3_QUANT_MODE=balanced
```

Points execute serially, one benchmark server and one harness at a time, with no concurrent GPU work and no concurrent build or analysis load on the host. A point measured while other work occupied the machine is not a canonical artifact and is re-measured.

One launch argument differs between roles: the candidate's server commands omit `--model-description`, which no longer exists because the checkpoint selects the profile. Workload, dataset selection, sampling and capacity settings are identical, so the points remain comparable.

## Acceptance

Every declared `decode-runtime` metric must have slowdown at most 1.05, computed as candidate/reference for lower-is-better metrics and reference/candidate for higher-is-better metrics. Every `fast_h3` point must have candidate-minus-reference at most 0.3 seconds for both mean video latency and reciprocal videos-per-second. A gain elsewhere cannot offset a failure.

These thresholds are inherited from the prior protocol and were fixed before any measurement in this series.

```bash
.venv/bin/python scripts/compare_eval.py decode-runtime \
  --config artifacts/audit-dismantle/benchmark-profiles.toml \
  --reference-root artifacts/audit-dismantle/reference \
  --candidate-root artifacts/audit-dismantle/candidate \
  --max-regression 0.05 \
  --output-dir artifacts/audit-dismantle/decode-comparison

.venv/bin/python scripts/compare_eval.py fast_h3 \
  --config artifacts/audit-dismantle/benchmark-profiles.toml \
  --reference-root artifacts/audit-dismantle/reference \
  --candidate-root artifacts/audit-dismantle/candidate \
  --max-latency-regression-ms 300 \
  --output-dir artifacts/audit-dismantle/h3-comparison
```

## Remaining audit scope

The following replacements from the audit are not delivered here. Each is a multi-file architectural change whose acceptance depends on its own end-to-end measurement.

- Worker input construction still resolves concrete model builders through the `_BUILDERS` table in `uniserve_worker/bootstrap/inputs.py`. The `bind` step of each builder constructs the model's own typed input and belongs on the model as a capability, leaving the worker owning buffers, bounds and stepping.
- Models still declare `entry_points` and `entry_paths` separately, and `describe_components` reverse-matches one against the other through its `anchors` table while re-importing the model package.
- Launch configuration is still translated from the Rust CLI through `WorkerProcessArgs` into argv and back through Python `argparse`, with two independent sets of defaults.
- The IPC boundary still maintains a typed submit path beside a schema-derived fallback.
- `Worker` and `ModelRunner` still concentrate IPC, execution, cache, transfer and model ownership.

`specs/tasks.md` is referenced as authoritative by the repository instructions but does not exist.
