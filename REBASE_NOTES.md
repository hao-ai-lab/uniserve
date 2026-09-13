# H3 architecture port map

## Source identities

- Upstream: `21f3204d42d26b5b44c1c25349164aa650615dc4`; feature source: `fab649d7` on `fv/omniref`; common ancestor: `2ef56d55`.
- Base model: `hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da`.
- Base numerical reference: FastVideo `a943220c115228ade5d57b3bab9a6a87fd600a10`; distilled manifests carry their own exact numerical reference commit.

## Ownership boundaries

`fdcab4dc` replaces runtime transactions with Scheduler-owned admission, inflight computations and allocations. IPC uses Start/Finish/Free, ScheduledRequest, RequestOutput and typed TensorPublication. `1b0f77a3` flattens execution into BatchState and PendingOutput; request progress, tensor storage and CPU tasks have shared runtime owners. `21f3204d` separates numerical models from execution: components declare calls, ModelSource resolves checkpoint sources, bootstrap prepares configuration, and models receive numerical batches and borrowed resource views.

The intervening staging/publication commits retain resident decode columns, sampling batches, immutable identities and physical tensor bindings through logical retirement. Reference inputs must participate in those owners, not add a parallel product registry or completion path.

## Module mapping

Every feature-source changed path is listed below. Destinations describe port ownership, not completed implementation. Existing behavioral coverage is retained only when it exercises the public numerical, protocol or HTTP contract.

| Feature-source path | Upstream destination | Port rule |
| --- | --- | --- |
| `Cargo.lock` | `Cargo.lock` | Regenerate only for required dependency changes; retain upstream dependency versions. |
| `crates/engine/src/runtime/admission.rs` | `crates/engine/src/scheduler/admission.rs` | Use Scheduler-owned request state, admission reservations and tensor dependencies. |
| `crates/engine/src/runtime/execution.rs` | `crates/engine/src/scheduler/execution.rs` | Use Scheduler-owned request state, admission reservations and tensor dependencies. |
| `crates/engine/src/runtime/inflight.rs` | `crates/engine/src/scheduler/inflight.rs` | Use Scheduler-owned request state, admission reservations and tensor dependencies. |
| `crates/engine/src/worker/executor.rs` | `crates/engine/src/worker/executor.rs` | Adapt reference request payload forwarding to current scheduler/executor interfaces. |
| `crates/engine/tests/multiproc_ipc.rs` | `crates/engine/tests/multiproc_ipc.rs` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `crates/foundation/core/src/events.rs` | `crates/foundation/core/src/events.rs` | Add ordered reference request descriptors at the public request boundary. |
| `crates/foundation/core/src/lib.rs` | `crates/foundation/core/src/lib.rs` | Add ordered reference request descriptors at the public request boundary. |
| `crates/foundation/core/tests/media_request.rs` | `crates/foundation/core/tests/media_request.rs` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `crates/server/Cargo.toml` | `crates/server/Cargo.toml` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/examples/h3_reference_cpu.rs` | `crates/server/examples/h3_reference_cpu.rs` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/src/engine_client/in_process.rs` | `crates/server/src/engine_client/in_process.rs` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/src/engine_client/media.rs` | `crates/server/src/engine_client/media.rs` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/src/openai/error.rs` | `crates/server/src/openai/error.rs` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/src/openai/types/videos.rs` | `crates/server/src/openai/types/videos.rs` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/src/openai/videos.rs` | `crates/server/src/openai/videos.rs` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/src/serving/mod.rs` | `crates/server/src/serving/mod.rs` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/src/serving/model.rs` | `crates/server/src/serving/model.rs` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/src/serving/references.rs` | `crates/server/src/serving/references.rs` | Retain upstream frontend/preprocessing ownership; add bounded contract-gated inline reference admission. |
| `crates/server/tests/model_preprocessing.rs` | `crates/server/tests/model_preprocessing.rs` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `crates/server/tests/video_requests.rs` | `crates/server/tests/video_requests.rs` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `crates/worker-ipc-py/src/convert.rs` | `crates/worker-ipc-py/src/convert.rs` | Extend current schema and conversion together; retain upstream operation and tensor identity model. |
| `crates/worker-ipc/schema/worker.fbs` | `crates/worker-ipc/schema/worker.fbs` | Extend current schema and conversion together; retain upstream operation and tensor identity model. |
| `crates/worker-ipc/src/codec.rs` | `crates/worker-ipc/src/codec.rs` | Extend current schema and conversion together; retain upstream operation and tensor identity model. |
| `crates/worker-ipc/src/iceoryx.rs` | `crates/worker-ipc/src/iceoryx.rs` | Extend current schema and conversion together; retain upstream operation and tensor identity model. |
| `crates/worker-ipc/src/operation.rs` | `crates/worker-ipc/src/operation.rs` | Extend current schema and conversion together; retain upstream operation and tensor identity model. |
| `crates/worker-ipc/src/product.rs` | `crates/worker-ipc/src/tensor.rs` | Extend typed tensor metadata; retain BufferId, TensorRef and TensorPublication ownership. |
| `crates/worker-ipc/src/tests.rs` | `crates/worker-ipc/src/tests.rs` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `docs/examples/h3-endpoint.json.template` | `docs/examples/h3-endpoint.json.template` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `docs/fast_h3/fast_h3.md` | `docs/fast_h3/fast_h3.md` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `scripts/h3/README.md` | `scripts/h3/README.md` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `scripts/h3/h3-endpoint.py` | `scripts/h3/h3-endpoint.py` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `scripts/h3/serve-h3-ref.sbatch` | `scripts/h3/serve-h3-ref.sbatch` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `scripts/h3/serve-h3.sbatch` | `scripts/h3/serve-h3.sbatch` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `scripts/h3/smoke-h3-ref.py` | `scripts/h3/smoke-h3-ref.py` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `scripts/h3/smoke-h3.py` | `scripts/h3/smoke-h3.py` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `scripts/h3/test_endpoint.py` | `scripts/h3/test_endpoint.py` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `specs/h3-reference-serving.md` | `specs/h3-reference-serving.md` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `specs/tasks.md` | `specs/tasks.md` | Reconcile with current CLI and supported behavior; do not modify external deployment scripts. |
| `tests/python/e2e/test_h3_reference_cpu.py` | `tests/python/e2e/test_h3_reference_cpu.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/fixtures/h3_reference_worker.py` | `tests/python/fixtures/h3_reference_worker.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/fixtures/models/fasth3-eight-step/fastvideo_inference.json` | `tests/python/fixtures/models/fasth3-eight-step/fastvideo_inference.json` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/integration/runtime/test_entry_execution.py` | `tests/python/integration/runtime/test_entry_execution.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/backends/test_video_sparse_metadata.py` | `tests/python/unit/backends/test_video_sparse_metadata.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_diffusion_modulation.py` | `tests/python/unit/models/test_diffusion_modulation.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_audio_encode.py` | `tests/python/unit/models/test_h3_audio_encode.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_base.py` | `tests/python/unit/models/test_h3_base.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_checkpoint_recipes.py` | `tests/python/unit/models/test_h3_checkpoint_recipes.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_contract.py` | `tests/python/unit/models/test_h3_contract.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_image_conditioning.py` | `tests/python/unit/models/test_h3_image_conditioning.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_media_plan.py` | `tests/python/unit/models/test_h3_media_plan.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_presentation_geometry.py` | `tests/python/unit/models/test_h3_presentation_geometry.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_reference_execution.py` | `tests/python/unit/models/test_h3_reference_execution.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_references.py` | `tests/python/unit/models/test_h3_references.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_resident_image.py` | `tests/python/unit/models/test_h3_resident_image.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/models/test_h3_target_geometry.py` | `tests/python/unit/models/test_h3_target_geometry.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/runtime/test_decoded_references.py` | `tests/python/unit/runtime/test_decoded_references.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/runtime/test_media_source.py` | `tests/python/unit/runtime/test_media_source.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/runtime/test_product_lifetime.py` | `tests/python/unit/runtime/test_product_lifetime.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `tests/python/unit/runtime/test_reference_image_decode.py` | `tests/python/unit/runtime/test_reference_image_decode.py` | Reassess against public contracts; rewrite removed operation/product and execution fixture APIs. |
| `uniserve_worker/backends/attention/video_sparse.py` | `uniserve_worker/nn/video_attention.py; backends/attention/video_sparse_provider.py` | Dense/VSA policy belongs to numerical attention; provider-specific implementations remain separate. |
| `uniserve_worker/bootstrap/catalog.py` | `uniserve_worker/bootstrap/catalog.py` | Retain upstream CatalogEntry and parsed configuration boundary; resolve checkpoint-specific recipes. |
| `uniserve_worker/bootstrap/inspect_model.py` | `uniserve_worker/bootstrap/inspect_model.py` | Retain upstream CatalogEntry and parsed configuration boundary; resolve checkpoint-specific recipes. |
| `uniserve_worker/execution/batch.py` | `uniserve_worker/execution/batch.py` | Adapt to ScheduledRequest, BatchState, PendingOutput and shared ModelRunner numerical calls. |
| `uniserve_worker/execution/bounded_storage.py` | `uniserve_worker/runtime/tensor_store.py; runtime/tensor_buffers.py; runtime/cpu.py` | Use upstream tensor leases and bounded CPU tasks; do not restore storage wrappers. |
| `uniserve_worker/execution/encode.py` | `uniserve_worker/execution/encode.py` | Adapt to ScheduledRequest, BatchState, PendingOutput and shared ModelRunner numerical calls. |
| `uniserve_worker/execution/model_runner.py` | `uniserve_worker/execution/model_runner.py` | Adapt to ScheduledRequest, BatchState, PendingOutput and shared ModelRunner numerical calls. |
| `uniserve_worker/execution/prepare.py` | `uniserve_worker/execution/prepare.py` | Adapt to ScheduledRequest, BatchState, PendingOutput and shared ModelRunner numerical calls. |
| `uniserve_worker/execution/rows.py` | `uniserve_worker/execution/rows.py` | Adapt to ScheduledRequest, BatchState, PendingOutput and shared ModelRunner numerical calls. |
| `uniserve_worker/execution/video.py` | `uniserve_worker/execution/video.py` | Adapt to ScheduledRequest, BatchState, PendingOutput and shared ModelRunner numerical calls. |
| `uniserve_worker/loader/loader.py` | `uniserve_worker/loader/source.py; bootstrap/metadata.py; loader/loader.py` | Resolve checkpoint metadata in ModelSource and bootstrap before numerical construction. |
| `uniserve_worker/media/decoded.py` | `uniserve_worker/media/decoded.py` | Bound external media decoding; publish through existing tensor transport and lifetime owners. |
| `uniserve_worker/media/source.py` | `uniserve_worker/media/source.py` | Bound external media decoding; publish through existing tensor transport and lifetime owners. |
| `uniserve_worker/models/minimax_h3/audio_vae.py` | `uniserve_worker/models/minimax_h3/audio_vae.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/base_contract.py` | `uniserve_worker/models/minimax_h3/base_contract.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/config.py` | `uniserve_worker/models/minimax_h3/config.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/encoder.py` | `uniserve_worker/models/minimax_h3/encoder.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/image_vae.py` | `uniserve_worker/models/minimax_h3/image_vae.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/layout.py` | `uniserve_worker/models/minimax_h3/layout.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/model.py` | `uniserve_worker/models/minimax_h3/model.py` | Keep EncoderMixin, DiffusionMixin, DecoderMixin, VideoMixin and borrowed TensorViews; port only numerical H3 behavior. |
| `uniserve_worker/models/minimax_h3/packing.py` | `uniserve_worker/models/minimax_h3/packing.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/presentation.py` | `uniserve_worker/models/minimax_h3/presentation.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/reference.py` | `uniserve_worker/models/minimax_h3/reference.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/transformer.py` | `uniserve_worker/models/minimax_h3/transformer.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/video_vae.py` | `uniserve_worker/models/minimax_h3/video_vae.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/vision.py` | `uniserve_worker/models/minimax_h3/vision.py` | Keep checkpoint-specific mathematics; consume numerical inputs and borrowed resources, not worker state. |
| `uniserve_worker/models/minimax_h3/weights.py` | `uniserve_worker/models/minimax_h3/weights.py; bootstrap/metadata.py` | CheckpointComponent declarations consume parsed config and BuildContext, not worker bindings. |
| `uniserve_worker/models/video.py` | `uniserve_worker/modeling/video.py; modeling/geometry.py` | Numerical output geometry remains independent of execution and publication. |
| `uniserve_worker/nn/decoder/qwen.py` | `uniserve_worker/nn/decoder/qwen.py` | Port reusable numerical behavior without introducing worker execution dependencies. |
| `uniserve_worker/nn/diffusion/schedule.py` | `uniserve_worker/nn/diffusion/schedule.py` | Port reusable numerical behavior without introducing worker execution dependencies. |
| `uniserve_worker/nn/quant/config.py` | `uniserve_worker/nn/quant/config.py` | Port reusable numerical behavior without introducing worker execution dependencies. |
| `uniserve_worker/worker/worker.py` | `uniserve_worker/worker/worker.py; runtime/tensors.py; bootstrap/capacity.py` | Keep flattened worker and shared request/tensor lifetime owners. |

## Launcher interface

`--supported-ops` selects computation capability groups, not wire operation enum values. Current H3 selectors are `encoder_text`, `diffusion_prepare`, `diffusion_step`, `diffusion_decode`, `media_append`, `diffusion_finalize`, and `transfer_product`. Decode covers both audio and video; append covers both encoding tracks. External launchers must use the current worker CLI and rebuild the Rust server and Python IPC extension together. External deployment files are not part of this worktree.

## Implementation review

The distilled checkpoint recipe port is implemented. Base dense H3, reference H3, reference IPC, reference worker execution, reference HTTP admission and CPU HTTP-to-transformer evidence are not yet ported. Detailed review of the complete refactor patch is incomplete; the ownership map is not a claim that the full 117,725-line patch has been reviewed.

### Modified upstream files

| Path | Required change |
| --- | --- |
| `uniserve_worker/bootstrap/catalog.py` | Bind each validated distilled checkpoint ladder and shifts to its schedule factory. |
| `uniserve_worker/bootstrap/metadata.py` | Read embedded or root-bound explicit manifests, enforce non-overriding sidecars, and validate scheduler shifts against the checkpoint recipe. Filesystem access remains outside numerical model modules. |
| `uniserve_worker/loader/source.py` | Fetch remote architecture sidecars before selecting the checkpoint-specific catalog entry, so loading uses the validated recipe rather than the default four-step schedule. |
| `uniserve_worker/models/minimax_h3/config.py` | Validate immutable eight-step release identity and explicit export recipes; expose normalized ladder, shifts and sparsity while retaining the four-step protocol. |
| `uniserve_worker/models/minimax_h3/layout.py` | Size sparse-selection scratch using checkpoint sparsity. |
| `uniserve_worker/models/minimax_h3/model.py` | Use checkpoint evaluation count and schedule in the numerical diffusion specification and worker-advertised step count. Direct numerical construction retains upstream's standard four-step defaults. |
| `uniserve_worker/models/minimax_h3/transformer.py` | Pass checkpoint sparsity into the shared sparse attention metadata. |
| `uniserve_worker/models/minimax_h3/weights.py` | Supply validated sparsity when constructing numerical layout and transformer components. CheckpointComponent ownership remains unchanged. |
| `uniserve_worker/nn/sparse_attention.py` | Parameterize shared selection cardinality by validated sparsity for both planning and numerical selection. |

No Rust runtime, IPC schema, upstream test, or deployment script is changed. Added behavioral tests cover four/eight-step schedules, independent Diffusers scheduler parity, immutable release identity, explicit sidecar binding, numerical diffusion specification, worker step advertisement and public sparse selection cardinality.

### Validation evidence

The isolated environment uses the locked Torch 2.13.0 CUDA-13 distribution and freshly built upstream IPC extension. Native compilation requires the read-only native toolchain directory on PATH, its shared libraries on LD_LIBRARY_PATH, LIBCLANG_PATH pointing at its LLVM-18 library directory, and local unversioned linker names for the installed libpython3.12.so.1 and libstdc++.so.6. Cargo targets are under `/tmp/uniserve-rebase-target` with 16 jobs.

The focused H3/bootstrap/checkpoint suite passes all 43 tests. The complete unit directory reports 416 passed, 69 skipped and 11 failed; all 11 failures require a CUDA device or CUDA pinned-memory allocation and report the missing NVIDIA driver on the login node. No tests were disabled or weakened. The reference CPU e2e test has not been ported or executed. No GPU correctness or performance claim is made.

`cargo build --workspace -j 16` passes. `cargo test --workspace -j 16` does not pass: the multiprocess IPC suite initially had two Python fixture import failures, resolved by setting PYTHONPATH to the worktree root, plus two persistent failures. A targeted confirmation reports eight passed and two failed: `failed_producer_retires_waiting_consumers_and_preserves_independent_work` submits TextEncoding to the upstream stub, whose component calls do not advertise text encoding; `multiprocess_topology_handles_rank_failure_and_capacity_limits` reports `fast submission did not complete` after its 30-second poll. The latter cause is not established. Neither test nor timeout was changed, and the workspace run stopped at that suite.
