"""Numerical preparation and model execution for the native executor."""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

from uniserve_worker._uniserve_ipc import BatchState
from uniserve_worker.execution.commit import (
    apply_decode_state,
    publish_predicates,
    validate_outputs,
)
from uniserve_worker.execution.dispatch import execute_calls
from uniserve_worker.execution.image import reserve_images
from uniserve_worker.execution.prepare import (
    capture_predicates,
    prepare_predicates,
    reserve_outputs,
)
from uniserve_worker.profiling import _forward_stats, record_component
from uniserve_worker.protocol.output import ForwardStats

if TYPE_CHECKING:
    from uniserve_worker.worker import Worker


class BatchRunner:
    """Prepare numerical work while Rust owns request and batch progression."""

    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.collective = any(
            group.size > 1
            for binding in worker.runner.bindings.values()
            if binding.owns
            for group in binding.communicators
        ) or (
            worker.sampling_group is not None and worker.sampling_group.size > 1
        )

    def prepare(self, state: BatchState) -> None:
        """Prepare inline image inputs while other batches can advance."""
        # Inline images prepare on the host lane while the executor advances
        # other batches. Their tasks join the native batch input dependencies.
        reserve_images(
            state,
            host_tasks=self.worker.host_tasks,
            request_pool=self.worker.requests,
            model_runner=self.worker.runner,
        )

    def prepare_predicates(self, state: BatchState) -> None:
        """Bind predicate views after native input admission."""
        prepare_predicates(
            state,
            tensor_store=self.worker.tensor_store,
            output_pool=self.worker.output_pool,
            model_runner=self.worker.runner,
        )

    def capture_predicates(self, state: BatchState) -> None:
        capture_predicates(state, self.worker.tensor_store)

    def profile_step(self, state: BatchState) -> AbstractContextManager[None]:
        """Scope one numerical step without advancing on lifecycle batches."""
        if self.worker.profiler is not None and state.batch.calls:
            first = state.batch.calls[0]
            return self.worker.profiler.step(
                f"batch:{first.kind}:{first.component}"
            )
        return nullcontext()

    def reserve(self, state: BatchState) -> None:
        reserve_outputs(
            state.batch,
            state.predicate_values(),
            kv_cache=self.worker.kv_cache,
            host_tasks=self.worker.host_tasks,
            tensor_store=self.worker.tensor_store,
            worker_info=self.worker.info,
            latent_pool=self.worker.latent_pool,
            output_pool=self.worker.output_pool,
            request_tables=self.worker.block_tables,
            request_pool=self.worker.requests,
            model_runner=self.worker.runner,
            config=self.worker.worker_config,
            state=state,
        )

    def execute(self, state: BatchState, indices: Sequence[int]) -> None:
        """Launch the selected homogeneous calls into reserved output views."""
        scheduled = tuple(state.batch.calls[index] for index in indices)

        execute_calls(
            scheduled,
            kv_cache=self.worker.kv_cache,
            tensor_store=self.worker.tensor_store,
            worker_info=self.worker.info,
            latent_pool=self.worker.latent_pool,
            media_mux=self.worker.media_mux,
            export_transports=self.worker.export_transports,
            transports=self.worker.transports,
            request_tables=self.worker.block_tables,
            request_pool=self.worker.requests,
            model_runner=self.worker.runner,
            decode_state=self.worker.decode_state,
            sampling_group=self.worker.sampling_group,
            tokenizer=self.worker.tokenizer,
            config=self.worker.worker_config,
            state=state,
        )

    def prepare_outputs(self, state: BatchState) -> None:
        """Write completion predicates and check numerical output bounds."""
        publish_predicates(state=state, tensor_store=self.worker.tensor_store)
        validate_outputs(state)

    def apply_outputs(self, state: BatchState) -> None:
        apply_decode_state(state=state, decode_state=self.worker.decode_state)

    def execution_stats(
        self, state: BatchState, started: int, commit_started: int | None
    ) -> tuple[int, ForwardStats]:
        """Snapshot timings before resource commit and device state updates."""
        if commit_started is not None:
            record_component(state.component_us, "commit_lane", commit_started)
        return (
            (time.perf_counter_ns() - started) // 1000,
            _forward_stats(state.forward_stats, state.component_us)
            if state.started_ns
            else ForwardStats(),
        )
