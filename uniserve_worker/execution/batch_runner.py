"""Numerical preparation and model execution for the native executor."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

from uniserve_worker.errors import WorkerError
from uniserve_worker.execution.batch import BatchState
from uniserve_worker.execution.image import reserve_images
from uniserve_worker.execution.prepare import (
    capture_predicates,
    prepare_batch,
    prepare_inputs,
)
from uniserve_worker.execution.step import execute_batch

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

    def prepare(self, state: BatchState) -> bool:
        """Reserve numerical inputs and report whether any require imports."""
        prepare_batch(
            state,
            kv_cache=self.worker.kv_cache,
            latent_pool=self.worker.latent_pool,
            request_tables=self.worker.block_tables,
            request_pool=self.worker.requests,
        )
        # Inline images prepare on the host lane while the executor advances
        # other batches. Their tasks join the native batch input dependencies.
        reserve_images(
            state,
            host_tasks=self.worker.host_tasks,
            request_pool=self.worker.requests,
            model_runner=self.worker.runner,
        )
        return bool(state.input_products or state.kv_inputs)

    def prepare_inputs(self, state: BatchState) -> None:
        """Bind tensor views and submit copies into reserved destinations."""
        prepare_inputs(
            state,
            kv_cache=self.worker.kv_cache,
            tensor_store=self.worker.tensor_store,
            latent_pool=self.worker.latent_pool,
            output_pool=self.worker.output_pool,
            request_tables=self.worker.block_tables,
            request_pool=self.worker.requests,
            model_runner=self.worker.runner,
            transfer_backends=self.worker.transports,
            config=self.worker.worker_config,
        )

    def capture_predicates(self, state: BatchState) -> None:
        capture_predicates(state, self.worker.tensor_store)

    def execute(self, state: BatchState) -> WorkerError | None:
        """Launch numerical work with the prepared inputs and profiler scope.

        Recoverable failures return their WorkerError for native result
        assembly; fatal or explicitly propagated errors raise. The executor
        releases input leases after this call, including on failure.
        """
        batch = state.batch
        # Lifecycle-only batches issue no model step and do not advance the
        # profiler's capture window.
        step: AbstractContextManager[None]
        if self.worker.profiler is not None and batch.calls:
            first = batch.calls[0]
            step = self.worker.profiler.step(
                f"batch:{first.kind}:{first.component}"
            )
        else:
            step = nullcontext()

        with step:
            return execute_batch(
                state,
                propagate_errors=state.propagate_errors,
                kv_cache=self.worker.kv_cache,
                host_tasks=self.worker.host_tasks,
                tensor_store=self.worker.tensor_store,
                worker_info=self.worker.info,
                latent_pool=self.worker.latent_pool,
                media_mux=self.worker.media_mux,
                output_pool=self.worker.output_pool,
                publication_transports=self.worker.publication_transports,
                request_tables=self.worker.block_tables,
                request_pool=self.worker.requests,
                model_runner=self.worker.runner,
                decode_state=self.worker.decode_state,
                sampling_group=self.worker.sampling_group,
                tokenizer=self.worker.tokenizer,
                transfer_backends=self.worker.transports,
                config=self.worker.worker_config,
            )

    def close(self, state: BatchState) -> None:
        """Release numerical owners while physical readers keep their leases."""
        state.close(
            self.worker.tensor_store,
            self.worker.latent_pool,
            self.worker.kv_cache,
        )
