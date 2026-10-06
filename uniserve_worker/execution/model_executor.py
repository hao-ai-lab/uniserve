"""Native model execution and numerical batch preparation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve_worker._uniserve_ipc import ModelExecutor as ModelExecutor
from uniserve_worker.errors import (
    ComputeError,
    InputError,
    ResourceError,
    WorkerError,
    WorkerErrorCode,
    classify,
)
from uniserve_worker.model_executor.diffusion_inputs import DiffusionRow
from uniserve_worker.model_executor.image_inputs import DecodeRow, VisionRow
from uniserve_worker.model_executor.input_batch import (
    CanvasStepRow,
    InputBatch,
    InputRow,
    TokenRow,
)
from uniserve_worker.model_executor.model_runner import ModelRunner
from uniserve_worker.protocol.call import (
    ForwardMode,
    MediaCall,
)
from uniserve_worker.protocol.identity import CallId
from uniserve_worker.sampling.metadata import TokenSelection

if TYPE_CHECKING:
    from uniserve_worker.storage.block_tables import BlockTables
    from uniserve_worker.storage.decode_state import DecodeState
    from uniserve_worker.storage.kv_cache import KVCacheManager


def _prepare_inputs(
    runner: ModelRunner,
    rows: tuple[InputRow, ...],
    partitions: tuple[tuple[InputRow, ...], ...],
    *,
    cache: KVCacheManager | None,
    tables: BlockTables | None,
    states: DecodeState | None,
) -> tuple[tuple[InputBatch | None, ...], torch.Tensor, bool]:
    """Pack native row partitions into each peer's numerical backing.

    All tensor work runs under the invocation's input-copy stream. Empty
    partitions retain their peer position for collective participation.
    """
    if runner.execution.microbatches is not None:
        runner.input_buffers.validate_rows(rows)

    batches = tuple(
        None
        if not part
        else peer.prepare_inputs(
            part,
            forward_mode=rows[0].forward_mode,
            cache=cache,
            tables=tables,
            states=states,
        )
        for peer, part in zip(runner.peers, partitions, strict=True)
    )
    populated = tuple(batch for batch in batches if batch is not None)
    slots = (
        populated[0].request_pool_indices
        if len(batches) == 1
        else torch.cat([batch.request_pool_indices for batch in populated])
    )
    # Only single-token last-logits rows can borrow graph output storage.
    borrow = all(
        isinstance(row, TokenRow)
        and row.query_tokens == 1
        and row.selection is TokenSelection.LAST_LOGITS
        for row in rows
    )
    return batches, slots, borrow


def _validate_outputs(
    values: tuple[torch.Tensor, ...],
    tasks: tuple[InputRow, ...],
    device: torch.device,
) -> None:
    """Validate one model output tensor per task on the expected device."""
    for value, task in zip(values, tasks, strict=True):
        if value.device != device:
            raise ValueError(
                f"model output is on {value.device}, expected {device}"
            )
        # A canvas step reports the sampler's decision, not a raw output.
        if isinstance(task, CanvasStepRow):
            if value.dtype != torch.int64 or value.shape != (
                1 + task.canvas_length,
            ):
                raise ValueError("a canvas step reports its stop and canvas")
            continue
        if not value.is_floating_point():
            raise ValueError("raw neural outputs must use a floating dtype")

        if isinstance(task, TokenRow) and value.ndim < 2:
            raise ValueError(
                "token output must retain token and feature dimensions"
            )

        if isinstance(task, DiffusionRow) and value.shape != task.latent.shape:
            raise ValueError("flow prediction shape does not match its latent")

        if isinstance(task, VisionRow) and value.numel() == 0:
            raise ValueError("encoder output must not be empty")

        # A decode (as opposed to denoise) latent task returns an image whose
        # trailing dimensions are [height, width].
        if isinstance(task, DecodeRow):
            if value.ndim < 2 or tuple(value.shape[-2:]) != (
                task.image_height,
                task.image_width,
            ):
                raise ValueError(
                    "decoded tensor does not match the requested image shape"
                )


def _input_failure(
    error: BaseException,
    forward_mode: ForwardMode | MediaCall,
    calls: tuple[tuple[int, int, int, CallId], ...],
) -> InputError:
    """Classify invalid model inputs with their phase and call identities.

    An ``InputError`` is returned unchanged.
    """
    if isinstance(error, InputError):
        return error
    return InputError(
        str(error) or type(error).__name__,
        phase="input_preparation",
        route=forward_mode.value,
        calls=calls,
    )


def _execution_failure(
    error: BaseException,
    forward_mode: ForwardMode | MediaCall,
    calls: tuple[tuple[int, int, int, CallId], ...],
) -> WorkerError:
    """Classify a model failure and attach the active phase and call identities.

    A ``WorkerError`` is returned unchanged. A CUDA graph failure, or an error
    ``classify`` maps to a resource or fatal worker failure, becomes a
    ``ResourceError`` that keeps the classified fatality; anything else becomes
    a ``ComputeError``.
    """
    if isinstance(error, WorkerError):
        return error
    classified = classify(error)
    if isinstance(error, CUDAGraphError) or classified.code in {
        WorkerErrorCode.RESOURCE_ERROR,
        WorkerErrorCode.FATAL_WORKER_FAILURE,
    }:
        return ResourceError(
            str(error) or type(error).__name__,
            phase="graph_or_device",
            route=forward_mode.value,
            calls=calls,
            fatal=classified.fatal,
        )
    return ComputeError(
        str(error) or type(error).__name__,
        phase="neural_execution",
        route=forward_mode.value,
        calls=calls,
    )
