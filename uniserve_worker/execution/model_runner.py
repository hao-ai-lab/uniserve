"""Device execution for one immutable forward plan."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypeVar, cast

import torch
from torch import nn

from uniserve_worker.forward import (
    AttnPlan,
    DecodeOutput,
    DecodeRow,
    EncodeOutput,
    EncodeRow,
    FlowOutput,
    FlowPatches,
    FlowRow,
    ForwardBatch,
    ForwardOutput,
    ForwardRow,
    NoAttention,
    PagedDecodePlan,
    PagedVarlenPlan,
    PatchInput,
    TokenEmbeddings,
    TokenHidden,
    TokenIds,
    TokenLogits,
    TokenOutput,
    TokenRow,
    TokenSegments,
    TowerInput,
    packed_tensor_views,
)
from uniserve_worker.foundation.errors import (
    ComputeError,
    ErrorCode,
    InputError,
    ResourceError,
    WorkerError,
    classify,
)
from uniserve_worker.runtime.execution_trace import (
    ExecutionPhase,
    ExecutionTrace,
    OperationTrace,
)
from uniserve_worker.runtime.graph_store import GraphExecutionError, GraphStore
from uniserve_worker.runtime.host_staging import (
    TensorStagingSlot,
    pack_integer_tensors,
)

from ._forward_plan import ForwardPlan, OutputSlot


class RunPath(StrEnum):
    EAGER = "eager"
    GRAPH_CAPTURE = "graph_capture"
    GRAPH_REPLAY = "graph_replay"
    GRAPH_FALLBACK = "graph_fallback"


@dataclass(frozen=True, slots=True)
class RunObservation:
    route: str
    row_count: int
    row_kind_counts: tuple[tuple[str, int], ...]
    path: RunPath
    model_forward_calls: int
    duration_us: int
    graph_unpadded_tokens: int
    graph_padded_tokens: int


class ModelRunner:
    """Stage one route, invoke the model once, and validate raw outputs."""

    def __init__(
        self,
        model: nn.Module,
        graph_store: GraphStore,
        trace: ExecutionTrace,
    ) -> None:
        if type(model).forward is nn.Module.forward:
            raise TypeError("runner model must implement forward(ForwardBatch)")
        self.model = model
        self.graph_store = graph_store
        self.trace = trace
        self._last_observation: RunObservation | None = None

    @property
    def last_observation(self) -> RunObservation | None:
        return self._last_observation

    def run(self, plan: ForwardPlan) -> ForwardOutput:
        started = time.perf_counter_ns()
        operations = _trace_operations(plan)
        counts = _row_kind_counts(plan.rows)
        try:
            batch = _stage(plan)
        except Exception as error:
            _mark_staging_submitted(plan)
            self.trace.emit(
                ExecutionPhase.ROUTE_EXECUTION,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
                route=str(plan.route),
                row_kind_counts=counts,
                error=error,
            )
            input_failure = _input_failure(error, plan, "input_staging")
            if input_failure is error:
                raise
            raise input_failure from error
        _mark_staging_submitted(plan)
        calls = 0

        def invoke(value: ForwardBatch) -> ForwardOutput:
            nonlocal calls
            calls += 1
            if plan.weights.version == 0:
                result = self.model(value)
            else:
                result = torch.func.functional_call(
                    self.model,
                    dict(plan.weights.tensors),
                    (value,),
                    strict=True,
                )
            if not isinstance(result, ForwardOutput):
                raise TypeError("model forward must return ForwardOutput")
            return result

        try:
            self.trace.emit(
                ExecutionPhase.ROUTE_EXECUTION,
                operations,
                route=str(plan.route),
                row_kind_counts=counts,
            )
            with torch.inference_mode():
                graph_run = self.graph_store.execute(
                    plan.graph_key,
                    batch,
                    invoke,
                    eligible=plan.graph_eligible,
                )
                output = graph_run.output
                graph_path = graph_run.path
        except Exception as error:
            self.trace.emit(
                ExecutionPhase.FORWARD_COMPLETION,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
                route=str(plan.route),
                row_kind_counts=counts,
                error=error,
            )
            execution_failure = _execution_failure(error, plan)
            if execution_failure is error:
                raise
            raise execution_failure from error
        try:
            output.validate_for(batch)
            _validate_tensors(batch, output, plan.outputs, torch.device(plan.device))
        except Exception as error:
            self.trace.emit(
                ExecutionPhase.FORWARD_COMPLETION,
                operations,
                duration_us=(time.perf_counter_ns() - started) // 1000,
                route=str(plan.route),
                row_kind_counts=counts,
                error=error,
                execution_path=graph_path,
            )
            output_failure = _compute_failure(error, plan, "output_validation")
            if output_failure is error:
                raise
            raise output_failure from error
        expected_calls = {
            RunPath.EAGER.value: 1,
            RunPath.GRAPH_CAPTURE.value: 1,
            RunPath.GRAPH_REPLAY.value: 0,
        }
        if graph_path in expected_calls and calls != expected_calls[graph_path]:
            raise ComputeError(
                f"route execution path {graph_path!r} made {calls} model forward calls",
                phase="graph_execution",
                route=str(plan.route),
                operations=plan.transaction.operations,
            )
        if graph_path == RunPath.GRAPH_FALLBACK.value and calls not in {1, 2}:
            raise ComputeError(
                f"graph fallback made {calls} model forward calls",
                phase="graph_execution",
                route=str(plan.route),
                operations=plan.transaction.operations,
            )
        path = RunPath(graph_path)
        duration_us = (time.perf_counter_ns() - started) // 1000
        self._last_observation = RunObservation(
            route=str(plan.route),
            row_count=len(plan.rows),
            row_kind_counts=tuple(sorted(counts.items())),
            path=path,
            model_forward_calls=calls,
            duration_us=duration_us,
            graph_unpadded_tokens=(
                graph_run.row_count if path is not RunPath.EAGER else 0
            ),
            graph_padded_tokens=(
                graph_run.padded_row_count - graph_run.row_count
                if path is not RunPath.EAGER
                else 0
            ),
        )
        self.trace.emit(
            ExecutionPhase.FORWARD_COMPLETION,
            operations,
            duration_us=duration_us,
            route=str(plan.route),
            row_kind_counts=counts,
            execution_path=path.value,
        )
        return output


def _trace_operations(plan: ForwardPlan) -> tuple[OperationTrace, ...]:
    return tuple(
        OperationTrace(session_id, epoch, op_id, version)
        for (session_id, epoch, op_id), version in zip(
            plan.transaction.operations,
            plan.transaction.base_versions,
            strict=True,
        )
    )


def _row_kind_counts(rows: tuple[ForwardRow, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        name = type(row).__name__.removesuffix("Row").lower()
        counts[name] = counts.get(name, 0) + 1
    return counts


def _input_failure(error: BaseException, plan: ForwardPlan, phase: str) -> InputError:
    if isinstance(error, InputError):
        return _enrich(error, plan, phase)
    return InputError(
        str(error) or type(error).__name__,
        phase=phase,
        route=str(plan.route),
        operations=plan.transaction.operations,
    )


def _compute_failure(error: BaseException, plan: ForwardPlan, phase: str) -> ComputeError:
    if isinstance(error, ComputeError):
        return _enrich(error, plan, phase)
    return ComputeError(
        str(error) or type(error).__name__,
        phase=phase,
        route=str(plan.route),
        operations=plan.transaction.operations,
    )


def _execution_failure(error: BaseException, plan: ForwardPlan) -> WorkerError:
    if isinstance(error, (InputError, ComputeError, ResourceError)):
        return _enrich(error, plan, "neural_execution")
    classified = classify(error)
    if isinstance(error, GraphExecutionError) or classified.code in {
        ErrorCode.RESOURCE_ERROR,
        ErrorCode.FATAL_WORKER_FAILURE,
    }:
        return ResourceError(
            str(error) or type(error).__name__,
            phase="graph_or_device",
            route=str(plan.route),
            operations=plan.transaction.operations,
            retryable=classified.retryable,
            fatal=classified.fatal,
        )
    return ComputeError(
        str(error) or type(error).__name__,
        phase="neural_execution",
        route=str(plan.route),
        operations=plan.transaction.operations,
    )


_WorkerFailure = TypeVar("_WorkerFailure", bound=WorkerError)


def _enrich(error: _WorkerFailure, plan: ForwardPlan, phase: str) -> _WorkerFailure:
    if error.phase is None:
        error.phase = phase
    if error.route is None:
        error.route = str(plan.route)
    if not error.operations:
        error.operations = plan.transaction.operations
    return error


def _stage(plan: ForwardPlan) -> ForwardBatch:
    device = torch.device(plan.device)
    rows = _stage_rows(plan.rows, device, plan.staging_slot)
    attention = plan.context.attention
    staged_attention: AttnPlan
    if isinstance(attention, NoAttention):
        staged_attention = attention
    elif isinstance(attention, PagedDecodePlan):
        staged_attention = replace(
            attention,
            block_table=attention.block_table.to(device=device, non_blocking=True),
            cache_seqlens=attention.cache_seqlens.to(device=device, non_blocking=True),
            kv_seqlens=attention.kv_seqlens.to(device=device, non_blocking=True),
            query_lens=attention.query_lens.to(device=device, non_blocking=True),
            decode_page_ids=attention.decode_page_ids.to(device=device, non_blocking=True),
            decode_page_offsets=attention.decode_page_offsets.to(device=device, non_blocking=True),
        )
    elif isinstance(attention, PagedVarlenPlan):
        staged_attention = replace(
            attention,
            block_table=attention.block_table.to(device=device, non_blocking=True),
            cache_seqlens=attention.cache_seqlens.to(device=device, non_blocking=True),
            query_lens=attention.query_lens.to(device=device, non_blocking=True),
            kv_seqlens=attention.kv_seqlens.to(device=device, non_blocking=True),
            cu_seqlens_q=attention.cu_seqlens_q.to(device=device, non_blocking=True),
            cu_seqlens_k=attention.cu_seqlens_k.to(device=device, non_blocking=True),
        )
    else:
        staged_attention = replace(
            attention,
            indexes=attention.indexes.to(device=device, non_blocking=True),
            route_indicators=attention.route_indicators.to(device=device, non_blocking=True),
            text_indices=attention.text_indices.to(device=device, non_blocking=True),
            visible_end=attention.visible_end.to(device=device, non_blocking=True),
            cu_seqlens_q=attention.cu_seqlens_q.to(device=device, non_blocking=True),
            page_table=attention.page_table.to(device=device, non_blocking=True),
            seqused_k=attention.seqused_k.to(device=device, non_blocking=True),
            write_page_ids=attention.write_page_ids.to(device=device, non_blocking=True),
            write_page_offsets=attention.write_page_offsets.to(device=device, non_blocking=True),
            write_token_indices=attention.write_token_indices.to(device=device, non_blocking=True),
        )
    return ForwardBatch(
        route=plan.route,
        rows=rows,
        context=replace(plan.context, attention=staged_attention),
    )


def _stage_rows(
    rows: tuple[ForwardRow, ...],
    device: torch.device,
    slot: TensorStagingSlot | None,
) -> tuple[ForwardRow, ...]:
    if rows and all(isinstance(row, TokenRow) for row in rows):
        token_rows = tuple(row for row in rows if isinstance(row, TokenRow))
        if all(isinstance(row.inputs, TokenIds) for row in token_rows):
            inputs = _pack_to_device(
                tuple(
                    row.inputs.values
                    for row in token_rows
                    if isinstance(row.inputs, TokenIds)
                ),
                device,
                slot=slot,
                name="token_ids",
            )
            positions = _pack_to_device(
                tuple(row.positions for row in token_rows),
                device,
                slot=slot,
                name="token_positions",
            )
            if inputs is not None and positions is not None:
                staged: list[ForwardRow] = []
                input_offset = 0
                position_offset = 0
                for row in token_rows:
                    input_count = int(cast(TokenIds, row.inputs).values.numel())
                    position_count = int(row.positions.numel())
                    staged.append(
                        replace(
                            row,
                            inputs=TokenIds(inputs[input_offset : input_offset + input_count]),
                            positions=positions[
                                position_offset : position_offset + position_count
                            ],
                        )
                    )
                    input_offset += input_count
                    position_offset += position_count
                return tuple(staged)
    return tuple(_stage_row(row, device) for row in rows)


def _pack_to_device(
    values: tuple[torch.Tensor, ...],
    device: torch.device,
    *,
    slot: TensorStagingSlot | None,
    name: str,
) -> torch.Tensor | None:
    if not values:
        return None
    flattened = tuple(value.reshape(-1) for value in values)
    if all(value.device == device for value in flattened):
        packed = packed_tensor_views(flattened)
        if packed is not None:
            return packed
    return pack_integer_tensors(
        flattened,
        device=device,
        slot=slot,
        name=name,
    )


def _mark_staging_submitted(plan: ForwardPlan) -> None:
    slot = plan.staging_slot
    if slot is not None:
        slot.stager.mark_submitted(slot, plan.device)


def _stage_row(row: ForwardRow, device: torch.device) -> ForwardRow:
    def move(value: torch.Tensor) -> torch.Tensor:
        return value.to(device=device, non_blocking=True)

    if isinstance(row, TokenRow):
        inputs = row.inputs
        staged_inputs: TokenIds | TokenEmbeddings | TokenSegments
        if isinstance(inputs, TokenIds):
            staged_inputs = TokenIds(move(inputs.values))
        elif isinstance(inputs, TokenEmbeddings):
            staged_inputs = TokenEmbeddings(move(inputs.values))
        else:
            staged_inputs = TokenSegments(
                tuple(
                    TokenIds(move(segment.values))
                    if isinstance(segment, TokenIds)
                    else TokenEmbeddings(move(segment.values))
                    for segment in inputs.values
                )
            )
        return replace(row, inputs=staged_inputs, positions=move(row.positions))
    if isinstance(row, FlowRow):
        return replace(
            row,
            conditioning=(
                FlowPatches(
                    move(row.conditioning.pixels),
                    move(row.conditioning.grid),
                    move(row.conditioning.noise_scale),
                )
                if isinstance(row.conditioning, FlowPatches)
                else row.conditioning
            ),
            positions=move(row.positions),
            timestep=move(row.timestep),
            latent=move(row.latent),
        )
    if isinstance(row, EncodeRow):
        encode_inputs = row.inputs
        staged_encode = (
            PatchInput(move(encode_inputs.pixels), move(encode_inputs.grid))
            if isinstance(encode_inputs, PatchInput)
            else TowerInput(move(encode_inputs.pixels))
        )
        return replace(row, inputs=staged_encode)
    return replace(row, latent=move(row.latent))


def _validate_tensors(
    batch: ForwardBatch,
    output: ForwardOutput,
    outputs: tuple[OutputSlot, ...],
    device: torch.device,
) -> None:
    for input_row, output_row, declared in zip(
        batch.rows, output.rows, outputs, strict=True
    ):
        expected_dtype = getattr(torch, declared.dtype.removeprefix("torch."), None)
        if not isinstance(expected_dtype, torch.dtype):
            raise ValueError(f"forward output declares unknown dtype {declared.dtype!r}")
        tensor = _output_tensor(output_row)
        if tensor.device != device:
            raise ValueError(
                f"output row {output_row.row_id} is on {tensor.device}, expected {device}"
            )
        if not tensor.is_floating_point():
            raise ValueError("raw neural outputs must use a floating dtype")
        if tensor.dtype is not expected_dtype:
            raise ValueError(
                f"output row {output_row.row_id} uses {tensor.dtype}, expected {expected_dtype}"
            )
        if isinstance(input_row, TokenRow):
            if tensor.ndim < 2:
                raise ValueError("token output must retain token/feature dimensions")
            if isinstance(output_row, TokenOutput):
                expected_hidden = input_row.selection.value == "hidden"
                if expected_hidden != isinstance(output_row.value, TokenHidden):
                    raise ValueError("token output representation does not match selection")
        elif isinstance(input_row, FlowRow):
            if tensor.shape != input_row.latent.shape:
                raise ValueError("flow prediction shape does not match its latent")
        elif isinstance(input_row, EncodeRow):
            if tensor.numel() == 0:
                raise ValueError("encoder output must not be empty")
        elif isinstance(input_row, DecodeRow):
            if tensor.ndim < 2 or tuple(tensor.shape[-2:]) != (
                input_row.image_height,
                input_row.image_width,
            ):
                raise ValueError("decoded tensor does not match declared image geometry")


def _output_tensor(output: TokenOutput | FlowOutput | EncodeOutput | DecodeOutput) -> torch.Tensor:
    if isinstance(output, TokenOutput):
        value = output.value
        return value.value if isinstance(value, (TokenLogits, TokenHidden)) else value.value
    if isinstance(output, FlowOutput):
        return output.prediction
    if isinstance(output, EncodeOutput):
        return output.features
    return output.tensor


__all__ = ["ModelRunner", "RunObservation", "RunPath"]
