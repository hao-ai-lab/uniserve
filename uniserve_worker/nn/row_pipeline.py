"""Row dependencies across projections, attention exchanges and decoder layers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch

from ..execution.forward_batch import AttentionMetadata
from .attention import RadixAttention
from .attention_storage import attention_exchange_storage
from .linear import GatheredLinear, LinearBase
from .parallel_attention import AttentionHeadRows, AttentionRowExchange, HeadRowPreparation
from .parallel_sequence import SequencePartition

RowConsumer = Callable[[slice, torch.Tensor], None]
_State = TypeVar("_State")
_Rows = TypeVar("_Rows")
_Input = TypeVar("_Input", contravariant=True)


class RowPreparation(Protocol[_Input]):
    """Accept completed rows while retaining the next operation's input state."""

    def append(self, interval: slice, rows: _Input) -> None: ...


RowTensors = tuple[torch.Tensor, ...]


class RowTensorSegments:
    """Own numerical row intervals and assemble storage only for a consumer.

    A following layer can consume prepared QKV directly, so its predecessor's
    output need not be copied into an otherwise unread complete tensor. Row
    tensors stay alive here until that dependency is consumed or discarded.
    """

    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.position = 0
        self.segments: list[RowTensors] = []

    @classmethod
    def complete(cls, values: RowTensors) -> RowTensorSegments:
        """Retain a complete interval without changing its numerical storage."""

        result = cls(values[0].shape[0])
        result.append(slice(0, result.rows), values)
        return result

    def append(self, interval: slice, values: RowTensors) -> None:
        """Retain one ordered interval with a common leading numerical row axis."""

        if interval.start != self.position or not self.position <= interval.stop <= self.rows:
            raise ValueError("row tensors require ordered, complete intervals")
        if not values or any(value.shape[0] != interval.stop - interval.start for value in values):
            raise ValueError("row tensors must preserve their declared interval")
        self.segments.append(values)
        self.position = interval.stop

    def materialize(self) -> RowTensors:
        """Return complete numerical tensors, joining each row axis at most once."""

        if self.position != self.rows or not self.segments:
            raise ValueError("row tensors did not cover their declared extent")
        if len(self.segments) > 1:
            self.segments = [
                tuple(torch.cat(values, dim=0) for values in zip(*self.segments, strict=True))
            ]
        return self.segments[0]


class PreparedRowTensors:
    """Project arriving rows while retaining the next layer's numerical state."""

    def __init__(
        self,
        rows: int,
        project: Callable[[slice, RowTensors], RowTensors],
        attention: RadixAttention,
        partition: SequencePartition | None,
    ) -> None:
        self.project = project
        self.output: RowTensorSegments | None = RowTensorSegments(rows)
        self.attention = attention
        self.partition = partition
        self.heads: HeadRowPreparation | None = None

    def append(self, interval: slice, rows: RowTensorSegments) -> None:
        if self.output is None:
            raise RuntimeError("prepared rows were already consumed")
        projected = self.project(interval, rows.materialize())
        if (
            interval.start == 0
            and interval.stop < self.output.rows
            and self.partition is not None
            and self.partition.group.world_size > 1
        ):
            self.heads = HeadRowPreparation(
                self.attention.exchange,
                self.partition,
                attention_exchange_storage(self.partition.group),
            )
        if self.heads is not None:
            self.heads.append(interval, projected[:3])
            projected = projected[3:]
        self.output.append(interval, projected)

    def finish(self) -> tuple[RowTensors | AttentionHeadRows, RowTensorSegments]:
        output, self.output = self.output, None
        if output is None:
            raise RuntimeError("prepared rows were already consumed")
        heads, self.heads = self.heads, None
        if heads is not None:
            return heads.finish(), output
        projected = output.materialize()
        return projected[:3], RowTensorSegments.complete(projected[3:])


def independent_linear_rows(*modules: torch.nn.Module) -> bool:
    """Whether linear activation scales permit splitting the leading row axis.

    Callers remain responsible for dependencies in their non-linear operations.
    Tensor-wide scales must see the complete domain before any row is consumed.
    """

    return all(
        child.quant_method.input_scale_domain != "tensor"
        for module in modules
        for child in module.modules()
        if isinstance(child, LinearBase)
    )


@dataclass
class ProjectedRows(Generic[_State]):
    """Own a streamed projection and the numerical state prepared from its rows."""

    projection: GatheredLinear
    state: _State | None
    transform: Callable[[slice, torch.Tensor], torch.Tensor] | None = None

    def append(self, interval: slice, rows: torch.Tensor) -> None:
        """Prepare and publish the next local row interval."""

        values = rows if self.transform is None else self.transform(interval, rows)
        self.projection.append(interval.start, values)

    def finish(self) -> tuple[torch.Tensor, _State]:
        """Consume outstanding transfers and hand numerical state to attention."""

        if self.state is None:
            raise RuntimeError("projected rows were already consumed")
        values = self.projection.finish()
        state, self.state = self.state, None
        self.transform = None
        return values, state


@dataclass(frozen=True)
class RowStage(Generic[_Rows]):
    """Bind one layer's mathematics and its legal row dependency boundaries.

    ``operation`` preserves row extent and accepts ``prepared_projection`` and
    ``row_consumer``. The latter receives each completed output interval once.
    Row-independent output permits the following stage to prepare its input
    before all prior rows finish; tensor-wide numerical domains wait for all rows.
    """

    operation: Callable[..., _Rows]
    row_independent: bool
    prepare_input: Callable[[_Rows], RowPreparation[_Rows]] | None = None


def run_row_pipeline(hidden: _Rows, stages: Sequence[RowStage[_Rows]]) -> _Rows:
    """Chain layer dependencies while retaining projection and scratch ownership."""

    prepared = None
    for index, stage in enumerate(stages):
        following = stages[index + 1] if index + 1 < len(stages) else None
        next_projection = (
            following.prepare_input(hidden)
            if stage.row_independent
            and following is not None
            and following.prepare_input is not None
            else None
        )
        hidden = stage.operation(
            hidden,
            prepared_projection=prepared,
            row_consumer=None if next_projection is None else next_projection.append,
        )
        prepared = next_projection
    return hidden


def packed_row_stage(
    project: Callable[[slice, RowTensors], RowTensors],
    attention: RadixAttention,
    finish: Callable[[slice, torch.Tensor, RowTensors], RowTensors],
    *,
    context: AttentionMetadata,
    partition: SequencePartition | None,
    causal: bool,
    scale: float,
    independent_input: bool,
    independent_output: bool,
) -> RowStage[RowTensorSegments]:
    """Bind packed decoder mathematics to the shared cross-layer row executor.

    Projected tensors contain Q/K/V followed by any row-local numerical state.
    The attention operation owns global visibility and cache writes. Completed
    head exchanges feed the output equations and then the next layer's input
    projection; complete-domain equations consume all exchanged rows together.
    """

    def prepare(values: RowTensorSegments) -> PreparedRowTensors:
        return PreparedRowTensors(values.rows, project, attention, partition)

    def operation(
        values: RowTensorSegments,
        *,
        prepared_projection: PreparedRowTensors | None,
        row_consumer: Callable[[slice, RowTensorSegments], None] | None,
    ) -> RowTensorSegments:
        count = values.rows
        full = slice(0, count)
        if prepared_projection is None:
            projected = project(full, values.materialize())
            qkv: RowTensors | AttentionHeadRows = projected[:3]
            state = RowTensorSegments.complete(projected[3:])
            del projected
        else:
            qkv, state = prepared_projection.finish()
        if isinstance(qkv, AttentionHeadRows):
            attended = attention.forward_heads(
                qkv,
                context,
                causal=causal,
                scale=scale,
                consume_row_intervals=independent_output,
            )
        else:
            attended = attention(
                *qkv,
                context,
                causal=causal,
                scale=scale,
                partition=partition,
                consume_row_intervals=independent_output,
            )
        del qkv
        state_rows = state.materialize()

        def consume(interval: slice, rows: torch.Tensor) -> RowTensorSegments:
            result = RowTensorSegments.complete(
                finish(interval, rows, tuple(value[interval] for value in state_rows))
            )
            if row_consumer is not None:
                row_consumer(interval, result)
            return result

        if not isinstance(attended, AttentionRowExchange):
            return consume(full, attended)
        if not independent_output:
            return consume(full, attended.materialize())
        output = RowTensorSegments(count)
        intervals = attended.chunks()
        del attended
        for interval, rows in intervals:
            output.append(interval, consume(interval, rows).materialize())
        return output

    return RowStage(operation, independent_output, prepare if independent_input else None)


def map_attention_rows(
    attention: torch.Tensor | AttentionRowExchange,
    output_like: torch.Tensor,
    workspace: torch.Tensor,
    transform: Callable[[slice, torch.Tensor], torch.Tensor],
    *,
    row_axis: int,
    row_independent: bool,
    consumer: RowConsumer | None,
) -> torch.Tensor:
    """Consume attention dependencies through row-local or complete-domain math."""

    if isinstance(attention, AttentionRowExchange) and row_independent:
        output = torch.empty_like(output_like)
        intervals = attention.chunks(workspace)
        # Only the iterator retains communication dependencies. Releasing the
        # producer here allows its input allocations to retire before consumers.
        del attention
        for interval, rows in intervals:
            target = output.narrow(row_axis, interval.start, interval.stop - interval.start)
            target.copy_(transform(interval, rows))
            if consumer is not None:
                consumer(interval, target)
        return output
    rows = attention.materialize() if isinstance(attention, AttentionRowExchange) else attention
    interval = slice(0, output_like.shape[row_axis])
    output = transform(interval, rows)
    if consumer is not None:
        consumer(interval, output)
    return output
