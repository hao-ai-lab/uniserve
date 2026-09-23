"""Execution staging envelopes and completion results around numerical calls."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from uniserve.model.logits import Logits, VocabShard
from uniserve.tensors import OutputLayout, adjacent_view
from uniserve_worker.model_executor.input_batch import InputBatch
from uniserve_worker.protocol.output import ForwardStats

if TYPE_CHECKING:
    from uniserve_worker.sampling.result import SamplerOutput


@dataclass(frozen=True, slots=True)
class ExecutionOutput:
    """Row-aligned numerical results with execution observations and reader.

    fences.
    """

    values: tuple[torch.Tensor, ...]
    vocabularies: tuple[VocabShard | None, ...] = ()
    request_pool_indices: torch.Tensor | None = None
    output_event: torch.cuda.Event | None = None
    stats: ForwardStats | None = None
    greedy: SamplerOutput | None = None

    layouts: tuple[OutputLayout | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.vocabularies:
            object.__setattr__(self, "vocabularies", (None,) * len(self.values))
        if len(self.vocabularies) != len(self.values):
            raise ValueError("vocabulary metadata must align with output rows")

        for value, vocab in zip(self.values, self.vocabularies, strict=True):
            if vocab is not None and (
                value.ndim != 2
                or value.shape[-1]
                != vocab.local_slice.stop - vocab.local_slice.start
            ):
                raise ValueError(
                    "vocabulary output rows disagree with their shard"
                )

        if not self.layouts:
            object.__setattr__(self, "layouts", (None,) * len(self.values))
        if len(self.layouts) != len(self.values):
            raise ValueError("output layouts must align with execution rows")

    def materialize(self) -> ExecutionOutput:
        """Gather global vocabulary rows.

        preserving their caller-visible shapes.
        """
        if self.output_event is not None:
            if not self.values:
                raise RuntimeError(
                    "forward output has a fence without a producer tensor"
                )
            torch.cuda.current_stream(self.values[0].device).wait_event(
                self.output_event
            )

        if not any(self.vocabularies):
            return self

        # Bucket rows that share one vocabulary shard so they gather together.
        groups: dict[
            tuple[int, int, int, int, int, torch.device, torch.dtype],
            tuple[VocabShard, list[int]],
        ] = {}
        for index, vocab in enumerate(self.vocabularies):
            if vocab is not None:
                key = (
                    vocab.size,
                    vocab.padded_size,
                    vocab.local_slice.start,
                    vocab.local_slice.stop,
                    id(vocab.group),
                    self.values[index].device,
                    self.values[index].dtype,
                )
                groups.setdefault(key, (vocab, []))[1].append(index)

        values = list(self.values)
        for vocab, indexes in groups.values():
            sources = tuple(self.values[index] for index in indexes)
            # Adjacent request rows already share backing; gather them together
            # without adding another allocation or collective per request.
            view = adjacent_view(sources)
            rows = (
                torch.cat(sources, dim=0)
                if view is None
                else view.reshape(
                    -1, vocab.local_slice.stop - vocab.local_slice.start
                )
            )
            gathered = Logits(rows, vocab).gather()
            for index, value in zip(
                indexes,
                gathered.split(tuple(source.shape[0] for source in sources)),
                strict=True,
            ):
                values[index] = value
        return replace(self, values=tuple(values), vocabularies=())

    def clone(self) -> ExecutionOutput:
        """Own detached copies that survive reuse of the producer's storage.

        Outputs on one device with one dtype share a contiguous allocation.
        Shapes and logical tensor values are preserved independently of source
        strides; storage remains live for as long as any returned tensor is
        retained.
        """
        if self.output_event is not None:
            if not self.values:
                raise RuntimeError(
                    "forward output has a fence without a producer tensor"
                )
            torch.cuda.current_stream(self.values[0].device).wait_event(
                self.output_event
            )

        # Tensors sharing a device and dtype copy through one flat allocation.
        groups: dict[tuple[torch.device, torch.dtype], list[int]] = defaultdict(
            list
        )
        for index, value in enumerate(self.values):
            groups[(value.device, value.dtype)].append(index)

        copied = list(self.values)
        for indexes in groups.values():
            if len(indexes) == 1:
                index = indexes[0]
                copied[index] = self.values[index].detach().clone()
                continue
            sources = [self.values[index] for index in indexes]
            storage = torch.cat(
                tuple(value.detach().reshape(-1) for value in sources)
            )
            views = storage.split(tuple(value.numel() for value in sources))
            for index, view in zip(indexes, views, strict=True):
                copied[index] = view.reshape(self.values[index].shape)

        greedy = self.greedy
        if greedy is not None:
            greedy = greedy.clone()
        return replace(
            self,
            values=tuple(copied),
            output_event=None,
            greedy=greedy,
            request_pool_indices=None
            if self.request_pool_indices is None
            else self.request_pool_indices.clone(),
        )

    def validate_for(self, batch: InputBatch) -> None:
        """Require one tensor result for every row in the originating batch."""
        if len(self.values) != batch.row_count:
            raise ValueError("model output count does not match forward rows")
        if any(not isinstance(value, torch.Tensor) for value in self.values):
            raise TypeError("model output values must be tensors")
