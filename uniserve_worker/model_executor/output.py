"""Numerical call inputs and completion results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from uniserve.model.logits import Logits, VocabShard
from uniserve.tensors import OutputLayout, adjacent_view
from uniserve_worker._uniserve_ipc import CUDAEvent
from uniserve_worker.model_executor.input_batch import InputBatch
from uniserve_worker.protocol.output import ForwardStats

if TYPE_CHECKING:
    from uniserve_worker.sampling.result import SamplerOutput


@dataclass(frozen=True, slots=True)
class ExecutionOutput:
    """Row-aligned numerical results with execution observations and fences.

    For a staged batch ``values`` holds one tensor per row, which
    ``validate_for`` checks; for a standalone call it holds the module
    call's tensors. A value with a ``VocabShard`` in ``vocabularies`` holds
    logits over this rank's vocabulary columns; ``materialize`` gathers them
    into the full vocabulary. ``layouts`` optionally describes each value as
    an ``OutputLayout``. For a staged batch, ``ModelExecutor`` sets
    ``request_pool_indices`` and ``output_event``, which is recorded on the
    lane stream after the forward and is None without a lane stream.
    ``greedy`` is the greedy decode a replayed text graph computed. It is
    None whenever ``graph_inputs.greedy_decode`` declines the batch, after
    eager execution, and for a batch staged without a force-finish column.
    """

    values: tuple[torch.Tensor, ...]
    vocabularies: tuple[VocabShard | None, ...] = ()
    request_pool_indices: torch.Tensor | None = None
    output_event: CUDAEvent | None = None
    stats: ForwardStats | None = None
    greedy: SamplerOutput | None = None

    layouts: tuple[OutputLayout | None, ...] = ()

    @classmethod
    def combine(cls, outputs):
        """Concatenate completed microbatch rows on their joined stream.

        Graph-greedy completion consists of four row-length sections, so
        concatenate each section independently. These results precede the
        general sampler and carry no speculative or logprob columns.
        """
        outputs = tuple(outputs)
        if len(outputs) == 1:
            return outputs[0]
        greedy = None
        if outputs and all(output.greedy is not None for output in outputs):
            parts = [output.greedy for output in outputs]
            greedy = replace(
                parts[0],
                **{
                    name: torch.cat([getattr(part, name) for part in parts])
                    for name in (
                        "tokens",
                        "valid",
                        "active",
                        "finish",
                        "continuation",
                        "tagged_tokens",
                    )
                },
                completion=torch.cat(
                    [part.completion.reshape(4, -1) for part in parts], dim=1
                ).reshape(-1),
            )
        return cls(
            values=tuple(
                value for output in outputs for value in output.values
            ),
            vocabularies=tuple(
                value for output in outputs for value in output.vocabularies
            ),
            layouts=tuple(
                value for output in outputs for value in output.layouts
            ),
            greedy=greedy,
            stats=ForwardStats.combine(
                [output.stats for output in outputs if output.stats is not None]
            ),
        )

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
        """Gather global vocabulary rows, preserving their row counts.

        The current stream first waits for ``output_event``. Each gather is
        an all-gather collective over the shard's tensor group, so every rank
        of that group must make the matching call. The result has no
        vocabulary metadata; it is ``self`` when no row is sharded.
        """
        if self.output_event is not None:
            if not self.values:
                raise RuntimeError(
                    "forward output has a fence without a producer tensor"
                )
            self.output_event.wait(
                torch.cuda.current_stream(self.values[0].device)
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

        The current stream first waits for ``output_event``, and the copy
        carries no event. Outputs on one device with one dtype share a
        contiguous allocation. Shapes and logical tensor values are preserved
        independently of source strides; storage remains live for as long as
        any returned tensor is retained. ``greedy`` and
        ``request_pool_indices`` are cloned as well.
        """
        if self.output_event is not None:
            if not self.values:
                raise RuntimeError(
                    "forward output has a fence without a producer tensor"
                )
            self.output_event.wait(
                torch.cuda.current_stream(self.values[0].device)
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
