"""Execution staging envelopes and completion results around numerical calls."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from uniserve.attention.metadata import AttentionMetadata
from uniserve.model.batch import TextOutput
from uniserve.model.tensors import FlowPatches, TokenSelection
from uniserve.tensors import OutputLayout
from uniserve_worker.protocol.batch import ForwardMode, ForwardStats, PipelineStage

if TYPE_CHECKING:
    from .sampling import SamplerOutput


@dataclass(frozen=True, slots=True)
class InputBatch:
    """One borrowed columnar view over execution-lane input buffers."""

    forward_mode: ForwardMode | PipelineStage
    row_count: int
    attention: AttentionMetadata
    request_pool_indices: torch.Tensor
    binding: int = 0
    cuda_graph_capture: bool = False
    decode_force_finish: torch.Tensor | None = None
    token_row_indices: tuple[int, ...] = ()
    flow_row_indices: tuple[int, ...] = ()
    input_ids: torch.Tensor | None = None
    input_embeddings: torch.Tensor | None = None
    embedding_mask: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    token_selections: tuple[TokenSelection, ...] = ()
    flow_positions: tuple[torch.Tensor, ...] = ()
    flow_timesteps: tuple[torch.Tensor, ...] = ()
    flow_latents: tuple[torch.Tensor, ...] = ()
    flow_conditioning: tuple[FlowPatches | None, ...] = ()
    flow_image_tokens: tuple[int, ...] = ()
    flow_heights: tuple[int, ...] = ()
    flow_widths: tuple[int, ...] = ()
    encode_pixels: tuple[torch.Tensor, ...] = ()
    encode_grids: tuple[torch.Tensor | None, ...] = ()
    encode_grid_shapes: tuple[tuple[int, int] | None, ...] = ()
    decode_latents: tuple[torch.Tensor, ...] = ()
    decode_heights: tuple[int, ...] = ()
    decode_widths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate borrowed batch columns against attention mode and row geometry."""

        if self.row_count < 1:
            raise ValueError("forward batch must contain at least one row")
        if int(self.request_pool_indices.numel()) != self.row_count:
            raise ValueError("forward request indices do not align with rows")
        if (
            int(self.attention.prefix_lens.numel()) != self.row_count
            or int(self.attention.query_lens.numel()) != self.row_count
        ):
            raise ValueError("forward KV lengths do not align with rows")
        if self.decode_force_finish is not None and (
            int(self.decode_force_finish.numel()) != self.row_count
            or self.decode_force_finish.dtype is not torch.bool
        ):
            raise ValueError("forward decode finish column does not align with rows")
        row_indices = (*self.token_row_indices, *self.flow_row_indices)
        if row_indices and (
            len(set(row_indices)) != len(row_indices)
            or min(row_indices) < 0
            or max(row_indices) >= self.row_count
        ):
            raise ValueError("forward row indexes are invalid")
        if len(self.token_row_indices) != len(self.token_selections):
            raise ValueError("forward token columns are not aligned")
        if any(
            len(lengths) != self.row_count
            for lengths in (
                self.attention.prefix_lens_cpu,
                self.attention.seq_lens_cpu,
                self.attention.query_lens_cpu,
            )
        ):
            raise ValueError("forward host KV lengths do not align with rows")
        if any(
            prefix < 0 or query < 0 or total != prefix + query
            for prefix, query, total in zip(
                self.attention.prefix_lens_cpu,
                self.attention.query_lens_cpu,
                self.attention.seq_lens_cpu,
                strict=True,
            )
        ):
            raise ValueError("forward sequence lengths must equal cached prefix plus query")
        flow_count = len(self.flow_row_indices)
        if any(
            len(values) != flow_count
            for values in (
                self.flow_positions,
                self.flow_timesteps,
                self.flow_latents,
                self.flow_conditioning,
                self.flow_image_tokens,
                self.flow_heights,
                self.flow_widths,
            )
        ):
            raise ValueError("forward flow columns are not aligned")
        encode_count = len(self.encode_pixels)
        if any(
            len(values) != encode_count for values in (self.encode_grids, self.encode_grid_shapes)
        ):
            raise ValueError("forward encoder columns are not aligned")
        decode_count = len(self.decode_latents)
        if any(len(values) != decode_count for values in (self.decode_heights, self.decode_widths)):
            raise ValueError("forward decoder columns are not aligned")


@dataclass(frozen=True, slots=True)
class ExecutionOutput(TextOutput):
    """Row-aligned numerical results with execution observations and reader fences."""

    request_pool_indices: torch.Tensor | None = None
    output_event: torch.cuda.Event | None = None
    stats: ForwardStats | None = None
    greedy: SamplerOutput | None = None

    layouts: tuple[OutputLayout | None, ...] = ()

    def __post_init__(self) -> None:
        TextOutput.__post_init__(self)
        if not self.layouts:
            object.__setattr__(self, "layouts", (None,) * len(self.values))
        if len(self.layouts) != len(self.values):
            raise ValueError("output geometry must align with execution rows")

    def materialize(self) -> ExecutionOutput:
        """Gather global vocabulary rows, preserving their caller-visible shapes."""

        if self.output_event is not None:
            if not self.values:
                raise RuntimeError("forward output has a fence without a producer tensor")
            torch.cuda.current_stream(self.values[0].device).wait_event(self.output_event)
        if not any(self.vocabularies):
            return self
        output = TextOutput(self.values, self.vocabularies).materialize()
        return replace(self, values=output.values, vocabularies=output.vocabularies)

    def clone(self) -> ExecutionOutput:
        """Own detached copies that survive reuse of the producer's storage.

        Outputs on one device with one dtype share a packed allocation. Shapes
        and logical tensor values are preserved independently of source strides;
        storage remains live for as long as any returned tensor is retained.
        """

        if self.output_event is not None:
            if not self.values:
                raise RuntimeError("forward output has a fence without a producer tensor")
            torch.cuda.current_stream(self.values[0].device).wait_event(self.output_event)
        groups: dict[tuple[torch.device, torch.dtype], list[int]] = defaultdict(list)
        for index, value in enumerate(self.values):
            groups[(value.device, value.dtype)].append(index)
        copied = list(self.values)
        for indexes in groups.values():
            if len(indexes) == 1:
                index = indexes[0]
                copied[index] = self.values[index].detach().clone()
                continue
            sources = [self.values[index] for index in indexes]
            packed = torch.cat(tuple(value.detach().reshape(-1) for value in sources))
            views = packed.split(tuple(value.numel() for value in sources))
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
