"""Numerical calls independent of request progress and execution resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, TypeVar

import torch

from uniserve.attention.metadata import AttentionMetadata
from uniserve.model.media import DecodeWindow
from uniserve.model.tensors import FlowPatches, TokenSelection, VocabularyPartition
from uniserve.tensors import OutputLayout

Size = TypeVar("Size")


@dataclass(frozen=True, slots=True)
class TextBatch:
    """Flattened token sequences and one output selection per logical sequence.

    Attention query lengths describe the live token spans. Input storage may
    contain trailing padding used by a captured numerical call. Zero-length
    query rows represent inactive padding and still retain their output row.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    attention: AttentionMetadata
    selections: tuple[TokenSelection, ...]
    inputs_embeds: torch.Tensor | None = None
    embedding_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        lengths = self.attention.query_lens_cpu
        if (
            not lengths
            or len(lengths) != len(self.selections)
            or any(n < 0 for n in lengths)
            or sum(lengths) < 1
        ):
            raise ValueError("text selections must align with nonnegative query spans")
        tokens = sum(lengths)
        if self.input_ids.numel() < tokens or self.positions.shape[-1] < tokens:
            raise ValueError("text inputs do not cover their query spans")
        if self.inputs_embeds is not None:
            if (
                self.inputs_embeds.ndim != 2
                or self.inputs_embeds.shape[0] != self.input_ids.numel()
            ):
                raise ValueError("input embeddings must align with token storage")
            if self.embedding_mask is None or self.embedding_mask.numel() != self.input_ids.numel():
                raise ValueError("input embeddings require a mask aligned with token storage")

    @property
    def row_count(self) -> int:
        return len(self.selections)


@dataclass(frozen=True, slots=True)
class DiffusionBatch(Generic[Size]):
    """Named modalities in logical row and CFG-branch order.

    Positions include the model's mathematical framing tokens. Latents contain
    only numerical state; timesteps identify the supplied network evaluation.
    Mapping order fixes modality order. Rows preserve the caller's ordered CFG
    branches; sizes and sequence lengths describe each branch independently.
    Latent preparation may omit network-only timestep and attention metadata.
    """

    latents: Mapping[str, tuple[torch.Tensor, ...]]
    sizes: tuple[Size, ...]
    timesteps: Mapping[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    conditioning: Mapping[str, tuple[torch.Tensor | FlowPatches | None, ...]] = field(
        default_factory=dict
    )
    positions: tuple[torch.Tensor, ...] = ()
    sequence_lengths: tuple[int, ...] = ()
    attention: AttentionMetadata | None = None
    ladder_index: int | None = None

    def __post_init__(self) -> None:
        rows = len(self.sizes)
        if not rows or not self.latents:
            raise ValueError("diffusion calls require named latent modalities and row shapes")
        for name in ("latents", "timesteps", "conditioning"):
            columns = getattr(self, name)
            if any(not key or len(values) != rows for key, values in columns.items()):
                raise ValueError("diffusion modalities must align with numerical rows")
            object.__setattr__(self, name, MappingProxyType(dict(columns)))
        if self.timesteps and tuple(self.timesteps) != tuple(self.latents):
            raise ValueError("diffusion timesteps must follow latent modality order")
        if any(
            column and len(column) != rows for column in (self.positions, self.sequence_lengths)
        ):
            raise ValueError("diffusion numerical columns must align with latent rows")
        if any(count < 1 for count in self.sequence_lengths):
            raise ValueError("diffusion sequence lengths must be positive")
        if self.attention is not None and len(self.attention.query_lens_cpu) != rows:
            raise ValueError("diffusion attention must align with numerical rows")
        if self.ladder_index is not None and self.ladder_index < 0:
            raise ValueError("diffusion ladder index cannot be negative")

    @property
    def row_count(self) -> int:
        return len(self.sizes)


@dataclass(frozen=True, slots=True)
class EncodeBatch:
    """Preprocessed numerical inputs with their logical grids and valid lengths."""

    values: tuple[torch.Tensor, ...]
    grids: tuple[torch.Tensor | None, ...] = ()
    grid_shapes: tuple[tuple[int, int] | None, ...] = ()
    lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.values or any(
            column and len(column) != len(self.values)
            for column in (
                self.grids,
                self.grid_shapes,
                self.lengths,
            )
        ):
            raise ValueError("encoder geometry must align with its numerical inputs")


@dataclass(frozen=True, slots=True)
class DecodeBatch(Generic[Size]):
    """Latents, entry-specific numerical sizes, and logical decode windows.

    Image rows use ImageSize, video rows use VideoSize, and audio rows use
    their PCM sample count. Each window corresponds to the input at the same
    row. Image and audio reconstruction omit windows; video supplies them.
    The caller resolves physical participation before constructing this batch.
    """

    latents: tuple[torch.Tensor, ...]
    sizes: tuple[Size, ...]
    windows: tuple[DecodeWindow, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.latents
            or len(self.latents) != len(self.sizes)
            or (self.windows and len(self.windows) != len(self.latents))
        ):
            raise ValueError("decoder geometry must align with latent inputs")


@dataclass(frozen=True, slots=True)
class TextOutput:
    """Hidden or vocabulary rows in the input sequence order, before sampling."""

    values: tuple[torch.Tensor, ...]
    vocabularies: tuple[VocabularyPartition | None, ...] = ()

    def __post_init__(self) -> None:
        if not self.vocabularies:
            object.__setattr__(self, "vocabularies", (None,) * len(self.values))
        if len(self.vocabularies) != len(self.values):
            raise ValueError("vocabulary metadata must align with output rows")
        for value, partition in zip(self.values, self.vocabularies, strict=True):
            if partition is not None and (value.ndim != 2 or value.shape[-1] != partition.width):
                raise ValueError("vocabulary output rows disagree with their partition")

    def materialize(self) -> TextOutput:
        """Collectively gather vocabulary shards in logical token order."""

        from uniserve.model.tensors import packed_tensor_views
        from uniserve.nn.logits import gather_vocabulary

        groups: dict[VocabularyPartition, list[int]] = {}
        for index, partition in enumerate(self.vocabularies):
            if partition is not None:
                groups.setdefault(partition, []).append(index)
        values = list(self.values)
        for partition, indexes in groups.items():
            sources = tuple(self.values[index] for index in indexes)
            rows = packed_tensor_views(sources)
            if rows is None:
                rows = torch.cat(sources, dim=0)
            gathered = gather_vocabulary(rows.reshape(-1, partition.width), partition)
            results = gathered.split(tuple(value.shape[0] for value in sources))
            for index, value in zip(indexes, results, strict=True):
                values[index] = value
        return TextOutput(tuple(values))


@dataclass(frozen=True, slots=True)
class TensorOutput:
    """Named numerical outputs in input-row order.

    A non-output pipeline stage returns None at the corresponding position;
    execution failures must raise instead of returning a missing value.
    """

    values: Mapping[str, tuple[torch.Tensor | None, ...]]
    layouts: Mapping[str, tuple[OutputLayout | None, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        if not self.values or len({len(rows) for rows in self.values.values()}) != 1:
            raise ValueError("named numerical outputs must have aligned rows")
        if self.layouts and (
            self.layouts.keys() != self.values.keys()
            or any(len(self.layouts[name]) != len(rows) for name, rows in self.values.items())
        ):
            raise ValueError("output geometry must align with named numerical rows")
        layouts = self.layouts or {name: (None,) * len(rows) for name, rows in self.values.items()}
        object.__setattr__(self, "layouts", MappingProxyType(dict(layouts)))
