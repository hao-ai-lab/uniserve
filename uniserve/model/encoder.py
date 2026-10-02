"""Shared homogeneous feature batching with explicit numerical boundaries."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import prod
from typing import Generic, TypeVar

import torch
from torch import nn

from uniserve.media import image
from uniserve.nn.attention import SequenceLengths, VarlenInput
from uniserve.tensors import OutputLayout

from .inputs import EmbeddingReplacement, VisionInput
from .transformer import TransformerDecoder

InputT = TypeVar("InputT")


class Encoder(nn.Module, Generic[InputT]):
    """Stack equal-shaped feature samples and restore their original order.

    The network preserves its leading sample dimension. It must apply token
    calls independently within each sample, as ordinary batched modules do.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def encode(self, inputs: InputT) -> tuple[torch.Tensor, ...] | None:
        """Encode each sample, or return None on a non-final pipeline stage.

        This default batches indexable tensor samples; encoders of structured
        inputs override it.
        """
        if not isinstance(inputs, (Sequence, torch.Tensor)):
            raise TypeError("the default encoder batches indexable samples")
        groups: defaultdict[
            tuple[torch.Size, torch.dtype, torch.device], list[int]
        ] = defaultdict(list)
        for index, value in enumerate(inputs):
            groups[(value.shape, value.dtype, value.device)].append(index)

        result: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            values = self.network(
                torch.stack(tuple(inputs[index] for index in indices))
            )
            if values.shape[0] != len(indices):
                raise ValueError(
                    "encoder network must preserve the sample dimension"
                )
            for index, value in zip(indices, values.unbind(), strict=True):
                result[index] = value
        return tuple(result[index] for index in range(len(inputs)))


class TextConditioner(Encoder[tuple[torch.Tensor, ...]]):
    """Refine the feature rows of text samples, padded or exact.

    ``encode`` takes ``[rows, width]`` samples. Without ``lengths`` every row
    is text, and samples of equal shape are refined together. With
    ``lengths``, an int32 device tensor holding each sample's text row count,
    every sample has the same shape and holds its text in its leading rows.
    Rows past a sample's length are padding: they may hold any finite
    values, and the network keeps every text row's output independent of
    them, as masking them out of attention does. Padding every text length
    to a shared row count lets one prepared or captured call serve all of
    them; the outputs of padding rows are unspecified. The network receives
    stacked samples ``[samples, rows, width]`` and their lengths or None.
    """

    def encode(
        self,
        inputs: tuple[torch.Tensor, ...],
        *,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        if any(value.ndim != 2 for value in inputs):
            raise ValueError("text conditioning samples are [rows, width]")
        if lengths is None:
            groups: defaultdict[torch.Size, list[int]] = defaultdict(list)
            for index, value in enumerate(inputs):
                groups[value.shape].append(index)
        else:
            if (
                lengths.shape != (len(inputs),)
                or lengths.dtype != torch.int32
                or any(value.shape != inputs[0].shape for value in inputs)
            ):
                raise ValueError(
                    "padded text conditioning requires samples of one shape "
                    "and one int32 length per sample"
                )
            groups = defaultdict(
                list, {inputs[0].shape: list(range(len(inputs)))}
            )

        result: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            values = self.network(
                torch.stack(tuple(inputs[index] for index in indices)),
                lengths,
            )
            if values.shape[0] != len(indices):
                raise ValueError(
                    "encoder network must preserve the sample dimension"
                )
            for index, value in zip(indices, values.unbind(), strict=True):
                result[index] = value
        return tuple(result[index] for index in range(len(inputs)))


class PatchEncoder(Encoder[VisionInput]):
    """Pack image samples and split their spatially downsampled features.

    downsample counts input patches per output feature on each spatial axis.
    Patch serialization for an architecture is defined by its network;
    complete CHW images are stacked without altering their pixel order.
    A packed sample with a ``(time, height, width)`` grid keeps its time
    axis: each output feature merges patches of one time step, so the sample
    yields ``time * height * width / downsample**2`` features.
    """

    def __init__(
        self,
        network: nn.Module,
        connector: nn.Module,
        *,
        patch_size: int,
        downsample: int,
        output_size: int,
        output_dtype: torch.dtype,
    ):
        super().__init__(network)
        if any(
            type(value) is not int or value < 1
            for value in (patch_size, downsample, output_size)
        ):
            raise ValueError(
                "patch encoding requires positive spatial and feature "
                "dimensions"
            )
        self.connector, self.patch_size, self.downsample = (
            connector,
            patch_size,
            downsample,
        )
        self._output_size, self._output_dtype = output_size, output_dtype

    def encode(self, inputs: VisionInput) -> tuple[torch.Tensor, ...]:
        groups = defaultdict(list)
        for index, value in enumerate(inputs.images):
            # Packed samples share one network call when their patch width
            # and grid rank agree, so their grids concatenate.
            key = (
                value.ndim,
                value.shape[1:] if value.ndim == 2 else value.shape,
                len(inputs.grid_shapes[index] or ()),
                value.dtype,
                value.device,
            )
            groups[key].append(index)

        result: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            pixels, grids, shapes = [], [], []
            for index in indices:
                value, grid, shape = (
                    inputs.images[index],
                    inputs.grids[index],
                    inputs.grid_shapes[index],
                )
                if value.ndim == 3:
                    if any(axis % self.patch_size for axis in value.shape[-2:]):
                        raise ValueError(
                            "image pixels must contain complete input patches"
                        )
                    shape = (
                        value.shape[-2] // self.patch_size,
                        value.shape[-1] // self.patch_size,
                    )
                    grid = value.new_empty((1, 2), dtype=torch.int64)
                    grid[:, 0].fill_(shape[0])
                    grid[:, 1].fill_(shape[1])
                elif value.ndim != 2 or grid is None or shape is None:
                    raise ValueError(
                        "packed vision samples require their grid and host "
                        "shape"
                    )
                elif value.shape[0] != prod(shape):
                    raise ValueError(
                        "vision grid dimensions must cover their patch rows"
                    )
                if (
                    len(shape) not in (2, 3)
                    or grid.shape != (1, len(shape))
                    or min(shape) < 1
                    or any(axis % self.downsample for axis in shape[-2:])
                ):
                    raise ValueError(
                        "vision grids must align with spatial downsampling"
                    )
                pixels.append(value)
                grids.append(grid)
                shapes.append(shape)

            values = (
                torch.stack(pixels)
                if pixels[0].ndim == 3
                else torch.cat(pixels)
            )
            features = self.network(values, torch.cat(grids), tuple(shapes))
            features = self.connector(features).to(self._output_dtype)

            counts = tuple(
                prod(shape) // self.downsample**2 for shape in shapes
            )
            if features.shape != (sum(counts), self._output_size):
                raise ValueError(
                    "vision features must cover the declared spatial output "
                    "grids"
                )
            for index, features in zip(
                indices, features.split(counts), strict=True
            ):
                result[index] = features
        return tuple(result[index] for index in range(inputs.batch_size))

    def pack_pixels(
        self, frames: torch.Tensor, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        """Return one image's or video's packed patch rows for ``encode``.

        ``frames`` holds ``[frames, height, width, 3]`` uint8 RGB on the
        host, an image being one frame; ``grid`` is the ``(time, height,
        width)`` patch grid they are resized to. The result is the sample
        ``encode`` takes with ``grid`` as its host shape. An encoder whose
        network reads packed patch rows defines this conversion; it reads no
        parameter, so a host rank holding the module's description runs it.
        """
        raise NotImplementedError

    def pixels_layout(self, num_tokens: int) -> OutputLayout:
        """Describe the packed patch rows of ``num_tokens`` merged tokens.

        These are the rows ``pack_pixels`` produces, ``downsample**2`` per
        merged token. An encoder that defines ``pack_pixels`` defines it.
        """
        raise NotImplementedError

    def features_layout(self, num_tokens: int) -> Mapping[str, OutputLayout]:
        """Describe ``encode``'s features of ``num_tokens`` merged tokens.

        The packed samples' rows are described under ``features``.
        """
        shape = (num_tokens, self._output_size)
        return {
            "features": OutputLayout(
                shape,
                self._output_dtype,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
            )
        }

    def output_layout(self, size: image.Config):
        stride = self.patch_size * self.downsample
        shape = (
            size.height // stride * (size.width // stride),
            self._output_size,
        )
        return {
            "features": OutputLayout(
                shape,
                self._output_dtype,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
            )
        }


class TextEncoder(Encoder[tuple[torch.Tensor, ...]]):
    """Encode independent token sequences through the retained decoder layers.

    retained_layers names the checkpoint layers participating in the numerical
    encoder, in execution order. Output normalization belongs to the supplied
    network; an Identity norm exposes its unnormalized final residual stream.
    """

    network: TransformerDecoder

    def __init__(
        self, network: TransformerDecoder, retained_layers: tuple[int, ...]
    ):
        super().__init__(network)
        if (
            not retained_layers
            or tuple(sorted(set(retained_layers))) != retained_layers
        ):
            raise ValueError(
                "retained decoder layers must be a nonempty increasing tuple"
            )
        if any(str(index) not in network.layers for index in retained_layers):
            raise ValueError(
                "retained decoder layers must exist in the supplied network"
            )
        self.retained_layers = retained_layers
        network.layers = nn.ModuleDict(
            {
                str(index): network.layers[str(index)]
                for index in retained_layers
            }
        )

    def encode(
        self,
        tokens: tuple[torch.Tensor, ...],
        *,
        positions: tuple[torch.Tensor, ...] | None = None,
        embeddings: tuple[EmbeddingReplacement | None, ...] | None = None,
        deepstack: tuple[tuple[torch.Tensor, ...] | None, ...] | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Return complete sequence features on the final pipeline stage.

        All participating stages execute the retained network. Earlier stages
        return None after forwarding their activations; sequence partitions
        are gathered by the final decoder stage before restoring sample bounds.

        The optional inputs hold one entry per sample of ``tokens``:

        - ``positions``: rotary coordinates, ``[length]`` or, for
          multimodal rotary positions, ``[axes, length]`` with the same axes
          in every sample. Without them a sample's tokens take positions
          ``0..length-1``.
        - ``embeddings``: an ``EmbeddingReplacement`` whose dense
          ``[length, hidden]`` values replace the token embeddings where its
          ``[length]`` mask is set, or None to keep every token embedding.
        - ``deepstack``: dense ``[length, hidden]`` features added to the
          residual stream after the leading retained layers, the ``j``-th
          after retained layer ``j`` (DeepStack), or None. Rows without
          features hold zeros. Every sample that has features has the same
          number of them, at most one per retained layer.

        Raises:
            ValueError: An optional input does not match the samples.
        """
        if not tokens:
            return (
                ()
                if self.network._pipeline.rank
                == self.network._pipeline.size - 1
                else None
            )
        if any(value.ndim != 1 for value in tokens):
            raise ValueError(
                "text encoding requires one-dimensional token sequences"
            )
        counts = tuple(value.numel() for value in tokens)
        packed = torch.cat(tokens)
        if positions is None:
            packed_positions = torch.cat(
                tuple(
                    torch.arange(count, device=packed.device)
                    for count in counts
                )
            )
        else:
            if (
                len(positions) != len(counts)
                or any(
                    value.shape[-1] != count
                    for value, count in zip(positions, counts, strict=True)
                )
                or len({value.shape[:-1] for value in positions}) != 1
            ):
                raise ValueError(
                    "positions must cover every sample's tokens with one "
                    "common axis layout"
                )
            packed_positions = torch.cat(positions, dim=-1)
        additions = self._deepstack(counts, deepstack)
        values = torch.cat(
            tuple(
                packed.new_full((1,), count, dtype=torch.int32)
                for count in counts
            )
        )
        offsets = torch.cat(
            (values.new_zeros(1), values.cumsum(0, dtype=torch.int32))
        )
        lengths = SequenceLengths(host=counts, values=values, offsets=offsets)
        attention = VarlenInput(lengths, lengths, (True,) * len(counts))

        embedded = None
        if self.network._pipeline.rank == 0:
            embedded = self._embed(packed, counts, embeddings)
        features = self.network(
            embedded, packed_positions, attention, deepstack=additions
        )
        if self.network._pipeline.rank != self.network._pipeline.size - 1:
            return None
        return tuple(features.split(counts))

    def _embed(
        self,
        packed: torch.Tensor,
        counts: tuple[int, ...],
        embeddings: tuple[EmbeddingReplacement | None, ...] | None,
    ) -> torch.Tensor:
        """Embed packed tokens, splicing each sample's replaced rows."""
        embedded = self.network.embed_input_ids(packed)
        if embeddings is None:
            return embedded
        if len(embeddings) != len(counts):
            raise ValueError("embedding replacements must align with samples")

        parts = list(embedded.split(counts))
        for index, replacement in enumerate(embeddings):
            if replacement is None:
                continue
            if replacement.values.shape != parts[index].shape or (
                replacement.mask.shape != (counts[index],)
            ):
                raise ValueError(
                    "embedding replacements must cover their sample's rows"
                )
            parts[index] = torch.where(
                replacement.mask.reshape(-1, 1),
                replacement.values.to(embedded.dtype),
                parts[index],
            )
        return torch.cat(parts)

    def _deepstack(
        self,
        counts: tuple[int, ...],
        deepstack: tuple[tuple[torch.Tensor, ...] | None, ...] | None,
    ) -> dict[str, torch.Tensor] | None:
        """Pack per-sample DeepStack features by the layer receiving them.

        Returns packed ``[tokens, hidden]`` additions keyed by the retained
        layer's network key, or None when no sample has features.
        """
        if deepstack is None:
            return None
        if len(deepstack) != len(counts):
            raise ValueError("DeepStack features must align with samples")
        provided = [values for values in deepstack if values]
        if not provided:
            return None
        depth, reference = len(provided[0]), provided[0][0]
        if (
            depth > len(self.retained_layers)
            or any(
                len(values) != depth
                for values in deepstack
                if values is not None
            )
            or any(
                feature.shape != (count, self.network.hidden_size)
                for values, count in zip(deepstack, counts, strict=True)
                if values is not None
                for feature in values
            )
        ):
            raise ValueError(
                "DeepStack features must give every sample the same leading "
                "retained layers and cover its rows"
            )

        additions = {}
        for layer in range(depth):
            additions[str(self.retained_layers[layer])] = torch.cat(
                tuple(
                    reference.new_zeros((count, self.network.hidden_size))
                    if values is None
                    else values[layer].to(reference.dtype)
                    for values, count in zip(deepstack, counts, strict=True)
                )
            )
        return additions

    def positions(
        self,
        token_ids: Sequence[int],
        *,
        image_grids: Sequence[tuple[int, int, int]] = (),
        video_grids: Sequence[tuple[int, int, int]] = (),
    ) -> torch.Tensor:
        """Return the rotary coordinates of a prompt holding vision blocks.

        ``image_grids`` and ``video_grids`` are the patch grids of the
        prompt's image and video blocks in prompt order. The result is the
        CPU int64 ``positions`` sample ``encode`` takes for the prompt. An
        encoder that splices vision tokens into prompts defines it.
        """
        raise NotImplementedError

    def output_layout(self, num_tokens: int, dtype: torch.dtype):
        """Describe the encoded conditioning this encoder emits.

        ``dtype`` is the numerical dtype the deployment runs the encoder at,
        which is what its hidden states carry. The caller supplies it because
        every rank of a deployment declares this product, including the ranks
        that never materialize the encoder's parameters and could read no
        representation from them.
        """
        if type(num_tokens) is not int or num_tokens < 0:
            raise ValueError("text output length must be a nonnegative integer")
        if not dtype.is_floating_point:
            raise ValueError("encoded conditioning requires a float dtype")
        shape = (num_tokens, self.network.hidden_size)
        return {
            "conditioning": OutputLayout(
                shape,
                dtype,
                tuple(slice(0, n) for n in shape),
                variable_axes=(0,),
            )
        }
