"""Shared homogeneous feature batching with explicit numerical boundaries."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Generic, TypeVar

import torch
from torch import nn

from uniserve.media import image
from uniserve.nn.attention import AttentionBatch, SequenceLengths, VarlenInput
from uniserve.tensors import OutputLayout

from .inputs import VisionInput
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

    With ``max_patches``, the most patch rows one image can have, the
    encoder also encodes images packed into fixed slots of that many rows
    (``encode_packed``). The network must then read image grids only as
    device values, accept ``grid_shapes=None``, and keep every image's
    output independent of the other segments its grids describe, as
    attention restricted to each segment and pooling within it do. One
    prepared or captured call then serves every packing of a slot count.
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
        max_patches: int | None = None,
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
        if max_patches is not None and (
            type(max_patches) is not int
            or max_patches < 1
            or max_patches % downsample**2
        ):
            raise ValueError(
                "packed image slots require a positive patch capacity that "
                "whole output features divide"
            )
        self.connector, self.patch_size, self.downsample = (
            connector,
            patch_size,
            downsample,
        )
        self._output_size, self._output_dtype = output_size, output_dtype
        self.max_patches = max_patches

    def packed_grids(
        self, shapes: Sequence[tuple[int, int]], slots: int
    ) -> tuple[tuple[int, int], ...]:
        """Lay out ``shapes`` in ``slots`` fixed slots for ``encode_packed``.

        Each slot holds ``max_patches`` rows: one image's patch rows first,
        then padding. It is described by two ``(rows, columns)`` grids: the
        image's patch grid, ``(0, 0)`` for an empty slot, followed by a
        ``downsample`` rows high grid covering the slot's remaining rows, or
        ``(0, 0)`` when the image fills it. Images take the leading slots in
        order.

        Raises:
            ValueError: When there are more images than slots, an image has
                more than ``max_patches`` patches, or a side is not a
                positive multiple of ``downsample``.
        """
        capacity = self.max_patches
        if capacity is None:
            raise ValueError("this patch encoder packs no image slots")
        if len(shapes) > slots:
            raise ValueError("packed images exceed their slots")

        grids: list[tuple[int, int]] = []
        for index in range(slots):
            shape = shapes[index] if index < len(shapes) else (0, 0)
            rows = shape[0] * shape[1]
            if index < len(shapes) and (
                min(shape) < 1
                or any(side % self.downsample for side in shape)
                or rows > capacity
            ):
                raise ValueError(
                    "packed image grids must align with spatial "
                    "downsampling within the slot capacity"
                )
            padding = capacity - rows
            grids.append(shape)
            grids.append(
                (self.downsample, padding // self.downsample)
                if padding
                else (0, 0)
            )
        return tuple(grids)

    def encode_packed(
        self, pixels: torch.Tensor, grids: torch.Tensor
    ) -> torch.Tensor:
        """Encode patch rows packed in fixed slots (see ``packed_grids``).

        ``pixels`` is ``[slots * max_patches, row]`` patch rows and
        ``grids`` the ``[2 * slots, 2]`` int64 device grids of their
        images and padding. Padding rows may hold any finite values.

        Returns:
            ``[slots * max_patches // downsample**2, output_size]`` features
            in row order: each slot's image features first, then features of
            its padding, which are unspecified.

        Raises:
            ValueError: When the encoder packs no slots or the tensors do not
                span whole slots.
        """
        capacity = self.max_patches
        if capacity is None:
            raise ValueError("this patch encoder packs no image slots")
        slots = grids.shape[0] // 2
        if (
            pixels.ndim != 2
            or pixels.shape[0] != slots * capacity
            or grids.shape != (2 * slots, 2)
        ):
            raise ValueError("packed patch rows must span whole image slots")
        features = self.network(pixels, grids, None)
        return self.connector(features).to(self._output_dtype)

    def encode(self, inputs: VisionInput) -> tuple[torch.Tensor, ...]:
        groups = defaultdict(list)
        for index, value in enumerate(inputs.images):
            key = (
                value.ndim,
                value.shape[1:] if value.ndim == 2 else value.shape,
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
                elif value.shape[0] != shape[0] * shape[1]:
                    raise ValueError(
                        "vision grid dimensions must cover their patch rows"
                    )
                if (
                    grid.shape != (1, 2)
                    or min(shape) < 1
                    or any(axis % self.downsample for axis in shape)
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
                height * width // self.downsample**2 for height, width in shapes
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
        self, tokens: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...] | None:
        """Return complete sequence features on the final pipeline stage.

        All participating stages execute the retained network. Earlier stages
        return None after forwarding their activations; sequence partitions
        are gathered by the final decoder stage before restoring sample bounds.
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
        positions = torch.cat(
            tuple(torch.arange(count, device=packed.device) for count in counts)
        )
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
        attention = AttentionBatch.single(
            VarlenInput(lengths, lengths, (True,) * len(counts))
        )

        embeddings = (
            self.network.embed_input_ids(packed)
            if self.network._pipeline.rank == 0
            else None
        )
        features = self.network(embeddings, positions, attention)
        if self.network._pipeline.rank != self.network._pipeline.size - 1:
            return None
        return tuple(features.split(counts))

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
