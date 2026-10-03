"""Shared homogeneous feature batching with explicit numerical boundaries."""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import prod
from typing import Generic, TypeVar

import torch
from torch import nn

from uniserve.media import image
from uniserve.nn.attention import AttentionBatch, SequenceLengths, VarlenInput
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


class TubeletEncoder(PatchEncoder, ABC):
    """Encode images and video blocks packed as spatiotemporal patches.

    A tubelet holds the RGB values of ``temporal_patch_size`` consecutive
    frames over one ``patch_size x patch_size`` patch; an image repeats its
    frame. Samples are packed tubelet rows with ``(time, height, width)``
    grids, ``time`` counting tubelets along the frames. ``pack_pixels``
    produces such a sample from decoded frames on the host, as the
    architecture's processor does. It reads no parameter, so a rank holding
    only the module's description runs it.
    """

    def __init__(
        self,
        network: nn.Module,
        connector: nn.Module,
        *,
        patch_size: int,
        temporal_patch_size: int,
        downsample: int,
        output_size: int,
        output_dtype: torch.dtype,
    ):
        super().__init__(
            network,
            connector,
            patch_size=patch_size,
            downsample=downsample,
            output_size=output_size,
            output_dtype=output_dtype,
        )
        if type(temporal_patch_size) is not int or temporal_patch_size < 1:
            raise ValueError("tubelets must span a positive frame count")
        self.temporal_patch_size = temporal_patch_size

    @abstractmethod
    def pack_pixels(
        self, frames: torch.Tensor, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        """Return one image's or video's packed tubelet rows for ``encode``.

        ``frames`` holds ``[frames, height, width, 3]`` uint8 RGB on the
        host, an image being one frame; ``grid`` is the ``(time, height,
        width)`` patch grid they are resized to. The result is the FP32
        ``[time * height * width, 3 * temporal_patch_size * patch_size**2]``
        sample ``encode`` takes with ``grid`` as its host shape
        (``pixels_layout``).
        """

    def pixels_layout(self, num_tokens: int) -> OutputLayout:
        """Describe the packed tubelet rows of ``num_tokens`` merged tokens.

        These are the FP32 rows ``pack_pixels`` produces, ``downsample**2``
        per merged token.
        """
        shape = (
            num_tokens * self.downsample**2,
            3 * self.temporal_patch_size * self.patch_size**2,
        )
        return OutputLayout(
            shape,
            torch.float32,
            tuple(slice(0, n) for n in shape),
            variable_axes=(0,),
        )


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
            return self._empty()
        packed, counts = self._pack(tokens)
        positions = torch.cat(
            tuple(torch.arange(count, device=packed.device) for count in counts)
        )
        embeddings = (
            self.network.embed_input_ids(packed)
            if self.network._pipeline.rank == 0
            else None
        )
        return self._decode(embeddings, positions, counts)

    def _empty(self) -> tuple[torch.Tensor, ...] | None:
        """Return the result of encoding no samples on this stage."""
        pipeline = self.network._pipeline
        return () if pipeline.rank == pipeline.size - 1 else None

    @staticmethod
    def _pack(
        tokens: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Concatenate token sequences and return them with their lengths."""
        if any(value.ndim != 1 for value in tokens):
            raise ValueError(
                "text encoding requires one-dimensional token sequences"
            )
        return torch.cat(tokens), tuple(value.numel() for value in tokens)

    def _decode(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        counts: tuple[int, ...],
        **inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        """Run the retained network over packed samples and split them.

        ``embeddings`` are the packed token embeddings on the first pipeline
        stage and None elsewhere; ``positions`` are the packed rotary
        coordinates, ``[tokens]`` or ``[axes, tokens]``. Each sample attends
        causally within itself. ``inputs`` are further keyword inputs of the
        network's forward.
        """
        values = torch.cat(
            tuple(
                positions.new_full((1,), count, dtype=torch.int32)
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

        features = self.network(embeddings, positions, attention, **inputs)
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


class MultimodalEncoder(TextEncoder, ABC):
    """Encode prompts whose placeholder tokens stand for vision tokens.

    ``vision`` encodes images and video blocks into one row per vision token
    (``PatchEncoder.encode``). A prompt marks each vision token with one of
    the ``placeholders`` token ids, in the order of the rows. ``encode``
    replaces each placeholder's token embedding with the leading ``hidden``
    values of its row; an architecture whose rows carry further values hands
    them to its network through ``_vision_inputs``. A prompt's rotary
    coordinates depend on the patch grids of its vision blocks, and
    ``positions`` computes them on the host so that callers stage them like
    any other input.
    """

    def __init__(
        self,
        network: TransformerDecoder,
        retained_layers: tuple[int, ...],
        vision: PatchEncoder,
        *,
        placeholders: tuple[int, ...],
    ):
        super().__init__(network, retained_layers)
        if (
            not isinstance(placeholders, tuple)
            or not placeholders
            or len(set(placeholders)) != len(placeholders)
            or any(
                type(token) is not int or token < 0 for token in placeholders
            )
        ):
            raise ValueError(
                "vision placeholders must be distinct nonnegative token ids"
            )
        self.vision = vision
        self.placeholders = placeholders

    @abstractmethod
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
        CPU int64 ``positions`` sample ``encode`` takes for the prompt.
        """

    def encode(
        self,
        tokens: tuple[torch.Tensor, ...],
        *,
        positions: tuple[torch.Tensor, ...] | None = None,
        visual: tuple[torch.Tensor | None, ...] | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """Encode prompts, splicing vision tokens at their placeholders.

        Args:
            tokens: One ``[length]`` token sequence per prompt.
            positions: One rotary coordinate tensor per prompt, ``[length]``
                or ``[axes, length]`` with the same axes for every prompt
                (``positions``). Without them every prompt takes the text
                coordinates ``0..length-1``.
            visual: Per prompt, the ``vision.encode`` rows of its vision
                tokens in placeholder order, ``[placeholders, width]`` with
                one width for every prompt, or None for a prompt without
                placeholders. Without it every token, placeholder ids
                included, is read as text, and the call reads no device
                value on the host.

        Returns:
            Per prompt ``[length, hidden]`` features on the final pipeline
            stage; None on earlier stages.

        Raises:
            ValueError: The positions or vision rows do not cover their
                prompts.
        """
        if not tokens:
            return self._empty()
        packed, counts = self._pack(tokens)
        if positions is None:
            packed_positions = torch.cat(
                tuple(
                    torch.arange(count, device=packed.device)
                    for count in counts
                )
            )
        elif (
            len(positions) != len(counts)
            or any(
                value.shape[-1] != count
                for value, count in zip(positions, counts, strict=True)
            )
            or len({value.shape[:-1] for value in positions}) != 1
        ):
            raise ValueError(
                "positions must cover every prompt's tokens with one common "
                "axis layout"
            )
        else:
            packed_positions = torch.cat(positions, dim=-1)

        rows = None if visual is None else self._vision_rows(tokens, visual)
        embeddings = None
        if self.network._pipeline.rank == 0:
            embeddings = self.network.embed_input_ids(packed)
            if rows is not None:
                values, mask = rows
                embeddings = EmbeddingReplacement(
                    values[:, : self.network.hidden_size], mask
                ).apply(embeddings)
        inputs = {} if rows is None else self._vision_inputs(rows[0])
        return self._decode(embeddings, packed_positions, counts, **inputs)

    def _vision_rows(
        self,
        tokens: tuple[torch.Tensor, ...],
        visual: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Place the prompts' vision rows at their packed placeholder tokens.

        Returns dense packed ``[tokens, width]`` rows, zero at text tokens,
        and the ``[tokens]`` placeholder mask, or None when no prompt holds
        vision tokens. One host read checks every prompt's placeholder count
        against its rows.
        """
        if len(visual) != len(tokens):
            raise ValueError("vision rows must align with the prompts")
        ids = tokens[0].new_tensor(self.placeholders)
        masks = tuple(torch.isin(value, ids) for value in tokens)
        counts = torch.stack(tuple(mask.sum() for mask in masks)).tolist()
        provided = [rows for rows in visual if rows is not None]
        if not provided:
            if any(counts):
                raise ValueError(
                    "a prompt with vision placeholders requires their rows"
                )
            return None

        # A prompt without rows holds no placeholders.
        width = provided[0].shape[-1]
        for rows, count in zip(visual, counts, strict=True):
            shape = (0, width) if rows is None else rows.shape
            if shape != (count, width):
                raise ValueError(
                    "vision rows must cover their prompt's placeholders with "
                    "one width"
                )
        mask = torch.cat(masks)
        values = provided[0].new_zeros((mask.numel(), width))
        values[mask] = torch.cat(provided)
        return values, mask

    def _vision_inputs(self, rows: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Return the network inputs the values of vision rows past their
        embedding carry.

        ``rows`` holds packed ``[tokens, width]`` vision rows, zero at text
        tokens. Here a row holds only its token's embedding; an architecture
        whose vision rows carry further values overrides this.
        """  # noqa: D205
        if rows.shape[-1] != self.network.hidden_size:
            raise ValueError("vision rows must hold one language embedding")
        return {}
