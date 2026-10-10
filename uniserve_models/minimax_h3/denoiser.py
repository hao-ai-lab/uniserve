"""H3's packed denoising computation for every schedule and attention kind.

``Denoiser`` implements the ``VideoDenoiser`` capability for H3's joint
video/audio transformer under one ``DenoiserConfig``: the released dense
DiTs with their full-step uniform schedule, a DMD student with sparse
attention, or a parallel-decoding (PDD) student. One sample's text, audio and
video rows share one packed sequence, split into equal row shards across the
sequence-parallel group. A modality's canonical sample on a rank is that
rank's rows of the modality, in packed order.

A layout fixes a frame count, a canvas and the text and condition
capacities. Every request whose frame count and canvas match and whose
prompt and conditions fit (``holds``) evaluates in that layout and shares its
constants, workspace and captured graphs; the tables that depend on the
exact prompt and conditions are request state, filled by ``prepare_state``.

Single-segment sparse attention packs ``packing.TilePacking`` (text in whole
64-row tiles, the media timeline at the exact prompt length). Dense attention
packs ``packing.DensePacking``: the generated rows lead at fixed offsets, so a
shard's sample rows are a layout constant, and each rank gathers its prefix
rows (text, then conditions) from the request's prefix source. Multi-segment
sparse attention packs ``packing.SegmentPacking``: text and condition tiles at
the layout's capacities, then the generated audio and video tiles at fixed
offsets; its prefix rows gather from the prefix source as dense attention's
do, and its tile and segment tables are request state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import torch

from uniserve.diffusion import (
    BlockGrid,
    CleanSampleEulerSolver,
)
from uniserve.media import image, video
from uniserve.model import (
    Condition,
    ConditionRole,
    LatentInput,
    VideoDenoiser,
)
from uniserve.nn import ColumnParallelLinear, RotaryEmbedding
from uniserve.nn.attention import SequenceLengths, VisibleInput, vsa
from uniserve.tensors import BufferConfig, OutputLayout, TensorOutput

from .conditioning import Conditioner
from .config import (
    NAMED_ASPECT_RATIOS,
    DenoiserConfig,
    DenseAttention,
    SparseAttention,
    canvas,
    rule_canvases,
)
from .inputs import (
    AttentionInput,
    DenoiserInput,
    DenoiserSize,
    SegmentInput,
    SequenceInput,
)
from .packing import (
    AUDIO_TAG,
    FPS,
    DensePacking,
    SegmentPacking,
    TilePacking,
    audio_latent_frames,
    condition_segments,
    dense_packing,
    dense_tables,
    latent_raster,
    patchify_video,
    segment_packing,
    segment_tables,
    segment_tiles,
    tile_packing,
    video_latent_frames,
)
from .processing import (
    AUDIO_SAMPLE_RATE,
    MAX_IMAGE_REFERENCES,
    MAX_VIDEO_REFERENCES,
    VAE_FRAMES_PER_CHUNK,
    VAE_LATENTS_PER_CHUNK,
    reference_image_size,
)
from .transformer import Transformer, TransformerLayer

# A visual condition is held at this clean time for every step whose
# generated video is noisier, as the released model was trained.
VISUAL_CONDITION_TIME = 0.999
# An audio reference conditions at the clean endpoint.
AUDIO_CONDITION_TIME = 1.0


def timestep_groups(config: DenoiserConfig) -> int:
    """Count the timestep groups a step of this denoiser reads.

    Text-only denoisers read the generated video and audio timesteps; a
    conditioned denoiser also reads the visual-condition and audio-reference
    levels.
    """
    return 2 if config.tasks == ("t2va",) else 4


def modulation_timesteps(
    config: DenoiserConfig,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """Enumerate each step's group timesteps and their distinct entries.

    Evaluates the checkpoint's video and audio grids on the host. Returns the
    per-step FP32 group timesteps (one [groups] tensor per step, video
    first), the distinct FP32 values in first-occurrence order of that
    step-major enumeration, and the [step, group] int64 entry of each.
    Distinct means distinct FP32 bit patterns, so an entry's products are
    exactly those of every occurrence.
    """
    groups = timestep_groups(config)
    values = {
        name: grid.endpoints("cpu") for name, grid in config.grids.items()
    }
    video, audio = (values[name].timesteps for name in ("video", "audio"))
    steps = []
    for index in range(values["video"].num_steps):
        row = [video[index], audio[index]]
        if groups == 4:
            # The comparison and the conditioning levels are FP32, as the
            # reference fills its per-row timestep table.
            row.append(
                torch.maximum(
                    video[index],
                    torch.tensor(
                        VISUAL_CONDITION_TIME, dtype=torch.float32, device="cpu"
                    ),
                )
            )
            row.append(
                torch.tensor(
                    AUDIO_CONDITION_TIME, dtype=torch.float32, device="cpu"
                )
            )
        steps.append(torch.stack(row))
    keys: dict[int, int] = {}
    entries: list[torch.Tensor] = []
    table = torch.empty((len(steps), groups), dtype=torch.int64, device="cpu")
    for step, step_times in enumerate(steps):
        for group, value in enumerate(step_times):
            key = int(value.view(torch.int32))
            if key not in keys:
                keys[key] = len(entries)
                entries.append(value)
            table[step, group] = keys[key]
    return tuple(steps), torch.stack(entries), table


class Denoiser(VideoDenoiser[DenoiserInput, DenoiserSize]):
    """Predict video/audio velocities from explicit latent and text tensors.

    Native random draws and their contiguous sample destinations use CPU FP32;
    callers transfer prepared samples into device views before evaluation.
    Text features are the output of ``conditioner.encode``. Packed padding is
    numerical attention input and never represents a second computation batch.
    """

    # A native temporal window covers 17 new frames and carries a 5-frame
    # overlap, so a legal input length is 5 + 17k with a 22-frame minimum.
    NATIVE_WINDOW_FRAMES = 17
    NATIVE_OVERLAP_FRAMES = 5

    def __init__(self, config: DenoiserConfig):
        super().__init__(
            modalities=("video", "audio"),
            prediction_dtype=torch.float32,
            solver=CleanSampleEulerSolver(),
            grids=config.grids,
        )
        self.config = config
        _, entries, table = modulation_timesteps(config)
        self.transformer = Transformer(
            config.transformer,
            attention=config.attention,
            entries=entries.numel(),
            steps=table.shape[0],
            groups=table.shape[1],
        )
        self.conditioner = Conditioner(config.transformer)
        self.rotary = RotaryEmbedding(
            2 * config.transformer.rope_frequency_dim,
            theta=config.transformer.rope_theta,
        )

    @property
    def canvases(self) -> tuple[image.Config, ...]:
        """Canvases a deployment may prepare layouts for.

        A checkpoint that generates only some canvases offers those; one
        that follows the canvas rule offers every named aspect ratio's
        canvas.
        """
        if self.config.canvases is not None:
            return self.config.canvases
        return tuple(
            dict.fromkeys(canvas(*ratio) for ratio in NAMED_ASPECT_RATIOS)
        )

    @property
    def tasks(self) -> tuple[str, ...]:
        return self.config.tasks

    @property
    def fixed_canvases(self) -> tuple[image.Config, ...] | None:
        return self.config.canvases

    def max_conditions(
        self, num_frames: int, canvas: image.Config
    ) -> tuple[Condition, ...]:
        """The largest condition set the checkpoint documents for a request.

        A ``ref2va`` request carries up to nine image and three video
        references, twelve references in all: nine images of the widest
        reference raster and three videos of the generated length with
        their soundtracks, on the canvas of the canvas rule where a video
        packs the most rows. A network that packs keyframes also holds the
        two keyframes of ``canvas``, which are all an ``fl2va`` request
        carries. Single-segment sparse attention takes no conditions.
        """
        tasks = set(self.config.tasks)
        if not (self.dense or self.segmented) or not tasks & {
            "fl2va",
            "ref2va",
        }:
            return ()

        keyframes: tuple[Condition, ...] = ()
        if not self.segmented:
            frame = video.Config(1, canvas)
            keyframes = (
                Condition(ConditionRole.FIRST_FRAME, frame),
                Condition(ConditionRole.LAST_FRAME, frame),
            )
        if "ref2va" not in tasks:
            return keyframes

        def rows(condition: Condition) -> int:
            size = self.make_size(
                num_frames, 1, canvas=canvas, conditions=(condition,)
            )
            return size.condition_rows

        # A reference video keeps the leading whole VAE windows of the
        # generated frames, and its soundtrack the generated duration.
        windows = max(
            1, (num_frames - VAE_LATENTS_PER_CHUNK) // VAE_FRAMES_PER_CHUNK
        )
        frames = windows * VAE_FRAMES_PER_CHUNK + VAE_LATENTS_PER_CHUNK
        samples = math.ceil(num_frames * AUDIO_SAMPLE_RATE / FPS)
        clip = max(
            (
                Condition(
                    ConditionRole.REFERENCE,
                    video.Config(frames, raster),
                    samples,
                )
                for raster in rule_canvases()
            ),
            key=rows,
        )
        still = max(
            (
                Condition(
                    ConditionRole.REFERENCE,
                    video.Config(1, reference_image_size(*aspect)),
                )
                for aspect in ((4, 1), (1, 4))
            ),
            key=rows,
        )
        return (
            *keyframes,
            *(still,) * MAX_IMAGE_REFERENCES,
            *(clip,) * MAX_VIDEO_REFERENCES,
        )

    @property
    def dense(self) -> bool:
        """Whether this denoiser attends densely (see ``DensePacking``)."""
        return isinstance(self.config.attention, DenseAttention)

    @property
    def segmented(self) -> bool:
        """Whether this denoiser's sparse attention selects per segment.

        Such a denoiser packs ``SegmentPacking``; a sparse denoiser without
        reference segments packs ``TilePacking``.
        """
        attention = self.config.attention
        return (
            isinstance(attention, SparseAttention)
            and attention.reference_keep is not None
        )

    @property
    def _tile(self) -> int:
        """Rows of one text and condition capacity unit of a layout."""
        attention = self.config.attention
        return attention.tile if isinstance(attention, SparseAttention) else 64

    def legal_frame_count(self, requested: int) -> int:
        """Round a requested duration up to the next complete native window."""
        window, overlap = self.NATIVE_WINDOW_FRAMES, self.NATIVE_OVERLAP_FRAMES
        return max(overlap + window, requested + (overlap - requested) % window)

    def make_size(
        self,
        num_frames: int,
        num_text_tokens: int,
        *,
        canvas: image.Config,
        conditions: tuple[Condition, ...] = (),
        vision_spans: tuple[tuple[int, int], ...] = (),
    ) -> DenoiserSize:
        """Build this network's size descriptor for one admitted request.

        The condition rows are the packed rows of ``conditions``
        (``packing.condition_segments``).

        Under multi-segment sparse attention the condition rows are the
        conditions' whole tiles (``packing.segment_tiles``), the rows the
        segment packing gives them.

        Raises:
            ValueError: A canvas the checkpoint does not generate, an invalid
                frame count, prompt length or vision span, or conditions this
                network does not take: keyframes without an ``fl2va`` or
                ``ref2va`` task, references without ``ref2va``, any condition
                under single-segment sparse attention, or a keyframe under
                multi-segment sparse attention, whose segment packing has no
                keyframe rows.
        """
        canvases = self.config.canvases
        if canvases is not None and canvas not in canvases:
            raise ValueError(
                f"this H3 checkpoint generates only {canvases}, not {canvas}"
            )
        segments = condition_segments(tuple(conditions), canvas)
        tasks = self.config.tasks
        if segments and not (self.dense or self.segmented):
            raise ValueError(
                "single-segment sparse H3 attention takes no conditions"
            )
        if any(segment.anchor is None for segment in segments) and (
            "ref2va" not in tasks
        ):
            raise ValueError("this H3 denoiser takes no references")
        if segments and not {"fl2va", "ref2va"} & set(tasks):
            raise ValueError("this H3 denoiser takes no keyframes")
        if self.segmented:
            rows = self._tile * sum(
                segment_tiles(segment, self._tile) for segment in segments
            )
        else:
            rows = sum(
                segment.video_rows + segment.audio_rows for segment in segments
            )
        return DenoiserSize(
            num_frames,
            canvas,
            num_text_tokens,
            rows,
            tuple(conditions),
            tuple(vision_spans),
        )

    def layout_size(self, size: DenoiserSize) -> DenoiserSize:
        """Return the smallest layout that holds ``size``.

        Text and conditions occupy whole tiles (64 rows, or the segment
        packing's tile), so the smallest segments are the prompt's and the
        conditions' own tiles.
        """
        tile = self._tile
        return DenoiserSize(
            size.num_frames,
            size.canvas,
            math.ceil(size.num_text_tokens / tile) * tile,
            math.ceil(size.condition_rows / tile) * tile,
        )

    def holds(self, layout: DenoiserSize, size: DenoiserSize) -> bool:
        """Whether ``layout`` evaluates a request of ``size``.

        A layout holds every prompt and condition set that fits its segments
        at its own frame count and canvas. The packed shapes and every
        constant depend on the layout alone; the request's state carries
        what the exact prompt and conditions change. Frame counts are not
        padded: the video extent sets the sparse selection budget and the
        audio timeline.
        """
        return (
            layout == self.layout_size(layout)
            and layout.num_frames == size.num_frames
            and layout.canvas == size.canvas
            and layout.num_text_tokens >= size.num_text_tokens
            and layout.condition_rows >= size.condition_rows
        )

    @property
    def text_condition_width(self) -> int:
        """Feature width of the retained text conditioning."""
        return self.config.transformer.hidden_size

    def text_condition_rows(self, layout: DenoiserSize) -> int:
        """Rows of a request's retained conditioning in ``layout``.

        Single-segment sparse attention retains the refined text over the
        layout's text rows. Dense and multi-segment sparse attention retain
        the prefix source: the text rows, the projected condition rows, and
        one zero row the generated and padding rows gather.
        """
        if not (self.dense or self.segmented):
            return layout.num_text_tokens
        return layout.num_text_tokens + layout.condition_rows + 1

    def condition_noise_shapes(
        self, size: DenoiserSize
    ) -> tuple[tuple[int, ...], ...]:
        """Native draws of the visual conditions, in packed order.

        Each is a ``[1, channels, frames, height, width]`` latent the
        condition's anchoring mixes in (``encode_conditions``). Audio
        references are conditioned at their posterior mean, so they draw
        none.
        """
        return tuple(
            (
                1,
                self.config.transformer.video_channels,
                segment.latent_frames,
                segment.latent_height,
                segment.latent_width,
            )
            for segment in condition_segments(size.conditions, size.canvas)
            if segment.video_rows
        )

    def condition_noise_capacity(self, layout: DenoiserSize) -> int:
        """FP32 elements of the condition draws of any size ``layout`` holds.

        A draw holds one element per channel of each latent pixel, as many
        as its rows hold, so the condition capacity bounds them.
        """
        return (
            layout.condition_rows * self.config.transformer.video_channels * 4
        )

    def condition_layout(
        self, layout: DenoiserSize, condition_rows: int
    ) -> DenoiserSize:
        """Return ``layout`` with a condition segment of ``condition_rows``.

        Conditions occupy whole tiles (64 rows, or the segment packing's
        tile), as ``layout_size`` rounds them, and a larger segment is kept.

        Raises:
            ValueError: Single-segment sparse attention, which takes no
                conditions, or a negative row count.
        """
        if not (self.dense or self.segmented):
            raise ValueError(
                "single-segment sparse H3 attention takes no conditions"
            )
        if type(condition_rows) is not int or condition_rows < 0:
            raise ValueError("H3 condition rows must be a nonnegative count")
        tile = self._tile
        return replace(
            layout,
            condition_rows=max(
                layout.condition_rows, math.ceil(condition_rows / tile) * tile
            ),
        )

    @torch.inference_mode()
    def encode_conditions(
        self,
        size: DenoiserSize,
        layout: DenoiserSize,
        *,
        latents: tuple[torch.Tensor, ...],
        noise: tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        """Project a request's conditions into its prefix source.

        ``latents`` lists each condition's encoded latents in request order:
        a visual condition's normalized ``[rows, channels * 4]`` FP32 patch
        rows, then, for a condition with audio, its ``[2 * frames,
        audio_channels]`` FP32 channel-major rows. ``noise`` holds the
        ``condition_noise_shapes`` draws on the host. A visual condition is
        anchored as ``t z + (1 - t) noise`` at ``t = VISUAL_CONDITION_TIME``
        in FP32; an audio reference is its latent unchanged. Each is then
        projected once by the layer that projects the generated rows of its
        modality, and the BF16 rows fill ``out`` past the layout's text
        capacity in packed order, where the request's prefix rows gather them
        (``dense_tables``). A pipeline stage without the input projections
        never reads the prefix source and writes nothing.

        Raises:
            ValueError: Inputs that do not match ``size`` or a retained
                conditioning that does not match ``layout``.
        """
        segments = condition_segments(size.conditions, size.canvas)
        visual_count = sum(1 for segment in segments if segment.video_rows)
        expected = sum(
            (condition.video is not None) + (condition.audio_samples > 0)
            for condition in size.conditions
        )
        if (
            not self.holds(layout, size)
            or len(latents) != expected
            or len(noise) != visual_count
            or out.shape
            != (self.text_condition_rows(layout), self.text_condition_width)
        ):
            raise ValueError(
                "H3 condition encoding requires each condition's latents, its "
                "draws and the retained conditioning of a layout that holds it"
            )
        if self.transformer.video_input is None:
            return

        # Each condition's visual and audio latents, in request order.
        remaining = iter(latents)
        encoded = [
            (
                next(remaining) if condition.video is not None else None,
                next(remaining) if condition.audio_samples else None,
            )
            for condition in size.conditions
        ]
        draws = iter(noise)
        level = torch.tensor(
            VISUAL_CONDITION_TIME, dtype=torch.float32, device=out.device
        )
        row = layout.num_text_tokens
        for segment in segments:
            visual, audio = encoded[segment.condition]
            if segment.audio_rows:
                if audio is None or audio.shape != (
                    segment.audio_rows,
                    self.config.transformer.audio_channels,
                ):
                    raise ValueError("H3 audio condition rows do not match")
                rows = audio.to(device=out.device, dtype=torch.float32)
                out[row : row + segment.audio_rows].copy_(
                    self.transformer.audio_input(
                        rows, output_dtype=torch.float32
                    ).to(out.dtype)
                )
                row += segment.audio_rows
            if segment.video_rows:
                draw = next(draws)
                width = self.config.transformer.video_channels * 4
                if (
                    visual is None
                    or visual.shape != (segment.video_rows, width)
                    or tuple(draw.shape)
                    != (
                        1,
                        self.config.transformer.video_channels,
                        segment.latent_frames,
                        segment.latent_height,
                        segment.latent_width,
                    )
                ):
                    raise ValueError("H3 visual condition rows do not match")
                clean = visual.to(device=out.device, dtype=torch.float32)
                draw = patchify_video(draw.to(device=out.device))[0]
                # The reference's anchoring, op for op in FP32; mixing after
                # patching is the same elementwise arithmetic.
                anchored = level * clean + (1.0 - level) * draw
                out[row : row + segment.video_rows].copy_(
                    self.transformer.video_input(
                        anchored, output_dtype=torch.float32
                    ).to(out.dtype)
                )
                row += segment.video_rows

    def bind_inputs(
        self,
        *,
        latents: Mapping[str, tuple[LatentInput, ...]],
        sizes: tuple[DenoiserSize, ...],
        step: torch.Tensor,
        text_features: tuple[torch.Tensor, ...],
    ) -> DenoiserInput:
        """Assemble one denoising step's typed input from resident tensors."""
        return DenoiserInput(
            latents=latents,
            sizes=sizes,
            step=step,
            text_features=text_features,
        )

    def _sequence_group(self):
        # The sequence group spans the mesh axes that shard the token
        # dimension of the attention input. Resident layers are
        # TransformerLayers whose merged attention branches are
        # column-parallel linears.
        layer = cast(
            TransformerLayer, next(iter(self.transformer.layers.values()))
        )
        query = cast(
            ColumnParallelLinear, layer.attention.projection.projections["q"]
        )
        distribution = query.input_distribution
        return distribution.mesh.get_group(distribution.shard_axes(0))

    def _tile_packing(
        self, size: DenoiserSize, layout: DenoiserSize | None = None
    ) -> TilePacking:
        # Packs ``size``'s prompt in ``layout``'s text segment (its own tiles
        # by default). The alignment is a multiple of both the default
        # 256-row alignment and one 64-row tile per sequence rank, so every
        # rank's shard holds whole tiles.
        return tile_packing(
            num_text_tokens=size.num_text_tokens,
            num_frames=size.num_frames,
            canvas=size.canvas,
            token_multiple=64 * math.lcm(4, self._sequence_group().size),
            text_rows=None if layout is None else layout.num_text_tokens,
        )

    def _dense_packing(self, layout: DenoiserSize) -> DensePacking:
        # Every rank's shard holds a whole number of 128-row blocks.
        return dense_packing(
            num_frames=layout.num_frames,
            canvas=layout.canvas,
            text_rows=layout.num_text_tokens,
            condition_rows=layout.condition_rows,
            token_multiple=128 * self._sequence_group().size,
        )

    def _segment_packing(self, layout: DenoiserSize) -> SegmentPacking:
        # Every rank's shard holds a whole number of tiles.
        return segment_packing(
            num_frames=layout.num_frames,
            canvas=layout.canvas,
            text_rows=layout.num_text_tokens,
            condition_rows=layout.condition_rows,
            tile=self._tile,
            token_multiple=self._tile * self._sequence_group().size,
        )

    def _padded_tokens(self, layout: DenoiserSize) -> int:
        if self.dense:
            return self._dense_packing(layout).padded_tokens
        if self.segmented:
            return self._segment_packing(layout).padded_tokens
        return self._tile_packing(layout).padded_tokens

    def _token_slice(self, padded_tokens: int) -> slice:
        group = self._sequence_group()
        width = padded_tokens // group.size
        return slice(group.rank * width, (group.rank + 1) * width)

    def latent_shape(
        self, modality: str, size: DenoiserSize
    ) -> tuple[int, ...]:
        # Complete canonical samples across all ranks; ``output_layout``
        # selects this rank's rows.
        if modality == "video":
            height, width = latent_raster(size.canvas)
            # Each 2x2 patch of the latent raster is one token.
            return video_latent_frames(size.num_frames) * (height // 2) * (
                width // 2
            ), self.config.transformer.video_channels * 4
        if modality == "audio":
            return 2 * audio_latent_frames(
                size.num_frames
            ), self.config.transformer.audio_channels
        raise ValueError(f"unknown H3 latent modality {modality!r}")

    def noise_shape(self, modality: str, size: DenoiserSize) -> tuple[int, ...]:
        if modality == "video":
            # Native draws keep the [C, T, H, W] latent layout before
            # patching.
            height, width = latent_raster(size.canvas)
            return (
                1,
                self.config.transformer.video_channels,
                video_latent_frames(size.num_frames),
                height,
                width,
            )
        return self.latent_shape(modality, size)

    def _sample_ranges(self, size: DenoiserSize) -> Mapping[str, slice]:
        """Each modality's canonical rows this rank holds in layout ``size``.

        Every modality's packed rows ascend, so the rows falling in this
        rank's token slice are one contiguous range of the sample.
        """
        if self.dense:
            packing = self._dense_packing(size)
            interval = self._token_slice(packing.padded_tokens)
            result = {}
            for name, start, count in (
                ("video", 0, packing.video_rows),
                ("audio", packing.video_rows, packing.audio_rows),
            ):
                low = min(max(interval.start - start, 0), count)
                high = min(max(interval.stop - start, 0), count)
                result[name] = slice(low, high)
            return result
        tiles = (
            self._segment_packing(size)
            if self.segmented
            else self._tile_packing(size)
        )
        interval = self._token_slice(tiles.padded_tokens)
        result = {}
        for name, indices in (
            ("video", tiles.video_indices),
            ("audio", tiles.audio_indices),
        ):
            start = int(torch.searchsorted(indices, interval.start))
            stop = int(torch.searchsorted(indices, interval.stop))
            result[name] = slice(start, stop)
        return result

    def output_layout(self, size: DenoiserSize) -> Mapping[str, OutputLayout]:
        """Locate this rank's predicted rows for requests of one layout.

        ``size`` is the layout the request is evaluated in: it fixes where the
        audio and video segments lie, hence which of their rows fall in this
        rank's shard. Every request the layout holds shares the result.
        """
        result = {}
        for name, rows in self._sample_ranges(size).items():
            shape = self.latent_shape(name, size)
            result[name] = OutputLayout(
                shape,
                self.prediction_dtype,
                local_slice=(rows, slice(0, shape[1])),
                variable_axes=(0,),
            )
        return result

    def _rotary_width(self) -> int:
        """Width of one packed row's flattened rotary coordinates."""
        cosine, _ = self.rotary(
            torch.zeros((1, 3), dtype=torch.float64),
            dtype=torch.float32,
            sequence_length=1,
        )
        return int(cosine.flatten(1).shape[1])

    def state_buffers(self, size: DenoiserSize) -> Mapping[str, BufferConfig]:
        """Describe one request's state on the caller's device.

        The state is each modality's local canonical sample and the tables
        that depend on the exact prompt (and conditions) within its layout.
        Sparse attention reads every tile's valid row count, the rotary
        ``cos``/``sin`` of every packed row, and the live dense prefix tiles
        (``vsa.Input``): the prefix key list, the dense key list and the
        live prefix count. Dense attention reads, for this rank's rows, the
        prefix source row each takes, its rotary ``cos``/``sin`` and its
        modulation row, and the used row count every query sees
        (``visible_end``). Multi-segment sparse attention reads this rank's
        prefix source rows and modulation rows, the rotary ``cos``/``sin`` of
        every packed row, and the segment tables of every tile
        (``vsa.Segments``). Their shapes follow the layout alone.
        """
        size = self.layout_size(size)
        samples = {
            name: BufferConfig(
                tuple(part.stop - part.start for part in layout.local_slice),
                layout.dtype,
            )
            for name, layout in self.output_layout(size).items()
        }
        width = self._rotary_width()
        packing: DensePacking | SegmentPacking | TilePacking
        if self.dense:
            packing = self._dense_packing(size)
            rows = packing.padded_tokens // self._sequence_group().size
            return {
                **samples,
                "prefix_index": BufferConfig((rows,), torch.int64),
                "cos": BufferConfig((rows, width), torch.float32),
                "sin": BufferConfig((rows, width), torch.float32),
                "modulation_indices": BufferConfig((rows,), torch.int64),
                # One endpoint that every query row shares.
                "visible_end": BufferConfig((1, 1), torch.int32),
            }
        if self.segmented:
            packing = self._segment_packing(size)
            rows = packing.padded_tokens
            local = rows // self._sequence_group().size
            tiles = rows // packing.tile
            return {
                **samples,
                "prefix_index": BufferConfig((local,), torch.int64),
                "cos": BufferConfig((rows, width), torch.float32),
                "sin": BufferConfig((rows, width), torch.float32),
                "modulation_indices": BufferConfig((local,), torch.int64),
                **{
                    name: BufferConfig((tiles,), torch.int32)
                    for name in (
                        "valid_sizes",
                        "tile_segments",
                        "segment_starts",
                        "segment_keep",
                    )
                },
            }
        packing = self._tile_packing(size)
        rows = packing.padded_tokens
        return {
            **samples,
            "tile_valid_sizes": BufferConfig((rows // 64,), torch.int32),
            "cos": BufferConfig((rows, width), torch.float32),
            "sin": BufferConfig((rows, width), torch.float32),
            "prefix_key_indices": BufferConfig(
                (packing.prefix_tiles,), torch.int32
            ),
            "dense_key_indices": BufferConfig(
                (packing.prefix_tiles + packing.video_tiles,), torch.int32
            ),
            "prefix_count": BufferConfig((1,), torch.int32),
        }

    def _dense_state(
        self, size: DenoiserSize, layout: DenoiserSize
    ) -> dict[str, torch.Tensor]:
        """Fill a request's dense tables for this rank's rows."""
        packing = self._dense_packing(layout)
        tables = dense_tables(
            packing,
            num_frames=size.num_frames,
            canvas=size.canvas,
            num_text_tokens=size.num_text_tokens,
            segments=condition_segments(size.conditions, size.canvas),
            vision_spans=size.vision_spans,
        )
        interval = self._token_slice(packing.padded_tokens)
        cosine, sine = self.rotary(
            tables.position_ids[interval],
            dtype=torch.float32,
            sequence_length=packing.padded_tokens,
        )
        groups, tags = tables.groups[interval], tables.token_tags[interval]
        return {
            "prefix_index": tables.prefix_index[interval],
            "cos": cosine.flatten(1),
            "sin": sine.flatten(1),
            # Row ``3 * group + tag`` of each step's modulation products.
            "modulation_indices": groups * 3 + tags,
            "visible_end": torch.full((1, 1), tables.used, dtype=torch.int32),
        }

    def _segment_state(
        self, size: DenoiserSize, layout: DenoiserSize
    ) -> dict[str, torch.Tensor]:
        """Fill a request's segment tables: this rank's rows, every tile."""
        packing = self._segment_packing(layout)
        attention = cast(SparseAttention, self.config.attention)
        assert attention.reference_keep is not None
        tables = segment_tables(
            packing,
            num_frames=size.num_frames,
            canvas=size.canvas,
            num_text_tokens=size.num_text_tokens,
            segments=condition_segments(size.conditions, size.canvas),
            vision_spans=size.vision_spans,
            sparsity=attention.sparsity,
            reference_keep=attention.reference_keep,
        )
        interval = self._token_slice(packing.padded_tokens)
        cosine, sine = self.rotary(
            tables.position_ids,
            dtype=torch.float32,
            sequence_length=packing.padded_tokens,
        )
        groups, tags = tables.groups[interval], tables.token_tags[interval]
        return {
            "prefix_index": tables.prefix_index[interval],
            "cos": cosine.flatten(1),
            "sin": sine.flatten(1),
            # Row ``3 * group + tag`` of each step's modulation products.
            "modulation_indices": groups * 3 + tags,
            "valid_sizes": tables.valid_sizes,
            "tile_segments": tables.tile_segments,
            "segment_starts": tables.segment_starts,
            "segment_keep": tables.segment_keep,
        }

    def _tile_state(
        self, size: DenoiserSize, layout: DenoiserSize
    ) -> dict[str, torch.Tensor]:
        """Fill a request's tile tables over every packed row."""
        packing = self._tile_packing(size, layout)
        cosine, sine = self.rotary(
            packing.position_ids,
            dtype=torch.float32,
            sequence_length=packing.padded_tokens,
        )
        # Live prefix tiles first; the unread tail of each list repeats key
        # tile 0 so every entry names a tile of the domain.
        prefix = packing.tile_valid_sizes[: packing.prefix_tiles]
        live = torch.nonzero(prefix).flatten().to(torch.int32)
        video = torch.arange(
            packing.prefix_tiles,
            packing.prefix_tiles + packing.video_tiles,
            dtype=torch.int32,
        )
        prefix_keys = torch.zeros(packing.prefix_tiles, dtype=torch.int32)
        prefix_keys[: live.numel()] = live
        dense_keys = torch.zeros(
            packing.prefix_tiles + packing.video_tiles, dtype=torch.int32
        )
        dense_keys[: live.numel() + video.numel()] = torch.cat((live, video))
        return {
            "tile_valid_sizes": packing.tile_valid_sizes,
            "cos": cosine.flatten(1),
            "sin": sine.flatten(1),
            "prefix_key_indices": prefix_keys,
            "dense_key_indices": dense_keys,
            "prefix_count": torch.tensor([live.numel()], dtype=torch.int32),
        }

    @torch.inference_mode()
    def prepare_state(
        self,
        sizes: tuple[DenoiserSize, ...],
        *,
        layouts: tuple[DenoiserSize, ...],
        out: Mapping[str, torch.Tensor],
    ) -> None:
        """Fill the prompt-dependent tables of one request on the host.

        Under sparse attention the prompt is packed in its layout's text
        segment: text tiles past the prompt hold no valid rows, the tile
        holding the prompt's end holds only its remaining rows, and every row
        after the text segment sits at the prompt's exact length on the rotary
        timeline; the dense prefix key lists name only the prefix tiles
        holding valid rows. Under dense attention the prefix rows follow the
        generated rows contiguously and every query sees exactly the used
        rows. Under multi-segment sparse attention the prompt and the
        conditions fill their tiles from the first and the segment tables
        name each tile's segment (``packing.segment_tables``).
        """
        tables = set(self.state_buffers(layouts[0]) if layouts else ()) - set(
            self.modalities
        )
        if (
            len(sizes) != 1
            or len(layouts) != 1
            or not self.holds(layouts[0], sizes[0])
            or set(out) != tables
        ):
            raise ValueError(
                "H3 request state covers one sample's prompt-dependent "
                "tables in a layout that holds it"
            )
        if self.dense:
            values = self._dense_state(sizes[0], layouts[0])
        elif self.segmented:
            values = self._segment_state(sizes[0], layouts[0])
        else:
            values = self._tile_state(sizes[0], layouts[0])
        for name, value in values.items():
            if out[name].shape != value.shape or out[name].dtype != value.dtype:
                raise ValueError(
                    f"H3 request table {name!r} does not match its layout"
                )
            out[name].copy_(value)

    def _metadata(self, size: DenoiserSize) -> dict[str, torch.Tensor]:
        # Constants depend on the layout alone. For each modality,
        # ``local_{name}_indices`` holds its rows within this rank's shard and
        # ``{name}_indices`` records, for each local row, the row of the
        # complete native draw it takes: the video raster row, or the
        # channel-major audio row.
        size = self.layout_size(size)
        packing: SegmentPacking | DensePacking
        if self.segmented:
            packing = self._segment_packing(size)
            interval = self._token_slice(packing.padded_tokens)
            values = {}
            for name, indices, raster in (
                (
                    "video",
                    packing.video_indices,
                    packing.video_raster_indices,
                ),
                (
                    "audio",
                    packing.audio_indices,
                    torch.arange(packing.audio_rows, dtype=torch.int64),
                ),
            ):
                selected = (indices >= interval.start) & (
                    indices < interval.stop
                )
                values[f"local_{name}_indices"] = (
                    indices[selected] - interval.start
                )
                values[f"{name}_indices"] = raster[selected]
            return values
        if self.dense:
            packing = self._dense_packing(size)
            interval = self._token_slice(packing.padded_tokens)
            values = {}
            for name, start in (
                ("video", 0),
                ("audio", packing.video_rows),
            ):
                rows = self._sample_ranges(size)[name]
                values[f"local_{name}_indices"] = torch.arange(
                    start + rows.start - interval.start,
                    start + rows.stop - interval.start,
                    dtype=torch.int64,
                )
                values[f"{name}_indices"] = torch.arange(
                    rows.start, rows.stop, dtype=torch.int64
                )
            # Attention treats every padded row as one sequence whose keys
            # each query bounds by the request's used rows.
            values["sequence_lengths"] = torch.tensor(
                [packing.padded_tokens], dtype=torch.int32
            )
            values["sequence_offsets"] = torch.tensor(
                [0, packing.padded_tokens], dtype=torch.int32
            )
            return values

        # Text also records the global packed row, which ``forward`` uses to
        # gather the text features.
        tiles = self._tile_packing(size)
        interval = self._token_slice(tiles.padded_tokens)
        values = {}
        for name, indices in (
            ("text", tiles.text_indices),
            ("video", tiles.video_indices),
            ("audio", tiles.audio_indices),
        ):
            selected = (indices >= interval.start) & (indices < interval.stop)
            values[f"local_{name}_indices"] = indices[selected] - interval.start
            if name == "text":
                values["global_text_indices"] = indices[selected]
            else:
                raster = (
                    tiles.video_raster_indices
                    if name == "video"
                    else torch.arange(indices.numel(), device="cpu")
                )
                values[f"{name}_indices"] = raster[selected]

        # Each layer's modulation products form six rows: the video
        # timestep's (video, text, audio) groups, then the audio timestep's.
        # Video and text tokens read their group under the video timestep
        # (rows 0 and 1); audio tokens read the audio group under the audio
        # timestep (row 5).
        tags = tiles.token_tags[interval]
        values["modulation_indices"] = (tags == AUDIO_TAG).long() * 3 + tags
        return values

    def constant_buffers(
        self, size: DenoiserSize
    ) -> Mapping[str, BufferConfig]:
        # The native-draw indices stay on the host, where ``prepare_latents``
        # gathers the CPU draws.
        return {
            name: BufferConfig(
                tuple(value.shape),
                value.dtype,
                host=name in {"video_indices", "audio_indices"},
            )
            for name, value in self._metadata(size).items()
        }

    @torch.inference_mode()
    def prepare_constants(
        self, size: DenoiserSize, *, out: Mapping[str, torch.Tensor]
    ) -> None:
        values = self._metadata(size)
        if values.keys() != out.keys():
            raise ValueError(
                "H3 constant views must cover every declared numerical field"
            )
        for name, value in values.items():
            target = out[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(
                    f"H3 constant {name!r} has incompatible shape or dtype"
                )
            if (
                name in {"video_indices", "audio_indices"}
                and target.device.type != "cpu"
            ):
                raise ValueError(
                    "H3 native draw indices require CPU representation"
                )

        for name, value in values.items():
            out[name].copy_(value)

    def workspace_buffers(
        self, size: DenoiserSize
    ) -> Mapping[str, BufferConfig]:
        tokens = self._padded_tokens(size)
        interval = self._token_slice(tokens)
        hidden = {
            "hidden": BufferConfig(
                (
                    interval.stop - interval.start,
                    self.config.transformer.hidden_size,
                ),
                torch.bfloat16,
            )
        }
        if self.dense or self.segmented:
            return hidden
        # Query rows are split across the group that shards the q
        # projection's output tokens; keys cover every packed row. Resident
        # layers are TransformerLayers whose merged attention branches are
        # column-parallel linears.
        layer = cast(
            TransformerLayer, next(iter(self.transformer.layers.values()))
        )
        attention = layer.attention
        query = cast(
            ColumnParallelLinear, attention.projection.projections["q"]
        )
        distribution = query.output_distribution
        context = distribution.mesh.get_group(distribution.shard_axes(0))
        return {
            **hidden,
            # Adjacent layers coexist while completed residual chunks feed the
            # next projection. Two numerical scratch sets keep their key pools
            # and fine-attention products independent until final consumption.
            **{
                f"attention.{slot}.{name}": config
                for slot in range(min(2, len(self.transformer.layers)))
                for name, config in attention.workspace_buffers(
                    tokens, tokens // context.size, dtype=torch.bfloat16
                ).items()
            },
        }

    @torch.inference_mode()
    def prepare_latents(
        self, sizes, *, noise, state, constants, workspace
    ) -> None:
        """Gather this rank's canonical sample rows from native CPU draws.

        ``noise`` holds each modality's complete FP32 draw in
        ``noise_shape`` with a leading sample axis of one; ``state`` holds
        CPU FP32 views of the local samples. Video is patchified to raster
        rows first; audio draws are already channel-major timeline rows. A
        PDD student starts from its first node's noise level, so its samples
        scale the draws by that node's sigma; the other schedules start from
        unit noise.

        Raises:
            ValueError: A sample count other than one, a modality order
                other than video/audio, or a draw, view or index table with
                the wrong shape, device or dtype.
        """
        if len(sizes) != 1 or tuple(noise) != self.modalities:
            raise ValueError(
                "H3 preparation requires one ordered video/audio sample"
            )
        size = sizes[0]
        for name in self.modalities:
            source, target, indices = (
                noise[name],
                state[name],
                constants[f"{name}_indices"],
            )
            if (
                tuple(source.shape) != (1, *self.noise_shape(name, size))
                or tuple(target.shape)
                != (1, indices.numel(), self.latent_shape(name, size)[1])
                or source.device.type != "cpu"
                or target.device.type != "cpu"
                or indices.device.type != "cpu"
                or source.dtype != torch.float32
                or target.dtype != torch.float32
                or indices.dtype != torch.int64
            ):
                raise ValueError(
                    "H3 initialization requires complete native CPU FP32 "
                    "draws and local sample views"
                )
        video = patchify_video(noise["video"][0])[0]
        starts = (
            self.make_schedules(None, shift=None, device="cpu")
            if isinstance(self.config.grids["video"], BlockGrid)
            else None
        )
        for name, source in (("video", video), ("audio", noise["audio"][0])):
            torch.index_select(
                source, 0, constants[f"{name}_indices"], out=state[name][0]
            )
            if starts is not None:
                state[name][0].mul_(float(starts[name].sigmas[0]))

    @torch.inference_mode()
    def forward(self, inputs: DenoiserInput, *, state, constants, workspace):
        """Predict one layout sample's video and audio velocities.

        The first pipeline stage writes the prefix rows and projected latents
        into the local packed rows of ``workspace["hidden"]``; ``Transformer``
        fills it with the preceding stage's output on later stages. The final
        stage maps each modality to a one-element tuple holding an FP32
        ``TensorOutput`` of this rank's rows; every other stage maps each
        modality to ``(None,)``.
        """
        # The step is device data; the schedule that named it checked that it
        # lies on the checkpoint schedule.
        if inputs.batch_size != 1:
            raise ValueError("H3 denoising requires one sample")
        size = inputs.sizes[0]
        if size != self.layout_size(size):
            raise ValueError(
                "H3 denoising evaluates a layout; prompt-dependent tables are "
                "request state"
            )
        attention: AttentionInput | SequenceInput | SegmentInput
        packing: DensePacking | SegmentPacking
        if self.dense:
            packing = self._dense_packing(size)
            tokens = packing.padded_tokens
            lengths = SequenceLengths(
                constants["sequence_lengths"],
                constants["sequence_offsets"],
                (tokens,),
            )
            attention = SequenceInput(
                self._token_slice(tokens),
                self._sequence_group(),
                VisibleInput(
                    lengths,
                    lengths,
                    state["visible_end"],
                    None,
                    prefix_bounds=True,
                    fully_visible=False,
                ),
                constants["local_video_indices"],
                constants["local_audio_indices"],
            )
            prefix_rows = self.text_condition_rows(size)
        elif self.segmented:
            packing = self._segment_packing(size)
            tokens = packing.padded_tokens
            attention = SegmentInput(
                self._token_slice(tokens),
                self._sequence_group(),
                vsa.Segments(
                    packing.tile,
                    tokens,
                    state["valid_sizes"],
                    state["tile_segments"],
                    state["segment_starts"],
                    state["segment_keep"],
                ),
                constants["local_video_indices"],
                constants["local_audio_indices"],
            )
            prefix_rows = self.text_condition_rows(size)
        else:
            tiles = self._tile_packing(size)
            tokens = tiles.padded_tokens
            attention = AttentionInput(
                tiles,
                self._token_slice(tokens),
                self._sequence_group(),
                vsa.Input(
                    tiles.padded_tokens,
                    tiles.prefix_tiles,
                    tiles.video_tiles,
                    tiles.prefix_tiles + tiles.video_tiles,
                    state["tile_valid_sizes"],
                    state["prefix_key_indices"],
                    state["dense_key_indices"],
                    state["prefix_count"],
                ),
                constants["local_text_indices"],
                constants["local_video_indices"],
                constants["local_audio_indices"],
            )
            prefix_rows = size.num_text_tokens
        interval = attention.token_slice
        hidden = workspace["hidden"]
        width = self.config.transformer.hidden_size
        if (
            hidden.shape != (interval.stop - interval.start, width)
            or hidden.dtype != torch.bfloat16
        ):
            raise ValueError(
                "H3 hidden workspace must match the local BF16 token shard"
            )

        pipeline = self.mesh.get_group("pp" if "pp" in self.mesh.axes else ())
        if pipeline.rank == 0:
            text = inputs.text_features[0]
            if text.shape != (prefix_rows, width):
                raise ValueError(
                    "H3 text features must be refined tokens "
                    "with the declared width"
                )
            if self.dense or self.segmented:
                # Every local row gathers its prefix source row; generated
                # and padding rows gather the zero row.
                torch.index_select(text, 0, state["prefix_index"], out=hidden)
            else:
                # Scatter text into the packed token rows; padding rows stay
                # zero. Text occupies the leading packed rows, so a text
                # row's global index is also its row in ``text``. Text
                # features cover the layout's text rows, zero past the
                # prompt, so every prompt length within one layout gathers
                # the same rows.
                hidden.zero_()
                hidden.index_copy_(
                    0,
                    constants["local_text_indices"],
                    text.index_select(0, constants["global_text_indices"]).to(
                        hidden.dtype
                    ),
                )
            # Scatter projected latents into their local rows.
            for name, projection, indices in (
                (
                    "video",
                    self.transformer.video_input,
                    attention.local_video_indices,
                ),
                (
                    "audio",
                    self.transformer.audio_input,
                    attention.local_audio_indices,
                ),
            ):
                sample = inputs.latents[name][0].tensor
                if (
                    sample.shape
                    != (indices.numel(), self.latent_shape(name, size)[1])
                    or sample.dtype != torch.float32
                ):
                    raise ValueError(
                        "H3 latent input must supply its local canonical "
                        "FP32 sample"
                    )
                hidden.index_copy_(
                    0,
                    indices,
                    projection(sample, output_dtype=torch.float32).to(
                        hidden.dtype
                    ),
                )

        tables = (
            state
            if self.dense or self.segmented
            else {
                **state,
                "modulation_indices": constants["modulation_indices"],
            }
        )
        predictions = self.transformer(
            hidden,
            attention,
            step=inputs.step,
            tables={
                "modulation_indices": tables["modulation_indices"],
                "cos": state["cos"],
                "sin": state["sin"],
            },
            workspace=workspace,
        )
        if not predictions:
            return dict.fromkeys(self.modalities, (None,))

        layouts = self.output_layout(size)
        return {
            name: (TensorOutput(value, layouts[name]),)
            for name, value in zip(self.modalities, predictions, strict=True)
        }
