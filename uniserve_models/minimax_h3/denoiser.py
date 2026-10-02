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

Sparse attention packs ``packing.TilePacking`` (text in whole 64-row tiles,
the media timeline at the exact prompt length). Dense attention packs
``packing.DensePacking``: the generated rows lead at fixed offsets, so a
shard's sample rows are a layout constant, and each rank gathers its prefix
rows (text, then conditions) from the request's prefix source.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch

from uniserve.diffusion import (
    CleanSampleEulerSolver,
    Schedule,
    block_grid,
    ladder,
    uniform_grid,
)
from uniserve.media import image
from uniserve.model import LatentInput, VideoDenoiser
from uniserve.nn import ColumnParallelLinear, RotaryEmbedding
from uniserve.nn.attention import SequenceLengths, VisibleInput, vsa
from uniserve.tensors import BufferConfig, OutputLayout, TensorOutput

from .conditioning import Conditioner
from .config import (
    NAMED_ASPECT_RATIOS,
    DenoiserConfig,
    DenseAttention,
    DmdLadder,
    PddGrid,
    UniformGrid,
    canvas,
)
from .inputs import AttentionInput, DenoiserInput, DenoiserSize, SequenceInput
from .packing import (
    AUDIO_TAG,
    DensePacking,
    TilePacking,
    audio_latent_frames,
    dense_packing,
    dense_tables,
    latent_raster,
    patchify_video,
    tile_packing,
    video_latent_frames,
)
from .transformer import Transformer, TransformerLayer

# A visual condition is held at this clean time for every step whose
# generated video is noisier, as the released model was trained.
VISUAL_CONDITION_TIME = 0.999
# An audio reference conditions at the clean endpoint.
AUDIO_CONDITION_TIME = 1.0


def schedules(
    config: DenoiserConfig, *, device: torch.device | str
) -> Mapping[str, Schedule]:
    """Build the checkpoint's video and audio schedules.

    The uniform grid follows the released diffusers schedule
    (``uniserve.diffusion.uniform_grid``); a DMD ladder shifts its trained
    rungs once per modality (``ladder``); a PDD grid evaluates its block
    nodes (``block_grid``).
    """
    schedule = config.schedule
    result = {}
    for name, shift in (
        ("video", schedule.video_shift),
        ("audio", schedule.audio_shift),
    ):
        if isinstance(schedule, UniformGrid):
            result[name] = uniform_grid(
                schedule.points, shift=shift, device=device
            )
        elif isinstance(schedule, DmdLadder):
            result[name] = ladder(
                schedule.rungs,
                shift=shift,
                clock=schedule.clock,
                device=device,
            )
        else:
            result[name] = block_grid(
                schedule.intervals,
                schedule.nodes,
                shift=shift,
                max_t=schedule.max_t,
                device=device,
            ).schedule
    if result["video"].num_steps != result["audio"].num_steps:
        raise ValueError("H3 video and audio schedules must align")
    return result


def timestep_groups(config: DenoiserConfig) -> int:
    """Count the timestep groups a step of this denoiser reads.

    Text-only denoisers read the generated video and audio timesteps; a
    conditioned denoiser also reads the visual-condition and audio-reference
    levels.
    """
    return 2 if config.tasks == ("t2va",) else 4


def modulation_timesteps(
    config: DenoiserConfig, values: Mapping[str, Schedule]
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """Enumerate each step's group timesteps and their distinct entries.

    Returns the per-step FP32 group timesteps (one [groups] tensor per
    step, video first), the distinct FP32 values in first-occurrence order of
    that step-major enumeration, and the [step, group] int64 entry of each.
    Distinct means distinct FP32 bit patterns, so an entry's products are
    exactly those of every occurrence.
    """
    groups = timestep_groups(config)
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
    entries = []
    table = torch.empty((len(steps), groups), dtype=torch.int64, device="cpu")
    for step, row in enumerate(steps):
        for group, value in enumerate(row):
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
        )
        self.config = config
        host = schedules(config, device="cpu")
        _, entries, table = modulation_timesteps(config, host)
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
        """Canvases a deployment prepares layouts for.

        A checkpoint that generates only some canvases prepares those; one
        that follows the canvas rule prepares every named aspect ratio's
        canvas, and other canvases evaluate unprepared.
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
    def schedule_shifts(self) -> Mapping[str, float]:
        schedule = self.config.schedule
        return {"video": schedule.video_shift, "audio": schedule.audio_shift}

    @property
    def fixed_canvases(self) -> tuple[image.Config, ...] | None:
        return self.config.canvases

    @property
    def max_sequence_rows(self) -> int | None:
        return self.config.max_sequence_rows

    @property
    def dense(self) -> bool:
        """Whether this denoiser attends densely (see ``DensePacking``)."""
        return isinstance(self.config.attention, DenseAttention)

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
        condition_rows: int,
    ) -> DenoiserSize:
        """Build this network's size descriptor for one admitted request.

        Raises:
            ValueError: A canvas the checkpoint does not generate, or an
                invalid frame count, prompt length or condition count.
        """
        canvases = self.config.canvases
        if canvases is not None and canvas not in canvases:
            raise ValueError(
                f"this H3 checkpoint generates only {canvases}, not {canvas}"
            )
        return DenoiserSize(num_frames, canvas, num_text_tokens, condition_rows)

    def layout_size(self, size: DenoiserSize) -> DenoiserSize:
        """Return the smallest layout that holds ``size``.

        Text and conditions occupy whole 64-row tiles, so the smallest
        regions are the prompt's and the conditions' own tiles.
        """
        return DenoiserSize(
            size.num_frames,
            size.canvas,
            math.ceil(size.num_text_tokens / 64) * 64,
            math.ceil(size.condition_rows / 64) * 64,
        )

    def holds(self, layout: DenoiserSize, size: DenoiserSize) -> bool:
        """Whether ``layout`` evaluates a request of ``size``.

        A layout holds every prompt and condition set that fits its regions
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
    def num_steps(self) -> int:
        """Number of network evaluations of the checkpoint's schedule."""
        return schedules(self.config, device="cpu")["video"].num_steps

    @property
    def text_condition_width(self) -> int:
        """Feature width of the retained text conditioning."""
        return self.config.transformer.hidden_size

    def text_condition_rows(self, layout: DenoiserSize) -> int:
        """Rows of a request's retained conditioning in ``layout``.

        Sparse attention retains the refined text over the layout's text
        rows. Dense attention retains the prefix source: the text rows, the
        projected condition rows, and one zero row the generated and padding
        rows gather.
        """
        if not self.dense:
            return layout.num_text_tokens
        return layout.num_text_tokens + layout.condition_rows + 1

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
        # Packs ``size``'s prompt in ``layout``'s text region (its own tiles
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

    def _padded_tokens(self, layout: DenoiserSize) -> int:
        if self.dense:
            return self._dense_packing(layout).padded_tokens
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

    def make_schedules(
        self, steps: int, *, shift: float | None, device
    ) -> Mapping[str, Schedule]:
        if (
            type(steps) is not int
            or steps != self.num_steps
            or shift is not None
        ):
            raise ValueError(
                f"H3 requires {self.num_steps} evaluations with its trained "
                "modality shifts"
            )
        return schedules(self.config, device=device)

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
        tiles = self._tile_packing(size)
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
        audio and video regions lie, hence which of their rows fall in this
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
        (``visible_end``). Their shapes follow the layout alone.
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
        if self.dense:
            packing = self._dense_packing(size)
            rows = packing.padded_tokens // self._sequence_group().size
            return {
                **samples,
                "prefix_index": BufferConfig((rows,), torch.int64),
                "cos": BufferConfig((rows, width), torch.float32),
                "sin": BufferConfig((rows, width), torch.float32),
                "modulation_indices": BufferConfig((rows,), torch.int64),
                "visible_end": BufferConfig(
                    (1, packing.padded_tokens), torch.int32
                ),
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
            "visible_end": torch.full(
                (1, packing.padded_tokens), tables.used, dtype=torch.int32
            ),
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
        region: text tiles past the prompt hold no valid rows, the tile
        holding the prompt's end holds only its remaining rows, and every row
        after the text region sits at the prompt's exact length on the rotary
        timeline; the dense prefix key lists name only the prefix tiles
        holding valid rows. Under dense attention the prefix rows follow the
        generated rows contiguously and every query sees exactly the used
        rows.
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
        values = (
            self._dense_state(sizes[0], layouts[0])
            if self.dense
            else self._tile_state(sizes[0], layouts[0])
        )
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
        if self.dense:
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
            schedules(self.config, device="cpu")
            if isinstance(self.config.schedule, PddGrid)
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
        if self.dense:
            packing = self._dense_packing(size)
            tokens = packing.padded_tokens
            lengths = SequenceLengths(
                constants["sequence_lengths"],
                constants["sequence_offsets"],
                (tokens,),
            )
            attention: AttentionInput | SequenceInput = SequenceInput(
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
            if self.dense:
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
            if self.dense
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
