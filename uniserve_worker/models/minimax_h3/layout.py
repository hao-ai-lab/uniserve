"""Tensor geometry and mathematical views for H3 computation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import torch

from uniserve_worker.nn.mesh import EntryBindings

from ...backends.attention.video_sparse import video_sparse_selected_tiles
from ...execution.batch import (
    DecodeRange,
    DeviceDim,
    DType,
    MediaGeometry,
    MediaTrack,
    ShapeBound,
    StaticDim,
    TensorSpec,
)
from ...execution.bounded_storage import BoundedTensorStorage, TensorSchema
from ...nn.mesh import DeviceMesh
from ...nn.parallel_attention import AttentionContextWorkspace
from ...transfer.layout import TensorRegion
from ..runtime import TensorOutputLayout
from .encoder import H3TextEncoderConfig
from .packing import H3PackedLayout, build_packed_layout, patchify_video

__all__ = [
    "H3Layout",
    "H3Scratch",
    "H3Tensors",
    "bind_request_tensors",
    "entry_output_schema",
    "tensor_output_layout",
    "warmup_geometries",
    "MIN_H3_FRAMES",
]

PROFILE_FPS = 24
PROFILE_AUDIO_RATE = 32_000
MIN_H3_FRAMES = 22


def entry_output_schema(layout: H3Layout) -> dict[str, tuple[TensorSpec, ...]]:
    """Declare bounded products for every H3 computation entry."""

    return {
        "text_encoder": (
            TensorSpec(
                "conditioning",
                DType.BF16,
                ShapeBound(
                    (
                        StaticDim(1),
                        DeviceDim(int(layout.packed.text_indices.numel())),
                        StaticDim(H3TextEncoderConfig().hidden_size),
                    )
                ),
            ),
            *(
                (
                    TensorSpec(
                        "presentation_tags",
                        DType.I64,
                        ShapeBound((DeviceDim(int(layout.packed.text_indices.numel())),)),
                    ),
                )
                if layout.packed.reference_shape is not None
                else ()
            ),
        ),
        "denoiser": (
            TensorSpec(
                "video_latents",
                DType.F32,
                ShapeBound((DeviceDim(int(layout.packed.video_indices.numel())), StaticDim(96))),
            ),
            TensorSpec(
                "audio_latents",
                DType.F32,
                ShapeBound((DeviceDim(int(layout.packed.audio_indices.numel())), StaticDim(32))),
            ),
        ),
        "video_decoder": (
            TensorSpec(
                "video_segments",
                DType.F32 if layout.video_dtype == torch.float32 else DType.F16,
                ShapeBound(
                    (
                        DeviceDim(layout.video_reconstruction_units),
                        StaticDim(1),
                        StaticDim(3),
                        StaticDim(25),
                        StaticDim(layout.height),
                        StaticDim(layout.width),
                    )
                ),
            ),
        ),
        "audio_decoder": (
            TensorSpec(
                "audio_samples",
                DType.I16,
                ShapeBound(
                    (
                        DeviceDim(round(layout.frame_count * PROFILE_AUDIO_RATE / PROFILE_FPS)),
                        StaticDim(2),
                    )
                ),
            ),
        ),
    }


def reconstruction_unit_frames(frames: int) -> tuple[int, ...]:
    """Convert generated frame count into overlapping decoder reconstruction units."""

    if frames < MIN_H3_FRAMES or frames % 17 != 5:
        raise ValueError("H3 frame count must have the form 17 * n + 5")
    units = (frames - 5) // 17
    return (17,) * (units - 1) + (MIN_H3_FRAMES,)


@dataclass(frozen=True, slots=True)
class H3Layout:
    """Combines packed multimodal geometry with sequence-parallel rank ownership."""

    packed: H3PackedLayout
    sp_rank: int
    sp_size: int
    tp_size: int
    ulysses_size: int
    sequence_kind: str
    context_col_size: int
    denoiser_participant: bool
    output_owner: bool
    local_start: int
    local_end: int
    frame_count: int
    reconstruction_unit_frames: tuple[int, ...]
    sparsity: float = 0.9
    attention_backend: str = "VIDEO_SPARSE_ATTN"
    attention: str = "vsa"
    video_dtype: torch.dtype = torch.float16
    local_video_rows: int = field(init=False)
    local_audio_rows: int = field(init=False)

    def __post_init__(self) -> None:
        """Resolve modality extents once for this immutable packed layout.

        Request tensor binding uses these extents on every bounded operation.
        Filtering the packed CPU indices there would allocate masks and invoke
        the CPU tensor thread pool between consecutive device submissions.
        """

        object.__setattr__(
            self, "local_video_rows", int(self.local_indices(self.packed.video_indices).numel())
        )
        object.__setattr__(
            self, "local_audio_rows", int(self.local_indices(self.packed.audio_indices).numel())
        )

    @classmethod
    def build(
        cls,
        bindings: EntryBindings,
        *,
        frames: int,
        text_rows: int,
        audio_frames: int,
        height: int = 768,
        width: int = 1344,
        sparsity: float = 0.9,
        attention_backend: str = "VIDEO_SPARSE_ATTN",
        attention: str = "vsa",
        video_dtype: torch.dtype = torch.float16,
        reference_shape: tuple[int, int] | None = None,
        presentation_tags: torch.Tensor | None = None,
    ) -> "H3Layout":
        """Partition one packed request evenly across the mesh sequence ranks."""

        if attention not in {"vsa", "dense"}:
            raise ValueError(f"unsupported H3 attention {attention!r}")
        mesh = bindings.meshes.get("denoiser")
        entry = bindings.entries.get("denoiser")
        config = None if entry is None else entry.parallel_config
        size = 1 if config is None else config.sequence_parallel_size
        rank = mesh.coord("sp") if mesh is not None else 0
        packed = build_packed_layout(
            text_rows=text_rows,
            num_frames=frames,
            height=height,
            width=width,
            audio_frames=audio_frames,
            reference_shape=reference_shape,
            presentation_tags=presentation_tags,
            # Four logical row partitions define tensorwise activation scales;
            # physical sequence ownership does not redefine those domains.
            row_multiple=64 * math.lcm(4, size),
        )
        shard = packed.padded_rows // size
        return cls(
            packed=packed,
            sparsity=sparsity,
            attention_backend=attention_backend,
            sp_rank=rank,
            sp_size=size,
            tp_size=1 if config is None else config.tensor_parallel_size,
            ulysses_size=1 if config is None else dict(config.dimensions)["ulysses"],
            sequence_kind="local" if config is None else config.sequence_parallel.kind,
            context_col_size=1 if config is None else dict(config.dimensions).get("cp_col", 1),
            denoiser_participant=mesh is not None,
            output_owner=bindings.owns("output"),
            local_start=rank * shard if mesh is not None else 0,
            local_end=(rank + 1) * shard if mesh is not None else 0,
            frame_count=int(frames),
            reconstruction_unit_frames=reconstruction_unit_frames(int(frames)),
            attention=attention,
            video_dtype=video_dtype,
        )

    @property
    def height(self) -> int:
        """Target raster height; the video VAE compresses space by sixteen."""

        return self.packed.latent_height * 16

    @property
    def width(self) -> int:
        """Target raster width in pixels."""

        return self.packed.latent_width * 16

    @property
    def video_rows_per_frame(self) -> int:
        """Transformer rows per latent frame, using the 2×2 spatial patch."""

        return (self.packed.latent_height // 2) * (self.packed.latent_width // 2)

    @property
    def attention_rows(self) -> int:
        """Rows owned by one context coordinate after Ulysses head exchange."""

        return self.packed.padded_rows // self.sp_size * self.ulysses_size

    @property
    def attention_start(self) -> int:
        return (self.sp_rank // self.ulysses_size) * self.attention_rows

    @property
    def attention_video_tiles(self) -> int:
        start = self.attention_start // 64
        end = start + self.attention_rows // 64
        prefix = self.packed.prefix_tiles
        return max(0, min(end, prefix + self.packed.video_tiles) - max(start, prefix))

    @property
    def shape_key(self) -> tuple:
        """Identify target geometry, padded text rows, and optional reference spans."""

        key = (
            self.frame_count,
            int(self.packed.text_indices.numel()),
            int(self.packed.audio_frames),
            self.height,
            self.width,
        )
        if self.packed.reference_shape is None:
            return key
        return (*key, self.packed.reference_shape, tuple(self.packed.presentation_tags.tolist()))

    @property
    def local_rows(self) -> int:
        """Count packed transport rows owned by this sequence rank."""

        return self.local_end - self.local_start

    @property
    def video_reconstruction_units(self) -> int:
        """Count overlapping temporal segments needed to reconstruct the video."""

        return len(self.reconstruction_unit_frames)

    def local_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Filter global packed indices to this rank and convert them to local offsets."""

        selected = indices[(indices >= self.local_start) & (indices < self.local_end)]
        return selected - self.local_start

    @property
    def local_video_raster_indices(self) -> torch.Tensor:
        """Map this rank's tile-major video rows back to global raster order."""

        selected = (self.packed.video_indices >= self.local_start) & (
            self.packed.video_indices < self.local_end
        )
        return self.packed.video_raster_indices[selected]

    @property
    def local_audio_raster_indices(self) -> torch.Tensor:
        """Map this rank's packed audio rows to the unsharded channel-major timeline."""

        selected = (self.packed.audio_indices >= self.local_start) & (
            self.packed.audio_indices < self.local_end
        )
        return torch.arange(self.packed.audio_indices.numel(), dtype=torch.long)[selected]


def request_tensor_schema(layout: H3Layout) -> dict[str, TensorSchema]:
    """Declare capacity shapes and representations for conditioning and media math.

    A short request can move a modality across a rank boundary, so modality
    capacity covers any such bindings within the maximum local row extent.
    Slot count, allocation, and device budgets belong to public execution.
    """

    packed = layout.packed
    denoiser = layout.denoiser_participant
    text = int(packed.text_indices.numel()) if denoiser else 0
    rows = packed.padded_rows if denoiser else 0
    prefix = packed.prefix_tiles if denoiser else 0
    dense = packed.prefix_tiles + packed.video_tiles if denoiser else 0
    return {
        "video_noise": TensorSchema(
            (int(denoiser), 24, packed.video_frames, packed.latent_height, packed.latent_width),
            torch.float32,
            memory="pinned",
        ),
        "audio_noise": TensorSchema(
            (packed.audio_indices.numel() if denoiser else 0, 32), torch.float32, memory="pinned"
        ),
        "video_source": TensorSchema(
            (min(packed.video_indices.numel(), layout.local_rows) if denoiser else 0, 96),
            torch.float32,
            memory="pinned",
        ),
        "audio_source": TensorSchema(
            (min(packed.audio_indices.numel(), layout.local_rows) if denoiser else 0, 32),
            torch.float32,
            memory="pinned",
        ),
        "text_condition": TensorSchema((1, text, 5376), torch.bfloat16),
        "reference_rows": TensorSchema(
            (min(packed.reference_indices.numel(), layout.local_rows) if denoiser else 0, 96),
            torch.float32,
        ),
        "video_rows": TensorSchema(
            (min(packed.video_indices.numel(), layout.local_rows), 96), torch.float32
        ),
        "audio_rows": TensorSchema(
            (min(packed.audio_indices.numel(), layout.local_rows), 32), torch.float32
        ),
        "tile_valid_sizes": TensorSchema(
            (packed.tile_valid_sizes.numel() if denoiser else 0,), torch.int32
        ),
        "prefix_key_indices": TensorSchema((prefix,), torch.int32),
        "dense_key_indices": TensorSchema((dense,), torch.int32),
        "prefix_count": TensorSchema((), torch.int32),
        "rotary_cosine": TensorSchema((rows, 96), torch.float32),
        "rotary_sine": TensorSchema((rows, 96), torch.float32),
        "video_overlap": TensorSchema(
            (int(layout.output_owner), 3, 5, layout.height, layout.width),
            layout.video_dtype,
        ),
    }


@dataclass(frozen=True, slots=True)
class H3Tensors:
    """Shape-bound inputs and outputs borrowed from public request storage.

    The layout is immutable; tensor contents are updated by the numerical
    operations. Admission, progress, and reclamation belong to RequestPool.
    """

    layout: H3Layout
    video_noise: torch.Tensor
    audio_noise: torch.Tensor
    video_source: torch.Tensor
    audio_source: torch.Tensor
    text_condition: torch.Tensor
    reference_rows: torch.Tensor
    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    tile_valid_sizes: torch.Tensor
    prefix_key_indices: torch.Tensor
    dense_key_indices: torch.Tensor
    prefix_count: torch.Tensor
    rotary_cosine: torch.Tensor
    rotary_sine: torch.Tensor
    video_overlap: torch.Tensor


def bind_request_tensors(storage: BoundedTensorStorage, layout: H3Layout) -> H3Tensors:
    """Borrow validated views without allocating or taking ownership of buffers."""

    denoiser = layout.denoiser_participant
    packed = layout.packed
    views = storage.bind(
        layout.shape_key,
        {
            "video_noise": (
                int(denoiser),
                24,
                packed.video_frames,
                packed.latent_height,
                packed.latent_width,
            ),
            "audio_noise": (packed.audio_indices.numel() if denoiser else 0, 32),
            "video_source": (layout.local_video_rows if denoiser else 0, 96),
            "audio_source": (layout.local_audio_rows if denoiser else 0, 32),
            "text_condition": (1, int(packed.text_indices.numel()) if denoiser else 0, 5376),
            "reference_rows": (int(layout.local_indices(packed.reference_indices).numel()), 96),
            "video_rows": (layout.local_video_rows, 96),
            "audio_rows": (layout.local_audio_rows, 32),
            "tile_valid_sizes": (int(packed.tile_valid_sizes.numel()) if denoiser else 0,),
            "prefix_key_indices": (int(packed.prefix_tiles) if denoiser else 0,),
            "dense_key_indices": (
                int(packed.prefix_tiles + packed.video_tiles) if denoiser else 0,
            ),
            "prefix_count": (),
            "rotary_cosine": (packed.padded_rows if denoiser else 0, 96),
            "rotary_sine": (packed.padded_rows if denoiser else 0, 96),
            "video_overlap": (int(layout.output_owner), 3, 5, layout.height, layout.width),
        },
    )
    return H3Tensors(
        layout=layout,
        video_noise=views["video_noise"],
        audio_noise=views["audio_noise"],
        video_source=views["video_source"],
        audio_source=views["audio_source"],
        text_condition=views["text_condition"],
        reference_rows=views["reference_rows"],
        video_rows=views["video_rows"],
        audio_rows=views["audio_rows"],
        tile_valid_sizes=views["tile_valid_sizes"],
        prefix_key_indices=views["prefix_key_indices"],
        dense_key_indices=views["dense_key_indices"],
        prefix_count=views["prefix_count"],
        rotary_cosine=views["rotary_cosine"],
        rotary_sine=views["rotary_sine"],
        video_overlap=views["video_overlap"],
    )


@dataclass(frozen=True, slots=True)
class H3Scratch:
    """Borrowed fixed-address intermediates for H3 projection, attention and modulation."""

    packed_hidden: torch.Tensor
    local_text_hidden: torch.Tensor
    projected_input: torch.Tensor
    projected_input_bf16: torch.Tensor
    video_velocity: torch.Tensor
    audio_velocity: torch.Tensor
    projection_peers: tuple[torch.Tensor, ...]
    projection_sync_input: torch.Tensor
    projection_sync_output: torch.Tensor
    attention_workspace: torch.Tensor
    attention_output: torch.Tensor
    context_workspace: AttentionContextWorkspace | None
    tile_scores: torch.Tensor
    block_indices: torch.Tensor
    block_counts: torch.Tensor
    pooled_query: torch.Tensor
    pooled_key: torch.Tensor
    pooled_value: torch.Tensor
    compressed_tiles: torch.Tensor
    topk_indices_i32: torch.Tensor
    block_adaln_params: torch.Tensor
    final_adaln_params: torch.Tensor
    rotary_positions: torch.Tensor
    rotary_frequencies: torch.Tensor


@dataclass(frozen=True, slots=True)
class H3MediaScratch:
    """Numerical reconstruction views borrowed from public execution storage."""

    video_input: torch.Tensor
    reconstruction_rows: torch.Tensor
    rgb_round: torch.Tensor
    audio_latents: torch.Tensor
    video_raster_order: torch.Tensor


def scratch_tensor_schema(
    layout: H3Layout,
    mesh: DeviceMesh,
    *,
    block_params_shape: tuple[int, ...],
    final_params_shape: tuple[int, ...],
    attention_workspace_dtype: torch.dtype,
) -> dict[str, TensorSchema]:
    """Declare intermediate capacities and the attention algorithm's shared storage."""

    rows = layout.local_rows
    heads = 56 // (layout.tp_size * layout.ulysses_size)
    global_rows = layout.packed.padded_rows
    tiles = (global_rows + 63) // 64
    query_rows = layout.attention_rows
    query_tiles = query_rows // 64
    local_text = min(int(layout.packed.text_indices.numel()), rows)
    local_video = min(int(layout.packed.video_indices.numel()), rows)
    local_audio = min(int(layout.packed.audio_indices.numel()), rows)
    projected = max(local_video, local_audio, min(layout.packed.reference_indices.numel(), rows))
    gather_group = (
        mesh.get_group("sp")
        if mesh.size("tp") == 1 and mesh.size("cp") == 1 and mesh.size("sp") > 1
        else None
    )
    output_group = (
        mesh.get_group("ulysses") if mesh.size("cp") == 1 and mesh.size("ulysses") > 1 else None
    )
    schema = {
        "packed_hidden": TensorSchema((1, rows, 5376), torch.bfloat16),
        "local_text_hidden": TensorSchema((1, local_text, 5376), torch.bfloat16),
        "projected_input": TensorSchema((projected, 5376), torch.float32),
        "projected_input_bf16": TensorSchema((projected, 5376), torch.bfloat16),
        "video_velocity": TensorSchema((local_video, 96), torch.float32),
        "audio_velocity": TensorSchema((local_audio, 32), torch.float32),
        "projection": TensorSchema(
            (rows, 56 // layout.tp_size, 128),
            torch.bfloat16,
            memory="symmetric",
            group=mesh.get_group("ulysses"),
        ),
        "projection_sync_input": TensorSchema((1,), torch.int32, fill=layout.sp_rank),
        "projection_sync_output": TensorSchema((layout.ulysses_size,), torch.int32),
        "attention_workspace": TensorSchema(
            (global_rows * 5376,),
            attention_workspace_dtype,
            memory="symmetric" if gather_group is not None else "device",
            group=gather_group,
        ),
        "attention_output": TensorSchema(
            (query_rows, heads, 128),
            torch.bfloat16,
            memory="symmetric" if output_group is not None else "device",
            group=output_group,
        ),
        "tile_scores": TensorSchema((heads, query_tiles, tiles), torch.float32),
        "block_indices": TensorSchema(
            (heads, query_tiles, layout.packed.prefix_tiles + layout.packed.video_tiles),
            torch.int32,
        ),
        "block_counts": TensorSchema((heads, query_tiles), torch.int32),
        "pooled_query": TensorSchema((query_tiles, heads, 128), torch.float32),
        "pooled_key": TensorSchema((tiles, heads, 128), torch.float32),
        "pooled_value": TensorSchema((tiles, heads, 128), torch.float32),
        "compressed_tiles": TensorSchema((heads, query_tiles, 128), torch.float32),
        "topk_indices_i32": TensorSchema(
            (
                heads,
                query_tiles,
                video_sparse_selected_tiles(layout.packed.video_tiles, layout.sparsity),
            ),
            torch.int32,
        ),
        "block_adaln_params": TensorSchema(block_params_shape, torch.bfloat16),
        "final_adaln_params": TensorSchema(final_params_shape, torch.bfloat16),
        "rotary_positions": TensorSchema((global_rows, 3), torch.float32),
        "rotary_frequencies": TensorSchema((global_rows, 3, 16), torch.float32),
    }
    if layout.attention == "dense":
        # Dense providers do not score, pool, select, or compress video tiles.
        for name in (
            "tile_scores",
            "block_indices",
            "block_counts",
            "pooled_query",
            "pooled_key",
            "pooled_value",
            "compressed_tiles",
            "topk_indices_i32",
        ):
            schema[name] = TensorSchema((0,), schema[name].dtype)
    return schema


def media_tensor_schema(layout: H3Layout, bindings: EntryBindings) -> dict[str, TensorSchema]:
    """Declare only the local numerical decoder and output intermediates."""

    video = bindings.owns("video_decoder")
    audio = bindings.owns("audio_decoder")
    output = bindings.owns("output")
    return {
        "video_input": TensorSchema(
            (int(video), 24, 7, layout.packed.latent_height, layout.packed.latent_width),
            torch.float32,
        ),
        "reconstruction_rows": TensorSchema(
            (7 * layout.video_rows_per_frame if video else 0, 96), torch.float32
        ),
        "rgb_round": TensorSchema(
            (layout.frame_count if output else 0, layout.height, layout.width, 3), torch.uint8
        ),
        "audio_latents": TensorSchema(
            (2 if audio else 0, 32, layout.packed.audio_frames), torch.float32
        ),
    }


def bind_compute_tensors(
    storage: BoundedTensorStorage,
    layout: H3Layout,
    mesh: DeviceMesh | None,
    context: AttentionContextWorkspace | None,
) -> tuple[H3Scratch | None, H3MediaScratch]:
    """Bind explicit borrowed views without allocating execution storage in the model."""

    shapes = {name: tuple(value.shape) for name, value in storage.capacity.items()}
    if mesh is not None:
        local_rows = int(layout.local_rows)
        global_rows = int(layout.packed.padded_rows)
        local_heads = 56 // (layout.tp_size * layout.ulysses_size)
        tiles = global_rows // 64
        query_rows = layout.attention_rows
        query_tiles = query_rows // 64
        prefix_width = int(layout.packed.prefix_tiles + layout.packed.video_tiles)
        local_text = int(layout.local_indices(layout.packed.text_indices).numel())
        local_video = int(layout.local_video_rows)
        local_audio = int(layout.local_audio_rows)
        projected_rows = max(
            local_video,
            local_audio,
            int(layout.local_indices(layout.packed.reference_indices).numel()),
        )
        shapes.update(
            {
                "packed_hidden": (1, local_rows, 5376),
                "local_text_hidden": (1, local_text, 5376),
                "projected_input": (projected_rows, 5376),
                "projected_input_bf16": (projected_rows, 5376),
                "video_velocity": (local_video, 96),
                "audio_velocity": (local_audio, 32),
                "attention_workspace": (global_rows * 5376,),
                "attention_output": (query_rows, local_heads, 128),
                "tile_scores": (local_heads, query_tiles, tiles),
                "block_indices": (local_heads, query_tiles, prefix_width),
                "block_counts": (local_heads, query_tiles),
                "pooled_query": (query_tiles, local_heads, 128),
                "pooled_key": (tiles, local_heads, 128),
                "pooled_value": (tiles, local_heads, 128),
                "compressed_tiles": (local_heads, query_tiles, 128),
                "topk_indices_i32": (
                    local_heads,
                    layout.attention_video_tiles,
                    video_sparse_selected_tiles(layout.packed.video_tiles, layout.sparsity),
                ),
                "rotary_positions": (global_rows, 3),
                "rotary_frequencies": (global_rows, 3, 16),
            }
        )
    if layout.attention == "dense":
        for name in (
            "tile_scores",
            "block_indices",
            "block_counts",
            "pooled_query",
            "pooled_key",
            "pooled_value",
            "compressed_tiles",
            "topk_indices_i32",
        ):
            if name in shapes:
                shapes[name] = (0,)
    video = bool(shapes["video_input"][0])
    shapes["video_input"] = (
        int(video),
        24,
        7,
        layout.packed.latent_height,
        layout.packed.latent_width,
    )
    shapes["reconstruction_rows"] = (7 * layout.video_rows_per_frame if video else 0, 96)
    shapes["audio_latents"] = (shapes["audio_latents"][0], 32, layout.packed.audio_frames)
    shapes["rgb_round"] = (
        layout.frame_count if shapes["rgb_round"][0] else 0,
        layout.height,
        layout.width,
        3,
    )
    views = storage.bind(layout.shape_key, shapes)
    device = views["video_input"].device
    media = H3MediaScratch(
        **{
            field.name: views[field.name]
            for field in fields(H3MediaScratch)
            if field.name != "video_raster_order"
        },
        video_raster_order=torch.argsort(layout.packed.video_raster_indices).to(device)
        if views["video_input"].numel()
        else torch.empty(0, dtype=torch.long, device=device),
    )
    if mesh is None:
        return None, media
    peer_shape = (layout.local_rows, 56 // layout.tp_size, 128)
    peer_elements = math.prod(peer_shape)
    scratch = H3Scratch(
        **{
            field.name: views[field.name]
            for field in fields(H3Scratch)
            if field.name not in {"projection_peers", "context_workspace"}
        },
        projection_peers=tuple(
            peer.reshape(-1)[:peer_elements].view(peer_shape)
            for peer in storage.peers("projection")
        ),
        context_workspace=context,
    )
    return scratch, media


if TYPE_CHECKING:
    from .transformer import H3TransformerMetadata, MiniMaxH3Transformer
    from .video_vae import MiniMaxH3VideoVAE


@dataclass(frozen=True, slots=True)
class H3ComputeInputs:
    """Binds an H3 layout to scratch storage, transformer indices, and sparse-attention page metadata."""

    layout: H3Layout
    scratch: H3Scratch | None
    media: H3MediaScratch
    transformer_metadata: H3TransformerMetadata | None
    base_tile_valid_sizes: torch.Tensor
    prompt_prefix_indices: torch.Tensor
    prompt_dense_indices: torch.Tensor
    prompt_prefix_counts: torch.Tensor

    @classmethod
    def bind(
        cls,
        bindings: EntryBindings,
        layout: H3Layout,
        storage: BoundedTensorStorage,
        context: AttentionContextWorkspace | None,
        transformer_metadata: H3TransformerMetadata | None,
        device: torch.device,
    ) -> H3ComputeInputs:
        """Bind one layout's sparse metadata and borrowed numerical scratch."""

        text_tiles = int(layout.packed.text_indices.numel()) // 64
        if text_tiles < 1:
            raise ValueError("H3 page execution requires at least one text page")
        prefix = torch.arange(layout.packed.prefix_tiles, dtype=torch.int32, device=device)
        dense = torch.arange(
            layout.packed.prefix_tiles + layout.packed.video_tiles,
            dtype=torch.int32,
            device=device,
        )
        scratch, media = bind_compute_tensors(
            storage, layout, bindings.meshes.get("denoiser"), context
        )
        return cls(
            layout=layout,
            scratch=scratch,
            media=media,
            transformer_metadata=transformer_metadata,
            base_tile_valid_sizes=layout.packed.tile_valid_sizes.to(device),
            prompt_prefix_indices=prefix,
            prompt_dense_indices=dense,
            prompt_prefix_counts=torch.tensor(
                layout.packed.prefix_tiles,
                dtype=torch.int32,
                device=device,
            ),
        )

    def _prepare_tile_metadata(self, slot: H3Tensors, text_rows: int) -> None:
        """Write request text validity over immutable packed tile metadata."""

        packed = self.layout.packed
        valid = slot.tile_valid_sizes
        text_tiles = int(packed.text_indices.numel()) // 64
        full_tiles, remaining = divmod(int(text_rows), 64)
        valid.copy_(self.base_tile_valid_sizes)
        valid[:text_tiles].zero_()
        if full_tiles:
            valid[:full_tiles].fill_(64)
        if remaining:
            valid[full_tiles].fill_(remaining)
        slot.prefix_key_indices.copy_(self.prompt_prefix_indices)
        slot.dense_key_indices.copy_(self.prompt_dense_indices)
        slot.prefix_count.copy_(self.prompt_prefix_counts)

    def _prepare_rotary(
        self,
        slot: H3Tensors,
        text_rows: int,
        transformer: MiniMaxH3Transformer,
    ) -> None:
        """Write prompt-relative multimodal rotary coordinates into request views."""

        assert self.scratch is not None and self.transformer_metadata is not None
        positions = self.scratch.rotary_positions
        positions.copy_(self.transformer_metadata.positions)
        non_text_start = int(self.layout.packed.text_indices.numel())
        if self.layout.packed.reference_shape is None:
            positions[non_text_start:, 0].add_(
                int(text_rows) - int(self.layout.packed.text_indices.numel())
            )
        transformer.rope.forward_into(
            positions,
            slot.rotary_cosine,
            slot.rotary_sine,
            self.scratch.rotary_frequencies,
        )

    @torch.inference_mode()
    def prepare_tensors(
        self,
        slot: H3Tensors,
        encoded: torch.Tensor | None,
        text_rows: int,
        transformer: MiniMaxH3Transformer | None,
        *,
        presentation_tags: torch.Tensor | None = None,
        reference_image: torch.Tensor | None = None,
        video_vae: MiniMaxH3VideoVAE | None = None,
    ) -> None:
        """Install presentation states and one fixed image into resident inputs.

        Image input is decoded HWC uint8 RGB. The layout must already reserve
        its geometry and presentation tags. VAE rows are separate from target
        solver state; all reference pages participate as dense attention keys.
        """

        packed = self.layout.packed
        if packed.reference_shape is not None:
            if (
                presentation_tags is None
                or not torch.equal(presentation_tags.cpu(), packed.presentation_tags)
                or text_rows != presentation_tags.numel()
            ):
                raise ValueError("presentation tags must match the bound image layout")
            if transformer is not None and transformer.pipeline.first:
                if reference_image is None or video_vae is None:
                    raise ValueError("image conditioning requires decoded pixels and a video VAE")
                if tuple(reference_image.shape) != (*packed.reference_shape, 3):
                    raise ValueError("reference raster does not match the bound layout")
                latents = video_vae.encode(reference_image)
                expected = (
                    1,
                    24,
                    1,
                    packed.reference_shape[0] // 16,
                    packed.reference_shape[1] // 16,
                )
                if tuple(latents.shape) != expected:
                    raise ValueError(f"image VAE latents must have shape {expected}")
                rows = patchify_video(latents)[0]
                owned = (packed.reference_indices >= self.layout.local_start) & (
                    packed.reference_indices < self.layout.local_end
                )
                slot.reference_rows.copy_(rows[owned.to(rows.device)])
        elif presentation_tags is not None or reference_image is not None:
            raise ValueError("image conditioning requires a reference layout")

        if transformer is not None:
            if transformer.pipeline.first:
                if encoded is None:
                    raise RuntimeError("denoiser input owner did not receive text conditioning")
                if encoded.shape != (1, text_rows, slot.text_condition.shape[2]):
                    raise ValueError(
                        "conditioning must contain exactly the declared presentation rows"
                    )
                slot.text_condition.zero_()
                slot.text_condition[:, : encoded.shape[1]].copy_(encoded)
            self._prepare_tile_metadata(slot, text_rows)
            self._prepare_rotary(slot, text_rows, transformer)
        slot.video_overlap.zero_()

    def prepare_warmup_slots(
        self,
        storage: tuple[BoundedTensorStorage, ...],
        transformer: MiniMaxH3Transformer,
    ) -> tuple[H3Tensors, ...]:
        """Bind and initialize every resident slot for one warmup geometry."""

        packed = self.layout.packed
        text_rows = int(
            packed.text_indices.numel()
            if packed.presentation_tags is None
            else packed.presentation_tags.numel()
        )
        views = tuple(bind_request_tensors(tensors, self.layout) for tensors in storage)
        for slot in views:
            # Warmup needs valid geometry and finite inputs, not an image encode
            # or request conditioning. Reference rows are fixed just like text.
            slot.text_condition.zero_()
            slot.reference_rows.zero_()
            slot.video_rows.zero_()
            slot.audio_rows.zero_()
            slot.video_overlap.zero_()
            self._prepare_tile_metadata(slot, text_rows)
            self._prepare_rotary(slot, text_rows, transformer)
        return views

    @staticmethod
    @torch.inference_mode()
    def initialize_tensors(
        slot: H3Tensors, seed: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Initialize locally owned rows with physical-layout-independent noise."""

        if not slot.layout.denoiser_participant:
            return ()
        layout = slot.layout
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        slot.video_noise.normal_(generator=generator)
        raster_rows = patchify_video(slot.video_noise)[0]
        tiled_rows = raster_rows.index_select(0, layout.packed.video_raster_indices)
        video_owned = (layout.packed.video_indices >= layout.local_start) & (
            layout.packed.video_indices < layout.local_end
        )
        torch.index_select(
            tiled_rows,
            0,
            video_owned.nonzero(as_tuple=False).flatten(),
            out=slot.video_source,
        )
        slot.audio_noise.normal_(generator=generator)
        audio_owned = (layout.packed.audio_indices >= layout.local_start) & (
            layout.packed.audio_indices < layout.local_end
        )
        torch.index_select(
            slot.audio_noise,
            0,
            audio_owned.nonzero(as_tuple=False).flatten(),
            out=slot.audio_source,
        )
        return ((slot.video_rows, slot.video_source), (slot.audio_rows, slot.audio_source))


def tensor_output_layout(
    bindings: EntryBindings,
    entry: str,
    output_index: int,
    decode: DecodeRange | None,
    *,
    frames: int,
    text_rows: int,
    prompt_tokens: int,
    audio_frames: int,
    height: int = 768,
    width: int = 1344,
    reference_shape: tuple[int, int] | None = None,
    presentation_tags: torch.Tensor | None = None,
) -> TensorOutputLayout | None:
    """Describe one entry's logical H3 tensor and this rank's produced region."""

    if bindings.process_group.rank not in bindings.output_ranks(entry):
        return None
    if entry == "text_encoder":
        if output_index == 1:
            return TensorOutputLayout((prompt_tokens,))
        return TensorOutputLayout((1, prompt_tokens, H3TextEncoderConfig().hidden_size))
    if entry == "denoiser":
        layout = H3Layout.build(
            bindings,
            frames=frames,
            text_rows=text_rows,
            audio_frames=audio_frames,
            height=height,
            width=width,
            reference_shape=reference_shape,
            presentation_tags=presentation_tags,
        )
        indices = layout.packed.video_indices if output_index == 0 else layout.packed.audio_indices
        count = layout.local_video_rows if output_index == 0 else layout.local_audio_rows
        if count == 0:
            return None
        width = 96 if output_index == 0 else 32
        start = int(torch.searchsorted(indices, layout.local_start))
        return TensorOutputLayout(
            (int(indices.numel()), width),
            TensorRegion((start, 0), (count, width)),
        )
    if entry == "video_decoder":
        if decode is None or decode.track is not MediaTrack.VIDEO:
            raise ValueError("video reconstruction requires a temporal range")
        rank = bindings.entries[entry].ranks.index(bindings.process_group.rank)
        if rank >= decode.max_units:
            return None
        shape = (decode.max_units, 1, 3, 25, height, width)
        return TensorOutputLayout(
            shape,
            TensorRegion((rank, 0, 0, 0, 0, 0), (1, *shape[1:])),
        )
    if entry == "audio_decoder":
        return TensorOutputLayout((round(frames * PROFILE_AUDIO_RATE / PROFILE_FPS), 2))
    raise ValueError(f"H3 entry {entry!r} has no Tensor result")


def warmup_geometries(capacity: H3Layout, denoise_steps: int) -> tuple[MediaGeometry, ...]:
    """Return unique representative request shapes within the configured capacity."""

    max_rows = int(capacity.packed.text_indices.numel())
    shapes = (
        (capacity.frame_count, 65),
        (MIN_H3_FRAMES, 1),
        *((capacity.frame_count, count) for count in (129, 193, 257)),
        (capacity.frame_count, max_rows),
    )
    result: list[MediaGeometry] = []
    keys: set[tuple[int, int]] = set()
    for frames, token_count in shapes:
        key = (frames, ((token_count + 63) // 64) * 64)
        if key in keys:
            continue
        keys.add(key)
        result.append(
            MediaGeometry(
                frame_count=frames,
                video_units=len(reconstruction_unit_frames(frames)),
                prompt_tokens=token_count,
                denoise_steps=denoise_steps,
            )
        )
    return tuple(result)
