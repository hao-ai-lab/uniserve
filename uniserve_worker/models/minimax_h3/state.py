"""Persistent request state and the single shared H3 execution scratch lane."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from ...backends.attention.video_sparse import video_sparse_selected_tiles
from ...execution.batch import RequestKey
from ...execution.bounded_storage import BoundedTensorStorage
from ...nn.mesh import DeviceMesh
from ...nn.parallel_attention import AttentionContextWorkspace
from .packing import H3PackedLayout, build_packed_layout
from .placement import H3Placement
from .schedule import H3Schedule

__all__ = [
    "H3Layout",
    "H3Scratch",
    "H3StatePool",
    "H3StateSlot",
    "MIN_H3_FRAMES",
]

PROFILE_HEIGHT = 768
PROFILE_WIDTH = 1344
PROFILE_FPS = 24
PROFILE_AUDIO_RATE = 32_000
FASTH3_STEPS = 4
MIN_H3_FRAMES = 22


def _reconstruction_unit_frames(frames: int) -> tuple[int, ...]:
    """Convert generated frame count into overlapping decoder reconstruction units."""

    if frames < MIN_H3_FRAMES or frames % 17 != 5:
        raise ValueError("H3 frame count must have the form 17 * n + 5")
    units = (frames - 5) // 17
    return (17,) * (units - 1) + (MIN_H3_FRAMES,)


@dataclass(frozen=True, slots=True)
class H3Layout:
    """Combines packed multimodal geometry with schedule and sequence-parallel rank ownership."""

    packed: H3PackedLayout
    schedule: H3Schedule
    sp_rank: int
    sp_size: int
    tp_size: int
    ulysses_size: int
    sequence_kind: str
    context_col_size: int
    denoiser_participant: bool
    output_owner: bool
    decoder_width: int
    local_start: int
    local_end: int
    frame_count: int
    reconstruction_unit_frames: tuple[int, ...]

    @classmethod
    def build(
        cls,
        placement: H3Placement,
        *,
        frames: int,
        text_rows: int,
        audio_frames: int,
        schedule: H3Schedule | None = None,
    ) -> "H3Layout":
        """Partition one packed request evenly across the mesh sequence ranks."""

        mesh = placement.denoiser_mesh
        config = placement.components["denoiser"].parallel_config
        size = config.sequence_parallel_size
        rank = mesh.coord("sp") if mesh is not None else 0
        packed = build_packed_layout(
            text_rows=text_rows,
            num_frames=frames,
            audio_frames=audio_frames,
            # Four logical row partitions define tensorwise activation scales;
            # physical sequence ownership does not redefine those domains.
            row_multiple=64 * math.lcm(4, size),
        )
        shard = packed.padded_rows // size
        return cls(
            packed=packed,
            schedule=H3Schedule.build(placement.process_group.device)
            if schedule is None
            else schedule,
            sp_rank=rank,
            sp_size=size,
            tp_size=config.tensor_parallel_size,
            ulysses_size=dict(config.dimensions)["ulysses"],
            sequence_kind=config.sequence_parallel.kind,
            context_col_size=dict(config.dimensions).get("cp_col", 1),
            denoiser_participant=mesh is not None,
            output_owner=placement.owns("output"),
            decoder_width=len(placement.decoder_ranks),
            local_start=rank * shard if mesh is not None else 0,
            local_end=(rank + 1) * shard if mesh is not None else 0,
            frame_count=int(frames),
            reconstruction_unit_frames=_reconstruction_unit_frames(int(frames)),
        )

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
    def shape_key(self) -> tuple[int, int, int]:
        """Identify layouts by video frames, padded text rows, and audio frames."""

        return (
            self.frame_count,
            int(self.packed.text_indices.numel()),
            int(self.packed.audio_frames),
        )

    @property
    def video_round_frames(self) -> int:
        """Bound the RGB frames produced by the largest reconstruction round."""

        return max(
            sum(self.reconstruction_unit_frames[start : start + self.decoder_width])
            for start in range(0, self.video_reconstruction_units, self.decoder_width)
        )

    @property
    def max_video_round_frames(self) -> int:
        """Bound one full sequence-parallel round at deployment capacity."""

        units = min(self.video_reconstruction_units, self.decoder_width)
        return (units - 1) * 17 + MIN_H3_FRAMES

    @property
    def local_rows(self) -> int:
        """Count packed transport rows owned by this sequence rank."""

        return self.local_end - self.local_start

    @property
    def video_reconstruction_units(self) -> int:
        """Count overlapping temporal segments needed to reconstruct the video."""

        return len(self.reconstruction_unit_frames)

    @property
    def persistent_units(self) -> int:
        """Express per-request conditioning and media storage in rounded MiB units."""

        video = self.packed.video_indices.numel() * 96 * 4
        audio = self.packed.audio_indices.numel() * 32 * 4
        text = self.packed.text_indices.numel() * 5120 * 2
        return (video + audio + text + (1 << 20) - 1) // (1 << 20)

    def local_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Filter global packed indices to this rank and convert them to local offsets."""

        selected = indices[(indices >= self.local_start) & (indices < self.local_end)]
        return selected - self.local_start

    @property
    def local_video_rows(self) -> int:
        """Count semantic video rows owned by this sequence rank."""

        return int(self.local_indices(self.packed.video_indices).numel())

    @property
    def local_audio_rows(self) -> int:
        """Count semantic audio rows owned by this sequence rank."""

        return int(self.local_indices(self.packed.audio_indices).numel())

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


@dataclass(frozen=True, slots=True)
class _H3StateBuffers:
    """Owns persistent conditioning, media rows, rotary tables, modulation plans, and overlap storage."""

    text_condition: torch.Tensor
    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    tile_valid_sizes: torch.Tensor
    prefix_key_indices: torch.Tensor
    dense_key_indices: torch.Tensor
    prefix_count: torch.Tensor
    rotary_cosine: torch.Tensor
    rotary_sine: torch.Tensor
    video_overlap: torch.Tensor | None


@dataclass(slots=True)
class H3StateSlot:
    """Tracks one admitted request’s persistent H3 state and generation progress."""

    index: int
    text_condition: torch.Tensor
    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    tile_valid_sizes: torch.Tensor
    prefix_key_indices: torch.Tensor
    dense_key_indices: torch.Tensor
    prefix_count: torch.Tensor
    rotary_cosine: torch.Tensor
    rotary_sine: torch.Tensor
    video_overlap: torch.Tensor | None
    bounded: BoundedTensorStorage
    request_key: RequestKey | None = None
    shape_key: tuple[int, int, int] | None = None
    denoise_step: int = 0
    next_video_unit: int = 0
    audio_reconstructed: bool = False

    @property
    def active(self) -> bool:
        """Indicate whether an admitted request currently owns this slot."""

        return self.request_key is not None

    def bind(self, layout: H3Layout) -> None:
        """Rebind capacity tensors to shape-bounded views for a compatible request layout."""

        text_rows = int(layout.packed.text_indices.numel()) if layout.denoiser_participant else 0
        if self.active and self.shape_key != layout.shape_key:
            raise RuntimeError("an active H3 state slot cannot change its execution layout")
        capacity = self.bounded.capacity
        views = self.bounded.bind(
            layout.shape_key,
            {
                "text_condition": (1, text_rows, 5376),
                "video_rows": (layout.local_video_rows, 96),
                "audio_rows": (layout.local_audio_rows, 32),
                "tile_valid_sizes": (
                    int(layout.packed.tile_valid_sizes.numel())
                    if layout.denoiser_participant
                    else 0,
                ),
                "prefix_key_indices": (
                    int(layout.packed.prefix_tiles) if layout.denoiser_participant else 0,
                ),
                "dense_key_indices": (
                    int(layout.packed.prefix_tiles + layout.packed.video_tiles)
                    if layout.denoiser_participant
                    else 0,
                ),
                "prefix_count": (),
                "rotary_cosine": (
                    layout.packed.padded_rows if layout.denoiser_participant else 0,
                    96,
                ),
                "rotary_sine": (
                    layout.packed.padded_rows if layout.denoiser_participant else 0,
                    96,
                ),
                "video_overlap": tuple(capacity["video_overlap"].shape),
            },
        )
        # Replace every exposed tensor with the shape-bounded view returned by the
        # common storage owner; no allocation changes ownership during rebinding.
        buffers = _H3StateBuffers(**views)
        self.text_condition = buffers.text_condition
        self.video_rows = buffers.video_rows
        self.audio_rows = buffers.audio_rows
        self.tile_valid_sizes = buffers.tile_valid_sizes
        self.prefix_key_indices = buffers.prefix_key_indices
        self.dense_key_indices = buffers.dense_key_indices
        self.prefix_count = buffers.prefix_count
        self.rotary_cosine = buffers.rotary_cosine
        self.rotary_sine = buffers.rotary_sine
        self.video_overlap = buffers.video_overlap
        self.shape_key = layout.shape_key

    def clear(self) -> None:
        """Release request ownership and reset progress while retaining resident buffers."""

        self.request_key = None
        self.shape_key = None
        self.denoise_step = 0
        self.next_video_unit = 0
        self.audio_reconstructed = False
        if self.video_overlap is not None:
            self.video_overlap.zero_()


class H3StatePool:
    """Owns bounded persistent H3 request slots and admits shape-compatible generation state atomically."""

    def __init__(
        self,
        layout: H3Layout,
        slot_count: int,
        device: torch.device,
    ) -> None:
        """Allocate a fixed set of reusable device-resident request-state slots."""

        if slot_count < 2:
            raise ValueError("the FastH3 serving topology requires at least two state slots")
        self.layout = layout
        self.slots = tuple(
            self._allocate_slot(
                index,
                layout,
                device,
            )
            for index in range(1, slot_count + 1)
        )

    @staticmethod
    def _allocate_slot(
        index: int,
        layout: H3Layout,
        device: torch.device,
    ) -> H3StateSlot:
        """Allocate all bounded device buffers owned by one reusable H3 request slot."""

        # Derive capacities from the largest packed layout; active request
        # shapes bind narrower views without reallocating slot storage.
        text_capacity = (
            int(layout.packed.text_indices.numel()) if layout.denoiser_participant else 0
        )
        video_capacity = min(int(layout.packed.video_indices.numel()), layout.local_rows)
        audio_capacity = min(int(layout.packed.audio_indices.numel()), layout.local_rows)
        tile_capacity = (
            int(layout.packed.tile_valid_sizes.numel()) if layout.denoiser_participant else 0
        )
        prefix_capacity = int(layout.packed.prefix_tiles) if layout.denoiser_participant else 0
        dense_capacity = (
            int(layout.packed.prefix_tiles + layout.packed.video_tiles)
            if layout.denoiser_participant
            else 0
        )
        row_capacity = int(layout.packed.padded_rows) if layout.denoiser_participant else 0

        # Conditioning, modality rows, sparse metadata, rotary tables, and
        # modulation plans remain resident for the entire slot lifetime.
        text_condition = torch.empty((1, text_capacity, 5376), dtype=torch.bfloat16, device=device)
        video_rows = torch.empty((video_capacity, 96), dtype=torch.float32, device=device)
        audio_rows = torch.empty((audio_capacity, 32), dtype=torch.float32, device=device)
        tile_valid_sizes = torch.empty((tile_capacity,), dtype=torch.int32, device=device)
        prefix_key_indices = torch.empty((prefix_capacity,), dtype=torch.int32, device=device)
        dense_key_indices = torch.empty((dense_capacity,), dtype=torch.int32, device=device)
        prefix_count = torch.zeros((), dtype=torch.int32, device=device)
        rotary_cosine = torch.empty((row_capacity, 96), dtype=torch.float32, device=device)
        rotary_sine = torch.empty((row_capacity, 96), dtype=torch.float32, device=device)
        video_overlap = torch.empty(
            (1 if layout.output_owner else 0, 3, 5, PROFILE_HEIGHT, PROFILE_WIDTH),
            dtype=torch.float16,
            device=device,
        )

        # The bounded inventory lets smaller layout bindings expose typed views
        # while keeping the overlap buffer under the same ownership boundary.
        tensors = locals()
        bounded = BoundedTensorStorage(
            {
                name: tensors[name]
                for name in _H3StateBuffers.__dataclass_fields__
                if name != "video_overlap"
            }
            | {"video_overlap": video_overlap}
        )
        slot = H3StateSlot(
            index=index,
            text_condition=text_condition,
            video_rows=video_rows,
            audio_rows=audio_rows,
            tile_valid_sizes=tile_valid_sizes,
            prefix_key_indices=prefix_key_indices,
            dense_key_indices=dense_key_indices,
            prefix_count=prefix_count,
            rotary_cosine=rotary_cosine,
            rotary_sine=rotary_sine,
            video_overlap=video_overlap,
            bounded=bounded,
        )

        # Initial binding establishes maximum views; clear shape ownership so
        # admission can claim the slot for its concrete request geometry.
        slot.bind(layout)
        slot.shape_key = None
        return slot

    @staticmethod
    def bytes_per_slot(
        layout: H3Layout,
    ) -> int:
        """Return the exact persistent CUDA tensor bytes owned by one state slot."""

        text_rows = int(layout.packed.text_indices.numel()) if layout.denoiser_participant else 0
        video_rows = min(int(layout.packed.video_indices.numel()), layout.local_rows)
        audio_rows = min(int(layout.packed.audio_indices.numel()), layout.local_rows)
        tile_count = (
            int(layout.packed.tile_valid_sizes.numel()) if layout.denoiser_participant else 0
        )
        prefix_tiles = int(layout.packed.prefix_tiles) if layout.denoiser_participant else 0
        dense_tiles = (
            int(layout.packed.prefix_tiles + layout.packed.video_tiles)
            if layout.denoiser_participant
            else 0
        )
        padded_rows = int(layout.packed.padded_rows) if layout.denoiser_participant else 0
        return sum(
            (
                text_rows * 5376 * 2,
                video_rows * 96 * 4,
                audio_rows * 32 * 4,
                tile_count * 4,
                prefix_tiles * 4,
                dense_tiles * 4,
                4,
                padded_rows * 96 * 4 * 2,
                (3 * 5 * PROFILE_HEIGHT * PROFILE_WIDTH * 2) if layout.output_owner else 0,
            )
        )

    @property
    def slot_count(self) -> int:
        """Expose the number of concurrently resident request states."""

        return len(self.slots)

    def get(self, index: int) -> H3StateSlot:
        """Resolve the one-based request-pool index to its resident H3 slot."""

        if index < 1 or index > len(self.slots):
            raise ValueError(f"H3 request-pool index {index} is outside resident capacity")
        return self.slots[index - 1]

    def drop_request(self, request_id: int) -> None:
        """Release every slot owned by a request id after completion or cancellation."""

        for slot in self.slots:
            if slot.request_key is not None and slot.request_key.request_id == int(request_id):
                slot.clear()

    def abort_admissions(self, admissions) -> None:
        """Atomically validate and release state slots for discarded admissions."""

        slots = tuple(
            (admission, self.get(int(admission.request_pool_idx))) for admission in admissions
        )
        for admission, slot in slots:
            if slot.request_key not in (None, admission.request_key):
                raise RuntimeError("discarded request admission no longer owns its state slot")
        for _admission, slot in slots:
            slot.clear()


@dataclass(slots=True)
class H3Scratch:
    """Owns fixed intermediate tensors for H3 projection, attention, exchange, reconstruction, and modulation."""

    packed_hidden: torch.Tensor
    local_text_hidden: torch.Tensor
    projected_input: torch.Tensor
    projected_input_bf16: torch.Tensor
    local_video_hidden: torch.Tensor
    local_audio_hidden: torch.Tensor
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
    bounded: BoundedTensorStorage | None

    def view(self, layout: H3Layout) -> "H3Scratch":
        """Create layout-bounded tensor views over the shared maximum-capacity scratch arena."""

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
        projected_rows = max(local_video, local_audio)
        workspace_elements = global_rows * 5376
        if self.bounded is None:
            raise RuntimeError("H3 scratch storage has no bounded-view owner")
        # Fixed reconstruction buffers retain capacity shape; packed execution
        # buffers contract to the active page geometry.
        shapes = {name: tuple(tensor.shape) for name, tensor in self.bounded.capacity.items()}
        shapes.update(
            {
                "packed_hidden": (1, local_rows, 5376),
                "local_text_hidden": (1, local_text, 5376),
                "projected_input": (projected_rows, 5376),
                "projected_input_bf16": (projected_rows, 5376),
                "local_video_hidden": (local_video, 5376),
                "local_audio_hidden": (local_audio, 5376),
                "video_velocity": (local_video, 96),
                "audio_velocity": (local_audio, 32),
                "attention_workspace": (workspace_elements,),
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
                    video_sparse_selected_tiles(layout.packed.video_tiles),
                ),
                "rotary_positions": (global_rows, 3),
                "rotary_frequencies": (global_rows, 3, 16),
            }
        )
        for rank in range(len(self.projection_peers)):
            shapes[f"projection_peer_{rank}"] = (local_rows, 56 // layout.tp_size, 128)
        views = self.bounded.bind(layout.shape_key, shapes)
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values.update({name: value for name, value in views.items() if name in values})
        values["projection_peers"] = tuple(
            views[f"projection_peer_{rank}"] for rank in range(len(self.projection_peers))
        )
        return H3Scratch(**values)

    @classmethod
    def allocate(
        cls,
        layout: H3Layout,
        mesh: DeviceMesh,
        *,
        block_params_shape: tuple[int, ...],
        final_params_shape: tuple[int, ...],
        attention_workspace_dtype: torch.dtype,
    ) -> "H3Scratch":
        """Allocate the maximum rank-local execution arena and symmetric attention exchange."""

        device = mesh.local_device
        local_rows = layout.local_rows
        local_heads = 56 // (layout.tp_size * layout.ulysses_size)
        global_rows = layout.packed.padded_rows
        tile_count = (global_rows + 63) // 64
        query_rows = layout.attention_rows
        query_tiles = query_rows // 64
        keep_video_tiles = video_sparse_selected_tiles(layout.packed.video_tiles)
        local_text = min(int(layout.packed.text_indices.numel()), local_rows)
        local_video = min(int(layout.packed.video_indices.numel()), local_rows)
        local_audio = min(int(layout.packed.audio_indices.numel()), local_rows)
        max_projected_rows = max(local_video, local_audio)
        # Q/K/V projection exchange is symmetric so every rank can address peer
        # slices directly during sequence-parallel sparse attention.
        projection_exchange = mesh.get_group("ulysses").symmetric_memory(
            (local_rows, 56 // layout.tp_size, 128),
            dtype=torch.bfloat16,
            name="video_attention_heads",
        )
        projection_peers = projection_exchange.peers
        context_workspace = None
        if layout.sp_size > layout.ulysses_size:
            key_group = mesh.get_group("cp_row" if layout.sequence_kind == "attention2d" else "cp")
            context_workspace = AttentionContextWorkspace.allocate(
                key_group,
                layout.attention_rows * layout.context_col_size,
                local_heads,
                mapped=layout.sequence_kind != "allgather",
            )
        scratch = cls(
            packed_hidden=torch.empty((1, local_rows, 5376), dtype=torch.bfloat16, device=device),
            local_text_hidden=torch.empty(
                (1, local_text, 5376), dtype=torch.bfloat16, device=device
            ),
            projected_input=torch.empty(
                (max_projected_rows, 5376), dtype=torch.float32, device=device
            ),
            projected_input_bf16=torch.empty(
                (max_projected_rows, 5376), dtype=torch.bfloat16, device=device
            ),
            local_video_hidden=torch.empty(
                (local_video, 5376), dtype=torch.bfloat16, device=device
            ),
            local_audio_hidden=torch.empty(
                (local_audio, 5376), dtype=torch.bfloat16, device=device
            ),
            video_velocity=torch.empty((local_video, 96), dtype=torch.float32, device=device),
            audio_velocity=torch.empty((local_audio, 32), dtype=torch.float32, device=device),
            projection_peers=projection_peers,
            projection_sync_input=torch.full(
                (1,), layout.sp_rank, dtype=torch.int32, device=device
            ),
            projection_sync_output=torch.empty(
                (layout.ulysses_size,), dtype=torch.int32, device=device
            ),
            attention_workspace=torch.empty(
                global_rows * 5376,
                dtype=attention_workspace_dtype,
                device=device,
            ),
            attention_output=torch.empty(
                (query_rows, local_heads, 128),
                dtype=torch.bfloat16,
                device=device,
            ),
            context_workspace=context_workspace,
            tile_scores=torch.empty(
                (local_heads, query_tiles, tile_count), dtype=torch.float32, device=device
            ),
            block_indices=torch.empty(
                (
                    local_heads,
                    query_tiles,
                    layout.packed.prefix_tiles + layout.packed.video_tiles,
                ),
                dtype=torch.int32,
                device=device,
            ),
            block_counts=torch.empty((local_heads, query_tiles), dtype=torch.int32, device=device),
            pooled_query=torch.empty(
                (query_tiles, local_heads, 128), dtype=torch.float32, device=device
            ),
            pooled_key=torch.empty(
                (tile_count, local_heads, 128), dtype=torch.float32, device=device
            ),
            pooled_value=torch.empty(
                (tile_count, local_heads, 128), dtype=torch.float32, device=device
            ),
            compressed_tiles=torch.empty(
                (local_heads, query_tiles, 128), dtype=torch.float32, device=device
            ),
            topk_indices_i32=torch.empty(
                (local_heads, query_tiles, keep_video_tiles),
                dtype=torch.int32,
                device=device,
            ),
            block_adaln_params=torch.empty(
                block_params_shape,
                dtype=torch.bfloat16,
                device=device,
            ),
            final_adaln_params=torch.empty(
                final_params_shape,
                dtype=torch.bfloat16,
                device=device,
            ),
            rotary_positions=torch.empty((global_rows, 3), dtype=torch.float32, device=device),
            rotary_frequencies=torch.empty(
                (global_rows, 3, 16), dtype=torch.float32, device=device
            ),
            bounded=None,
        )
        # Only tensors whose active shapes depend on request geometry participate
        # in bounded rebinding; decoder and collective buffers remain fixed.
        variable_fields = (
            "packed_hidden",
            "local_text_hidden",
            "projected_input",
            "projected_input_bf16",
            "local_video_hidden",
            "local_audio_hidden",
            "video_velocity",
            "audio_velocity",
            "attention_workspace",
            "attention_output",
            "tile_scores",
            "block_indices",
            "block_counts",
            "pooled_query",
            "pooled_key",
            "pooled_value",
            "compressed_tiles",
            "topk_indices_i32",
            "rotary_positions",
            "rotary_frequencies",
        )
        tensors = {name: getattr(scratch, name) for name in variable_fields}
        tensors.update(
            {f"projection_peer_{rank}": peer for rank, peer in enumerate(scratch.projection_peers)}
        )
        scratch.bounded = BoundedTensorStorage(tensors)
        return scratch


@dataclass(slots=True)
class H3MediaScratch:
    """Own component-transfer and decode buffers according to actual placement."""

    encoder_hidden: torch.Tensor
    video_send: torch.Tensor
    video_receive: torch.Tensor
    video_input: torch.Tensor
    reconstruction_rows: torch.Tensor
    segment_placeholder: torch.Tensor
    empty_segments: torch.Tensor
    segment_receive: torch.Tensor
    rgb_round: torch.Tensor
    audio_send: torch.Tensor
    audio_receive: torch.Tensor
    audio_input: torch.Tensor
    audio_latents: torch.Tensor
    pcm: torch.Tensor
    local_video_raster: torch.Tensor
    local_audio_raster: torch.Tensor
    bounded: BoundedTensorStorage

    @classmethod
    def allocate(cls, layout: H3Layout, placement: H3Placement) -> "H3MediaScratch":
        """Reserve only assigned producer, decoder, and output products."""

        device = placement.process_group.device
        producer = placement.process_group.rank in placement.latent_producers
        video = placement.owns("video_decoder")
        audio = placement.owns("audio_decoder")
        output = placement.owns("output")
        encoder_consumer = placement.owns("denoiser") and not placement.owns("text_encoder")
        sources = len(placement.latent_producers)
        destinations = len(placement.decoder_ranks)
        audio_rows = layout.packed.audio_indices.numel()
        samples = round(layout.frame_count * PROFILE_AUDIO_RATE / PROFILE_FPS)
        tensors = {
            "encoder_hidden": torch.empty(
                (1 if encoder_consumer else 0, layout.packed.text_indices.numel(), 5120),
                dtype=torch.bfloat16,
                device=device,
            ),
            "video_send": torch.empty(
                (destinations if producer else 0, 24, 7, 48, 84), dtype=torch.float32, device=device
            ),
            "video_receive": torch.empty(
                (sources if video else 0, 24, 7, 48, 84), dtype=torch.float32, device=device
            ),
            "video_input": torch.empty(
                (1 if video else 0, 24, 7, 48, 84), dtype=torch.float32, device=device
            ),
            "reconstruction_rows": torch.empty(
                (7 * 24 * 42 if producer else 0, 96), dtype=torch.float32, device=device
            ),
            "segment_placeholder": torch.zeros(
                (1 if video else 0, 3, 25, PROFILE_HEIGHT, PROFILE_WIDTH),
                dtype=torch.float16,
                device=device,
            ),
            "empty_segments": torch.empty(
                (0, 1, 3, 25, PROFILE_HEIGHT, PROFILE_WIDTH), dtype=torch.float16, device=device
            ),
            "segment_receive": torch.empty(
                (destinations if output else 0, 1, 3, 25, PROFILE_HEIGHT, PROFILE_WIDTH),
                dtype=torch.float16,
                device=device,
            ),
            "rgb_round": torch.empty(
                (layout.max_video_round_frames if output else 0, PROFILE_HEIGHT, PROFILE_WIDTH, 3),
                dtype=torch.uint8,
                device=device,
            ),
            "audio_send": torch.empty(
                (1 if producer else 0, audio_rows, 32), dtype=torch.float32, device=device
            ),
            "audio_receive": torch.empty(
                (sources if audio else 0, audio_rows, 32), dtype=torch.float32, device=device
            ),
            "audio_input": torch.empty(
                (audio_rows if audio else 0, 32), dtype=torch.float32, device=device
            ),
            "audio_latents": torch.empty(
                (2 if audio else 0, 32, layout.packed.audio_frames),
                dtype=torch.float32,
                device=device,
            ),
            "pcm": torch.empty((samples if output else 0, 2), dtype=torch.int16, device=device),
        }
        return cls(
            **tensors,
            local_video_raster=layout.local_video_raster_indices.to(device),
            local_audio_raster=layout.local_audio_raster_indices.to(device),
            bounded=BoundedTensorStorage(tensors),
        )

    def view(self, layout: H3Layout) -> "H3MediaScratch":
        """Borrow contiguous active-shape views from component-owned transfer storage."""

        shapes = {name: tuple(tensor.shape) for name, tensor in self.bounded.capacity.items()}
        audio_rows = layout.packed.audio_indices.numel()
        shapes["encoder_hidden"] = (
            shapes["encoder_hidden"][0],
            layout.packed.text_indices.numel(),
            5120,
        )
        for name in ("audio_send", "audio_receive"):
            shapes[name] = (shapes[name][0], audio_rows, 32)
        shapes["audio_input"] = (audio_rows if shapes["audio_input"][0] else 0, 32)
        shapes["audio_latents"] = (shapes["audio_latents"][0], 32, layout.packed.audio_frames)
        shapes["pcm"] = (
            round(layout.frame_count * PROFILE_AUDIO_RATE / PROFILE_FPS) if shapes["pcm"][0] else 0,
            2,
        )
        shapes["rgb_round"] = (
            layout.video_round_frames if shapes["rgb_round"][0] else 0,
            PROFILE_HEIGHT,
            PROFILE_WIDTH,
            3,
        )
        values = self.bounded.bind(layout.shape_key, shapes)
        return H3MediaScratch(
            **values,
            local_video_raster=layout.local_video_raster_indices.to(self.video_send.device),
            local_audio_raster=layout.local_audio_raster_indices.to(self.video_send.device),
            bounded=self.bounded,
        )
