"""Request-bound denoising with exact geometry and schedule variants."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeVar, cast

import torch

from uniserve.attention.context import sparse_attention_scope
from uniserve.attention.inputs import physical_columns
from uniserve.attention.metadata import AttentionMode
from uniserve.distributed.mesh import Communicator
from uniserve.math import bucketed_length, ceil_div
from uniserve.model.batch import DiffusionBatch
from uniserve.model.denoising import DenoisingStep
from uniserve.model.diffusion import DiffusionMixin
from uniserve.model.image_diffusion import ImageDiffusion, LatentLayout
from uniserve.model.media import ImageSize
from uniserve.model.tensors import FlowPatches, TensorViews, TokenSelection
from uniserve.model.text import TextMixin
from uniserve.nn.diffusion.cfg import Branch, CfgPlan, build_flow_cfg_plan
from uniserve.nn.diffusion.config import DiffusionConfig
from uniserve.nn.diffusion.integrator import CleanSampleEulerSolver, EulerSolver
from uniserve.nn.diffusion.schedule import DiffusionSchedule
from uniserve.nn.parallel_attention import (
    AttentionBuffers,
    ParallelAttention,
    context_scope,
    output_scope,
)
from uniserve.nn.vision import get_flattened_position_ids_extrapolate
from uniserve.runtime.cuda_graph import CudaGraph, GraphExecutionError, capture_pools

from ..protocol.batch import PipelineStage
from ..runtime.latent_pool import LatentPool
from .batch import InputBatch
from .denoising import numerical_signature
from .device_transfer import tensor_to_device
from .diffusion_state import DiffusionState, resolve_prefix
from .graph_inputs import DiffusionShape
from .input_buffers import InputBuffers
from .model_entry import ModelEntry, TensorOutput, capture_required
from .rows import ForwardRow
from .runners.prefill import stage_text

if TYPE_CHECKING:
    from uniserve.attention.video_sparse_provider import SparseAttentionProvider
    from uniserve.nn.sparse_attention import SparseAttention
    from uniserve.runtime.attention_storage import OutputStorage

    from .model_runner import ModelRunner

InputT = TypeVar("InputT")
logger = logging.getLogger(__name__)


def restore_samples(operation: DenoisingStep) -> Callable[[], None]:
    """Snapshot only solver state; learned predictions are disposable scratch."""

    snapshots = tuple(value.clone() for value in operation.samples)

    def restore() -> None:
        for value, saved in zip(operation.samples, snapshots, strict=True):
            value.copy_(saved)

    return restore


class DiffusionRunner:
    """Own graph-bound request slots for standard diffusion capability calls.

    Physical slot identity and backing addresses participate in residency. Pooled
    storage can be reused by subsequent requests; graphs retain only numerical
    views, never request identities. Replacing backing or geometry retires the
    dependent graphs before their metadata is freed.
    """

    def __init__(
        self,
        model: DiffusionMixin | None = None,
        *,
        device: torch.device,
        capture_stream: torch.cuda.Stream | None,
        groups: tuple[Communicator, ...],
        capacity: int,
        generation: ImageDiffusion | None = None,
        solver: EulerSolver | CleanSampleEulerSolver | None = None,
        sparse_layers: tuple[SparseAttention, ...] = (),
        additional_devices: tuple[torch.device, ...] = (),
        context_buffers: Mapping[ParallelAttention, AttentionBuffers] | None = None,
        output_storage: OutputStorage | None = None,
    ) -> None:
        self.solver = model.solver if model is not None else solver
        if generation is not None and self.solver is None:
            raise ValueError("image diffusion requires its numerical solver")
        self.generation = generation
        self.model = model
        self.device = device
        self.capture_stream = capture_stream
        self.graph_pool = torch.cuda.graph_pool_handle() if capture_stream is not None else None
        self.device_pools = capture_pools(additional_devices) if capture_stream is not None else {}
        self.graphs: dict[tuple[Hashable, Hashable, Hashable], CudaGraph[TensorOutput]] = {}
        self.groups = groups
        self.capacity = max(2, capacity)
        self._slots: OrderedDict[Hashable, tuple[Hashable, Hashable]] = OrderedDict()
        self.prepared_inputs: OrderedDict[Hashable, object] = OrderedDict()
        self._context_buffers = dict(context_buffers or {})
        self._output_storage = output_storage
        self._sparse_layers = sparse_layers
        self._sparse: dict[Hashable, SparseAttentionProvider] = {}
        self._closed = False

    @contextmanager
    def attention_scope(self, input_key: Hashable) -> Iterator[None]:
        """Bind sparse plans for this runner and its prepared-input key.

        Plans may contain mutable device indices. Separate runners own distinct
        providers; input retirement releases them after dependent graphs.
        Direct eager numerical callers use the same scope as capture and replay.
        """

        if self._closed:
            raise GraphExecutionError("denoising runner is closed")
        provider = self._sparse.get(input_key)
        if self._sparse_layers and provider is None:
            from uniserve.attention.video_sparse_provider import resolve_sparse_provider

            provider = resolve_sparse_provider(self.device)
            self._sparse[input_key] = provider
        with (
            sparse_attention_scope(provider),
            context_scope(self._context_buffers),
            output_scope(self._output_storage.views if self._output_storage is not None else {}),
        ):
            yield

    def initialize(self, size: ImageSize, config: DiffusionConfig) -> DiffusionState:
        """Prepare a request's actual schedule independently of accepted progress."""

        if self.generation is None:
            raise ValueError("model has no image generation pipeline")
        return DiffusionState(
            size=size,
            config=config,
            schedule=self.generation.create_schedule(config, device="cpu"),
        )

    def flow_rows(
        self,
        trajectory: DiffusionState,
        current: torch.Tensor,
        guide: CfgPlan,
        timestep: torch.Tensor,
        *,
        conditioning_position: int,
        height: int,
        width: int,
        patch_size: int | None,
        device: torch.device,
    ) -> tuple[ForwardRow, ...]:
        """Bind each guided prediction to this operation's current latent workspace."""

        generation = self.generation
        if generation is None:
            raise ValueError("model has no image generation pipeline")
        # The solver's resident latent may be on another device. All CFG
        # branches share this one numerical input copy for the learned call.
        current = tensor_to_device(current, device)
        rows = []
        for branch in guide.branches:
            entry = trajectory.entries[branch]
            temporal = conditioning_position if branch is Branch.COND else entry[2]
            positions, indexes, conditioning, query_tokens, text_local = prepare_denoise_inputs(
                generation,
                current,
                height,
                width,
                temporal=temporal,
                patch_size=patch_size,
                positions=trajectory.positions,
            )
            rows.append(
                ForwardRow(
                    forward_mode=PipelineStage.DENOISING,
                    flow_conditioning=conditioning,
                    positions=positions,
                    timestep=timestep.reshape(1),
                    latent=current,
                    image_tokens=query_tokens,
                    image_height=height,
                    image_width=width,
                    request_pool_idx=entry[0],
                    seq_len=entry[2],
                    group_id=entry[1],
                    write_kv=False,
                    causal=False,
                    attention_indexes=indexes,
                    text_local_indices=text_local,
                )
            )
        return tuple(rows)

    def integrate(
        self,
        current: torch.Tensor,
        outputs: tuple[torch.Tensor, ...],
        guide: CfgPlan,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
    ) -> None:
        """Apply the declared prediction conversion and one Euler step in place."""

        generation = self.generation
        if generation is None:
            raise ValueError("model has no image generation pipeline")
        velocity = guide.combine(
            {
                branch: tensor_to_device(output, current.device)
                for branch, output in zip(guide.branches, outputs, strict=True)
            }
        )
        if self.solver is None:
            raise ValueError("image integration requires its numerical solver")
        self.solver.step(velocity, current, timestep, next_timestep)

    @torch.inference_mode()
    def warmup(
        self,
        batch: DiffusionBatch,
        schedule: DiffusionSchedule,
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
        input_key: Hashable,
    ) -> None:
        if self.model is None:
            raise ValueError("runner has no diffusion model")
        operation = DenoisingStep(self.model, batch, state, constants, scratch, schedule)
        restore = restore_samples(operation)
        try:
            with self.attention_scope(input_key):
                operation()
        finally:
            restore()
            torch.cuda.current_stream(self.device).synchronize()

    @torch.inference_mode()
    def step(
        self,
        batch: DiffusionBatch,
        schedule: DiffusionSchedule,
        *,
        state: TensorViews,
        constants: TensorViews,
        scratch: TensorViews,
        slot: Hashable,
        input_key: Hashable,
    ) -> tuple[TensorOutput, str]:
        if self._closed:
            raise GraphExecutionError("denoising runner is closed")
        if self.model is None:
            raise ValueError("runner has no diffusion model")
        with self.attention_scope(input_key):
            operation = DenoisingStep(self.model, batch, state, constants, scratch, schedule)
            if self.capture_stream is None:
                return operation(), "eager"
            signature = numerical_signature(
                (
                    batch.latents,
                    batch.sizes,
                    batch.conditioning,
                    batch.positions,
                    batch.sequence_lengths,
                    batch.attention,
                    state,
                    constants,
                    scratch,
                )
            )
            variant = (batch.ladder_index, numerical_signature(batch.timesteps), id(schedule))
            key = (slot, signature, variant)
            missing = capture_required(key not in self.graphs, self.groups, self.device)
            if missing:
                torch.cuda.current_stream(self.device).synchronize()
                resident = self._slots.get(slot)
                if resident is not None and resident[0] != signature:
                    self.release_slot(slot)
                    resident = None
                if resident is None and len(self._slots) >= self.capacity:
                    self.release_slot(next(iter(self._slots)))
                self._discard_graph(key)
                restore = restore_samples(operation)
                try:
                    graph = CudaGraph[TensorOutput](
                        device=self.device,
                        stream=self.capture_stream,
                        pool=self.graph_pool,
                        device_pools=self.device_pools,
                    )
                    graph.capture(
                        operation,
                        keepalive=(batch, state, constants, scratch, schedule),
                        restore=restore,
                    )
                    self.graphs[key] = graph
                except BaseException:
                    self._discard_graph(key)
                    raise
                finally:
                    del restore
                if resident is None:
                    self._slots[slot] = (signature, input_key)
            self._slots.move_to_end(slot)
            return self.graphs[key].replay(), "graph_capture" if missing else "graph_replay"

    def _discard_graph(self, key: tuple[Hashable, Hashable, Hashable]) -> None:
        graph = self.graphs.pop(key, None)
        if graph is not None:
            graph.close()
        if not self.graphs and self.graph_pool is not None:
            self.graph_pool = torch.cuda.graph_pool_handle()
            self.device_pools = capture_pools(self.device_pools)

    def release_slot(self, slot: Hashable) -> None:
        """Release a drained slot's graphs before its numerical backing is reused."""

        self._slots.pop(slot, None)
        for key in tuple(self.graphs):
            if key[0] == slot:
                self._discard_graph(key)

    def release_inputs(self, key: Hashable) -> None:
        """Release drained prepared inputs after retiring their dependent graphs."""

        for slot, (_signature, resident_key) in tuple(self._slots.items()):
            if resident_key == key:
                self.release_slot(slot)
        self._sparse.pop(key, None)
        self.prepared_inputs.pop(key, None)

    def prepare_inputs(self, key: Hashable, build: Callable[[], InputT]) -> InputT:
        """Retain prepared numerical inputs and retire dependent graphs before their backing."""

        if key not in self.prepared_inputs:
            if len(self.prepared_inputs) >= self.capacity:
                if self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).synchronize()
                retired = next(iter(self.prepared_inputs))
                self.release_inputs(retired)
            self.prepared_inputs[key] = build()
        self.prepared_inputs.move_to_end(key)
        return cast(InputT, self.prepared_inputs[key])

    def close(self) -> None:
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        self.graph_pool = None
        self.device_pools.clear()
        self._slots.clear()
        self.prepared_inputs.clear()
        self._sparse.clear()
        self._context_buffers.clear()
        self._output_storage = None
        self._closed = True

    @torch.inference_mode()
    def prepare_flow(
        self,
        runner: ModelRunner,
        entry: ModelEntry,
        latent_pool: LatentPool,
        tokenizer: object,
        patch_size: int | None,
        shapes: tuple[DiffusionShape, ...],
        *,
        capture: bool,
    ) -> None:
        """Prepare independent diffusion calls using their bound input storage."""

        from functools import partial

        from .model_runner import capture_image_parameters

        generation = self.generation
        if generation is None or entry.input_buffers is None:
            raise ValueError("flow preparation requires a generation binding and input buffers")
        buffers = entry.input_buffers
        forward = partial(runner.batch_forward, entry)
        packed = cast(TextMixin, runner.model).text_backbone.attention_mode is AttentionMode.PACKED

        cache = runner.kv_cache
        for shape in sorted(
            shapes,
            key=lambda value: value.rows * value.height * value.width * value.cfg_branches,
            reverse=True,
        ):
            image = capture_image_parameters(
                shape.cfg_branches, steps=2, height=shape.height, width=shape.width
            )
            guide = build_flow_cfg_plan(
                cfg_text_scale=image.cfg_text_scale,
                cfg_img_scale=image.cfg_img_scale,
                recipe=generation.cfg_recipe,
                renorm=image.cfg_renorm_type,
                renorm_min=image.cfg_renorm_min,
                use_cfg=True,
            )
            prefixes = tuple(
                resolve_prefix(
                    runner.flow_prompt,
                    generation.branch_source(branch),
                    image_prompt="",
                    negative_prompt="",
                    negative_token_ids=(),
                    tokenizer=tokenizer,
                )[0]
                for branch in guide.branches
            )
            page_counts = tuple(ceil_div(len(prefix), cache.cache.page_size) for prefix in prefixes)
            with (
                cache.startup_pages(shape.rows * sum(page_counts)) as scratch,
                latent_pool.startup_values(
                    shape.rows, generation.image_tokens(shape.height, shape.width)
                ) as latents,
            ):
                # Private startup values use the same numerical latent representation.
                for value in latents:
                    value.zero_()
                branch_pages = []
                cursor = 0
                for _ in range(shape.rows):
                    for count in page_counts:
                        branch_pages.append(scratch[cursor : cursor + count])
                        cursor += count
                repeated = prefixes * shape.rows
                selected = tuple(index for index, prefix in enumerate(repeated) if prefix)
                if selected:
                    prefix = stage_text(
                        buffers,
                        cache,
                        tuple(repeated[index] for index in selected),
                        tuple(branch_pages[index] for index in selected),
                        packed=packed,
                        selection=TokenSelection.HIDDEN,
                        slots=tuple(1 + index // shape.cfg_branches for index in selected),
                    )
                    if capture:
                        runner.capture_batch(entry, prefix, forward)
                    # Prefix KV is the numerical input of the subsequent denoiser.
                    runner.run_batch(entry, prefix, forward, eligible=True)
                batch = _stage_flow(
                    buffers,
                    generation,
                    patch_size,
                    packed,
                    runner.worker_config.block_size,
                    shape,
                    latents,
                    repeated,
                    tuple(branch_pages),
                )
                if capture:
                    runner.capture_batch(entry, batch, forward)
                else:
                    runner.eager_batch(entry, batch, forward)


def _stage_flow(
    buffers: InputBuffers,
    generation: ImageDiffusion,
    patch_size: int | None,
    packed: bool,
    block_size: int,
    shape: DiffusionShape,
    latents: tuple[torch.Tensor, ...],
    prefixes: tuple[tuple[int, ...], ...],
    pages: tuple[tuple[int, ...], ...],
) -> InputBatch:
    rows = shape.rows * shape.cfg_branches
    geometry = tuple(
        prepare_denoise_inputs(
            generation,
            latents[index // shape.cfg_branches],
            shape.height,
            shape.width,
            temporal=len(prefixes[index]),
            patch_size=patch_size,
        )
        for index in range(rows)
    )
    positions, indexes, conditioning, queries, local_text = zip(*geometry, strict=True)
    attention = physical_columns(
        pages=pages,
        prefix_lens=tuple(map(len, prefixes)),
        query_lens=queries,
        causal_rows=(False,) * rows,
        write_rows=(False,) * rows,
        positions=indexes,
        token_rows=(False,) * rows,
        text_local_indices=local_text,
        width=min(buffers.max_blocks_per_row, bucketed_length(max(1, max(map(len, pages))))),
        block_size=block_size,
        packed=packed,
    )
    flow_rows = tuple(
        ForwardRow(
            forward_mode=PipelineStage.DENOISING,
            positions=positions[index],
            timestep=torch.tensor(
                [generation.timestep(DiffusionConfig(steps=2, timestep_shift=3.0), 0)]
            ),
            latent=latents[index // shape.cfg_branches],
            flow_conditioning=conditioning[index],
            image_tokens=queries[index],
            image_height=shape.height,
            image_width=shape.width,
            request_pool_idx=1 + index // shape.cfg_branches,
            seq_len=len(prefixes[index]),
            causal=False,
        )
        for index in range(rows)
    )
    return buffers.stage(
        flow_rows,
        forward_mode=PipelineStage.DENOISING,
        attention=attention,
    )


def prepare_denoise_inputs(
    flow: ImageDiffusion,
    latent: torch.Tensor,
    height: int,
    width: int,
    *,
    temporal: int,
    patch_size: int | None,
    positions: dict[int, tuple[torch.Tensor, torch.Tensor, int, tuple[int, ...]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, FlowPatches | None, int, tuple[int, ...]]:
    """Share static coordinates while rebuilding conditioning from the current latent."""

    cached = None if positions is None else positions.get(temporal)
    if cached is None:
        image_tokens = flow.image_tokens(height, width)
        text_local: tuple[int, ...]
        if flow.latent_layout is LatentLayout.PATCH_TOKENS:
            latent_positions = get_flattened_position_ids_extrapolate(
                height,
                width,
                int(flow.latent_downsample),
                int(math.isqrt(flow.max_latent_tokens)),
            )
            query_tokens = image_tokens + int(flow.marker_tokens)
            attention_indexes = torch.stack(
                (
                    torch.full((query_tokens,), temporal, dtype=torch.long),
                    torch.zeros(query_tokens, dtype=torch.long),
                    torch.zeros(query_tokens, dtype=torch.long),
                )
            )
            text_local = (0, query_tokens - 1)
        else:
            latent_positions = _spatial_positions(
                height,
                width,
                int(flow.latent_patch_size),
                temporal,
            )
            query_tokens = image_tokens
            attention_indexes = latent_positions
            text_local = ()
        cached = latent_positions, attention_indexes, query_tokens, text_local
        if positions is not None:
            positions[temporal] = cached
    latent_positions, attention_indexes, query_tokens, text_local = cached
    conditioning = (
        None
        if flow.latent_layout is LatentLayout.PATCH_TOKENS
        else flow.conditioning(
            flow.unpatchify(latent, height, width), height, width, patch_size=patch_size
        )
    )
    return latent_positions, attention_indexes, conditioning, query_tokens, text_local


def _spatial_positions(
    height: int,
    width: int,
    patch: int,
    temporal: int,
) -> torch.Tensor:
    """Build flattened temporal-height-width coordinates for latent patches."""

    grid_height = height // patch
    grid_width = width // patch
    y = torch.arange(grid_height, dtype=torch.long).repeat_interleave(grid_width)
    x = torch.arange(grid_width, dtype=torch.long).repeat(grid_height)
    return torch.stack((torch.full_like(x, int(temporal)), y, x))
