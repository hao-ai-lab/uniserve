"""Request-bound denoising with exact geometry and schedule variants."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, TypeVar, cast

import torch

from ..foundation.errors import invalid_descriptor
from ..foundation.math import bucketed_length, ceil_div
from ..models.generation import GenerationPipeline, LatentLayout
from ..nn.diffusion.cfg import Branch, CfgPlan, build_flow_cfg_plan
from ..nn.diffusion.integrator import euler_step
from ..nn.diffusion.schedule import DiffusionSchedule, x_pred_to_velocity
from ..nn.mesh import Communicator
from ..nn.vision import get_flattened_position_ids_extrapolate
from ..protocol.batch import ForwardMode, PipelineStage
from ..runtime.latent_pool import LatentPool
from .attention import physical_columns
from .cuda_graph import CudaGraph, GraphExecutionError
from .denoising import DenoisingStep
from .device_transfer import tensor_to_device
from .diffusion_state import DiffusionState
from .forward_batch import FlowPatches, ForwardBatch, TokenSelection
from .graph_inputs import DiffusionShape, MixedShape
from .input_buffers import InputBuffers
from .model_entry import ModelEntry, TensorOutput, capture_required
from .rows import ForwardRow
from .runners.prefill import stage_text

if TYPE_CHECKING:
    from .model_runner import ModelRunner

GeometryT = TypeVar("GeometryT")
logger = logging.getLogger(__name__)
MIN_MIXED_SERVICE_SPEEDUP = 1.03


def restore_samples(operation: DenoisingStep) -> Callable[[], None]:
    """Snapshot only solver state; learned predictions are disposable scratch."""

    snapshots = tuple(value.clone() for value in operation.samples)

    def restore() -> None:
        for value, saved in zip(operation.samples, snapshots, strict=True):
            value.copy_(saved)

    return restore


class DiffusionRunner:
    """Own graph-bound request slots while sharing the model's step binding.

    Physical slot identity and backing addresses participate in residency. Pooled
    storage can be reused by subsequent requests; graphs retain only numerical
    views, never request identities. Replacing backing or geometry retires the
    dependent graphs before their metadata is freed.
    """

    def __init__(
        self,
        bind_step: Callable[..., DenoisingStep] | None = None,
        signature: Callable[..., Hashable] | None = None,
        *,
        device: torch.device,
        capture_stream: torch.cuda.Stream | None,
        groups: tuple[Communicator, ...],
        capacity: int,
        generation: GenerationPipeline | None = None,
    ) -> None:
        if (bind_step is None) != (signature is None):
            raise ValueError("diffusion execution requires its numerical model binding")
        self.generation = generation
        self.bind_step = bind_step
        self.signature = signature
        self.device = device
        self.capture_stream = capture_stream
        self.graph_pool = torch.cuda.graph_pool_handle() if capture_stream is not None else None
        self.graphs: dict[tuple[Hashable, Hashable, Hashable], CudaGraph[TensorOutput]] = {}
        self.groups = groups
        self.capacity = max(2, capacity)
        self._slots: OrderedDict[Hashable, tuple[Hashable, Hashable]] = OrderedDict()
        self._geometry: OrderedDict[Hashable, object] = OrderedDict()
        self._closed = False

    def initialize(
        self, height: int, width: int, steps: int, timestep_shift: float
    ) -> DiffusionState:
        """Prepare a request's numerical schedule independently of accepted progress."""

        generation = self.generation
        if generation is None:
            raise ValueError("model has no image generation pipeline")
        if steps <= 0:
            raise ValueError("num_steps must be positive")
        return DiffusionState(
            geometry=(height, width, steps, timestep_shift),
            timesteps=tuple(
                generation.schedule_pair(steps, timestep_shift, step) for step in range(steps)
            ),
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
            positions, indexes, conditioning, query_tokens, text_local = denoise_geometry(
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
        if generation.prediction in {"x", "x_prediction", "x_pred"}:
            velocity = x_pred_to_velocity(velocity, current, timestep)
        elif generation.prediction != "velocity":
            raise invalid_descriptor(f"unsupported flow prediction {generation.prediction!r}")
        current.copy_(euler_step(current, velocity, timestep, next_timestep))

    @torch.inference_mode()
    def warmup(self, tensors: object, metadata: object, schedule: DiffusionSchedule) -> None:
        if self.bind_step is None:
            raise ValueError("model has no tensor-bound denoising step")
        operation = self.bind_step(tensors, metadata, 0, schedule)
        restore = restore_samples(operation)
        try:
            operation()
        finally:
            restore()
            torch.cuda.current_stream(self.device).synchronize()

    @torch.inference_mode()
    def step(
        self,
        tensors: object,
        metadata: object,
        step: int,
        schedule: DiffusionSchedule,
        *,
        slot: Hashable,
        geometry: Hashable,
    ) -> tuple[TensorOutput, str]:
        if self._closed:
            raise GraphExecutionError("denoising runner is closed")
        if self.bind_step is None or self.signature is None:
            raise ValueError("model has no tensor-bound denoising step")
        operation = self.bind_step(tensors, metadata, step, schedule)
        if self.capture_stream is None:
            return operation(), "eager"
        backing = tuple(
            (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
            for value in operation.samples
        )
        signature = (self.signature(tensors, metadata), backing, id(metadata))
        variant = (step, id(schedule))
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
                    device=self.device, stream=self.capture_stream, pool=self.graph_pool
                )
                graph.capture(operation, keepalive=(tensors, metadata, schedule), restore=restore)
                self.graphs[key] = graph
            except BaseException:
                self._discard_graph(key)
                raise
            finally:
                del restore
            if resident is None:
                self._slots[slot] = (signature, geometry)
        self._slots.move_to_end(slot)
        return self.graphs[key].replay(), "graph_capture" if missing else "graph_replay"

    def _discard_graph(self, key: tuple[Hashable, Hashable, Hashable]) -> None:
        graph = self.graphs.pop(key, None)
        if graph is not None:
            graph.close()
        if not self.graphs and self.graph_pool is not None:
            self.graph_pool = torch.cuda.graph_pool_handle()

    def release_slot(self, slot: Hashable) -> None:
        """Release a drained slot's graphs before its numerical backing is reused."""

        self._slots.pop(slot, None)
        for key in tuple(self.graphs):
            if key[0] == slot:
                self._discard_graph(key)

    def discard_geometry(self, geometry: Hashable) -> None:
        for slot, (_signature, resident_geometry) in tuple(self._slots.items()):
            if resident_geometry == geometry:
                self.release_slot(slot)

    def prepare_geometry(self, key: Hashable, build: Callable[[], GeometryT]) -> GeometryT:
        """Retain immutable numerical geometry and retire dependent graph bindings first."""

        if key not in self._geometry:
            if len(self._geometry) >= self.capacity:
                if self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).synchronize()
                retired = next(iter(self._geometry))
                self.discard_geometry(retired)
                del self._geometry[retired]
            self._geometry[key] = build()
        self._geometry.move_to_end(key)
        return cast(GeometryT, self._geometry[key])

    def close(self) -> None:
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        self.graph_pool = None
        self._slots.clear()
        self._geometry.clear()
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
        mixed: tuple[MixedShape, ...],
        *,
        capture: bool,
    ) -> None:
        """Prepare flow and mixed calls using their bound input storage and fixed qualification rule."""

        from functools import partial

        from .model_runner import capture_image_parameters

        generation = self.generation
        if generation is None or entry.input_buffers is None:
            raise ValueError("flow preparation requires a generation binding and input buffers")
        buffers = entry.input_buffers
        forward = partial(runner.batch_forward, entry)
        tensorized = runner.model.tensorized_mixed

        if not capture:
            shapes = tuple(
                dict.fromkeys(
                    (
                        *shapes,
                        *(
                            DiffusionShape(x.flow_rows, x.height, x.width, x.cfg_branches)
                            for x in mixed
                        ),
                    )
                )
            )
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
                generation.prefix(
                    generation.branch_source(branch),
                    image_prompt="",
                    negative_prompt="",
                    negative_token_ids=(),
                    tokenizer=tokenizer,
                )[0]
                for branch in guide.branches
            )
            mixed_shapes = tuple(
                value
                for value in mixed
                if value.flow_rows == shape.rows
                and value.height == shape.height
                and value.width == shape.width
                and value.cfg_branches == shape.cfg_branches
            )
            text_count = max((value.decode_rows for value in mixed_shapes), default=0)
            page_counts = tuple(ceil_div(len(prefix), cache.block_size) for prefix in prefixes)
            with (
                cache.startup_pages(shape.rows * sum(page_counts) + text_count) as scratch,
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
                token_pages = tuple((page,) for page in scratch[cursor:])
                repeated = prefixes * shape.rows
                selected = tuple(index for index, prefix in enumerate(repeated) if prefix)
                if selected:
                    prefix = stage_text(
                        buffers,
                        cache,
                        tuple(repeated[index] for index in selected),
                        tuple(branch_pages[index] for index in selected),
                        packed=tensorized,
                        selection=TokenSelection.HIDDEN,
                        slots=tuple(1 + index // shape.cfg_branches for index in selected),
                    )
                    if capture:
                        runner.capture_batch(entry, prefix, forward)
                    # Prefix KV is the numerical input of the subsequent denoiser.
                    runner.run_batch(entry, prefix, forward, eligible=True)
                if text_count:
                    prompt = stage_text(
                        buffers,
                        cache,
                        ((0,),) * text_count,
                        token_pages,
                        packed=tensorized,
                    )
                    forward(prompt)

                def stage(decode_rows: int, include_flow: bool = True) -> ForwardBatch:
                    if not include_flow:
                        return stage_text(
                            buffers,
                            cache,
                            ((0,),) * decode_rows,
                            token_pages[:decode_rows],
                            packed=tensorized,
                            prefixes=(1,) * decode_rows,
                            decode=True,
                        )
                    return _stage_flow(
                        buffers,
                        generation,
                        patch_size,
                        tensorized,
                        runner.worker_config.block_size,
                        shape,
                        latents,
                        repeated,
                        tuple(branch_pages),
                        token_pages[:decode_rows],
                    )

                if capture:
                    runner.capture_batch(entry, stage(0), forward)
                else:
                    runner.eager_batch(entry, stage(0), forward)
                for item in mixed_shapes:
                    if capture:
                        runner.capture_batch(entry, stage(item.decode_rows), forward)
                    else:
                        runner.eager_batch(entry, stage(item.decode_rows), forward)
                    # Preserve the existing service-time qualification criterion,
                    # now measured after all involved graphs are already resident.
                    mixed_us, graph = _measure_flow(runner, entry, lambda: stage(item.decode_rows))
                    decode_us, _ = _measure_flow(
                        runner, entry, lambda: stage(item.decode_rows, False), eligible=graph
                    )
                    flow_us, _ = _measure_flow(runner, entry, lambda: stage(0), eligible=graph)
                    serial_us = decode_us + flow_us
                    eligible = (
                        buffers.device.type != "cuda"
                        or serial_us / mixed_us >= MIN_MIXED_SERVICE_SPEEDUP
                    )
                    runner.qualify_mixed(item, eligible)
                    logger.info(
                        "evaluated mixed execution bucket=%r mixed_us=%d serial_us=%d service_eligible=%s",
                        item,
                        mixed_us,
                        serial_us,
                        eligible,
                    )


def _stage_flow(
    buffers: InputBuffers,
    generation: GenerationPipeline,
    patch_size: int | None,
    tensorized: bool,
    block_size: int,
    shape: DiffusionShape,
    latents: tuple[torch.Tensor, ...],
    prefixes: tuple[tuple[int, ...], ...],
    pages: tuple[tuple[int, ...], ...],
    token_pages: tuple[tuple[int, ...], ...],
) -> ForwardBatch:
    rows = shape.rows * shape.cfg_branches
    text = len(token_pages)
    geometry = tuple(
        denoise_geometry(
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
    token_positions = tuple(torch.tensor([1], dtype=torch.int64) for _ in range(text))
    token_indexes = tuple(torch.tensor([[1], [0], [0]]) for _ in range(text))
    all_pages = (*token_pages, *pages)
    attention = physical_columns(
        pages=all_pages,
        prefix_lens=(*(1,) * text, *map(len, prefixes)),
        query_lens=(*(1,) * text, *queries),
        causal_rows=(*(True,) * text, *(False,) * rows),
        write_rows=(*(True,) * text, *(False,) * rows),
        positions=(*token_indexes, *indexes),
        token_rows=(*(True,) * text, *(False,) * rows),
        text_local_indices=(*((),) * text, *local_text),
        width=min(buffers.max_blocks_per_row, bucketed_length(max(1, max(map(len, all_pages))))),
        block_size=block_size,
        packed=tensorized,
    )
    token_rows = tuple(
        ForwardRow(
            forward_mode=ForwardMode.DECODE,
            token_ids=torch.zeros(1, dtype=torch.int64),
            positions=token_positions[index],
            selection=TokenSelection.LAST_LOGITS,
            request_pool_idx=index + 1,
            seq_len=1,
            write_kv=True,
        )
        for index in range(text)
    )
    flow_rows = tuple(
        ForwardRow(
            forward_mode=PipelineStage.DENOISING,
            positions=positions[index],
            timestep=torch.tensor([generation.schedule_pair(2, 3.0, 0)[0]]),
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
        (*token_rows, *flow_rows),
        forward_mode=ForwardMode.MIXED if text else PipelineStage.DENOISING,
        attention=attention,
    )


def _measure_flow(
    runner: ModelRunner,
    entry: ModelEntry,
    prepare: Callable[[], ForwardBatch],
    *,
    eligible: bool = True,
) -> tuple[int, bool]:
    import time
    from functools import partial

    device = entry.device
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    started = time.perf_counter_ns()
    result = runner.run_batch(
        entry, prepare(), partial(runner.batch_forward, entry), eligible=eligible
    )
    if device.type == "cuda":
        end.record()
        end.synchronize()
        elapsed = max(1, round(start.elapsed_time(end) * 1000))
    else:
        elapsed = max(1, (time.perf_counter_ns() - started) // 1000)
    return elapsed, result.stats is not None and bool(
        result.stats.cuda_graph_captures or result.stats.cuda_graph_replays
    )


def denoise_geometry(
    flow: GenerationPipeline,
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
            query_tokens = image_tokens + int(flow.commit_marker_tokens)
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
            flow.materialization_latent(latent, height, width), height, width, patch_size=patch_size
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
