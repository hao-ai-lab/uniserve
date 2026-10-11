"""Build WorkerInfo from a loaded model and worker resource settings.

``Worker.__init__`` calls ``build_worker_layout`` after ``ModelExecutor`` has
bound the model and ``resolve_request_capacity`` has fitted the request pool.
The resulting ``WorkerLayout`` carries the ``WorkerInfo`` the worker reports
to the engine (supported calls, batch bounds, request slots, KV cache
geometry, latent pages, buffer pool size, component products) together with
the allocation sizes the worker then reserves.

Two sizing paths exist. A token worker (no ``VideoPostprocessor`` and no
media request state) sizes paged KV, latent pages and text input staging;
given a storage grant, its KV pool is sized from, or with a configured
``max_total_tokens`` checked against, the grant left after its fixed
allocations. A request-tensor worker (media) holds fixed per-request state
and products and has no KV cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from torch import nn

from uniserve.distributed.mesh import Communicator
from uniserve.math import ceil_div
from uniserve.media import image
from uniserve.model import (
    CausalLM,
    PatchEncoder,
    VideoDenoiser,
    VideoPostprocessor,
)
from uniserve.processing import ImageProcessor
from uniserve.runtime.kv_cache import Planes
from uniserve.tensors import BufferConfig
from uniserve_worker.bootstrap.cache import (
    cache_info,
    plan_cache,
    resident_width,
)
from uniserve_worker.bootstrap.capacity import (
    ArenaCapacity,
    LatentPoolPlan,
    call_window,
    derive_runtime_kv_capacity,
    device_total_bytes,
    input_buffer_config,
    latent_pool_plan,
    latent_trajectory_bytes,
    local_product_storage_bytes,
    model_arena_capacity,
    product_storage_bytes,
    request_tensor_window,
    vision_tokens,
)
from uniserve_worker.bootstrap.components import (
    call_kinds,
    describe_components,
    holds_host_components,
    media_components,
    supported_calls,
)
from uniserve_worker.bootstrap.inputs import (
    capability,
    image_builder,
    media_builder,
)
from uniserve_worker.bootstrap.outputs import resolve_outputs
from uniserve_worker.config.deployment import ComponentConfig
from uniserve_worker.config.execution import (
    WorkerConfig,
    graph_storage_budget_bytes,
)
from uniserve_worker.errors import unsupported_setup
from uniserve_worker.model_executor.canvas_runner import canvas_staging_rows
from uniserve_worker.model_executor.component_binding import ComponentBinding
from uniserve_worker.model_executor.input_buffers import (
    TokenBufferConfig,
    buffered_kinds,
)
from uniserve_worker.model_executor.input_buffers import (
    input_buffer_config as call_buffer_config,
)
from uniserve_worker.model_executor.resources import media_state_buffers
from uniserve_worker.protocol.call import CALL_KINDS
from uniserve_worker.protocol.transfer import WorkerEndpoint
from uniserve_worker.protocol.worker_info import (
    ComponentInfo,
    VideoDenoiserInfo,
    WorkerInfo,
)
from uniserve_worker.storage.block_tables import BlockTables, GroupShape
from uniserve_worker.storage.cache_imports import cache_transfer_workspace_bytes
from uniserve_worker.storage.canvas_slots import (
    CanvasSlots,
    generating_denoiser,
)
from uniserve_worker.storage.decode_state import DecodeState

__all__ = [
    "WorkerLayout",
    "build_worker_layout",
]


@dataclass(frozen=True, slots=True)
class WorkerLayout:
    """Resolved worker allocations and their public capacity report.

    Attributes:
        info: The capacity report sent to the engine.
        arena: Latent pool, tensor store, device-product, transfer and
            host-lane budgets.
        input_config: Token input staging bounds; ``None`` without a
            ``CausalLM`` and on a request-tensor worker.
        fixed_device_bytes: ``(device, bytes)`` reserved outside the KV pool
            on each device of a token worker; empty on a request-tensor
            worker. With a KV cache, the primary device's entry is already
            charged against the KV grant; ``Worker.__init__`` checks each
            other CUDA device against its own grant.
        physical_buffer_pool_bytes: Bytes this rank's ``BufferPool`` backs.
            On a request-tensor rank that backs only some products it is
            smaller than ``info.buffer_pool_bytes``, and the worker builds a
            compact pool.
        planes: The K/V unit pool plan the worker allocates its pool from
            with ``info.kv_cache.num_units`` units; ``None`` without a
            ``CausalLM``.
        latent_pool: The latent pool plan the worker allocates when it
            advances samples, which ``info.latent_pages`` and
            ``info.latent_page_units`` advertise; ``None`` without a
            denoiser.
    """

    info: WorkerInfo
    arena: ArenaCapacity
    input_config: TokenBufferConfig | None
    fixed_device_bytes: tuple[tuple[str, int], ...]
    physical_buffer_pool_bytes: int
    planes: Planes | None
    latent_pool: LatentPoolPlan | None


def _exports_fabric_handles(device: object) -> bool:
    """Report whether this rank's device exports a handle another host imports.

    A rank without a CUDA device publishes nothing across a device transport,
    so it reports no fabric capability rather than probing one it cannot use.
    """
    import torch

    resolved = torch.device(str(device))
    if resolved.type != "cuda":
        return False
    from uniserve_kernels.peer_storage import exports_fabric_handles

    return exports_fabric_handles(resolved.index or 0)


def build_worker_layout(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    image_processor: ImageProcessor | None = None,
    queue_depth: int = 1,
    endpoint: WorkerEndpoint | None = None,
    capacity_group: Communicator | None = None,
    components: tuple[tuple[str, ComponentConfig], ...] = (),
    bindings: Mapping[str, ComponentBinding] | None = None,
    state_buffers: Mapping[str, BufferConfig] | None = None,
) -> WorkerLayout:
    """Resolve resource dimensions and the capacity report used by the worker.

    Model dimensions determine storage reservations. Admission additionally
    obeys every lane's bounds; these restrictions do not shrink the storage
    needed by warmup and graph capture.

    ``state_buffers`` defaults to the media request state derived from
    ``bindings``; a non-empty value, or a ``VideoPostprocessor`` in the model,
    selects the request-tensor path. For a token worker with a ``CausalLM``, a
    ``capacity_group`` spanning several ranks makes them agree on one KV page
    count through a collective, so every member must call this.

    Raises:
        WorkerError: ``queue_depth`` is not positive.
        ValueError: The KV pool does not fit the storage grant. Errors from
            the sizing helpers propagate.
    """
    if queue_depth <= 0:
        raise unsupported_setup("worker pipeline depth must be positive")

    dedicated = worker_config.role == "experts"
    supported = (
        frozenset()
        if dedicated
        else supported_calls(model, (name for name, _ in components))
    )

    endpoint = endpoint or WorkerEndpoint.local(rank=int(worker_config.rank))

    if state_buffers is None:
        state_buffers = media_state_buffers(
            bindings or {}, media_builder(model, worker_config)
        )

    # Media workers hold fixed request tensors; token workers size paged KV,
    # latent pools, and text staging instead.
    if capability(model, VideoPostprocessor) is not None or state_buffers:
        layout = _request_tensor_worker_layout(
            model,
            worker_config,
            state_buffers=state_buffers,
            bindings=bindings,
            queue_depth=queue_depth,
            endpoint=endpoint,
        )
    else:
        layout = _token_worker_layout(
            model,
            worker_config,
            image_processor=image_processor,
            queue_depth=queue_depth,
            endpoint=endpoint,
            capacity_group=capacity_group,
            bindings=bindings,
        )

    # A shared admission limit must be safe on every eligible lane. Keep it
    # in the layout so runtime allocation and the IPC handshake read the
    # same value.
    max_calls = layout.info.max_batch_calls
    max_tokens = layout.info.max_batch_tokens

    for lane in worker_config.lanes:
        max_calls = min(max_calls, lane.max_batch_calls or max_calls)
        max_tokens = min(max_tokens, lane.max_batch_tokens or max_tokens)

    # The per-path layouts report every call and media route the model
    # declares; the fields below narrow them to this placement's held
    # components.
    outputs = resolve_outputs(model, worker_config)
    held = tuple(name for name, _ in components)
    info = replace(
        layout.info,
        supported_calls=tuple(code for code in CALL_KINDS if code in supported),
        # Advertised routing and advertised work describe the same placement:
        # a call this worker will not accept is a call it does not route.
        media_components={}
        if dedicated
        else {
            call: component
            for call, component in media_components(model, held).items()
            if call in supported
        },
        fabric_handles=_exports_fabric_handles(worker_config.device),
        max_batch_calls=max_calls,
        max_batch_tokens=max_tokens,
        components=tuple(
            ComponentInfo(name, outputs.get(name, ())) for name, _ in components
        ),
    )
    return replace(layout, info=info)


def _token_worker_layout(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    image_processor: ImageProcessor | None,
    queue_depth: int,
    endpoint: WorkerEndpoint,
    capacity_group: Communicator | None,
    bindings: Mapping[str, ComponentBinding] | None,
) -> WorkerLayout:
    """Size token inputs, paged KV, and latent storage before admission.

    Without a configured ``max_total_tokens``, the KV pool takes the whole
    pages of its storage grant (``pool_storage_bytes``) left after every
    fixed allocation on the primary device: the buffer pool and its share of
    the device products, the latent pool and input staging placed there,
    block tables, decode state, KV import workspaces and the graph storage
    budget. A CUDA device requires that grant; another
    device without one takes ``derive_runtime_kv_capacity``'s default page
    count. A configured capacity is checked against the grant when one is
    given.
    """
    encoder_cache_entries = (
        worker_config.encoder_cache_entries
        if image_processor is not None
        else 0
    )

    # The unit pool is planned once; its unit size sizes the grant below and
    # the worker allocates the pool from the same plan.
    text = capability(model, CausalLM)
    planes = None if text is None else plan_cache(text, worker_config)
    unit_bytes = 0 if planes is None else planes.unit_bytes

    flow = image_builder(model)
    plan = latent_pool_plan(model, worker_config) if flow is not None else None
    latent_pool_bytes = 0 if plan is None else plan.capacity_bytes

    max_vit_grid_tokens = vision_tokens(model, image_processor)
    vision = capability(model, PatchEncoder)
    # Encoder representation remains meaningful on a worker without a text
    # decoder. Query the numerical output instead of borrowing language width.
    if vision is None or not max_vit_grid_tokens:
        max_vision_feature_bytes = 0
    else:
        stride = vision.patch_size * vision.downsample
        feature = vision.output_layout(image.Config(stride, stride))["features"]
        max_vision_feature_bytes = (
            max_vit_grid_tokens * feature.shape[1] * feature.dtype.itemsize
        )
    # One image's whole latent trajectory in the pool's dtype.
    max_latent_feature_bytes = (
        0
        if flow is None or plan is None
        else latent_trajectory_bytes(
            flow.max_tokens, plan.latent_width, plan.dtype.itemsize
        )
    )

    capacity = None
    kv_cache = None
    unresolved_window = call_window(
        int(queue_depth), int(worker_config.max_batch_calls)
    )
    # The scheduler's encoder cache retains up to ``encoder_cache_entries``
    # features and places the next image's feature before it evicts the
    # least recently used one, so the pool holds one feature beyond them.
    buffer_pool_bytes = (encoder_cache_entries + 1) * max(
        max_latent_feature_bytes, max_vision_feature_bytes
    )
    # Standalone encoders also publish persistent tensors. Their declared
    # products retain one bounded allocation per resident request.
    buffer_pool_bytes += (
        worker_config.max_request_pool_size
        * product_storage_bytes(resolve_outputs(model, worker_config))
    )

    arena_args = {
        "queue_depth": int(queue_depth),
        "max_latent_feature_bytes": max_latent_feature_bytes,
        "max_vision_feature_bytes": max_vision_feature_bytes,
        "unit_bytes": unit_bytes,
    }

    # A provisional arena without KV units supplies the device-product bytes,
    # which do not depend on the unit count; the final arena below is
    # recomputed with the granted units, which bound its transfer bytes.
    arena = model_arena_capacity(
        model,
        worker_config,
        num_units=0,
        bindings=None,
        state_buffers=None,
        **arena_args,
    )
    input_config = (
        input_buffer_config(
            model, worker_config, planes=planes, processor=image_processor
        )
        if planes is not None
        else None
    )

    devices = tuple(
        dict.fromkeys(
            (
                worker_config.device,
                worker_config.generation_device or worker_config.device,
            )
        )
    )
    # Every device has its own persistent and product backing; the latent
    # pool lives on the generation device, or the primary device without one.
    fixed_bytes = dict.fromkeys(
        devices, buffer_pool_bytes + arena.device_product_bytes // len(devices)
    )
    fixed_bytes[worker_config.generation_device or worker_config.device] += (
        latent_pool_bytes
    )

    if text is not None:
        assert planes is not None and input_config is not None
        from uniserve_worker.model_executor.canvas_runner import CanvasRunner
        from uniserve_worker.model_executor.graph_inputs import (
            prefill_captures,
        )
        from uniserve_worker.protocol.call import ForwardMode, MediaCall

        # Reserve the input staging ``ModelExecutor`` allocates for each call
        # this rank owns and each lane serving it, including the prefill row
        # widening for captured graphs. The worker-wide row and token bounds
        # are used here, so the reservation covers lanes with narrower bounds.
        max_rows = min(
            worker_config.max_batch_calls, worker_config.max_request_pool_size
        )
        staged = buffered_kinds(diffusion=flow is not None)
        for name, calls in describe_components(model).items():
            placement = None if bindings is None else bindings.get(name)
            if bindings is not None and (
                placement is None or not placement.owns
            ):
                continue
            for call in calls:
                kinds = call_kinds((call,)) & staged
                target = (
                    worker_config.generation_device or worker_config.device
                    if kinds
                    & {MediaCall.LATENT_ENCODING, MediaCall.IMAGE_DECODING}
                    else worker_config.device
                    if placement is None
                    else str(placement.device)
                )
                for lane in worker_config.lanes or (None,):
                    selected = (
                        kinds
                        if lane is None
                        else kinds.intersection(lane.call_kinds)
                    )
                    if not selected:
                        continue
                    fields = input_config
                    if (
                        ForwardMode.PREFILL in selected
                        and torch.device(target).type == "cuda"
                    ):
                        captures = prefill_captures(
                            worker_config,
                            max_rows=max_rows,
                            max_tokens=input_config.max_text_tokens,
                            image_builder=flow is not None,
                            feature_injection=image_processor is not None
                            and image_processor.feature_injection is not None,
                            # Mask variants share the same storage bounds.
                            device_causality=True,
                        )
                        fields = (
                            replace(
                                fields,
                                max_rows=max(
                                    fields.max_rows,
                                    *(shape.row_bucket for shape in captures),
                                ),
                            )
                            if captures
                            else fields
                        )
                    elif (
                        ForwardMode.TOKEN_DENOISING in selected
                        and torch.device(target).type == "cuda"
                        and worker_config.graph_policy != "off"
                    ):
                        # Canvas readout graphs stage padding sequences
                        # beyond their canvases.
                        fields = replace(
                            fields,
                            max_rows=canvas_staging_rows(fields.max_rows),
                        )
                    _, allocation = call_buffer_config(
                        next(iter(selected)), fields
                    )
                    copies = (
                        worker_config.expert_microbatches
                        if selected
                        & {
                            ForwardMode.PREFILL,
                            ForwardMode.DECODE,
                            ForwardMode.TOKEN_DENOISING,
                        }
                        else 1
                    )
                    fixed_bytes[target] = fixed_bytes.get(
                        target, 0
                    ) + copies * sum(
                        field.nbytes for field in allocation.buffers().values()
                    )

                # A token denoiser that generates the deployment's canvases
                # keeps every request slot's canvas resident. Staging and
                # sampler scratch are private to each concurrent execution.
                denoiser = (
                    generating_denoiser(call.module)
                    if ForwardMode.TOKEN_DENOISING in kinds
                    else None
                )
                sampling = worker_config.canvas_sampling
                if denoiser is not None and sampling is not None:
                    fixed_bytes[target] = (
                        fixed_bytes.get(target, 0)
                        + CanvasSlots.denoiser_bytes(
                            denoiser,
                            request_pool_size=worker_config.max_request_pool_size,
                            history_depth=sampling.stability_threshold,
                        )
                        + worker_config.expert_microbatches
                        * CanvasRunner.sampler_bytes(
                            denoiser,
                            max_rows=input_config.max_rows,
                            history_depth=sampling.stability_threshold,
                            device_type=torch.device(target).type,
                        )
                    )

        # Block tables, decode state and the KV import workspaces are charged
        # to the primary device and shared by every lane. The workspaces are
        # sized by the same unresolved-call window the worker gives
        # ``CacheImports``, for the largest page of any group.
        schemas = (
            BlockTables.buffers(
                groups=tuple(
                    GroupShape(
                        group.page_tokens, group.units_per_page, group.window
                    )
                    for group in planes.groups
                ),
                request_pool_size=worker_config.max_request_pool_size,
                width=resident_width(
                    planes,
                    max_sequence_tokens=worker_config.max_model_len,
                ),
            ),
            DecodeState.buffers(
                request_pool_size=worker_config.max_request_pool_size,
                vocab_size=text.backbone.vocab_size,
                continuation_width=1,
                logits_dtype=getattr(torch, worker_config.model_dtype),
            ),
        )
        fixed_bytes[worker_config.device] += sum(
            field.nbytes for schema in schemas for field in schema.values()
        ) + cache_transfer_workspace_bytes(
            planes.groups, capacity=unresolved_window
        )

        page_shapes = tuple(
            (group.page_tokens, group.units_per_page) for group in planes.groups
        )
        resident_copies, co_resident_units = _kv_residency_shape(
            worker_config,
            unit_bytes=unit_bytes,
            co_resident_bytes=fixed_bytes[worker_config.device],
        )
        capacity = derive_runtime_kv_capacity(
            pages=page_shapes,
            max_total_tokens=worker_config.max_total_tokens,
            unit_bytes=unit_bytes,
            device=worker_config.device,
            available_bytes=worker_config.pool_storage_bytes,
            resident_copies=resident_copies,
            co_resident_units=co_resident_units,
        )

        # Ranks sharing one logical pool must agree on its unit count; the
        # minimum keeps every member within its own storage grant.
        if capacity_group is not None and capacity_group.size > 1:
            units = torch.tensor(
                capacity.num_units,
                dtype=torch.int64,
                device=capacity_group.device,
            )
            capacity_group.all_reduce(units, op="min")
            capacity = replace(capacity, num_units=int(units.item()))
        kv_cache = cache_info(text, planes, num_units=capacity.num_units)

    # ``build_worker_layout`` replaces the supported calls and media routes
    # with the placement's narrowed values.
    supported = (
        ()
        if worker_config.role == "experts"
        else tuple(
            code for code in CALL_KINDS if code in supported_calls(model)
        )
    )
    arena = model_arena_capacity(
        model,
        worker_config,
        bindings=None,
        state_buffers=None,
        num_units=0 if capacity is None else capacity.num_units,
        **arena_args,
    )
    info = WorkerInfo(
        endpoint=endpoint,
        supported_calls=supported,
        max_batch_calls=int(worker_config.max_batch_calls),
        max_batch_tokens=int(worker_config.max_batch_tokens),
        request_slots=int(worker_config.max_request_pool_size),
        kv_cache=kv_cache,
        latent_page_units=0 if plan is None else plan.page_units,
        latent_pages=0 if plan is None else plan.num_pages,
        buffer_pool_bytes=buffer_pool_bytes,
        max_unresolved_calls=unresolved_window,
        media_components={}
        if worker_config.role == "experts"
        else dict(media_components(model, worker_config.deployment_components)),
        # The rank's host executor and transfer capacity are the ones the
        # arena reserved.
        host_lane_capacity=int(arena.host_lane_inflight),
        transfer_tickets=int(arena.transfer_tickets),
        encoder_cache_entries=encoder_cache_entries,
        encoder_entry_bytes=max(
            max_latent_feature_bytes, max_vision_feature_bytes
        )
        if encoder_cache_entries
        else 0,
    )
    return WorkerLayout(
        info=info,
        arena=arena,
        input_config=input_config,
        fixed_device_bytes=tuple(fixed_bytes.items()),
        physical_buffer_pool_bytes=buffer_pool_bytes,
        planes=planes,
        latent_pool=plan,
    )


def video_denoiser_info(
    denoiser: VideoDenoiser, *, canvases: tuple[image.Config, ...] = ()
) -> VideoDenoiserInfo:
    """Describe what the deployment's video denoiser serves.

    The tasks are the denoiser's, in its order. The schedule's sigma points
    include the clean endpoint the network never evaluates. The canvases are
    ``canvases``, those the deployment prepares and admits, or without any
    every canvas the denoiser offers (``VideoDenoiser.canvases``).
    """
    served = canvases or denoiser.canvases
    shifts = denoiser.schedule_shifts
    return VideoDenoiserInfo(
        tasks=denoiser.tasks,
        schedule_points=denoiser.num_steps + 1,
        video_shift=float(shifts["video"]),
        audio_shift=float(shifts["audio"]),
        canvases=tuple((canvas.width, canvas.height) for canvas in served),
    )


def _request_tensor_worker_layout(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    state_buffers: Mapping[str, BufferConfig],
    queue_depth: int,
    endpoint: WorkerEndpoint,
    bindings: Mapping[str, ComponentBinding] | None,
) -> WorkerLayout:
    """Describe request tensors, products, and persistent capacity.

    ``info.buffer_pool_bytes`` covers every product of every resident request
    and is the logical pool the engine assigns buffer offsets in; the physical
    pool backs only the products this rank produces or holds a consumer of
    (``local_product_storage_bytes``).
    """
    slots = int(worker_config.max_request_pool_size)
    depth = int(queue_depth)
    unresolved_window = request_tensor_window(depth, slots)
    max_calls = min(slots, int(worker_config.max_batch_calls))
    # Every rank advertises the pool geometry; only a rank that advances the
    # samples allocates it.
    plan = latent_pool_plan(model, worker_config)
    builder = media_builder(model, worker_config)

    arena = model_arena_capacity(
        model,
        worker_config,
        bindings=bindings,
        state_buffers=state_buffers,
        queue_depth=depth,
        num_units=0,
        max_latent_feature_bytes=0,
        max_vision_feature_bytes=0,
        unit_bytes=0,
    )

    # A host rank is one codec slot, the same on each of a host worker's
    # ranks, so the engine's ledger admits one host task per rank. Any other
    # worker's host lane is the arena's executor. ``Worker.__init__`` sizes
    # its ``HostLane`` by the same rule, so the two must change together.
    host = bindings is not None and holds_host_components(bindings)
    info = WorkerInfo(
        endpoint=endpoint,
        supported_calls=tuple(
            code for code in CALL_KINDS if code in supported_calls(model)
        ),
        max_batch_calls=max_calls,
        max_batch_tokens=max_calls,
        request_slots=slots,
        kv_cache=None,
        latent_page_units=0 if plan is None else plan.page_units,
        latent_pages=0 if plan is None else plan.num_pages,
        buffer_pool_bytes=slots
        * product_storage_bytes(resolve_outputs(model, worker_config)),
        max_unresolved_calls=unresolved_window,
        media_components=dict(
            media_components(model, worker_config.deployment_components)
        ),
        video_denoiser=video_denoiser_info(
            builder.denoiser, canvases=builder.canvases
        ),
        host_lane_capacity=1 if host else int(arena.host_lane_inflight),
        transfer_tickets=int(arena.transfer_tickets),
    )
    return WorkerLayout(
        info=info,
        arena=arena,
        input_config=None,
        fixed_device_bytes=(),
        planes=None,
        latent_pool=plan,
        physical_buffer_pool_bytes=slots
        * local_product_storage_bytes(
            resolve_outputs(model, worker_config),
            bindings=bindings or {},
            media_components=media_components(
                model, worker_config.deployment_components
            ),
        ),
    )


def _kv_residency_shape(
    worker_config: WorkerConfig,
    *,
    unit_bytes: int,
    co_resident_bytes: int,
) -> tuple[int, int]:
    """Describe the single KV pool and co-resident fixed allocations.

    Returns:
        ``(resident_copies, co_resident_units)`` for
        ``derive_runtime_kv_capacity``: one resident KV pool, and the units
        of the graph storage budget and ``co_resident_bytes``, each rounded
        up to whole units.
    """
    # Captured executables are held for the worker's lifetime, so they occupy
    # the same static budget as the KV pools and are reserved before the
    # request pool is sized.
    size = max(1, int(unit_bytes))
    graph_units = ceil_div(
        graph_storage_budget_bytes(device_total_bytes(worker_config.device)),
        size,
    )
    fixed_owner_units = ceil_div(max(0, int(co_resident_bytes)), size)

    return 1, graph_units + fixed_owner_units
