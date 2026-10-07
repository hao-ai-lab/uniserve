"""Size the fixed physical storage of one worker process.

Worker startup bounds each fixed storage owner of a worker process: text
input staging (``input_buffer_config``), the request pool
(``resolve_request_capacity``), the paged KV pool
(``derive_runtime_kv_capacity``), the latent pool (``latent_pool_plan``), and
the product, relay, transfer and host-lane bounds ``ArenaCapacity`` collects.
``uniserve_worker.bootstrap.report`` turns these into the ``WorkerInfo`` the
worker advertises and the sizes ``Worker.__init__`` allocates, and
``Worker.warmup`` rechecks the device grant with ``check_startup_storage``
once warmup has run.

Two kinds of worker are sized differently. A token worker (no
``VideoPostprocessor`` and no media request state) derives its bounds from
the queue depth and batch bounds and, without a configured
``kv_token_capacity``, gives paged KV what remains of the device grant. A
request-tensor worker holds fixed per-request state and products; on CUDA,
its request-slot count is fitted collectively so every rank of the worker
agrees on it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from uniserve.diffusion import Branch
from uniserve.distributed.mesh import Communicator
from uniserve.math import ceil_div
from uniserve.model import (
    CausalLM,
    PatchEncoder,
    VideoDecoder,
    VideoPostprocessor,
)
from uniserve.processing import (
    FeatureLayout,
    ImageProcessor,
    PatchTransform,
)
from uniserve.runtime.device import (
    canonical_device,
    device_storage_budget,
    process_device_bytes,
)
from uniserve.tensors import BufferConfig
from uniserve_worker.bootstrap.cache import (
    page_size,
    plan_cache,
    table_widths,
)
from uniserve_worker.bootstrap.components import media_components
from uniserve_worker.bootstrap.inputs import (
    capability,
    image_builder,
    media_builder,
)
from uniserve_worker.bootstrap.outputs import resolve_outputs
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.errors import unsupported_setup
from uniserve_worker.model_executor.component_binding import ComponentBinding
from uniserve_worker.model_executor.input_buffers import TokenBufferConfig
from uniserve_worker.model_executor.resources import (
    holds_samples,
    media_state_buffers,
)
from uniserve_worker.protocol.call import MediaCall
from uniserve_worker.protocol.tensor import OutputInfo
from uniserve_worker.storage.kv_cache import KVCacheManager
from uniserve_worker.storage.tensor_store import (
    TensorStore,
    device_product_capacity_bytes,
)

# ``ArenaCapacity.tensor_store`` allows this many generic persistent products
# per call slot of the arena, plus ``_DEVICE_PRODUCT_RETIREMENT_BATCHES``
# further batches of ``max_batch_calls`` calls.
_DEVICE_PRODUCTS_PER_CALL = 6
_DEVICE_PRODUCT_RETIREMENT_BATCHES = 1
# Upper bound on the reads ``ArenaCapacity.transfer_tickets`` lets a rank
# keep in flight for its queued calls; see ``request_tensor_arena_capacity``
# for the floor a request-tensor rank keeps above it.
_MAX_TRANSFER_ENTRIES = 256
# A token worker's host-lane bound; request-tensor workers derive theirs.
_HOST_LANE_INFLIGHT = 256
# Relay-arena bytes budgeted per (request slot, relay lane) row across one
# device's ``TensorStore`` relay arenas.
_REQUEST_RELAY_ROW_BYTES = 18
# Relay lanes beyond the unresolved-call window. ``Worker.__init__`` builds
# ``TensorStore`` with ``relay_depth = max_unresolved_calls + 1`` to match.
_REQUEST_RELAY_RETIREMENT_LANES = 1

DEFAULT_NUM_UNITS_FALLBACK = 4096
DEFAULT_MAX_BATCH_OPS = 1024
DEFAULT_MAX_REQUEST_POOL_SIZE = 128


def input_buffer_config(
    model: nn.Module,
    config: WorkerConfig,
    *,
    processor: ImageProcessor | None = None,
) -> TokenBufferConfig:
    """Size text input staging for one call on this worker.

    Rows are bounded by the request pool and by the worker's and every lane's
    call bound; with an image denoiser, the row bound is multiplied by the
    number of guidance ``Branch`` members. Tokens cover the admitted text span
    plus, when the processor declares feature injection, one image's feature
    span (encoder features or a generated image, with start and end tokens
    for ``FeatureLayout.FRAMED``), or the longest image denoising sequence,
    including ``flow_graph_shapes``, in every guidance branch, whichever is
    larger.

    Raises:
        ValueError: The model has no ``CausalLM``.
    """
    text = capability(model, CausalLM)
    if text is None:
        raise ValueError("text input staging requires a causal language model")

    max_rows = min(
        config.max_request_pool_size,
        config.max_batch_calls,
        *(
            lane.max_batch_calls or config.max_batch_calls
            for lane in config.lanes
        ),
    )
    # ``min`` of a single integer raises, so a worker without lanes takes
    # its own token bound directly.
    max_tokens = (
        min(
            config.max_batch_tokens,
            *(
                lane.max_batch_tokens or config.max_batch_tokens
                for lane in config.lanes
            ),
        )
        if config.lanes
        else config.max_batch_tokens
    )

    flow = image_builder(model)
    branches = 1 if flow is None else len(Branch)
    injection = None if processor is None else processor.feature_injection
    image_span = (
        0
        if injection is None
        else max(
            vision_tokens(model, processor),
            0 if flow is None else flow.max_tokens,
        )
        + (2 if injection.layout is FeatureLayout.FRAMED else 0)
    )
    text_tokens = max_tokens + image_span

    if flow is None:
        flow_tokens = 0
    else:
        from uniserve.media import image

        flow_tokens = branches * max(
            (
                flow.max_tokens + flow.framing,
                *(
                    flow.sequence_length(image.Config(height, width))
                    for height, width in config.flow_graph_shapes
                ),
            )
        )

    # Text and diffusion use separate homogeneous calls on this lane.
    call_tokens = max(text_tokens, flow_tokens)
    return TokenBufferConfig(
        max_rows=max_rows * branches,
        max_tokens=call_tokens,
        max_text_tokens=text_tokens,
        table_widths=table_widths(
            plan_cache(text, config),
            max_sequence_tokens=config.max_sequence_tokens,
            max_query_tokens=call_tokens,
        ),
        hidden_size=text.backbone.hidden_size,
        embedding_dtype=getattr(
            torch, config.model_dtype.removeprefix("torch.")
        ),
    )


def vision_tokens(model: nn.Module, processor: ImageProcessor | None) -> int:
    """Bound feature rows from the actual image transform and encoder stride.

    Returns zero when the model has no ``PatchEncoder``, or there is no
    processor or it has no ViT transform.
    """
    encoder = capability(model, PatchEncoder)
    if encoder is None or processor is None or processor.vit is None:
        return 0
    transform = processor.vit
    pixels = (
        transform.pixel_bound()
        if isinstance(transform, PatchTransform)
        else min(transform.resize.max_pixels, transform.resize.max_size**2)
    )
    return pixels // (encoder.patch_size * encoder.downsample) ** 2


def tensor_slot_capacity(
    schema: Mapping[str, BufferConfig],
    group: Communicator,
    *,
    maximum: int,
    minimum: int,
    available_bytes: int,
    auxiliary_bytes: Callable[[int], int],
) -> int:
    """Choose the largest slot count whose complete storage fits on every rank.

    ``auxiliary_bytes(slots)`` includes products and runtime arenas. Its cost
    may increase when fewer slots allow more in-flight outputs per request,
    so ranks agree on feasible counts rather than reducing local maxima.

    This is a collective over ``group``: every member must call it with the
    same bounds, because the per-count feasibility vectors are min-reduced
    elementwise. Only device fields of ``schema`` count against
    ``available_bytes``.

    Raises:
        ValueError: ``minimum`` is below one or exceeds ``maximum``.
        RuntimeError: No count in the range fits on every rank.
    """
    bytes_per_slot = sum(
        field.nbytes for field in schema.values() if not field.host
    )
    if minimum < 1 or maximum < minimum:
        raise ValueError("request tensor capacity requires valid slot bounds")

    candidates = range(minimum, maximum + 1)
    requirements = [
        count * bytes_per_slot + auxiliary_bytes(count) for count in candidates
    ]
    agreed = torch.tensor(
        [required <= available_bytes for required in requirements],
        dtype=torch.int32,
        device=group.device,
    )
    group.all_reduce(agreed, op="min")

    feasible = [
        count for count, fits in zip(candidates, agreed.cpu().tolist()) if fits
    ]
    if feasible:
        return feasible[-1]

    raise RuntimeError(
        "insufficient device storage for a common request tensor slot count: "
        f"candidate range {minimum}..{maximum}, local requirements "
        f"{requirements}, {available_bytes} bytes available"
    )


def request_tensor_window(queue_depth: int, request_slots: int) -> int:
    """Return how many unresolved calls one request slot may have in flight.

    Each slot's even share of ``queue_depth`` keeps one position reserved, so
    the window is that share minus one. ``loaded_worker_config`` in
    ``uniserve_worker.bootstrap.model_loader`` applies the same three
    positions per slot when it resolves the slot count.

    Raises:
        ValueError: ``request_slots`` is below one or ``queue_depth`` leaves
            fewer than three positions per slot.
    """
    if request_slots < 1 or queue_depth < 3 * request_slots:
        raise ValueError(
            "request tensor pipeline requires two unresolved outputs per slot"
        )
    return queue_depth // request_slots - 1


def product_storage_bytes(
    entry_outputs: Mapping[str, tuple[OutputInfo, ...]],
) -> int:
    """Size one request's complete product set.

    Each output's ``max_bytes`` is rounded up to ``BufferPool``'s 256-byte
    binding alignment.
    """
    return sum(
        ((output.max_bytes + 255) // 256) * 256
        for outputs in entry_outputs.values()
        for output in outputs
    )


def active_latent_capacity_tokens(
    per_image_tokens: int, concurrency_token_budget: int | None
) -> int:
    """Return the latent token capacity for concurrently denoised images.

    The capacity is ``concurrency_token_budget`` raised to at least one
    image's tokens, or one image's tokens without a budget; it is zero when
    an image has no tokens.
    """
    per_image = max(0, int(per_image_tokens))
    if per_image == 0:
        return 0
    if concurrency_token_budget is None:
        return per_image
    return max(per_image, int(concurrency_token_budget))


def local_product_storage_bytes(
    entry_outputs: Mapping[str, tuple[OutputInfo, ...]],
    *,
    bindings: Mapping[str, ComponentBinding],
    media_components: Mapping[MediaCall, str],
) -> int:
    """Size the persistent products this rank backs, from its placement.

    A rank backs every product it produces and every product a consumer it
    holds reads, each at its complete declared extent. The engine reserves a
    request's products whole when it admits the request, and the store returns
    a product's storage only once every reader on this rank has retired it, so
    a product that streams in rounds of media units is still backed whole: the
    rounds a consumer has not yet read stay bound while later rounds land.
    Producers and remote consumers each need the complete logical allocation
    because disjoint regions may subsequently be imported into it. Alignment
    follows BufferPool's allocation rules.

    A rank produces an entry's products when it is one of the producer's
    ``output_ranks``. Empty ``bindings`` count every entry's products.

    Returns:
        Bytes of one request's products backed on this rank.
    """
    consumers: dict[str, set[str]] = {}
    # These are the concrete persistent Tensor consumers of the video path.
    # Denoising state is resident; encoders consume decoded output buffers.
    for source_call, destination_call in (
        (MediaCall.MEDIA_READING, MediaCall.VISION_ENCODING),
        (MediaCall.MEDIA_READING, MediaCall.LATENT_ENCODING),
        (MediaCall.VISION_ENCODING, MediaCall.TEXT_ENCODING),
        (MediaCall.LATENT_ENCODING, MediaCall.LATENT_PREPARATION),
        (MediaCall.TEXT_ENCODING, MediaCall.LATENT_PREPARATION),
        (MediaCall.DENOISING, MediaCall.VIDEO_DECODING),
        (MediaCall.DENOISING, MediaCall.AUDIO_DECODING),
        (MediaCall.VIDEO_DECODING, MediaCall.VIDEO_ENCODING),
        (MediaCall.AUDIO_DECODING, MediaCall.AUDIO_ENCODING),
        (MediaCall.VIDEO_ENCODING, MediaCall.MUXING),
    ):
        source = media_components.get(source_call)
        destination = media_components.get(destination_call)
        if source is not None and destination is not None:
            consumers.setdefault(source, set()).add(destination)

    total = 0
    for entry, outputs in entry_outputs.items():
        if bindings:
            producer = bindings.get(entry)
            produces = (
                producer is not None
                and producer.process_group.global_rank in producer.output_ranks
            )
            consumes = any(
                consumer in bindings and bindings[consumer].owns
                for consumer in consumers.get(entry, ())
            )
            if not produces and not consumes:
                continue

        total += sum(
            ceil_div(output.max_bytes, 256) * 256 for output in outputs
        )

    return total


@dataclass(frozen=True)
class RuntimeKVCapacity:
    """Resolved KV unit capacity within the physical budget.

    Attributes:
        unit_bytes: Physical bytes one unit occupies on this rank.
        num_units: Units in the pool, including the unit-zero sentinel.
    """

    unit_bytes: int
    num_units: int


def latent_trajectory_bytes(
    latent_units: int,
    latent_width: int,
    dtype_bytes: int,
) -> int:
    """Calculate storage for one latent trajectory.

    The trajectory is ``latent_units`` rows of ``latent_width`` elements of
    ``dtype_bytes`` bytes each.

    Raises:
        ValueError: ``latent_units`` is negative or ``latent_width`` or
            ``dtype_bytes`` is below one.
    """
    units = int(latent_units)
    width = int(latent_width)
    element_bytes = int(dtype_bytes)
    if units < 0 or width < 1 or element_bytes < 1:
        raise ValueError("latent trajectory dimensions are invalid")
    return units * width * element_bytes


def latent_pool_capacity_bytes(
    *,
    request_pool_size: int,
    num_pages: int,
    page_units: int,
    latent_width: int,
    dtype_bytes: int,
    staging: bool = True,
) -> int:
    """Calculate double-buffered latent pool storage.

    Covers latent pages, step storage when the pool stages steps, page
    tables, and timestep metadata. The result must equal the
    ``persistent_bytes`` of the ``LatentPool`` built from the same
    arguments: ``Worker.__init__`` refuses a pool whose allocation disagrees
    with ``ArenaCapacity.latent_pool_bytes``.

    Raises:
        ValueError: A dimension is below one or ``num_pages`` is below two.
    """
    slots = int(request_pool_size)
    pages = int(num_pages)
    units = int(page_units)
    width = int(latent_width)
    element_bytes = int(dtype_bytes)
    if min(slots, units, width, element_bytes) < 1 or pages < 2:
        raise ValueError("latent pool dimensions are invalid")

    # One page stays reserved as the sentinel; only usable pages hold
    # trajectories.
    usable_pages = pages - 1
    # Two banks of every page, sentinel included; one int64 page index per
    # usable page; one float32 timestep per one-based slot plus an unused
    # row zero.
    storage = 2 * pages * units * width * element_bytes
    step_buffer = usable_pages * units * width * element_bytes if staging else 0
    page_table = usable_pages * 8
    timesteps = (slots + 1) * 4
    return storage + step_buffer + page_table + timesteps


@dataclass(frozen=True, slots=True)
class LatentPoolPlan:
    """The latent pool a worker's denoiser advances its samples in.

    A worker advertises this geometry, budgets its bytes and allocates exactly
    it. A KV-conditioned image denoiser's trajectories take pages of
    ``block_size`` latent tokens that the scheduler allocates, and the pool
    stages each step. A standalone denoiser's unit is one sample element:
    every request slot owns the same run of consecutive pages after the
    sentinel page, and the denoiser's runners stage their own steps.
    """

    request_pool_size: int
    num_pages: int
    page_units: int
    latent_width: int
    dtype: torch.dtype
    staging: bool

    @property
    def capacity_bytes(self) -> int:
        """Device bytes the ``LatentPool`` of this plan allocates."""
        return latent_pool_capacity_bytes(
            request_pool_size=self.request_pool_size,
            num_pages=self.num_pages,
            page_units=self.page_units,
            latent_width=self.latent_width,
            dtype_bytes=self.dtype.itemsize,
            staging=self.staging,
        )


def latent_pool_plan(
    model: nn.Module, worker_config: WorkerConfig
) -> LatentPoolPlan | None:
    """Plan the latent pool of a model with a denoiser.

    The plan follows the model and the resolved configuration alone, so every
    rank of a worker plans the same pool; ``max_request_pool_size`` must be
    resolved.

    An image denoiser's pages hold ``block_size`` patchified latent tokens,
    enough for ``kv_token_capacity`` tokens, when set, and at least one
    image. A video denoiser's pages hold ``page_units`` sample elements each,
    a fixed run of pages per request slot. Both reserve one extra sentinel
    page.

    Returns:
        The plan, or ``None`` when the model has neither denoiser.

    Raises:
        WorkerError: With ``UNSUPPORTED_SETUP`` when an image denoiser's
            ``model_dtype`` names no torch dtype.
        ValueError: ``image_builder`` or ``media_builder`` rejects the
            model's capabilities, an image denoiser's ``block_size`` is
            unresolved, or the video denoiser's sample modalities do not
            share one dtype.
    """
    slots = int(worker_config.max_request_pool_size)
    flow = image_builder(model)
    if flow is not None:
        dtype = getattr(
            torch, str(worker_config.model_dtype).removeprefix("torch."), None
        )
        if not isinstance(dtype, torch.dtype):
            raise unsupported_setup(
                f"unsupported latent dtype {worker_config.model_dtype!r}"
            )
        page_units = page_size(worker_config)
        units = active_latent_capacity_tokens(
            flow.max_tokens, worker_config.kv_token_capacity
        )
        return LatentPoolPlan(
            request_pool_size=slots,
            num_pages=ceil_div(units, page_units) + 1,
            page_units=page_units,
            latent_width=flow.denoiser.latent_channels
            * flow.denoiser.patch_size**2,
            dtype=dtype,
            staging=True,
        )

    builder = media_builder(model, worker_config)
    if builder is None:
        return None
    pages = builder.sample_pages
    return LatentPoolPlan(
        request_pool_size=slots,
        num_pages=slots * pages.pages + 1,
        page_units=pages.page_units,
        latent_width=1,
        dtype=pages.dtype,
        staging=False,
    )


@dataclass(frozen=True, slots=True)
class ArenaCapacity:
    """Budgets latent storage, device products, transfers, and CPU tasks.

    All budgets are for one worker arena.

    Attributes:
        latent_pool_bytes: Exact ``LatentPool`` allocation, or zero when this
            rank holds no latent pool.
        tensor_store: ``TensorStore`` capacity in resident generic persistent
            products per device.
        device_product_bytes: Device bytes of ``TensorStore``'s scalar
            product backing plus its request-relay arenas, summed over the
            worker's devices for a token worker and sized for one device for
            a request-tensor worker; also the store's relay byte bound.
        transfer_bytes: Byte budget of the rank's ``TransferCapacity``, which
            ``Worker.__init__`` scales further for request-tensor workers.
        transfer_tickets: Read tickets of the rank's ``TransferCapacity``.
        host_lane_inflight: In-flight task bound of a rank's host lane,
            except on a rank holding host components, which runs one task.
    """

    latent_pool_bytes: int
    tensor_store: int
    device_product_bytes: int
    transfer_bytes: int
    transfer_tickets: int
    host_lane_inflight: int


def call_window(queue_depth: int, max_calls: int) -> int:
    """Bound simultaneously live calls.

    The bound follows pipeline depth and per-batch capacity: it is the queue
    depth, except that a depth of one allows two calls when a batch holds at
    least two. A token worker advertises it as ``max_unresolved_calls``.

    Raises:
        ValueError: Either bound is below one.
    """
    depth = int(queue_depth)
    calls = int(max_calls)
    if depth < 1 or calls < 1:
        raise ValueError("call-window sizing requires positive bounds")
    return min(depth * calls, max(2, depth))


def request_tensor_arena_capacity(
    worker_config: WorkerConfig,
    *,
    queue_depth: int,
    product_bytes_per_request: int,
    concurrent_imports: int = 0,
    latent_pool_bytes: int = 0,
) -> ArenaCapacity:
    """Bound product, relay and transfer storage for fixed request tensors.

    ``concurrent_imports`` is how many remote product regions one request's
    calls can have in flight at once beyond its call window, which is what
    the artifact's assembly costs: it reads every encode round of the request,
    and each round was written by every rank that held a media unit in it.

    ``product_bytes_per_request`` is the bytes of one request's products
    this rank backs (``local_product_storage_bytes``).
    """
    depth = int(queue_depth)
    max_calls = int(worker_config.max_batch_calls)
    state_slots = int(worker_config.max_request_pool_size)
    # Product records and transfer tickets cover every queued call, or every
    # resident request's concurrent artifact imports when those are more.
    slots = max(depth * max_calls, state_slots * int(concurrent_imports))
    unresolved_window = request_tensor_window(depth, state_slots)
    tensor_store = _DEVICE_PRODUCTS_PER_CALL * (
        slots + _DEVICE_PRODUCT_RETIREMENT_BATCHES * max_calls
    )
    # Relay arenas index one-based request slots directly, so they hold one
    # extra row.
    relay_bytes = (
        (state_slots + 1)
        * (max(1, unresolved_window) + _REQUEST_RELAY_RETIREMENT_LANES)
        * _REQUEST_RELAY_ROW_BYTES
    )
    return ArenaCapacity(
        latent_pool_bytes=int(latent_pool_bytes),
        tensor_store=tensor_store,
        device_product_bytes=(
            device_product_capacity_bytes(tensor_store, 1, max_value_bytes=1)
            + relay_bytes
        ),
        transfer_bytes=max(1, state_slots * product_bytes_per_request),
        # Read tickets bound the reads in flight: a ticket returns when its
        # read retires, and a batch whose imports need more than are free
        # waits for returns (``Executor.advance_inputs``). One product's reads
        # start together, and a product is written by at most every rank of
        # the group producing it, so the rank keeps at least its own group's
        # world size: the products it reads in several regions come from its
        # own group's sequence-parallel or distributed components.
        transfer_tickets=max(
            int(worker_config.world_size), min(slots, _MAX_TRANSFER_ENTRIES)
        ),
        host_lane_inflight=state_slots * (unresolved_window + 1),
    )


def artifact_import_regions(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    bindings: Mapping[str, ComponentBinding],
) -> int:
    """Count the remote product regions assembling one artifact reads.

    Media units are encoded on the ranks that reconstruct them, one round at a
    time, and the muxer reads every round. Each round holds one media unit per
    participating rank and the muxer produced one of them itself.
    """
    components = media_components(model, worker_config.deployment_components)
    encoder = components.get(MediaCall.VIDEO_ENCODING)
    binding = None if encoder is None else bindings.get(encoder)
    decoder = capability(model, VideoDecoder)
    builder = media_builder(model, worker_config)
    if binding is None or decoder is None or builder is None:
        return 0
    ranks = len(binding.config.ranks)
    units = len(decoder.frame_slices(builder.maximum.num_frames))
    return ceil_div(units, ranks) * max(0, ranks - 1)


def model_arena_capacity(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    queue_depth: int,
    completion_payload_bytes: int,
    num_units: int,
    request_pool_size: int,
    num_latent_pages: int,
    latent_page_units: int,
    latent_width: int,
    max_latent_feature_bytes: int,
    max_vision_feature_bytes: int,
    unit_bytes: int,
    bindings: Mapping[str, ComponentBinding] | None = None,
    state_buffers: Mapping[str, BufferConfig] | None = None,
) -> ArenaCapacity:
    """Derive arena bounds from worker settings.

    Covers device-product, transfer, latent, and CPU arenas. A request-tensor
    worker (a ``VideoPostprocessor`` or non-empty ``state_buffers``) is sized
    by ``request_tensor_arena_capacity`` from its placement; of the keyword
    arguments, only ``queue_depth``, ``bindings`` and ``state_buffers``
    shape its bounds. Otherwise the remaining keyword arguments size the
    token-worker bounds. ``state_buffers`` defaults to the media request
    state of ``bindings``.

    Raises:
        ValueError: A runtime bound or latent-pool dimension is invalid, or
            an image denoiser's ``model_dtype`` is not FP16, BF16 or FP32.
    """
    depth = int(queue_depth)
    payload_bytes = int(completion_payload_bytes)
    max_calls = int(worker_config.max_batch_calls)
    if depth < 1 or payload_bytes < 1 or max_calls < 1:
        raise ValueError("model arena sizing requires positive runtime bounds")

    slots = depth * max_calls
    if state_buffers is None:
        state_buffers = media_state_buffers(
            bindings or {}, media_builder(model, worker_config)
        )

    if capability(model, VideoPostprocessor) is not None or state_buffers:
        # A rank that holds a standalone denoiser's request state advances
        # its samples in the latent pool.
        plan = latent_pool_plan(model, worker_config)
        return request_tensor_arena_capacity(
            worker_config,
            latent_pool_bytes=plan.capacity_bytes
            if plan is not None
            and holds_samples(
                state_buffers, media_builder(model, worker_config)
            )
            else 0,
            queue_depth=depth,
            product_bytes_per_request=local_product_storage_bytes(
                resolve_outputs(model, worker_config),
                bindings=bindings or {},
                media_components=media_components(
                    model, worker_config.deployment_components
                ),
            ),
            concurrent_imports=artifact_import_regions(
                model, worker_config, bindings=bindings or {}
            ),
        )

    transfer_tickets = min(slots, _MAX_TRANSFER_ENTRIES)
    flow = image_builder(model)

    latent_pool_bytes = 0
    latent_transfer_bytes = 0
    if flow is not None:
        dtype_bytes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
        }.get(str(worker_config.model_dtype).removeprefix("torch.").lower())
        if dtype_bytes is None:
            raise ValueError(
                f"unsupported latent dtype {worker_config.model_dtype!r}"
            )
        latent_pool_bytes = latent_pool_capacity_bytes(
            request_pool_size=int(request_pool_size),
            num_pages=int(num_latent_pages),
            page_units=int(latent_page_units),
            latent_width=int(latent_width),
            dtype_bytes=dtype_bytes,
        )
        latent_transfer_bytes = latent_trajectory_bytes(
            flow.max_tokens,
            int(latent_width),
            dtype_bytes,
        )
    # Every ticket is budgeted for the largest single transfer: the whole KV
    # pool, one image's latent trajectory, or one latent or vision feature.
    max_transfer_bytes = max(
        int(num_units) * int(unit_bytes),
        latent_transfer_bytes,
        int(max_latent_feature_bytes),
        int(max_vision_feature_bytes),
        1,
    )

    device_product_slots = (
        slots + _DEVICE_PRODUCT_RETIREMENT_BATCHES * max_calls
    )
    tensor_store = _DEVICE_PRODUCTS_PER_CALL * device_product_slots
    device_count = len(
        {
            str(worker_config.device),
            str(worker_config.generation_device or worker_config.device),
        }
    )
    # Resident images and tensor products borrow scheduler-assigned storage
    # from BufferPool, whose complete grant ``build_worker_layout`` counts.
    # TensorStore owns scalar backing and request relays separately, with
    # relay arenas on each device.
    device_product_bytes = device_product_capacity_bytes(
        tensor_store,
        device_count,
        max_value_bytes=1,
    )
    device_product_bytes += (
        (int(request_pool_size) + 1)
        * (call_window(depth, max_calls) + _REQUEST_RELAY_RETIREMENT_LANES)
        * _REQUEST_RELAY_ROW_BYTES
        * device_count
    )

    return ArenaCapacity(
        latent_pool_bytes=latent_pool_bytes,
        tensor_store=tensor_store,
        device_product_bytes=device_product_bytes,
        transfer_bytes=max_transfer_bytes * transfer_tickets,
        transfer_tickets=transfer_tickets,
        host_lane_inflight=_HOST_LANE_INFLIGHT,
    )


def units_for_tokens(pages: Sequence[tuple[int, int]], tokens: int) -> int:
    """Return the units whole pages of ``tokens`` tokens occupy in every group.

    ``pages`` holds each group's ``(page_tokens, units_per_page)``; each group
    counts the whole pages ``tokens`` fills, rounded down.
    """
    return sum(
        int(tokens) // int(page_tokens) * int(units_per_page)
        for page_tokens, units_per_page in pages
    )


def device_total_bytes(device: str | torch.device) -> int:
    """Return CUDA capacity for a device.

    A non-CUDA device reports zero. Device errors propagate so an unknown
    budget cannot become zero.
    """
    target = torch.device(device)
    if target.type != "cuda":
        return 0
    _free, total = torch.cuda.mem_get_info(target)
    return int(total)


def derive_runtime_kv_capacity(
    *,
    pages: Sequence[tuple[int, int]],
    kv_token_capacity: int | None,
    unit_bytes: int,
    device: Any = None,
    available_bytes: int | None = None,
    floor: int = 1,
    default_units: int | None = None,
    resident_copies: int = 1,
    co_resident_units: int = 0,
) -> RuntimeKVCapacity:
    """Size one KV unit pool from explicit tokens or a host-owned byte grant.

    ``pages`` holds each cache group's ``(page_tokens, units_per_page)``. The
    unit count, which includes the unit-zero sentinel, comes from the first
    source present:

    - ``kv_token_capacity``: the units whole pages of that many tokens
      occupy in every group;
    - ``available_bytes``: the whole units the grant holds, less
      ``co_resident_units`` units of other fixed allocations, divided among
      ``resident_copies`` pools;
    - without a device or on one other than CUDA, ``default_units`` or
      ``DEFAULT_NUM_UNITS_FALLBACK``.

    The pool holds at least ``floor`` units and at least the sentinel plus
    one page of every group, the least any request can be admitted with.
    Automatic CUDA sizing requires a granted budget. Whenever a grant is
    given, the resulting pools and co-resident units must fit it.

    Raises:
        ValueError: A dimension is out of range, ``kv_token_capacity`` or
            ``available_bytes`` is invalid, a grant yields fewer than the
            minimum units or is exceeded, or a CUDA device has no grant.
    """
    size = int(unit_bytes)
    if (
        not pages
        or any(min(shape) < 1 for shape in pages)
        or size < 1
        or floor < 1
        or resident_copies < 1
        or co_resident_units < 0
    ):
        raise ValueError("KV capacity dimensions must be positive")
    if available_bytes is not None and available_bytes < 0:
        raise ValueError("KV storage grant must not be negative")
    minimum = max(int(floor), 1 + sum(int(per_page) for _, per_page in pages))

    if kv_token_capacity is not None:
        if kv_token_capacity <= 0:
            raise ValueError("configured KV token capacity must be positive")
        units = max(minimum, units_for_tokens(pages, kv_token_capacity))
    elif available_bytes is not None:
        units = (available_bytes // size - co_resident_units) // resident_copies
        if units < minimum:
            raise ValueError(
                "device storage grant cannot hold the required KV pool"
            )
    elif device is not None and torch.device(device).type == "cuda":
        raise ValueError(
            "automatic CUDA KV sizing requires a host storage grant"
        )
    else:
        units = max(
            minimum,
            DEFAULT_NUM_UNITS_FALLBACK
            if default_units is None
            else int(default_units),
        )

    if (
        available_bytes is not None
        and (resident_copies * units + co_resident_units) * size
        > available_bytes
    ):
        raise ValueError(
            "configured KV storage exceeds the device storage grant"
        )

    return RuntimeKVCapacity(unit_bytes=size, num_units=units)


__all__ = [
    "ArenaCapacity",
    "DEFAULT_MAX_BATCH_OPS",
    "DEFAULT_MAX_REQUEST_POOL_SIZE",
    "DEFAULT_NUM_UNITS_FALLBACK",
    "RuntimeKVCapacity",
    "derive_runtime_kv_capacity",
    "device_total_bytes",
    "graph_table_widths",
    "LatentPoolPlan",
    "latent_pool_capacity_bytes",
    "latent_pool_plan",
    "latent_trajectory_bytes",
    "model_arena_capacity",
    "call_window",
    "units_for_tokens",
]


def resolve_request_capacity(
    model: nn.Module,
    worker_config: WorkerConfig,
    *,
    queue_depth: int,
    capacity_group: Communicator | None,
    bindings: Mapping[str, ComponentBinding] | None = None,
    state_buffers: Mapping[str, BufferConfig] | None = None,
) -> WorkerConfig:
    """Fit request tensors within the rank's fixed storage grant.

    Product arenas are fitted together with the request tensors.

    On a CUDA device, the grant measured now becomes ``pool_storage_bytes``;
    ``Worker.__init__`` calls this after the runner has bound its persistent
    inputs and workspaces, so they are already excluded. A request-tensor
    worker also receives the largest slot count, from
    ``min_request_pool_size`` up to ``max_request_pool_size`` and a third of
    ``queue_depth``, whose request state, products, relays and latent pool fit
    on every rank of ``capacity_group``, and its batch bounds are clamped to
    that count. On any other device the configuration is returned unchanged.

    Raises:
        WorkerError: With ``UNSUPPORTED_SETUP`` when a request-tensor worker
            has no ``capacity_group``.
        ValueError: The slot bounds are invalid.
        RuntimeError: No slot count fits on every rank.
    """
    if canonical_device(worker_config.device).type == "cuda":
        available, _free = device_storage_budget(
            worker_config.device, worker_config.kv_storage_fraction
        )
        worker_config = replace(worker_config, pool_storage_bytes=available)
        schema = (
            media_state_buffers(
                bindings or {}, media_builder(model, worker_config)
            )
            if state_buffers is None
            else state_buffers
        )

        if capability(model, VideoPostprocessor) is not None or schema:
            if capacity_group is None:
                raise unsupported_setup(
                    "request tensor sizing requires its rank group"
                )

            samples = holds_samples(schema, media_builder(model, worker_config))

            # Each candidate count is priced with the batch bounds it would
            # impose, the same clamping applied to the chosen count below.
            def auxiliary_bytes(count: int) -> int:
                capacity_config = replace(
                    worker_config,
                    max_request_pool_size=count,
                    max_batch_calls=min(count, worker_config.max_batch_calls),
                    max_batch_tokens=min(count, worker_config.max_batch_tokens),
                )
                plan = latent_pool_plan(model, capacity_config)
                latent_bytes = (
                    plan.capacity_bytes if samples and plan is not None else 0
                )
                product_bytes = local_product_storage_bytes(
                    resolve_outputs(model, worker_config),
                    bindings=bindings or {},
                    media_components=media_components(
                        model, worker_config.deployment_components
                    ),
                )
                arena = request_tensor_arena_capacity(
                    capacity_config,
                    queue_depth=queue_depth,
                    product_bytes_per_request=product_bytes,
                )
                return (
                    count * product_bytes
                    + arena.device_product_bytes
                    + latent_bytes
                )

            slots = tensor_slot_capacity(
                schema,
                capacity_group,
                maximum=min(
                    worker_config.max_request_pool_size, queue_depth // 3
                ),
                minimum=worker_config.min_request_pool_size,
                available_bytes=available,
                auxiliary_bytes=auxiliary_bytes,
            )
            worker_config = replace(
                worker_config,
                max_request_pool_size=slots,
                max_batch_calls=min(slots, worker_config.max_batch_calls),
                max_batch_tokens=min(slots, worker_config.max_batch_tokens),
            )
    return worker_config


def graph_table_widths(
    model: nn.Module, worker_config: WorkerConfig, pool: KVCacheManager | None
) -> tuple[int, ...]:
    """Return the column width every captured graph stages per block table.

    A full-attention table spans the longest sequence, capped at the pages
    its group can draw from the pool; a sliding-window table spans the pages
    a window of history and one call's queries intersect. Every captured
    text graph uses these widths, so graph shapes vary only in rows and
    tokens. Empty without a ``CausalLM``, a KV pool, or a positive
    ``max_sequence_tokens``.
    """
    if capability(model, CausalLM) is None or pool is None:
        return ()
    if worker_config.max_sequence_tokens < 1:
        return ()
    widths = table_widths(
        pool.cache.planes,
        max_sequence_tokens=worker_config.max_sequence_tokens,
        max_query_tokens=worker_config.max_batch_tokens,
    )
    # Unit 0 is the pool's padding sentinel, so one group's table can hold
    # at most every remaining unit, a whole page at a time.
    limits = tuple(
        max(1, (int(pool.info.num_units) - 1) // group.units_per_page)
        for group in pool.cache.groups
        for _ in range(group.units_per_page)
    )
    return tuple(
        min(width, limit) for width, limit in zip(widths, limits, strict=True)
    )


def check_startup_storage(
    worker_config: WorkerConfig,
    product_capacity_bytes: int,
    tensor_store: TensorStore,
) -> None:
    """Check resident startup allocations against device grants.

    Reserved products are checked as well. ``product_capacity_bytes`` is the
    arena's ``device_product_bytes``, split evenly across the worker's
    devices; on each CUDA device, the part not yet resident in
    ``tensor_store`` must fit both its current storage budget and, together
    with everything the process holds on the device
    (``process_device_bytes``), the ``kv_storage_fraction`` share of the
    device.

    Raises:
        WorkerError: With ``UNSUPPORTED_SETUP`` when a device lacks room.
    """
    # Warmup may retain backend plans, graph pools, graph executables and
    # communicator resources in addition to the explicit arenas, several of
    # them outside the caching allocator. Readiness requires that these
    # resident allocations leave room for every still-lazy public product
    # within the same grant.
    devices = tuple(
        dict.fromkeys(
            (
                worker_config.device,
                worker_config.generation_device or worker_config.device,
            )
        )
    )
    product_bytes = product_capacity_bytes // len(devices)
    for device in devices:
        if canonical_device(device).type != "cuda":
            continue
        available, free = device_storage_budget(
            device, worker_config.kv_storage_fraction
        )
        grant = int(
            device_total_bytes(device) * worker_config.kv_storage_fraction
        )
        remaining = max(0, product_bytes - tensor_store.resident_bytes(device))
        process_bytes = process_device_bytes(device)
        if remaining > available or process_bytes + remaining > grant:
            raise unsupported_setup(
                f"initialized runtime on {device} exceeds its static storage "
                f"grant: it requires {process_bytes + remaining} bytes "
                f"({process_bytes} process-resident bytes and {remaining} "
                f"reserved product bytes) of a {grant}-byte grant, with "
                f"{free} device-free bytes"
            )
