"""Prepare bounded text inputs and CUDA graphs during worker startup.

``ModelExecutor.capture`` calls these functions before serving begins, and
before ``ModelExecutor.complete_startup`` seals graph capture. They stage
startup rows (token ID zero for text; for image denoising, guidance-branch
prefixes resolved through the model's ``FlowPrompt``) through the same
``TokenBuffers``, ``CanvasBuffers`` and ``DiffusionBuffers`` serving uses, on
scratch KV units and latent values leased from the idle pools, and capture
each configured bucket; when graphs are disabled, representative calls run
eagerly as warmup instead.
Image encoders and decoders, which run eagerly, evaluate one synthetic image
each (``prepare_images``), so every staged call kind has prepared its call
sites and chosen its kernels before serving.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, TypedDict

import torch

from uniserve.diffusion import Renorm
from uniserve.math import ceil_div
from uniserve.media import image as media_image
from uniserve_worker.model_executor.attention import from_tables, table_pages
from uniserve_worker.model_executor.diffusion_inputs import (
    DiffusionRow,
    resolve_prefix,
)
from uniserve_worker.model_executor.encoder_runner import EncoderRunner
from uniserve_worker.model_executor.image_inputs import (
    DecodeRow,
    VisionRow,
    prepare_tensor_image,
)
from uniserve_worker.model_executor.input_batch import (
    CanvasRow,
    CanvasStepRow,
    InputBatch,
    TokenRow,
)
from uniserve_worker.model_executor.input_buffers import (
    CanvasBuffers,
    TokenBuffers,
)
from uniserve_worker.model_executor.model_runner import ModelRunner
from uniserve_worker.model_executor.output import ExecutionOutput
from uniserve_worker.protocol.call import ForwardMode, ImageParams, MediaCall
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.block_tables import GroupTable
from uniserve_worker.storage.kv_cache import KVCacheManager

if TYPE_CHECKING:
    from uniserve.diffusion.canvas import CanvasSampling
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.model_executor.canvas_runner import CanvasRunner
    from uniserve_worker.model_executor.graph_inputs import PrefillShape


def scratch_tables(
    cache: KVCacheManager, units: Sequence[int], lengths: Sequence[int]
) -> tuple[tuple[GroupTable, ...], ...]:
    """Partition scratch units into each row's tables of every cache group.

    Row ``i`` receives, in every group, whole pages from page zero covering
    ``lengths[i]`` tokens (at least one page), taken from ``units`` in order;
    ``units`` must hold ``sum(map(cache.page_units, lengths))`` units.
    """
    result, cursor = [], 0
    for length in lengths:
        row = []
        for shape in cache.shapes:
            pages = max(1, -(-int(length) // shape.page_tokens))
            count = pages * shape.units_per_page
            row.append(
                GroupTable(
                    shape,
                    0,
                    tuple(units[cursor : cursor + count]),
                    pages * shape.page_tokens,
                )
            )
            cursor += count
        result.append(tuple(row))
    return tuple(result)


def stage_text(
    buffers: TokenBuffers,
    tokens: tuple[tuple[int, ...], ...],
    tables: Sequence[Sequence[GroupTable]],
    *,
    prefixes: tuple[int, ...] | None = None,
    decode: bool = False,
    selection: TokenSelection = TokenSelection.LAST_LOGITS,
    causal: bool | None = True,
    slots: tuple[int, ...] | None = None,
    embeddings: bool = False,
) -> InputBatch:
    """Stage synthetic token rows through serving's staging path.

    ``tokens`` holds each row's token IDs and ``tables`` its scratch tables
    of every cache group. Rows default to empty prefixes and to request
    slots ``1..rows``; slot 0 is the inactive sentinel. Every row writes KV.
    With ``embeddings``, every token replaces its embedding with zeros, as an
    image feature row replaces its placeholders. A decode batch also carries
    the staging's cleared force-finish column. ``causal=None`` stages device
    flags for a mixed-context graph family, even for a one-row warmup.
    """
    rows = len(tokens)
    flags = (
        tuple(index % 2 == 0 for index in range(rows))
        if causal is None
        else (causal,) * rows
    )
    lengths = tuple(len(value) for value in tokens)
    prefixes = (0,) * rows if prefixes is None else prefixes
    positions = tuple(
        torch.arange(prefix, prefix + length, dtype=torch.int64)
        for prefix, length in zip(prefixes, lengths, strict=True)
    )

    attention = from_tables(
        table_pages(tables, prefix_lengths=prefixes, query_lengths=lengths),
        query_lengths=lengths,
        prefix_lengths=prefixes,
        causal=flags,
        write=(True,) * rows,
    )

    slots = slots or tuple(range(1, rows + 1))
    mode = ForwardMode.DECODE if decode else ForwardMode.PREFILL
    # Replaced embeddings take the staging column's dtype; staging rejects
    # them on a lane without one.
    replaced = buffers.input_embeddings if embeddings else None
    if embeddings and replaced is None:
        raise ValueError("embedding replacement requires an embedding column")
    batch = buffers.prepare_inputs(
        tuple(
            TokenRow(
                forward_mode=mode,
                token_ids=torch.tensor(value, dtype=torch.int64),
                token_embeddings=None
                if replaced is None
                else replaced.new_zeros((len(value), buffers.hidden_size)),
                positions=position,
                selection=selection,
                request_pool_idx=slot,
                seq_len=prefix,
                write_kv=True,
                causal=flag,
            )
            for value, position, slot, prefix, flag in zip(
                tokens, positions, slots, prefixes, flags, strict=True
            )
        ),
        forward_mode=mode,
        attention=attention,
    )
    if causal is None:
        # Even a one-row warmup must compile the device-flag specialization;
        # later replays can carry any mixture of text and image rows.
        values = buffers.stage_causality(flags, dynamic=True)
        current = batch.inputs.attention
        batch = replace(
            batch,
            inputs=replace(
                batch.inputs,
                attention=replace(
                    current,
                    entries={
                        table: replace(entry, causal_values=values)
                        for table, entry in current.entries.items()
                    },
                ),
            ),
        )
    if decode:
        # Startup captures the same force-finish address used by live decode.
        force_finish = buffers.decode_force_finish[:rows]
        force_finish.zero_()
        batch = replace(batch, decode_force_finish=force_finish)
    return batch


def capture_lengths(
    shape: PrefillShape,
    row_tokens: int,
    pages: Sequence[tuple[int, int]],
    units: int,
) -> tuple[int, ...]:
    """Return the row lengths of the startup batch that captures ``shape``.

    ``pages`` holds ``(page_tokens, units_per_page)`` of every cache group
    and ``units`` the unit pool's allocatable units, which the batch's pages
    fit. The batch numbers the bucket's ``live_rows`` rows, or more when a
    row, which holds at most ``row_tokens`` (the most one request appends in
    a call), cannot hold its share, up to one less than the bucket's row
    count and at most the rows the pool holds a page of every group for. It
    holds the most tokens up to the bucket's whose rows fit the pool;
    tokens it does not hold become the bucket's padding sequence. Its rows
    hold exactly ``prefill_units`` units: the fewest any batch of as many
    rows and tokens holds (see ``_spread``).
    """
    row_units = sum(count for _, count in pages)
    rows = max(shape.live_rows, ceil_div(shape.token_bucket, row_tokens))
    rows = max(1, min(rows, shape.row_bucket - 1, units // row_units))

    def cost(lengths):
        return sum(
            count * ceil_div(length, page)
            for length in lengths
            for page, count in pages
        )

    # One token per row always fits, since the pool holds a page of every
    # group for each row; the most tokens whose rows fit is searched above
    # it. A spread that no row cap admits counts as not fitting.
    fitted = (1,) * rows
    low, high = rows, min(shape.token_bucket, rows * row_tokens)
    while low < high:
        tokens = (low + high + 1) // 2
        lengths = _spread(rows, tokens, row_tokens, pages)
        if lengths is not None and cost(lengths) <= units:
            low, fitted = tokens, lengths
        else:
            high = tokens - 1
    return fitted


def _spread(
    rows: int,
    tokens: int,
    row_tokens: int,
    pages: Sequence[tuple[int, int]],
) -> tuple[int, ...] | None:
    """Spread ``tokens`` over ``rows`` rows in the fewest pages of each group.

    A group whose pages ``tokens`` overfills at one page per row needs
    ``ceil(tokens / page_tokens)`` pages, which it reaches when every row but
    one holds whole pages; the others need one page per row, which they
    reach when no row exceeds their page. Page sizes are powers of two, so
    the overfilled groups' pages are the smaller ones and divide the largest
    of them, the step: every row holds whole steps up to the smallest other
    page, and the last row also holds the remainder below one step. Returns
    None when rows of at most ``row_tokens`` tokens cannot be spread so.
    """
    packed = [page for page, _ in pages if ceil_div(tokens, page) > rows]
    step = max(packed, default=1)
    cap = min(
        (page for page, _ in pages if ceil_div(tokens, page) <= rows),
        default=row_tokens,
    )
    cap = min(cap, row_tokens)
    if cap < step:
        return None

    steps, remainder = divmod(tokens, step)
    base, extra = divmod(steps, rows)
    lengths = [(base + (index < extra)) * step for index in range(rows)]
    lengths[-1] += remainder
    if min(lengths) < 1 or max(lengths) > cap:
        return None
    return tuple(lengths)


def prepare_prefill(
    runner: ModelExecutor,
    entry: ModelRunner,
    buffers: TokenBuffers,
    forward: Callable[[InputBatch], ExecutionOutput],
    shapes: tuple[PrefillShape, ...],
) -> None:
    """Capture each selected physical token/row bucket, largest first.

    Buckets are ordered by token-times-row footprint, so the first capture
    of hidden states sizes the runner's shared prefill output. Each capture
    stages the ``capture_lengths`` rows of its bucket on zeroed scratch KV
    units, which the pool's allocatable units hold, with the bucket's
    causality, embedding replacement and outputs (rows of a cache-only
    bucket select ``TokenSelection.CACHE``), through serving's staging with
    its real paged
    attention input: per-table block tables, device start pages for windowed
    tables and their host mirrors; the runner's ``select_graph_shape`` pads
    them to the bucket.

    Raises:
        ValueError: A bucket is prepared on a worker without a KV cache.
    """
    config = runner.worker_config
    row_tokens = max(
        1, min(config.max_sequence_tokens, config.max_batch_tokens)
    )
    cache = runner.kv_cache
    if cache is None:
        raise ValueError("prefill capture requires the worker KV cache")
    pages = tuple(
        (group.page_tokens, group.units_per_page) for group in cache.shapes
    )
    units = cache.info.num_units - 1
    for shape in sorted(
        shapes,
        key=lambda item: (
            item.token_bucket * item.row_bucket,
            item.token_bucket,
        ),
        reverse=True,
    ):
        lengths = capture_lengths(shape, row_tokens, pages, units)
        count = sum(map(cache.page_units, lengths))
        with cache.startup_units(count) as scratch:
            batch = stage_text(
                buffers,
                tuple((0,) * length for length in lengths),
                scratch_tables(cache, scratch, lengths),
                selection=TokenSelection.LAST_LOGITS
                if shape.outputs
                else TokenSelection.CACHE,
                causal=shape.causal,
                embeddings=shape.embeddings,
            )
            entry.capture_batch(batch, forward)


def prepare_decode(
    runner: ModelExecutor,
    entry: ModelRunner,
    buffers: TokenBuffers,
    forward: Callable[[InputBatch], ExecutionOutput],
) -> None:
    """Prepare valid one-token prefixes, then capture each decode bucket.

    Buckets are captured from the largest row count down. An entry without
    decode buckets (graphs disabled, or no CUDA device) runs one single-row
    decode eagerly instead.

    Raises:
        ValueError: A bucket is prepared on a worker without a KV cache.
    """
    row_counts = tuple(reversed(runner.decode_shapes[entry])) or (1,)
    for rows in row_counts:
        cache = runner.kv_cache
        if cache is None:
            raise ValueError("decode capture requires the worker KV cache")
        # Each row holds its one-token prompt and one decoded token.
        with cache.startup_units(rows * cache.page_units(2)) as scratch:
            tables = scratch_tables(cache, scratch, (2,) * rows)

            # Prefill the one-token prompt eagerly so the decode capture below
            # attends over K/V the model wrote rather than zeroed scratch.
            prompt = stage_text(buffers, ((0,),) * rows, tables)
            entry.eager_batch(prompt, forward)

            batch = stage_text(
                buffers,
                ((0,),) * rows,
                tables,
                prefixes=(1,) * rows,
                decode=True,
            )

            # Capture with every row live (slots 1..rows, as staged above),
            # then restore the caller's predicates.
            predicates = runner.decode_predicates
            saved = None if predicates is None else predicates.clone()
            try:
                if predicates is not None:
                    predicates[1 : rows + 1] = True
                entry.capture_batch(batch, forward)
            finally:
                if saved is not None and predicates is not None:
                    predicates.copy_(saved)


class _RowFields(TypedDict):
    """The ``AttentionRow`` fields every synthetic canvas row shares."""

    forward_mode: ForwardMode
    positions: torch.Tensor
    seq_len: int
    write_kv: bool
    causal: bool


def stage_canvas(
    buffers: CanvasBuffers,
    tables: Sequence[Sequence[GroupTable]],
    *,
    length: int,
    sampling: CanvasSampling | None = None,
    step: int = 0,
) -> InputBatch:
    """Stage synthetic canvas rows through serving's staging path.

    Row ``i`` is a canvas of ``length`` tokens in request slot ``i + 1``
    over a one-token prefix on its scratch ``tables``, read-only and
    non-causal as every canvas is. Without ``sampling`` the rows are
    readout canvases of token zero reading one slot; with it they are step
    ``step`` of block zero of the slots' resident canvases under that
    sampling.
    """
    rows = len(tables)
    prefix = 1
    attention = from_tables(
        table_pages(
            tables,
            prefix_lengths=(prefix,) * rows,
            query_lengths=(length,) * rows,
        ),
        query_lengths=(length,) * rows,
        prefix_lengths=(prefix,) * rows,
        causal=(False,) * rows,
        write=(False,) * rows,
    )
    positions = torch.arange(prefix, prefix + length, dtype=torch.int64)
    common: _RowFields = {
        "forward_mode": ForwardMode.TOKEN_DENOISING,
        "positions": positions,
        "seq_len": prefix,
        "write_kv": False,
        "causal": False,
    }
    staged = tuple(
        CanvasRow(
            request_pool_idx=slot,
            token_ids=torch.zeros(length, dtype=torch.int64),
            slot_tokens=(0,),
            candidate_offsets=(0, 1),
            candidate_ids=(0,),
            **common,
        )
        if sampling is None
        else CanvasStepRow(
            request_pool_idx=slot,
            canvas_length=length,
            seed=slot,
            step=step,
            sampling=sampling,
            **common,
        )
        for slot in range(1, rows + 1)
    )
    return buffers.prepare_inputs(
        staged, forward_mode=ForwardMode.TOKEN_DENOISING, attention=attention
    )


def prepare_canvas(runner: ModelExecutor, entry: CanvasRunner) -> None:
    """Capture every canvas row bucket of a token denoiser, largest first.

    Each bucket captures readout passes at the supported canvas lengths and,
    when the entry generates canvases, two canvas steps of the served
    sampling over request slots one
    upward, one whose rows all start their canvas and one whose rows
    continue it; their resident state stays scratch until a request starts
    its canvas there. The rows are staged through serving's staging with the
    real read-only attention input over one-token prefixes on scratch KV
    units, and the runner pads them to their bucket. The largest readout
    bucket, captured first, sizes the runner's shared readout output. A
    step's warm call runs the sampler's chunk shapes before their capture.
    Without graph pools, one row of each kind runs eagerly.

    Raises:
        ValueError: A bucket is prepared on a worker without a KV cache.
    """
    cache = runner.kv_cache
    if cache is None:
        raise ValueError("canvas capture requires the worker KV cache")
    slots = entry.canvas_slots
    # Readouts can have shorter numerical canvases. Generating steps keep
    # the model's length because it defines the resident sampler state.
    kinds = (
        tuple((None, 0, length) for length in reversed(entry.readout_lengths))
    ) + (
        ()
        if slots is None
        else (
            (slots.constants, 0, entry.canvas_length),
            (slots.constants, 1, entry.canvas_length),
        )
    )
    for rows in reversed(entry.canvas_rows if entry.pools else (1,)):
        for sampling, step, length in kinds:
            with cache.startup_units(rows * cache.page_units(1)) as scratch:
                batch = stage_canvas(
                    entry.input_buffers,
                    scratch_tables(cache, scratch, (1,) * rows),
                    length=length,
                    sampling=sampling,
                    step=step,
                )
                entry.capture_batch(batch, entry.batch_forward)


def capture_image_parameters(cfg_branches, *, steps, height, width):
    """Build fixed image-generation params whose CFG scales fit a branch count.

    One branch runs unconditioned, two add text guidance, and three add text
    and image guidance; each count maps to its (text, image) scale pair.
    ``prepare_flow`` checks that the resulting guidance realizes the count.
    """
    text, image = {1: (1.0, 1.0), 2: (4.0, 1.0), 3: (4.0, 2.0)}[cfg_branches]
    return ImageParams(
        steps=steps,
        cfg_text_scale=text,
        cfg_img_scale=image,
        height=height,
        width=width,
        seed=0,
    )


@torch.inference_mode()
def prepare_flow(runner, entry, latent_pool, tokenizer):
    """Warm and capture configured image shapes on real conditioning prefixes.

    For each shape, the prefix entry first prefills every nonempty branch
    prompt prefix eagerly into scratch KV units, then the denoising entry
    stages rows that read those prefixes and captures them, or runs them
    eagerly when graph capture or flow graphs are disabled. Rows are
    ordered by request with guidance branches varying fastest, and the
    branches of one request share its slot and latent.
    """
    import math

    from uniserve_worker.model_executor.graph_inputs import DiffusionShape
    from uniserve_worker.protocol.call import MediaCall

    builder, cache = runner.image_builder, runner.kv_cache
    capture = (
        runner.worker_config.graph_policy != "off"
        and runner.worker_config.flow_cuda_graph
    )
    capacity = min(builder.max_tokens, latent_pool.capacity_units)
    side = max(1, math.isqrt(capacity)) * builder.denoiser.downsample

    # Without capture, or without selected capture shapes, prepare one shape
    # per guidance-branch count: the last selected capture with that count,
    # or a one-row square image whose side derives from the smaller of the
    # builder's token bound and the latent pool capacity. With capture
    # enabled, the loop below still captures that fallback shape.
    shapes = (
        runner.flow_captures
        if capture and runner.flow_captures
        else tuple(
            next(
                (
                    shape
                    for shape in reversed(runner.flow_captures)
                    if shape.cfg_branches == branches
                ),
                DiffusionShape(1, side, side, branches),
            )
            for branches in runner.flow_cfg_branches
        )
    )

    # Conditioning prefixes are prefilled by the same component's prefill
    # entry.
    forward = entry.batch_forward
    prefix_entry = runner._forward_calls[(entry.name, ForwardMode.PREFILL)]
    stream = entry.context.stream
    if stream is not None:
        stream.wait(torch.cuda.current_stream(entry.device))

    with entry.context.activate():
        for shape in sorted(
            shapes,
            key=lambda item: (
                item.rows * item.height * item.width * item.cfg_branches
            ),
            reverse=True,
        ):
            size = media_image.Config(shape.height, shape.width)
            image = capture_image_parameters(
                shape.cfg_branches,
                steps=2,
                height=shape.height,
                width=shape.width,
            )
            schedule = builder.denoiser.make_schedules(
                image.steps,
                shift=image.timestep_shift
                if image.timestep_shift > 0
                else None,
                device="cpu",
            )["image"]
            guidance = builder.denoiser.make_guidance(
                text_scale=image.cfg_text_scale,
                image_scale=image.cfg_img_scale,
                interval=image.cfg_interval,
                renorm=Renorm(image.cfg_renorm_type),
                renorm_min=image.cfg_renorm_min,
            )
            branches = guidance.branches(schedule, 0)
            if len(branches) != shape.cfg_branches:
                raise ValueError(
                    "capture guidance does not realize its configured branch "
                    "count"
                )

            # One prefix per (request, branch), branch-fastest. A prefix can
            # be empty (see ``resolve_prefix``).
            prefixes = (
                tuple(
                    resolve_prefix(
                        runner.flow_prompt,
                        builder.branch_source(branch),
                        image_prompt="",
                        negative_prompt="",
                        negative_token_ids=(),
                        tokenizer=tokenizer,
                    )[0]
                    for branch in branches
                )
                * shape.rows
            )
            lengths = tuple(map(len, prefixes))

            with (
                cache.startup_units(
                    sum(map(cache.page_units, lengths))
                ) as scratch,
                latent_pool.startup_values(
                    shape.rows, builder.denoiser.latent_shape("image", size)[0]
                ) as latents,
            ):
                for value in latents:
                    value.zero_()

                tables = scratch_tables(cache, scratch, lengths)

                # Only nonempty prefixes are prefilled, with hidden-state
                # selection whose output is discarded; the denoising rows
                # below read the written KV without writing.
                selected = tuple(
                    index for index, prefix in enumerate(prefixes) if prefix
                )
                if selected:
                    batch = stage_text(
                        prefix_entry.input_buffers,
                        tuple(prefixes[index] for index in selected),
                        tuple(tables[index] for index in selected),
                        selection=TokenSelection.HIDDEN,
                        slots=tuple(
                            1 + index // shape.cfg_branches
                            for index in selected
                        ),
                    )
                    prefix_stream = prefix_entry.context.stream
                    if prefix_stream is not None:
                        prefix_stream.wait(
                            torch.cuda.current_stream(entry.device)
                        )
                    prefix_entry.eager_batch(
                        batch,
                        prefix_entry.batch_forward,
                    )
                    # The prefix activation has restored the caller's stream.
                    if prefix_stream is not None:
                        torch.cuda.current_stream(entry.device).wait_stream(
                            prefix_stream.stream
                        )

                queries = (builder.sequence_length(size),) * len(prefixes)
                attention = from_tables(
                    table_pages(
                        tables, prefix_lengths=lengths, query_lengths=queries
                    ),
                    query_lengths=queries,
                    prefix_lengths=lengths,
                    causal=(False,) * len(prefixes),
                    write=(False,) * len(prefixes),
                )
                rows = tuple(
                    DiffusionRow(
                        forward_mode=MediaCall.DENOISING,
                        positions=builder.positions(
                            size, len(prefix), device=entry.device
                        ),
                        timestep=schedule.timesteps[:1].to(entry.device),
                        latent=latents[index // shape.cfg_branches].to(
                            entry.device
                        ),
                        image_tokens=builder.sequence_length(size),
                        image_height=size.height,
                        image_width=size.width,
                        request_pool_idx=1 + index // shape.cfg_branches,
                        seq_len=len(prefix),
                        causal=False,
                    )
                    for index, prefix in enumerate(prefixes)
                )

                batch = entry.input_buffers.prepare_inputs(
                    rows,
                    forward_mode=MediaCall.DENOISING,
                    attention=attention,
                )
                if capture:
                    entry.capture_batch(batch, forward)
                else:
                    entry.eager_batch(batch, forward)

    if stream is not None:
        torch.cuda.current_stream(entry.device).wait_stream(stream.stream)


@torch.inference_mode()
def prepare_images(runner: ModelExecutor, latents) -> None:
    """Evaluate one synthetic image in every image input and output call.

    Vision and latent encoding stage a black square through the model's
    image processor, as a request image is staged; image decoding decodes a
    zero latent of an image that square's size; and a model that writes an
    input image's latent into its prompt (framed latent features, as
    ``execution.image.latent_state_row`` builds them) writes one on scratch
    KV units through its denoising entry. Latents come from the latent pool
    ``latents``. These calls run eagerly, so this prepares their call sites
    and resolves their kernels, and a call without a native kernel fails
    here rather than at the first request. Kernel selection depends on each
    call's representation, not its extent, so the square is small: two
    latent patches of the image denoiser on a side, which any latent pool
    holds, or 64 pixels without a denoiser. Each transform resizes it to its
    tower's bounds, as it would a request image.
    """
    builder = runner.image_builder
    side = 64 if builder is None else 2 * builder.denoiser.downsample
    size = media_image.Config(side, side)
    # Only a model whose input images reach the prompt as framed latents
    # (and not only as vision features) writes latent features.
    latent_features = (
        builder is not None
        and builder.framing == 2
        and latents is not None
        and any(
            MediaCall.LATENT_ENCODING in entry.call_kinds
            for entry in runner.entries.values()
        )
    )
    for entry in runner.entries.values():
        stream = entry.context.stream
        if stream is not None:
            stream.wait(torch.cuda.current_stream(entry.device))
        for kind in (MediaCall.VISION_ENCODING, MediaCall.LATENT_ENCODING):
            if kind not in entry.call_kinds or runner.processor is None:
                continue
            prepared = prepare_tensor_image(
                runner.processor,
                kind,
                torch.zeros((3, side, side)),
                device=entry.device,
                signed_unit=False,
            )
            row = VisionRow(
                forward_mode=kind,
                encode_pixels=prepared.pixels,
                encode_grid=prepared.grid,
                encode_grid_shape=prepared.grid_shape,
            )
            with entry.context.activate():
                batch = entry.prepare_inputs((row,), forward_mode=kind)
            entry.eager_batch(batch, entry.batch_forward)
            # A patch encoder that packs image slots serves vision calls
            # only through graphs of every slot count a batch can need.
            if (
                kind is MediaCall.VISION_ENCODING
                and isinstance(entry, EncoderRunner)
                and entry.packs_images
            ):
                entry.capture_packed(
                    max_images=runner.worker_config.max_batch_calls,
                    dtype=prepared.pixels.dtype,
                )
        if (
            MediaCall.IMAGE_DECODING in entry.call_kinds
            and builder is not None
            and latents is not None
        ):
            units = builder.denoiser.latent_shape("image", size)[0]
            with latents.startup_values(1, units) as (latent,):
                latent.zero_()
                with entry.context.activate():
                    batch = entry.prepare_inputs(
                        (
                            DecodeRow(
                                forward_mode=MediaCall.IMAGE_DECODING,
                                latent=latent.to(entry.device),
                                image_height=side,
                                image_width=side,
                            ),
                        ),
                        forward_mode=MediaCall.IMAGE_DECODING,
                    )
                entry.eager_batch(batch, entry.batch_forward)
        if latent_features and MediaCall.DENOISING in entry.call_kinds:
            _write_latent_feature(runner, entry, latents, size)
        # Activation has restored the caller's stream.
        if stream is not None:
            torch.cuda.current_stream(entry.device).wait_stream(stream.stream)


def _write_latent_feature(runner, entry, latents, size):
    """Write one zero image latent, framed, into scratch KV non-causally.

    The row is the one ``execution.image.latent_state_row`` builds for an
    input image at the start of an empty prompt: the latent at timestep
    zero between the builder's two framing tokens, attending to itself in
    both directions and writing its KV.
    """
    builder, cache = runner.image_builder, runner.kv_cache
    query = builder.sequence_length(size)
    units = builder.denoiser.latent_shape("image", size)[0]
    with (
        cache.startup_units(cache.page_units(query)) as scratch,
        latents.startup_values(1, units) as (latent,),
    ):
        latent.zero_()
        positions = builder.positions(size, 1, device=entry.device)
        positions[0, 0] = 0
        positions[0, -1] = builder.rope_advance
        attention = from_tables(
            table_pages(
                scratch_tables(cache, scratch, (query,)),
                prefix_lengths=(0,),
                query_lengths=(query,),
            ),
            query_lengths=(query,),
            prefix_lengths=(0,),
            causal=(False,),
            write=(True,),
        )
        row = DiffusionRow(
            forward_mode=MediaCall.DENOISING,
            positions=positions,
            timestep=latent.new_zeros(1),
            latent=latent.to(entry.device),
            image_tokens=query,
            image_height=size.height,
            image_width=size.width,
            request_pool_idx=1,
            seq_len=0,
            write_kv=True,
            causal=False,
        )
        with entry.context.activate():
            batch = entry.prepare_inputs(
                (row,), forward_mode=MediaCall.DENOISING, attention=attention
            )
        entry.eager_batch(batch, entry.batch_forward)
