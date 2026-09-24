"""Prepare bounded text inputs and CUDA graphs during worker startup.

``ModelExecutor.capture`` calls these functions before serving begins, and
before ``ModelExecutor.complete_startup`` seals graph capture. They stage
startup rows (token ID zero for text; for image denoising, guidance-branch
prefixes resolved through the model's ``FlowPrompt``) through the same
``TokenBuffers`` and ``DiffusionBuffers`` serving uses, on scratch KV pages and
latent values leased from the idle pools, and capture each configured bucket;
when graphs are disabled, representative calls run eagerly as warmup instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.diffusion import Renorm
from uniserve.math import ceil_div
from uniserve.media import image as media_image
from uniserve_worker.model_executor.attention import from_blocks
from uniserve_worker.model_executor.diffusion_inputs import (
    DiffusionRow,
    resolve_prefix,
)
from uniserve_worker.model_executor.input_batch import InputBatch, TokenRow
from uniserve_worker.model_executor.input_buffers import TokenBuffers
from uniserve_worker.model_executor.model_runner import ModelRunner
from uniserve_worker.model_executor.output import ExecutionOutput
from uniserve_worker.protocol.call import ForwardMode, ImageParams
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.kv_cache import KVCacheManager

if TYPE_CHECKING:
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.model_executor.graph_inputs import PrefillShape


def stage_text(
    buffers: TokenBuffers,
    cache: KVCacheManager,
    tokens: tuple[tuple[int, ...], ...],
    pages: Sequence[Sequence[int]],
    *,
    prefixes: tuple[int, ...] | None = None,
    decode: bool = False,
    selection: TokenSelection = TokenSelection.LAST_LOGITS,
    causal: bool = True,
    slots: tuple[int, ...] | None = None,
) -> InputBatch:
    """Stage synthetic token rows through serving's staging path.

    ``tokens`` holds each row's token IDs and ``pages`` its physical KV
    pages. Rows default to empty prefixes and to request slots ``1..rows``;
    slot 0 is the inactive sentinel. Every row writes KV. A decode batch
    also carries the staging's cleared force-finish column.
    """
    rows = len(tokens)
    lengths = tuple(len(value) for value in tokens)
    prefixes = (0,) * rows if prefixes is None else prefixes
    positions = tuple(
        torch.arange(prefix, prefix + length, dtype=torch.int64)
        for prefix, length in zip(prefixes, lengths, strict=True)
    )

    attention = from_blocks(
        pages=pages,
        query_lengths=lengths,
        prefix_lengths=prefixes,
        causal=(causal,) * rows,
        write=(True,) * rows,
        block_size=cache.info.block_size,
    )

    slots = slots or tuple(range(1, rows + 1))
    mode = ForwardMode.DECODE if decode else ForwardMode.PREFILL
    batch = buffers.prepare_inputs(
        tuple(
            TokenRow(
                forward_mode=mode,
                token_ids=torch.tensor(value, dtype=torch.int64),
                positions=position,
                selection=selection,
                request_pool_idx=slot,
                seq_len=prefix,
                write_kv=True,
                causal=causal,
            )
            for value, position, slot, prefix in zip(
                tokens, positions, slots, prefixes, strict=True
            )
        ),
        forward_mode=mode,
        attention=attention,
    )
    if decode:
        # Startup captures the same force-finish address used by live decode.
        force_finish = buffers.decode_force_finish[:rows]
        force_finish.zero_()
        batch = replace(batch, decode_force_finish=force_finish)
    return batch


def prepare_prefill(
    runner: ModelExecutor,
    entry: ModelRunner,
    buffers: TokenBuffers,
    forward: Callable[[InputBatch], ExecutionOutput],
    shapes: tuple[PrefillShape, ...],
) -> None:
    """Capture each selected physical token/row bucket, largest first.

    Buckets are ordered by token-times-row footprint. Each capture stages
    ``live_rows`` rows totaling the bucket's token count on zeroed scratch
    KV pages; the runner's ``select_graph_shape`` pads them to the bucket.

    Raises:
        ValueError: A bucket is prepared on a worker without a KV cache.
    """
    for shape in sorted(
        shapes,
        key=lambda item: (
            item.token_bucket * item.row_bucket,
            item.token_bucket,
        ),
        reverse=True,
    ):
        # One row holds the long prompt; the remaining live rows hold one token.
        lengths = (
            shape.token_bucket - shape.live_rows + 1,
            *(1,) * (shape.live_rows - 1),
        )
        counts = tuple(
            ceil_div(length, runner.worker_config.block_size)
            for length in lengths
        )

        cache = runner.kv_cache
        if cache is None:
            raise ValueError("prefill capture requires the worker KV cache")
        with cache.startup_pages(sum(counts)) as scratch:
            pages = tuple(
                scratch[sum(counts[:index]) : sum(counts[: index + 1])]
                for index in range(len(counts))
            )
            batch = stage_text(
                buffers,
                cache,
                tuple((0,) * count for count in lengths),
                pages,
                causal=shape.causal,
                selection=shape.selection,
            )
            entry.capture_batch(batch, forward)


def prepare_decode(
    runner: ModelExecutor,
    entry: ModelRunner,
    buffers: TokenBuffers,
    forward: Callable[[InputBatch], ExecutionOutput],
) -> None:
    """Prepare valid one-token prefixes, then capture each decode bucket.

    Buckets are captured from the largest row count down. With graphs
    disabled, one single-row decode runs eagerly instead.

    Raises:
        ValueError: A bucket is prepared on a worker without a KV cache.
    """
    row_counts = (
        tuple(reversed(runner.decode_shapes[entry]))
        if runner.worker_config.graph_policy != "off"
        else (1,)
    )
    for rows in row_counts:
        cache = runner.kv_cache
        if cache is None:
            raise ValueError("decode capture requires the worker KV cache")
        with cache.startup_pages(rows) as scratch:
            pages = tuple((page,) for page in scratch)

            # Prefill the one-token prompt eagerly so the decode capture below
            # attends over K/V the model wrote rather than zeroed scratch.
            prompt = stage_text(buffers, cache, ((0,),) * rows, pages)
            entry.eager_batch(prompt, forward)

            batch = stage_text(
                buffers,
                cache,
                ((0,),) * rows,
                pages,
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
    prompt prefix eagerly into scratch KV pages, then the denoising entry
    stages rows that read those prefixes and captures them, or runs them
    eagerly when graph capture or prefill graphs are disabled. Rows are
    ordered by request with guidance branches varying fastest, and the
    branches of one request share its slot and latent.
    """
    import math

    from uniserve.math import ceil_div
    from uniserve_worker.model_executor.attention import from_blocks
    from uniserve_worker.model_executor.graph_inputs import DiffusionShape
    from uniserve_worker.protocol.call import MediaCall

    builder, cache = runner.image_builder, runner.kv_cache
    capture = (
        runner.worker_config.graph_policy != "off"
        and runner.worker_config.prefill_cuda_graph
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
            page_counts = tuple(
                ceil_div(len(prefix), cache.info.block_size)
                for prefix in prefixes
            )

            with (
                cache.startup_pages(sum(page_counts)) as scratch,
                latent_pool.startup_values(
                    shape.rows, builder.denoiser.latent_shape("image", size)[0]
                ) as latents,
            ):
                for value in latents:
                    value.zero_()

                pages, cursor = [], 0
                for count in page_counts:
                    pages.append(tuple(scratch[cursor : cursor + count]))
                    cursor += count

                # Only nonempty prefixes are prefilled, with hidden-state
                # selection whose output is discarded; the denoising rows
                # below read the written KV without writing.
                selected = tuple(
                    index for index, prefix in enumerate(prefixes) if prefix
                )
                if selected:
                    batch = stage_text(
                        prefix_entry.input_buffers,
                        cache,
                        tuple(prefixes[index] for index in selected),
                        tuple(pages[index] for index in selected),
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

                attention = from_blocks(
                    pages=tuple(pages),
                    query_lengths=(builder.sequence_length(size),)
                    * len(prefixes),
                    prefix_lengths=tuple(map(len, prefixes)),
                    block_size=cache.info.block_size,
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
