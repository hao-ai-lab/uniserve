"""Numerical startup inputs built through the same buffers as serving.

The native ModelExecutor chooses capture buckets, owns scratch resource scopes
and submits eager or captured calls. These helpers construct token, canvas and
image tensors, mathematical conditioning and attention views for those calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from uniserve.diffusion import Renorm
from uniserve.media import image as media_image
from uniserve_worker._uniserve_ipc import resolve_prefix
from uniserve_worker.model_executor.attention import from_tables, table_pages
from uniserve_worker.model_executor.diffusion_inputs import DiffusionRow
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
from uniserve_worker.protocol.call import ForwardMode, ImageParams, MediaCall
from uniserve_worker.sampling.metadata import TokenSelection
from uniserve_worker.storage.block_tables import GroupTable

if TYPE_CHECKING:
    from uniserve.diffusion.canvas import CanvasSampling


def make_text_batch(
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
    """Prepare synthetic token rows through serving's input preparation.

    ``tokens`` holds each row's token IDs and ``tables`` its scratch tables
    of every cache group. Rows default to empty prefixes and to request
    slots ``1..rows``; slot 0 is the inactive sentinel. Every row writes KV.
    With ``embeddings``, every token replaces its embedding with zeros, as an
    image feature row replaces its placeholders. A decode batch also carries
    the buffer's cleared force-finish column. ``causal=None`` supplies device
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
    # Replaced embeddings take the buffer column's dtype; input preparation
    # rejects them on a lane without one.
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
        values = buffers.prepare_causality(flags, dynamic=True)
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


def make_canvas_batch(
    buffers: CanvasBuffers,
    tables: Sequence[Sequence[GroupTable]],
    *,
    length: int,
    sampling: CanvasSampling | None = None,
    step: int = 0,
) -> InputBatch:
    """Prepare synthetic canvas rows through serving's input preparation.

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
    common = {
        "forward_mode": ForwardMode.TOKEN_DENOISING,
        "positions": positions,
        "seq_len": prefix,
        "write_kv": False,
        "causal": False,
    }
    inputs = tuple(
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
        inputs, forward_mode=ForwardMode.TOKEN_DENOISING, attention=attention
    )


def capture_image_parameters(cfg_branches, *, steps, height, width):
    """Build fixed image-generation params whose CFG scales fit a branch count.

    One branch runs unconditioned, two add text guidance, and three add text
    and image guidance; each count maps to its (text, image) scale pair.
    The numerical guidance resolves the branches used by the startup rows.
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


def flow_prefixes(runner, shape, tokenizer):
    """Resolve the solver schedule and branch-fastest conditioning prefixes."""
    builder = runner.image_builder
    image = capture_image_parameters(
        shape.cfg_branches, steps=2, height=shape.height, width=shape.width
    )
    schedule = builder.denoiser.make_schedules(
        image.steps,
        shift=image.timestep_shift if image.timestep_shift > 0 else None,
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
    return schedule, prefixes


def make_flow_batch(runner, entry, shape, schedule, prefixes, tables, latents):
    """Build denoising rows over branch prefixes and zero latents."""
    builder = runner.image_builder
    size = media_image.Config(shape.height, shape.width)
    lengths = tuple(map(len, prefixes))
    queries = (builder.sequence_length(size),) * len(prefixes)
    for latent in latents:
        latent.zero_()

    attention = from_tables(
        table_pages(tables, prefix_lengths=lengths, query_lengths=queries),
        query_lengths=queries,
        prefix_lengths=lengths,
        causal=(False,) * len(prefixes),
        write=(False,) * len(prefixes),
    )
    rows = tuple(
        DiffusionRow(
            forward_mode=MediaCall.DENOISING,
            positions=builder.positions(size, len(prefix), device=entry.device),
            timestep=schedule.timesteps[:1].to(entry.device),
            latent=latents[index // shape.cfg_branches].to(entry.device),
            image_tokens=builder.sequence_length(size),
            image_height=size.height,
            image_width=size.width,
            request_pool_idx=1 + index // shape.cfg_branches,
            seq_len=len(prefix),
            causal=False,
        )
        for index, prefix in enumerate(prefixes)
    )
    return entry.input_buffers.prepare_inputs(
        rows, forward_mode=MediaCall.DENOISING, attention=attention
    )


def make_image_batch(processor, entry, kind, side):
    """Prepare a black image through the model's normal input transform."""
    prepared = prepare_tensor_image(
        processor,
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
    return entry.prepare_inputs(
        (row,), forward_mode=kind
    ), prepared.pixels.dtype


def make_decoding_batch(entry, side, latent):
    """Build one image-decoding row from a zero latent."""
    latent.zero_()
    row = DecodeRow(
        forward_mode=MediaCall.IMAGE_DECODING,
        latent=latent.to(entry.device),
        image_height=side,
        image_width=side,
    )
    return entry.prepare_inputs((row,), forward_mode=MediaCall.IMAGE_DECODING)


def make_latent_feature_batch(runner, entry, size, tables, latent):
    """Write one framed latent into scratch KV as an input image would."""
    builder = runner.image_builder
    query = builder.sequence_length(size)
    latent.zero_()
    positions = builder.positions(size, 1, device=entry.device)
    positions[0, 0] = 0
    positions[0, -1] = builder.rope_advance
    attention = from_tables(
        table_pages(tables, prefix_lengths=(0,), query_lengths=(query,)),
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
    return entry.prepare_inputs(
        (row,), forward_mode=MediaCall.DENOISING, attention=attention
    )
