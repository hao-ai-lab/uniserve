"""What a worker asks the engine to warm before it serves.

The engine drives startup warmup through its own scheduler and allocators:
after the workers report, it admits real requests of the shapes below, so
every captured row count, guidance-branch count and canvas chunk has run
through the serving path before the first client request. The worker only
states the shapes its runtime was prepared for, as `WorkerInfo` fields:

- `image_warmups`: for each guidance-branch count the flow runtime captured,
  the last captured shape with that count, or one image of a square raster
  sized from the latent pool;
- `canvas_step_rows`: the rows one sampler chunk of the token denoiser
  covers, every row count of which a canvas step can run.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from uniserve_worker.model_executor.graph_inputs import DiffusionShape
from uniserve_worker.model_executor.startup import (
    capture_branch_prefixes,
    capture_image_parameters,
)
from uniserve_worker.protocol.call import MediaCall
from uniserve_worker.protocol.worker_info import ImageWarmup

if TYPE_CHECKING:
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.protocol.worker_info import WorkerInfo


def image_warmups(
    runner: ModelExecutor, info: WorkerInfo, tokenizer: object | None = None
) -> tuple[ImageWarmup, ...]:
    """Return the image shapes the engine warms this rank's flow runtime with.

    One shape per guidance-branch count the runtime captures, at the last
    captured shape with that count, or else one image whose side, in latent
    patches, is the integer square root of the latent capacity, capped at the
    image builder's token bound. The engine's denoiser lane runs one request
    at a time, so one request exercises the shape; a worker without a request
    slot to spare for a guided request's alternative prefix states none.

    A guided shape also states the token ids of its alternative guidance
    prefix. The worker encodes the empty negative prompt through the model's
    flow prompt, so a request carries those ids as its negative prompt tokens,
    which is how the engine sizes that prefix's KV the way the worker reads it.

    Empty without an image builder or without latent preparation and
    denoising.
    """
    builder = runner.image_builder
    if builder is None or not {
        MediaCall.LATENT_PREPARATION,
        MediaCall.DENOISING,
    }.issubset(info.supported_calls):
        return ()

    capacity = min(
        int(info.latent_capacity_units), builder.max_tokens + builder.framing
    )
    side = max(1, math.isqrt(max(1, capacity))) * builder.denoiser.downsample

    shapes = []
    for branches in runner.flow_cfg_branches:
        shape = next(
            (
                shape
                for shape in reversed(runner.flow_captures)
                if shape.cfg_branches == branches
            ),
            DiffusionShape(1, side, side, branches),
        )
        if int(info.request_slots) < (2 if branches > 1 else 1):
            continue

        # The scales that realize the branch count are the capture's own.
        guidance = capture_image_parameters(
            branches, steps=2, height=shape.height, width=shape.width
        )
        _, resolved = capture_branch_prefixes(
            builder, runner.flow_prompt, tokenizer, guidance
        )
        alternatives = {
            prefix
            for prefix, reuses_conditioning in resolved
            if not reuses_conditioning
        }
        if len(alternatives) > 1:
            raise ValueError(
                "image warmup guidance has more than one alternative prefix"
            )
        shapes.append(
            ImageWarmup(
                height=shape.height,
                width=shape.width,
                cfg_text_scale=guidance.cfg_text_scale,
                cfg_img_scale=guidance.cfg_img_scale,
                negative_token_ids=next(iter(alternatives), ()),
            )
        )
    return tuple(shapes)


def canvas_step_rows(runner: ModelExecutor, generates: bool) -> int:
    """Return the rows of one sampler chunk of the canvas runner.

    Zero when the rank's token denoiser generates no canvases (``generates``
    is false) or the runner has no canvas entry.
    """
    entry = runner.canvas_runner
    if not generates or entry is None:
        return 0
    return int(entry.step_rows)
