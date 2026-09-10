"""Flow prefix and joint decode/flow startup preparation."""

import logging
from collections.abc import Callable

import torch

from ...foundation.math import bucketed_length, ceil_div
from ...models.generation import GenerationPipeline
from ...nn.diffusion.cfg import build_flow_cfg_plan
from ...runtime.latent_pool import LatentPool
from ..attention import physical_columns
from ..flow import denoise_geometry
from ..forward_batch import ForwardBatch, ForwardOutput, ModelPhase, TokenSelection
from ..input_buffers import InputBuffers
from .packed import FlowCapture, MixedCapture, PackedRunner
from .prefill import stage_text

logger = logging.getLogger(__name__)
MIN_MIXED_SERVICE_SPEEDUP = 1.03


class FlowRunner:
    """Prepare configured numerical flow inputs over a lane's packed owner.

    The packed owner retains inputs, attention metadata, backend, and publication
    lifetime. This owner binds generation geometry and its bounded startup leases.
    """

    def __init__(
        self,
        packed: PackedRunner,
        buffers: InputBuffers,
        forward: Callable[[ForwardBatch], ForwardOutput],
        *,
        generation: GenerationPipeline,
        latents: LatentPool,
        tokenizer: object,
        patch_size: int | None,
        tensorized: bool,
    ) -> None:
        self.packed = packed
        self.buffers = buffers
        self.forward = forward
        self.generation = generation
        self.latents = latents
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.tensorized = tensorized

    def warmup(
        self,
        shapes: tuple[FlowCapture, ...],
        mixed: tuple[MixedCapture, ...],
        qualify: Callable[[MixedCapture, bool], bool],
    ) -> None:
        """Prepare representative eager geometry when no catalog covers this path."""

        required = tuple(
            dict.fromkeys(
                (
                    *shapes,
                    *(FlowCapture(x.flow_rows, x.height, x.width, x.cfg_branches) for x in mixed),
                )
            )
        )
        self._prepare(required, mixed, qualify, capture=False)

    def capture(
        self,
        shapes: tuple[FlowCapture, ...],
        mixed: tuple[MixedCapture, ...],
        qualify: Callable[[MixedCapture, bool], bool],
    ) -> None:
        self._prepare(shapes, mixed, qualify, capture=True)

    @torch.inference_mode()
    def _prepare(
        self,
        shapes: tuple[FlowCapture, ...],
        mixed: tuple[MixedCapture, ...],
        qualify: Callable[[MixedCapture, bool], bool],
        *,
        capture: bool,
    ) -> None:
        from ..model_runner import capture_image_parameters

        cache = self.packed.cache_pool
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
                recipe=self.generation.cfg_recipe,
                renorm=image.cfg_renorm_type,
                renorm_min=image.cfg_renorm_min,
                use_cfg=True,
            )
            prefixes = tuple(
                self.generation.prefix(
                    self.generation.branch_source(branch),
                    image_prompt="",
                    negative_prompt="",
                    negative_token_ids=(),
                    tokenizer=self.tokenizer,
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
                self.latents.startup_values(
                    shape.rows, self.generation.image_tokens(shape.height, shape.width)
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
                        self.buffers,
                        cache,
                        tuple(repeated[index] for index in selected),
                        tuple(branch_pages[index] for index in selected),
                        packed=self.tensorized,
                        selection=TokenSelection.HIDDEN,
                        slots=tuple(1 + index // shape.cfg_branches for index in selected),
                    )
                    if capture:
                        self.packed.capture(prefix, self.forward)
                    # Prefix KV is the numerical input of the subsequent denoiser.
                    self.packed.run(prefix, self.forward, eligible=True)
                if text_count:
                    prompt = stage_text(
                        self.buffers,
                        cache,
                        ((0,),) * text_count,
                        token_pages,
                        packed=self.tensorized,
                    )
                    self.forward(prompt)

                def stage(decode_rows: int, include_flow: bool = True) -> ForwardBatch:
                    if not include_flow:
                        return stage_text(
                            self.buffers,
                            cache,
                            ((0,),) * decode_rows,
                            token_pages[:decode_rows],
                            packed=self.tensorized,
                            prefixes=(1,) * decode_rows,
                            decode=True,
                        )
                    return self._stage(
                        shape, latents, repeated, tuple(branch_pages), token_pages[:decode_rows]
                    )

                if capture:
                    self.packed.capture(stage(0), self.forward)
                else:
                    self.packed.warmup(stage(0), self.forward)
                for item in mixed_shapes:
                    if capture:
                        self.packed.capture(stage(item.decode_rows), self.forward)
                    else:
                        self.packed.warmup(stage(item.decode_rows), self.forward)
                    # Preserve the existing service-time qualification criterion,
                    # now measured after all involved graphs are already resident.
                    mixed_us, graph = self._measure(lambda: stage(item.decode_rows))
                    decode_us, _ = self._measure(
                        lambda: stage(item.decode_rows, False), eligible=graph
                    )
                    flow_us, _ = self._measure(lambda: stage(0), eligible=graph)
                    serial_us = decode_us + flow_us
                    eligible = (
                        self.buffers.device.type != "cuda"
                        or serial_us / mixed_us >= MIN_MIXED_SERVICE_SPEEDUP
                    )
                    qualify(item, eligible)
                    logger.info(
                        "evaluated mixed execution bucket=%r mixed_us=%d serial_us=%d service_eligible=%s",
                        item,
                        mixed_us,
                        serial_us,
                        eligible,
                    )

    def _stage(self, shape, latents, prefixes, pages, token_pages) -> ForwardBatch:
        rows = shape.rows * shape.cfg_branches
        text = len(token_pages)
        geometry = tuple(
            denoise_geometry(
                self.generation,
                latents[index // shape.cfg_branches],
                shape.height,
                shape.width,
                temporal=len(prefixes[index]),
                patch_size=self.patch_size,
            )
            for index in range(rows)
        )
        positions, indexes, conditioning, queries, local_text = zip(*geometry, strict=True)
        token_positions = tuple(torch.tensor([1], dtype=torch.int64) for _ in range(text))
        token_indexes = tuple(torch.tensor([[1], [0], [0]]) for _ in range(text))
        all_pages = (*token_pages, *pages)
        attention = physical_columns(
            pages=all_pages,
            seq_lens=(*(1,) * text, *map(len, prefixes)),
            query_lens=(*(1,) * text, *queries),
            causal_rows=(*(True,) * text, *(False,) * rows),
            write_rows=(*(True,) * text, *(False,) * rows),
            positions=(*token_indexes, *indexes),
            token_rows=(*(True,) * text, *(False,) * rows),
            text_local_indices=(*((),) * text, *local_text),
            width=min(
                self.buffers.max_blocks_per_row, bucketed_length(max(1, max(map(len, all_pages))))
            ),
            block_size=self.packed.block_size,
            packed=self.tensorized,
        )
        return self.buffers.stage(
            phase=ModelPhase.DENOISE,
            row_count=text + rows,
            request_pool_indices=(
                *range(1, text + 1),
                *(1 + index // shape.cfg_branches for index in range(rows)),
            ),
            token_row_indices=tuple(range(text)),
            token_ids=tuple(torch.zeros(1, dtype=torch.int64) for _ in range(text)),
            token_embeddings=(None,) * text,
            token_embedding_masks=(None,) * text,
            token_positions=token_positions,
            token_selections=(TokenSelection.LAST_LOGITS,) * text,
            flow_row_indices=tuple(range(text, text + rows)),
            flow_positions=positions,
            flow_timesteps=tuple(
                torch.tensor([self.generation.schedule_pair(2, 3.0, 0)[0]]) for _ in range(rows)
            ),
            flow_latents=tuple(latents[index // shape.cfg_branches] for index in range(rows)),
            flow_conditioning=conditioning,
            flow_image_tokens=queries,
            flow_heights=(shape.height,) * rows,
            flow_widths=(shape.width,) * rows,
            attention=attention,
        )

    def _measure(
        self, prepare: Callable[[], ForwardBatch], *, eligible: bool = True
    ) -> tuple[int, bool]:
        import time

        device = self.buffers.device
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        started = time.perf_counter_ns()
        result = self.packed.run(prepare(), self.forward, eligible=eligible)
        if device.type == "cuda":
            end.record()
            end.synchronize()
            elapsed = max(1, round(start.elapsed_time(end) * 1000))
        else:
            elapsed = max(1, (time.perf_counter_ns() - started) // 1000)
        return elapsed, result.path != "eager"
