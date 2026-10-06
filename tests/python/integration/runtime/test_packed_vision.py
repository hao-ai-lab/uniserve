"""A packed vision call serves every packing of its image slots.

``PatchEncoder.encode_packed`` encodes images laid out in fixed slots of
``max_patches`` patch rows (``packed_grids``). Each image's soft tokens must
equal the features ``encode`` returns for it alone, whatever the images'
aspect ratios, and one CUDA graph captured over empty slots must replay any
later packing with those same features. The DiffusionGemma vision tower
runs on the SM100 kernels at the released head width.
"""

import pytest
import torch

from tests.python.fixtures.checkpoints import diffusion_gemma_checkpoint
from tests.python.fixtures.model_runner import forward_batch
from uniserve.loading import weights
from uniserve.model import ComponentEntry, EntryPoint, VisionInput
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve.runtime.cuda_graph import CUDAGraph
from uniserve_models import loading as models
from uniserve_worker.bootstrap.cache import cache_info
from uniserve_worker.bootstrap.capacity import input_buffer_config
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.image_inputs import VisionRow
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    MediaCall,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.storage.kv_cache import KVCacheManager

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

DEVICE = torch.device("cuda:0")
# Vision heads of the released width, 72, which FlashAttention-4 serves.
VISION = {
    "hidden_size": 144,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 72,
    "intermediate_size": 96,
}
# Patch grids of three aspect ratios, sides in multiples of the 3 x 3
# pooling kernel.
SHAPES = ((3, 6), (6, 3), (6, 6))


def _encoder(root):
    diffusion_gemma_checkpoint(root, vision=VISION)
    return models.load_model(
        models.read_config(root),
        device=DEVICE,
        weights=weights.Config(dtype=torch.bfloat16),
    ).model.vision_encoder


def _pack(encoder, pixels, images, shapes, slots):
    """Write ``images`` into the leading slots of ``pixels``; return grids."""
    capacity = encoder.max_patches
    for index, image in enumerate(images):
        pixels[index * capacity : index * capacity + image.shape[0]] = image
    return torch.tensor(
        encoder.packed_grids(shapes, slots), dtype=torch.long, device=DEVICE
    )


def _features(encoder, output, shapes):
    """Each packed image's soft tokens, which lead its slot."""
    tokens = encoder.max_patches // encoder.downsample**2
    return tuple(
        output[index * tokens : index * tokens + rows * columns // 9]
        for index, (rows, columns) in enumerate(shapes)
    )


@torch.inference_mode()
def test_packed_images_encode_as_they_do_alone(tmp_path):
    encoder = _encoder(tmp_path)
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    images = tuple(
        torch.rand(rows * columns, 48, device=DEVICE, generator=generator)
        for rows, columns in SHAPES
    )
    context = ExecutionContext(encoder, derive_host_lengths=False)
    context.prepare(None)
    with context, context.activate():
        alone = tuple(
            encoder.encode(
                VisionInput(
                    (image,),
                    (torch.tensor([shape], device=DEVICE),),
                    (shape,),
                )
            )[0]
            for image, shape in zip(images, SHAPES, strict=True)
        )

        # Four slots hold the three images and one empty slot of padding.
        slots = 4
        pixels = torch.zeros(slots * encoder.max_patches, 48, device=DEVICE)
        grids = _pack(encoder, pixels, images, SHAPES, slots)
        packed = _features(
            encoder, encoder.encode_packed(pixels, grids), SHAPES
        )
        for actual, expected in zip(packed, alone, strict=True):
            torch.testing.assert_close(actual, expected)

        # A graph captured over empty slots replays two later packings.
        static_pixels = torch.zeros_like(pixels)
        static_grids = torch.tensor(
            encoder.packed_grids((), slots), dtype=torch.long, device=DEVICE
        )
        graph = CUDAGraph(context=context)
        graph.capture(
            lambda: encoder.encode_packed(static_pixels, static_grids)
        )
        try:
            for order in ((0, 1, 2), (2, 0)):
                shapes = tuple(SHAPES[index] for index in order)
                static_grids.copy_(
                    _pack(
                        encoder,
                        static_pixels,
                        tuple(images[index] for index in order),
                        shapes,
                        slots,
                    )
                )
                replayed = _features(encoder, graph.replay(), shapes)
                for actual, index in zip(replayed, order, strict=True):
                    torch.testing.assert_close(actual, alone[index])
        finally:
            graph.close()


@torch.inference_mode()
def test_worker_packed_vision_preserves_rows_and_reports_replay(tmp_path):
    diffusion_gemma_checkpoint(tmp_path, vision=VISION)
    source = models.read_config(tmp_path)
    model = models.load_model(
        source, device=DEVICE, weights=weights.Config(dtype=torch.bfloat16)
    ).model
    encoder = model.vision_encoder
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    images = tuple(
        torch.rand(rows * columns, 48, device=DEVICE, generator=generator)
        for rows, columns in SHAPES
    )
    grids = tuple(torch.tensor([shape], device=DEVICE) for shape in SHAPES)
    with ExecutionContext(encoder) as context:
        context.prepare(None)
        with context.activate():
            expected = encoder.encode(VisionInput(images, grids, SHAPES))

    config = WorkerConfig(
        device=str(DEVICE),
        model_dtype="bfloat16",
        graph_policy="full",
        block_size=16,
        max_sequence_tokens=64,
        max_batch_tokens=64,
        max_batch_calls=4,
        max_request_pool_size=4,
    )
    runner = ModelExecutor(
        model,
        config,
        image_processor=source.image_processor,
        entry_points={
            "vision_encoder": ComponentEntry(
                "vision_encoder", (EntryPoint("encode"),)
            )
        },
    )
    cache = PrefixCache(
        model.text.cache_config, num_units=16, block_size=16, device=DEVICE
    )
    manager = KVCacheManager(
        cache,
        info=cache_info(model.text, config, num_units=16),
        request_pool_size=4,
        table_width=4,
    )
    try:
        runner.configure_inputs(
            input_config=input_buffer_config(
                model, config, processor=source.image_processor
            ),
            kv_cache=manager,
            latent_pool=None,
            decode_predicates=None,
            max_calls=4,
            request_slots=4,
            latent_capacity_units=0,
            table_widths=(4,) * len(manager.shapes),
            max_inflight=1,
        )
        runner.capture(tokenizer=None, latents=None)
        runner.complete_startup()
        retained = []
        for order in ((0, 1, 2), (2, 0)):
            rows = tuple(
                VisionRow(
                    MediaCall.VISION_ENCODING,
                    encode_pixels=images[index],
                    encode_grid=grids[index],
                    encode_grid_shape=SHAPES[index],
                )
                for index in order
            )
            calls = tuple(
                Call(
                    RequestKey(1, slot, 0),
                    CallId(1, 0),
                    CallCoordinates(),
                    MediaCall.VISION_ENCODING,
                    Bounds(),
                    component="vision_encoder",
                )
                for slot in range(len(rows))
            )
            result = forward_batch(
                runner, rows, calls=calls, cache=None, tables=None, states=None
            )
            assert result.stats.cuda_graph_replays == 1
            assert result.stats.cuda_graph_runtime_mode_counts == {
                "graph_replay": 1
            }
            retained.extend(
                zip(
                    result.values,
                    (expected[index] for index in order),
                    strict=True,
                )
            )

        for actual, reference in retained:
            torch.testing.assert_close(actual, reference)
    finally:
        runner.close()
        manager.close()
