"""Worker image batches preserve the codec's spatial numerical domain."""

import pytest
import torch

from tests.python.integration.model_loading.test_bagel import _checkpoint
from uniserve.media import image
from uniserve.runtime import PrefixCache
from uniserve_models import bagel
from uniserve_worker.bootstrap.cache import cache_info
from uniserve_worker.bootstrap.capacity import input_buffer_config
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.image_inputs import DecodeRow, VisionRow
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    MediaCall,
)
from uniserve_worker.protocol.identity import CallId, RequestKey
from uniserve_worker.storage.kv_cache import KVCacheManager
from uniserve_worker.storage.latent_pool import LatentPool

pytestmark = pytest.mark.integration


@torch.inference_mode()
def test_batched_codec_queries_are_independent_of_text_token_capacity(tmp_path):
    _, _, architecture = _checkpoint(tmp_path)
    model = bagel.Model(architecture).eval()
    config = WorkerConfig(
        device="cpu",
        attention_backend="torch",
        model_dtype="float32",
        block_size=16,
        max_sequence_tokens=32,
        max_batch_calls=3,
        max_request_pool_size=3,
        max_batch_tokens=64,
        graph_policy="off",
        flow_graph_shapes=((16, 16),),
        flow_graph_batch_sizes=(1,),
    )
    runner = ModelExecutor(model, config)
    cache = PrefixCache(
        model.text.cache_config, num_blocks=8, block_size=16, device="cpu"
    )
    manager = KVCacheManager(
        cache,
        info=cache_info(model.text, config, num_blocks=8),
        request_pool_size=3,
        max_blocks_per_request=2,
    )
    latents = LatentPool(
        request_pool_size=3,
        num_pages=4,
        page_units=16,
        latent_width=8,
        dtype=torch.bfloat16,
        device="cpu",
    )
    try:
        runner.configure_inputs(
            input_config=input_buffer_config(model, config),
            kv_cache=manager,
            latent_pool=latents,
            decode_predicates=torch.tensor([False, True, True, True]),
            max_calls=3,
            request_slots=3,
            max_tokens=64,
            latent_capacity_units=16,
            decode_context_blocks=2,
            max_inflight=1,
        )
        runner.complete_startup()
        size = image.Config(16, 16)
        retained = []
        for count in (1, 3, 1):
            # Three images contain 48 canonical latent patches but 192
            # spatial attention queries, beyond the text token grant of 64.
            pixels = (
                torch.arange(count * 3 * 16 * 16)
                .reshape(count, 3, 16, 16)
                .float()
                .sin()
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(129)
                expected = model.latent_encoder.encode(pixels)
                torch.manual_seed(129)
                rows = tuple(
                    VisionRow(MediaCall.LATENT_ENCODING, encode_pixels=value)
                    for value in pixels
                )
                encoded = _run(runner, manager, rows)
            torch.testing.assert_close(torch.stack(encoded), expected)
            expected_pixels = model.image_decoder.decode(
                tuple(expected.unbind(0)), sizes=(size,) * count
            )
            rows = tuple(
                DecodeRow(
                    MediaCall.IMAGE_DECODING,
                    latent=value,
                    image_height=size.height,
                    image_width=size.width,
                )
                for value in encoded
            )
            decoded = _run(runner, manager, rows)
            for actual, reference in zip(decoded, expected_pixels, strict=True):
                torch.testing.assert_close(actual, reference)
                retained.append((actual, reference.clone()))
        for actual, reference in retained:
            torch.testing.assert_close(actual, reference)
    finally:
        runner.close()
        latents.close()
        manager.close()


def _run(runner, manager, rows):
    calls = tuple(
        Call(
            RequestKey(1, index, 0),
            CallId(1, 0),
            CallCoordinates(),
            row.forward_mode,
            Bounds(),
        )
        for index, row in enumerate(rows)
    )
    return (
        runner.run_forward_group(
            rows,
            calls=calls,
            cache=manager,
            tables=manager.block_tables,
            states=None,
        )
        .materialize()
        .values
    )
