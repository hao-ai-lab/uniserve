from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.worker_config import stub_worker_config
from uniserve_models.stub import Model, image_processor
from uniserve_worker.bootstrap.capacity import derive_runtime_kv_capacity
from uniserve_worker.bootstrap.worker_info_builder import build_worker_info

TEST_WORKER_CONFIG = stub_worker_config(64, max_batch_tokens=8192)

pytestmark = pytest.mark.unit

BLOCK_SIZE = 64
TOTAL_BYTES = 184 * 1024**3
STATIC_FRACTION = 0.70


@pytest.fixture(autouse=True)
def fixed_device_total(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True, raising=False)
    monkeypatch.setattr(
        "torch.cuda.mem_get_info",
        lambda device: (TOTAL_BYTES, TOTAL_BYTES),
        raising=False,
    )


def _model() -> Model:
    return Model()


def _worker_config(*, token_capacity: int | None):
    return replace(
        TEST_WORKER_CONFIG,
        device="cuda:0",
        block_size=BLOCK_SIZE,
        kv_token_capacity=token_capacity,
        kv_memory_fraction=STATIC_FRACTION,
    )


def test_explicit_kv_capacity_provisions_one_physical_page_pool():
    info = build_worker_info(_model(), _worker_config(token_capacity=131072))

    assert info.kv_cache is not None
    assert info.kv_cache.num_blocks * BLOCK_SIZE == 131072


def test_automatic_kv_capacity_reserves_all_storage_within_the_grant():
    capacity = derive_runtime_kv_capacity(
        block_size=64,
        kv_token_capacity=None,
        bytes_per_token=128,
        device="cuda:0",
        available_bytes=10 * 64 * 128 + 100,
        resident_copies=2,
        co_resident_blocks=2,
    )
    assert capacity.num_blocks == 4
    assert capacity.token_capacity == 256


@pytest.mark.parametrize("tokens", [None, 256])
def test_kv_storage_cannot_exceed_its_grant(tokens):
    with pytest.raises(ValueError, match="grant"):
        derive_runtime_kv_capacity(
            block_size=64,
            kv_token_capacity=tokens,
            bytes_per_token=128,
            available_bytes=64 * 128 - 1,
        )


def test_automatic_cuda_kv_capacity_requires_a_host_grant():
    with pytest.raises(ValueError, match="host memory grant"):
        derive_runtime_kv_capacity(
            block_size=64,
            kv_token_capacity=None,
            bytes_per_token=128,
            device="cuda:0",
        )


def test_automatic_capacity_charges_request_and_input_storage() -> None:
    model = _model()
    config = replace(
        _worker_config(token_capacity=None),
        pool_memory_bytes=32 * 1024**3,
        max_request_pool_size=4,
        max_batch_operations=4,
        max_batch_tokens=64,
    )
    small = build_worker_info(model, config)
    larger_requests = build_worker_info(model, replace(config, max_request_pool_size=128))
    larger_input = build_worker_info(model, replace(config, max_batch_tokens=65536))
    assert small.kv_cache is not None
    assert larger_requests.kv_cache is not None
    assert larger_input.kv_cache is not None
    assert larger_requests.kv_cache.num_blocks < small.kv_cache.num_blocks
    assert larger_input.kv_cache.num_blocks < small.kv_cache.num_blocks


def test_explicit_pages_cannot_displace_resident_encoder_storage() -> None:
    model = _model()
    config = replace(
        _worker_config(token_capacity=64),
        pool_memory_bytes=32 * 1024**3,
        max_request_pool_size=4,
        max_batch_operations=4,
        max_batch_tokens=64,
    )
    processor = image_processor()
    build_worker_info(model, config, image_processor=processor)
    # The processor's admitted image area owns the complete resident feature bound.
    processor = replace(processor, vit=replace(processor.vit, max_pixels=1024**3 * 16**2))
    with pytest.raises(ValueError, match="grant"):
        build_worker_info(model, config, image_processor=processor)
