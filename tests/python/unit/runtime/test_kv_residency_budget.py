from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.worker_config import stub_worker_config
from uniserve_models.stub import Model, image_processor
from uniserve_worker.bootstrap.capacity import derive_runtime_kv_capacity
from uniserve_worker.bootstrap.report import build_worker_layout

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
    monkeypatch.setattr(
        "torch.cuda.memory_reserved", lambda device: 0, raising=False
    )


def _model() -> Model:
    return Model()


def _worker_config(*, token_capacity: int | None):
    return TEST_WORKER_CONFIG.replace(
        device="cuda:0",
        block_size=BLOCK_SIZE,
        kv_token_capacity=token_capacity,
        kv_storage_fraction=STATIC_FRACTION,
    )


# Layout planning queries the device properties of its CUDA grant.
@pytest.mark.gpu
def test_explicit_kv_capacity_provisions_one_physical_unit_pool():
    info = build_worker_layout(
        _model(), _worker_config(token_capacity=131072)
    ).info

    # The stub's layers share one group whose pages are single units.
    assert info.kv_cache is not None
    assert info.kv_cache.num_units * BLOCK_SIZE == 131072


def test_explicit_kv_capacity_counts_the_units_of_every_group():
    # A full group of one-unit 64-token pages and a windowed group of
    # five-unit 128-token pages hold 1024 tokens in 16 + 8 * 5 units.
    capacity = derive_runtime_kv_capacity(
        pages=((64, 1), (128, 5)),
        kv_token_capacity=1024,
        unit_bytes=4096,
    )
    assert (capacity.num_units, capacity.unit_bytes) == (56, 4096)


def test_automatic_kv_capacity_reserves_all_storage_within_the_grant():
    capacity = derive_runtime_kv_capacity(
        pages=((64, 1),),
        kv_token_capacity=None,
        unit_bytes=64 * 128,
        device="cuda:0",
        available_bytes=10 * 64 * 128 + 100,
        resident_copies=2,
        co_resident_units=2,
    )
    assert capacity.num_units == 4


@pytest.mark.parametrize("tokens", [None, 256])
def test_kv_storage_cannot_exceed_its_grant(tokens):
    with pytest.raises(ValueError, match="grant"):
        derive_runtime_kv_capacity(
            pages=((64, 1),),
            kv_token_capacity=tokens,
            unit_bytes=64 * 128,
            available_bytes=64 * 128 - 1,
        )


def test_automatic_cuda_kv_capacity_requires_a_host_grant():
    with pytest.raises(ValueError, match="host storage grant"):
        derive_runtime_kv_capacity(
            pages=((64, 1),),
            kv_token_capacity=None,
            unit_bytes=64 * 128,
            device="cuda:0",
        )


# Layout planning queries the device properties of its CUDA grant.
@pytest.mark.gpu
def test_automatic_capacity_charges_request_and_input_storage() -> None:
    model = _model()
    config = _worker_config(token_capacity=None).replace(
        pool_storage_bytes=32 * 1024**3,
        max_request_pool_size=4,
        max_batch_calls=4,
        max_batch_tokens=64,
    )
    small = build_worker_layout(model, config).info
    larger_requests = build_worker_layout(
        model, config.replace(max_request_pool_size=128)
    ).info
    larger_input = build_worker_layout(
        model, config.replace(max_batch_tokens=65536)
    ).info
    assert small.kv_cache is not None
    assert larger_requests.kv_cache is not None
    assert larger_input.kv_cache is not None
    assert larger_requests.kv_cache.num_units < small.kv_cache.num_units
    assert larger_input.kv_cache.num_units < small.kv_cache.num_units


# Layout planning queries the device properties of its CUDA grant.
@pytest.mark.gpu
def test_explicit_pages_cannot_displace_resident_encoder_storage() -> None:
    model = _model()
    config = _worker_config(token_capacity=64).replace(
        pool_storage_bytes=32 * 1024**3,
        max_request_pool_size=4,
        max_batch_calls=4,
        max_batch_tokens=64,
    )
    processor = image_processor()
    build_worker_layout(model, config, image_processor=processor).info
    # The processor's admitted image area owns the complete resident feature
    # bound.
    vit = processor.vit
    processor = replace(
        processor,
        vit=replace(
            vit, resize=replace(vit.resize, max_pixels=1024**3 * 16**2)
        ),
    )
    with pytest.raises(ValueError, match="grant"):
        build_worker_layout(model, config, image_processor=processor).info
