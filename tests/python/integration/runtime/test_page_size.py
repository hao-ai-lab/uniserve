"""Cache allocation chooses page sizes supported by every numerical reader."""

import pytest
import torch

from tests.python.fixtures.hybrid import hybrid_model
from uniserve.runtime.backends.attention import resolve
from uniserve_worker.bootstrap.cache import plan_cache, resolve_page_size
from uniserve_worker.config.execution import WorkerConfig

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_hybrid_page_size_serves_every_attention_group():
    model = hybrid_model(
        dtype=torch.bfloat16, heads=((16, 8, 256), (16, 2, 512))
    )
    backend = resolve("auto", device=torch.device("cuda:0"))
    config = resolve_page_size(model, WorkerConfig(device="cuda"), backend)

    # Full-attention rows occupy half the bytes. A 64-token base page
    # would give them 128-token pages, beyond their native reader's limit.
    assert config.block_size == 32
    assert sorted(
        group.page_tokens for group in plan_cache(model, config).groups
    ) == [32, 64]

    explicit = config.replace(block_size=64)
    assert resolve_page_size(model, explicit, backend) == explicit


def test_uniform_page_size_keeps_the_largest_supported_base():
    model = hybrid_model(
        layers=1, dtype=torch.bfloat16, heads=((16, 8, 256), (16, 2, 512))
    )
    config = resolve_page_size(
        model,
        WorkerConfig(device="cuda"),
        resolve("auto", device=torch.device("cuda:0")),
    )

    assert config.block_size == 64
