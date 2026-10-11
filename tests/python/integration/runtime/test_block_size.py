"""Cache allocation chooses a block size supported by every numerical reader."""

from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.hybrid import hybrid_model
from tests.python.fixtures.worker_config import cpu_worker_config
from uniserve.runtime.backends.attention import resolve
from uniserve_worker.bootstrap.cache import plan_cache, resolve_block_size

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_hybrid_block_size_serves_every_attention_group():
    model = hybrid_model(
        dtype=torch.bfloat16, heads=((16, 8, 256), (16, 2, 512))
    )
    backend = resolve("auto", device=torch.device("cuda:0"))
    config = resolve_block_size(
        model, cpu_worker_config(device="cuda"), backend
    )

    # Full-attention rows occupy half the bytes. A 64-token block
    # would give them 128-token pages, beyond their native reader's limit.
    assert config.block_size == 32
    assert sorted(
        group.page_tokens for group in plan_cache(model, config).groups
    ) == [32, 64]

    explicit = replace(config, block_size=64)
    assert resolve_block_size(model, explicit, backend) == explicit


def test_uniform_block_size_keeps_the_largest_supported_base():
    model = hybrid_model(
        layers=1, dtype=torch.bfloat16, heads=((16, 8, 256), (16, 2, 512))
    )
    config = resolve_block_size(
        model,
        cpu_worker_config(device="cuda"),
        resolve("auto", device=torch.device("cuda:0")),
    )

    assert config.block_size == 64
