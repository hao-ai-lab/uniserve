"""A worker whose operator sets no KV page size chooses one its kernels read.

The base page size belongs to the cache group with the widest token rows;
every other group's page holds as many tokens as fit the same unit plane.
Unset, the worker takes the largest power of two up to 64 tokens at which
the automatic attention backend serves every cache layer's calls on its
group's pages. DiffusionGemma's full-attention layers keep a quarter of the
sliding layers' K/V heads at twice their width, so their pages hold twice
as many tokens, and the SM100 kernels of their head width read pages of at
most 64 tokens.
"""

from dataclasses import replace

import pytest
import torch

from tests.python.fixtures.checkpoints import (
    diffusion_gemma_checkpoint,
    qwen_checkpoint,
)
from uniserve.loading import weights
from uniserve.runtime.backends import attention as attention_backend
from uniserve_models import loading as models
from uniserve_worker.bootstrap.cache import plan_cache, resolve_page_size
from uniserve_worker.config.execution import WorkerConfig

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# Attention shapes of the released DiffusionGemma checkpoints: sliding
# layers of 16 query and 8 K/V heads of width 256, full layers of 16 query
# and 2 K/V heads of width 512.
RELEASED_ATTENTION = {
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 256,
    "global_head_dim": 512,
    "num_global_key_value_heads": 2,
}


def _load(root):
    return models.load_model(
        models.read_config(root),
        device="cpu",
        weights=weights.Config(dtype=torch.bfloat16),
    ).model


def _automatic():
    return attention_backend.resolve("auto", device=torch.device("cuda:0"))


def test_unset_page_size_is_the_largest_every_group_reads(tmp_path):
    diffusion_gemma_checkpoint(tmp_path, text=RELEASED_ATTENTION)
    model = _load(tmp_path)
    attention = _automatic()

    config = resolve_page_size(model, WorkerConfig(device="cuda"), attention)

    # 64-token sliding pages would give full-attention layers 128-token
    # pages, which no native kernel of their width reads.
    assert config.block_size == 32
    pages = sorted(
        group.page_tokens for group in plan_cache(model.text, config).groups
    )
    assert pages == [32, 64]

    # An explicit size is the operator's, whatever it gives the groups.
    explicit = replace(config, block_size=64)
    assert resolve_page_size(model, explicit, attention) == explicit


def test_a_model_every_kernel_reads_keeps_the_default_page_size(tmp_path):
    qwen_checkpoint(tmp_path)

    config = resolve_page_size(
        _load(tmp_path), WorkerConfig(device="cuda"), _automatic()
    )

    assert config.block_size == 64
