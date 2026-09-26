"""A small causal LM whose windowed and full-attention K/V shapes differ.

Five windowed layers precede each full-attention layer. A windowed K/V row
(four heads of four elements) is twice a full-attention row (one head of
eight elements), as DiffusionGemma's sliding rows are twice its full rows,
so a full-attention page holds twice the tokens of a windowed page. The
model only declares its cache layers; it computes nothing.
"""

from __future__ import annotations

import torch
from torch import nn

from uniserve.model import CausalLM, TransformerDecoder
from uniserve.nn.attention import Attention
from uniserve.runtime import PrefixCache
from uniserve_worker.bootstrap.cache import (
    cache_info,
    group_layers,
    plan_cache,
    resident_width,
    storage,
)
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.storage.kv_cache import KVCacheManager


class _Layer(nn.Module):
    def __init__(self, index: int, window: int, dtype: torch.dtype):
        super().__init__()
        # ``TransformerDecoder.cache_config`` takes the cache dtype from a
        # layer's first parameter.
        self.scale = nn.Parameter(torch.zeros((), dtype=dtype))
        name = f"backbone.layers.{index}.attention"
        self.attention = (
            Attention(2, 1, 8, cache_name=name)
            if index % 6 == 5
            else Attention(4, 4, 4, cache_name=name, window=window)
        )


def hybrid_model(
    *, layers: int = 12, window: int = 8, dtype: torch.dtype = torch.float32
) -> CausalLM:
    """Build the declaration-only hybrid model."""
    backbone = TransformerDecoder(
        nn.Embedding(8, 4),
        nn.ModuleDict(
            {
                str(index): _Layer(index, window, dtype)
                for index in range(layers)
            }
        ),
        nn.Identity(),
    )
    return CausalLM(backbone, nn.Identity())


def hybrid_pool(
    model: CausalLM,
    config: WorkerConfig,
    *,
    num_units: int,
    request_pool_size: int = 1,
    import_capacity: int = 1,
) -> KVCacheManager:
    """Own the model's unit pool as a worker does, sized to ``num_units``."""
    planes = plan_cache(model, config)
    dtype, fp8 = storage(config)
    if fp8:
        raise ValueError("the hybrid fixture stores unquantized K/V")
    cache = PrefixCache(
        model.cache_config,
        num_units=num_units,
        block_size=config.block_size,
        device=config.device,
        dtype=dtype,
    )
    return KVCacheManager(
        cache,
        info=cache_info(model, config, num_units=num_units),
        group_layers=group_layers(model, planes),
        import_capacity=import_capacity,
        request_pool_size=request_pool_size,
        table_width=resident_width(
            planes, max_sequence_tokens=config.max_sequence_tokens
        ),
    )
