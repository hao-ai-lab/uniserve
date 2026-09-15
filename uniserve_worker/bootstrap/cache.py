"""Describe the scheduler's rectangular K/V protocol from state layouts."""

from dataclasses import replace

import torch

from uniserve.cache import mha
from uniserve.math import ceil_div
from uniserve.model import CausalLM
from uniserve.quantization import Quantizer

from ..config import WorkerConfig
from .worker_info import KVCacheInfo, KvGroup, KvGroupKind


def cache_info(
    model: CausalLM, config: WorkerConfig, *, num_blocks: int
) -> KVCacheInfo:
    """Map resident named MHA layers to the global token/layer/head axes.

    The wire protocol describes contiguous homogeneous layer/head intervals.
    Library caches remain free to use other state layouts; unsupported wire
    layouts fail here before any scheduler grants or transfers are advertised.
    """
    layers = model.cache_config.layers
    if not layers or any(
        not isinstance(value, mha.Config) for value in layers.values()
    ):
        raise ValueError(
            "the worker K/V protocol requires resident MHA state layers"
        )

    names = model.backbone.cache_names
    indexes = tuple(names.index(name) for name in layers)
    if indexes != tuple(range(indexes[0], indexes[0] + len(indexes))):
        raise ValueError(
            "the worker K/V protocol requires consecutive logical cache layers"
        )

    layout = next(iter(layers.values()))
    if any(value != layout for value in layers.values()):
        raise ValueError(
            "the worker K/V protocol requires matching per-layer head layouts"
        )

    heads = layout.head_indices
    if heads != tuple(range(heads[0], heads[0] + len(heads))):
        raise ValueError(
            "the worker K/V protocol requires consecutive logical cache heads"
        )

    dtype = (
        layout.compute_dtype
        if config.kv_cache_dtype is None
        else getattr(torch, config.kv_cache_dtype.removeprefix("torch."))
    )
    quantizer = (
        Quantizer("fp8", axis=0) if dtype is torch.float8_e4m3fn else None
    )
    fields = layout.buffers(
        num_blocks=1,
        block_size=config.block_size,
        dtype=layout.compute_dtype if quantizer is not None else dtype,
        quantizer=quantizer,
    )
    bytes_per_token = ceil_div(
        len(layers) * sum(field.nbytes for field in fields.values()),
        config.block_size,
    )

    return KVCacheInfo(
        block_size=config.block_size,
        num_blocks=num_blocks,
        num_layers=len(layers),
        total_layers=len(names),
        layer_offset=indexes[0],
        num_kv_heads=len(heads),
        total_kv_heads=layout.num_kv_heads,
        kv_head_offset=heads[0],
        head_dim=layout.head_dim,
        bytes_per_token=bytes_per_token,
        groups=(KvGroup(num_blocks, KvGroupKind.FULL, 0, 0),),
        dtype=str(dtype).removeprefix("torch."),
    )


def resize_cache(info: KVCacheInfo, num_blocks: int) -> KVCacheInfo:
    """Attach the granted capacity to one complete physical page group."""
    return replace(
        info,
        num_blocks=num_blocks,
        groups=(replace(info.groups[0], num_blocks=num_blocks),),
    )
