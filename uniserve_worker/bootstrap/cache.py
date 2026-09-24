"""Describe a rank's paged K/V cache to the engine scheduler.

The worker reports a ``KVCacheInfo`` in its ``WorkerInfo``: one rectangular
region of the model's logical cache, a consecutive interval of layers times a
consecutive interval of K/V heads, together with page geometry and the bytes
one token occupies. The token-worker path of ``build_worker_layout`` in
``uniserve_worker.bootstrap.report`` calls ``cache_info`` with a one-page
pool to learn the per-token size, sizes the pool, and attaches the granted
page count with ``resize_cache``. When the engine assembles a worker group
from its ranks' startup reports, it requires their regions to cover every
layer's K/V heads without a gap.
"""

from dataclasses import replace

import torch

from uniserve.cache import mha
from uniserve.math import ceil_div
from uniserve.model import CausalLM
from uniserve.quantization import Quantizer
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.protocol.worker_info import (
    KVCacheInfo,
    KvGroup,
    KvGroupKind,
)


def cache_info(
    model: CausalLM, config: WorkerConfig, *, num_blocks: int
) -> KVCacheInfo:
    """Map resident named MHA layers to the global token/layer/head axes.

    The wire protocol describes contiguous homogeneous layer/head intervals.
    Library caches remain free to use other state layouts; unsupported wire
    layouts fail here before any scheduler grants or transfers are advertised.

    Args:
        model: The causal LM whose ``cache_config`` lists this rank's resident
            cache layers and whose backbone orders all logical layers.
        config: Supplies ``block_size`` and the optional ``kv_cache_dtype``
            override of the layers' compute dtype.
        num_blocks: Physical pages to advertise.

    Returns:
        The cache description with one full-context group of ``num_blocks``
        pages.

    Raises:
        ValueError: There are no resident layers; they are not all MHA
            layouts, not consecutive in ``cache_names`` order, not one
            identical layout, or hold non-consecutive heads; a layer is not
            named in ``cache_names``; ``block_size`` is below one; or the
            configured dtype cannot store K/V values.
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

    # FP8 storage is expressed to the layout as its compute dtype plus a
    # per-block FP8 quantizer; the published dtype names the stored FP8 type.
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
    # One block's fields include per-block metadata (initialized flags and,
    # with FP8, scales), so the per-token figure amortizes it and rounds up.
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
    """Attach the granted capacity to one complete physical page group.

    ``info`` must carry the single group ``cache_info`` produces.
    """
    return replace(
        info,
        num_blocks=num_blocks,
        groups=(replace(info.groups[0], num_blocks=num_blocks),),
    )
