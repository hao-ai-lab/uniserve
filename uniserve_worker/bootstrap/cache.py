"""Describe a rank's paged K/V unit pool to the engine scheduler.

The worker reports a ``KVCacheInfo`` in its ``WorkerInfo``: the unit pool's
size and unit bytes, and one group per history window and K/V page shape.
Each group places this rank's layers by global cache-layer id and its K/V
heads by interval. The token-worker path of ``build_worker_layout`` in
``uniserve_worker.bootstrap.report`` calls ``cache_info`` with a minimal
pool to learn the unit size, sizes the pool, and attaches the granted unit
count with ``resize_cache``. When the engine assembles a worker group from
its ranks' startup reports, it requires every logical layer to belong to one
group and its K/V heads to be covered without a gap.
"""

from dataclasses import replace

import torch

from uniserve.math import ceil_div
from uniserve.model import CausalLM
from uniserve.quantization import Quantizer
from uniserve.runtime.prefix_cache import Planes, plan_units
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.protocol.worker_info import (
    KVCacheInfo,
    KvGroup,
    KvGroupKind,
)


def storage(config: WorkerConfig) -> tuple[torch.dtype | None, bool]:
    """Resolve the configured K/V storage as ``(dtype, fp8)``.

    ``dtype`` is the logical storage dtype passed to the unit pool, or
    ``None`` for each layer's compute dtype. FP8 storage is expressed as the
    compute dtype plus a per-block FP8 quantizer.
    """
    if config.kv_cache_dtype is None:
        return None, False
    dtype = getattr(torch, config.kv_cache_dtype.removeprefix("torch."))
    if dtype is torch.float8_e4m3fn:
        return None, True
    return dtype, False


def plan_cache(model: CausalLM, config: WorkerConfig) -> Planes:
    """Plan the rank's unit pool from its resident cache layers.

    The page size of the group with the widest token rows is the configured
    ``block_size``; every other group's page holds as many tokens as fit
    the same plane.

    Raises:
        ValueError: As ``plan_units``, or when the model has no resident
            cache layer.
    """
    layers = model.cache_config.layers
    if not layers:
        raise ValueError(
            "the worker K/V protocol requires resident cache layers"
        )
    dtype, fp8 = storage(config)
    return plan_units(
        model.cache_config,
        block_size=config.block_size,
        dtype=dtype,
        quantization={name: Quantizer("fp8", axis=0) for name in layers}
        if fp8
        else None,
    )


def table_widths(
    planes: Planes, *, max_sequence_tokens: int, max_query_tokens: int
) -> tuple[int, ...]:
    """Bound the columns one call stages per numerical block table.

    A full-attention table spans the longest sequence. A sliding-window
    table spans only the pages a reader's window of history and one call's
    queries intersect, at most ``ceil((window + queries) / page_tokens) +
    1``. Tables are in table order, one per unit position of every group.
    """
    widths = []
    for group in planes.groups:
        width = ceil_div(max(1, max_sequence_tokens), group.page_tokens)
        if group.window is not None:
            width = min(
                width,
                ceil_div(
                    group.window + max(1, max_query_tokens), group.page_tokens
                )
                + 1,
            )
        widths.extend((max(1, width),) * group.units_per_page)
    return tuple(widths)


def resident_width(planes: Planes, *, max_sequence_tokens: int) -> int:
    """Return the most pages one slot's installed group table may hold.

    A table may cover the longest sequence in any group, including a
    sliding-window group whose worst-case reservation holds every page.
    """
    return max(
        1,
        *(
            ceil_div(max(1, max_sequence_tokens), group.page_tokens)
            for group in planes.groups
        ),
    )


def group_layers(
    model: CausalLM, planes: Planes
) -> tuple[tuple[int, ...], ...]:
    """Return every group's global cache-layer ids across pipeline stages.

    A logical cache layer joins the group whose history window, logical K/V
    heads and head width it shares; the decoder records these for every
    layer, including those resident on other stages. A group's layers in
    ascending global order form its publication layer axis.

    Raises:
        ValueError: A resident group's layers are not the logical layers of
            one shape.
    """
    backbone = model.backbone
    names = backbone.cache_names
    layouts = model.cache_config.layers
    result = []
    for group in planes.groups:
        layout = layouts[group.layers[0]]
        key = (layout.window, layout.num_kv_heads, layout.head_dim)
        members = tuple(
            index
            for index, layer in enumerate(backbone.cache_layers)
            if (layer.window, layer.num_kv_heads, layer.head_dim) == key
        )
        local = tuple(names.index(name) for name in group.layers)
        if not set(local).issubset(members):
            raise ValueError(
                "resident cache layers disagree with their logical geometry"
            )
        result.append(members)
    if len({layer for members in result for layer in members}) != sum(
        map(len, result)
    ):
        raise ValueError(
            "logical cache layers belong to several resident groups"
        )
    return tuple(result)


def cache_info(
    model: CausalLM, config: WorkerConfig, *, num_units: int
) -> KVCacheInfo:
    """Map resident MHA layers to the unit pool's groups and global axes.

    Args:
        model: The causal LM whose ``cache_config`` lists this rank's resident
            cache layers and whose backbone orders all logical layers.
        config: Supplies ``block_size`` and the optional ``kv_cache_dtype``
            override of the layers' compute dtype.
        num_units: Units to advertise, including the unit-zero sentinel.

    Returns:
        The unit pool description with one group per history window and page
        shape; a history window becomes a sliding-window group.

    Raises:
        ValueError: As ``plan_cache``; or a group's layers hold
            non-consecutive heads.
    """
    planes = plan_cache(model, config)
    names = model.backbone.cache_names
    layouts = model.cache_config.layers
    dtype, fp8 = storage(config)

    groups = []
    for group in planes.groups:
        layout = layouts[group.layers[0]]
        heads = layout.head_indices
        if heads != tuple(range(heads[0], heads[0] + len(heads))):
            raise ValueError(
                "the worker K/V protocol requires consecutive logical cache "
                "heads"
            )
        groups.append(
            KvGroup(
                kind=KvGroupKind.FULL
                if group.window is None
                else KvGroupKind.SLIDING_WINDOW,
                window=0 if group.window is None else group.window,
                sink=0,
                page_tokens=group.page_tokens,
                units_per_page=group.units_per_page,
                layer_ids=tuple(names.index(name) for name in group.layers),
                num_kv_heads=len(heads),
                total_kv_heads=layout.num_kv_heads,
                kv_head_offset=heads[0],
                head_dim=group.head_dim,
            )
        )

    # The published dtype names the stored element type: FP8 for encoded
    # storage, otherwise the configured or compute dtype.
    stored = (
        torch.float8_e4m3fn
        if fp8
        else dtype
        if dtype is not None
        else layouts[planes.groups[0].layers[0]].compute_dtype
    )
    return KVCacheInfo(
        num_units=num_units,
        unit_bytes=planes.unit_bytes,
        dtype=str(stored).removeprefix("torch."),
        groups=tuple(groups),
    )


def resize_cache(info: KVCacheInfo, num_units: int) -> KVCacheInfo:
    """Attach the granted unit count to a cache description."""
    return replace(info, num_units=num_units)
