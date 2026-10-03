"""Real named MHA storage and its scheduler transfer declaration.

``mha_pool`` builds one full-attention group whose layers share a page
shape, so each logical page is one unit and unit ids are page ids.
"""

import torch

from uniserve.cache import Config, mha
from uniserve.quantization import Quantizer
from uniserve.runtime import PrefixCache
from uniserve_worker.protocol.worker_info import (
    KVCacheInfo,
    KvGroup,
    KvGroupKind,
)
from uniserve_worker.storage.kv_cache import KVCacheManager


def mha_pool(
    *,
    num_layers,
    num_kv_heads,
    head_dim,
    dtype,
    total_layers,
    total_kv_heads,
    num_pages,
    page_size,
    device,
    request_pool_size=1,
    table_width=None,
    layer_offset=0,
    kv_head_offset=0,
    store_dtype=None,
    import_capacity=1,
):
    layers = {
        f"layers.{index}.attention": mha.Config(
            total_kv_heads,
            head_dim,
            tuple(range(kv_head_offset, kv_head_offset + num_kv_heads)),
            dtype,
        )
        for index in range(layer_offset, layer_offset + num_layers)
    }
    encoded = store_dtype is torch.float8_e4m3fn
    quantization = (
        {name: Quantizer("fp8", axis=0) for name in layers} if encoded else None
    )
    cache = PrefixCache(
        Config(layers),
        num_units=num_pages,
        block_size=page_size,
        dtype=dtype if encoded else store_dtype or dtype,
        quantization=quantization,
        device=device,
    )
    info = KVCacheInfo(
        num_units=num_pages,
        unit_bytes=cache.planes.unit_bytes,
        dtype=str(store_dtype or dtype).removeprefix("torch."),
        groups=(
            KvGroup(
                kind=KvGroupKind.FULL,
                window=0,
                sink=0,
                page_tokens=page_size,
                units_per_page=1,
                layer_ids=tuple(range(layer_offset, layer_offset + num_layers)),
                num_kv_heads=num_kv_heads,
                total_kv_heads=total_kv_heads,
                kv_head_offset=kv_head_offset,
                head_dim=head_dim,
            ),
        ),
    )
    return KVCacheManager(
        cache,
        info=info,
        group_layers=(tuple(range(total_layers)),),
        request_pool_size=request_pool_size,
        table_width=table_width,
        import_capacity=import_capacity,
    )
