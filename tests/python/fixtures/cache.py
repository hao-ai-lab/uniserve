"""Real named MHA storage and its scheduler transfer declaration."""

import torch

from uniserve.cache import Config, mha
from uniserve.quantization import Quantizer
from uniserve.runtime import PrefixCache
from uniserve_worker.bootstrap.worker_info import KVCacheInfo, KvGroup, KvGroupKind
from uniserve_worker.runtime.cache_manager import CacheManager


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
    max_blocks_per_request=None,
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
    quantization = {name: Quantizer("fp8", axis=0) for name in layers} if encoded else None
    cache = PrefixCache(
        Config(layers),
        num_blocks=num_pages,
        block_size=page_size,
        dtype=dtype if encoded else store_dtype or dtype,
        quantization=quantization,
        device=device,
    )
    byte_count = sum(
        field.nbytes
        for layout in layers.values()
        for field in layout.buffers(
            num_blocks=1,
            block_size=page_size,
            dtype=dtype if encoded else store_dtype or dtype,
            quantizer=Quantizer("fp8", axis=0) if encoded else None,
        ).values()
    )
    info = KVCacheInfo(
        block_size=page_size,
        num_blocks=num_pages,
        num_layers=num_layers,
        total_layers=total_layers,
        layer_offset=layer_offset,
        num_kv_heads=num_kv_heads,
        total_kv_heads=total_kv_heads,
        kv_head_offset=kv_head_offset,
        head_dim=head_dim,
        bytes_per_token=(byte_count + page_size - 1) // page_size,
        groups=(KvGroup(num_pages, KvGroupKind.FULL, 0, 0),),
        dtype=str(store_dtype or dtype).removeprefix("torch."),
    )
    return CacheManager(
        cache,
        info=info,
        request_pool_size=request_pool_size,
        max_blocks_per_request=max_blocks_per_request,
        import_capacity=import_capacity,
    )
