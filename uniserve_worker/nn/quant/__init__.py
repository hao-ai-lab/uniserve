"""Quantization seam for shared layers."""
from .base import QuantizeMethodBase, UnquantizedLinearMethod
from .config import (
    QuantizationConfig,
    get_current_kv_cache_dtype,
    get_current_quantization_config,
    is_quant_context_active,
    kv_cache_dtype_from_model_config,
    use_quantization_config,
    warn_if_no_quant_context,
)
from .fp8 import W8A8Fp8LinearMethod
from .kv_cache import (
    dequantize_fp8_block,
    is_fp8_kv_dtype,
    kv_cache_bytes_per_token,
    kv_store_dtype_name,
    kv_store_itemsize,
    quantize_fp8_block,
    resolve_kv_store_dtype,
    scale_for_fp8_block,
)

__all__ = [
    "QuantizationConfig",
    "QuantizeMethodBase",
    "UnquantizedLinearMethod",
    "W8A8Fp8LinearMethod",
    "dequantize_fp8_block",
    "get_current_kv_cache_dtype",
    "get_current_quantization_config",
    "is_fp8_kv_dtype",
    "is_quant_context_active",
    "kv_cache_bytes_per_token",
    "kv_cache_dtype_from_model_config",
    "kv_store_dtype_name",
    "kv_store_itemsize",
    "quantize_fp8_block",
    "resolve_kv_store_dtype",
    "scale_for_fp8_block",
    "use_quantization_config",
    "warn_if_no_quant_context",
]
