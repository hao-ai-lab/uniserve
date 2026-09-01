"""Quantization seam for shared layers."""

from .base import QuantizeMethodBase, UnquantizedLinearMethod, process_quantized_modules
from .config import QuantizationConfig
from .fp8 import DynamicW8A8Fp8LinearMethod, W8A8Fp8LinearMethod
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
from .mxfp8 import DynamicW8A8MxFp8LinearMethod
from .nvfp4 import DynamicW4A4NvFp4LinearMethod

__all__ = [
    "QuantizationConfig",
    "QuantizeMethodBase",
    "UnquantizedLinearMethod",
    "process_quantized_modules",
    "DynamicW8A8Fp8LinearMethod",
    "DynamicW8A8MxFp8LinearMethod",
    "DynamicW4A4NvFp4LinearMethod",
    "W8A8Fp8LinearMethod",
    "dequantize_fp8_block",
    "is_fp8_kv_dtype",
    "kv_cache_bytes_per_token",
    "kv_store_dtype_name",
    "kv_store_itemsize",
    "quantize_fp8_block",
    "resolve_kv_store_dtype",
    "scale_for_fp8_block",
]
