"""Quantization seam for shared layers."""

from uniserve.nn.quant.base import LinearMethod, UnquantizedLinearMethod, process_quantized_modules
from uniserve.nn.quant.config import QuantizationConfig
from uniserve.nn.quant.fp8 import DynamicW8A8Fp8LinearMethod, W8A8Fp8LinearMethod
from uniserve.nn.quant.kv_cache import (
    dequantize_fp8_block,
    is_fp8_kv_dtype,
    kv_cache_bytes_per_token,
    kv_store_dtype_name,
    kv_store_itemsize,
    quantize_fp8_block,
    resolve_kv_store_dtype,
    scale_for_fp8_block,
)
from uniserve.nn.quant.mxfp8 import DynamicW8A8MxFp8LinearMethod
from uniserve.nn.quant.nvfp4 import DynamicW4A4NvFp4LinearMethod

__all__ = [
    "QuantizationConfig",
    "LinearMethod",
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
