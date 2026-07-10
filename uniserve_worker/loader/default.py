"""Default checkpoint loader for new-style UniModel classes."""
from __future__ import annotations

import logging
import time
from typing import Any, Type, cast

from ..contracts.model_protocols import UniModel
from ..nn.quant import QuantizationConfig, use_quantization_config
from ..nn.quant.base import process_quantized_modules
from .base import BaseModelLoader, LoadResult
from .registry import register_loader
from .weight_utils import iter_weights, resolve_weight_files, strict_load_weights

__all__ = [
    'DefaultModelLoader',
]

logger = logging.getLogger(__name__)


class DefaultModelLoader(BaseModelLoader):
    def load_model(
        self,
        model_cls: Type[UniModel],
        config: Any,
        *,
        device: str = "cpu",
        model_path: str | None = None,
        **kwargs: Any,
    ) -> LoadResult:
        del kwargs  # config-driven streaming ignores serving-wrapper extras
        if model_path is None:
            raise ValueError("DefaultModelLoader requires model_path")
        start = time.perf_counter()
        quant_config = QuantizationConfig.from_model_config(config)
        with use_quantization_config(quant_config):
            model = cast(Any, model_cls)(config=config)
        files = resolve_weight_files(model_path)
        summary = strict_load_weights(model, iter_weights(files))
        modules = getattr(model, "modules", None)
        if callable(modules):
            process_quantized_modules(modules())
        # Fold weights to the model's serving dtype on CPU before the device
        # move so only the serving-dtype weights are copied to the GPU. Without
        # this, a model materialized in fp32 would transfer the full fp32 copy
        # to the device and only downcast afterwards, doubling the device-side
        # peak and overflowing GPUs that can hold the bf16 model but not fp32.
        prepare_dtype = getattr(model, "prepare_serving_dtype", None)
        if callable(prepare_dtype):
            prepare_dtype()
        move_to = getattr(model, "to", None)
        if callable(move_to):
            move_to(device)
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        dtype, real_device = _first_parameter_dtype_device(model)
        loaded = summary.loaded_count if summary.loaded_count is not None else summary.tensors_seen
        elapsed = time.perf_counter() - start
        logger.info(
            "loaded model checkpoint",
            extra={
                "model_class": model_cls.__name__,
                "model_path": str(model_path),
                "files": [str(path) for path in files],
                "tensors_seen": summary.tensors_seen,
                "loaded_tensors": loaded,
                "ignored_tensors": len(summary.ignored),
                "dtype": dtype,
                "device": real_device or str(device),
                "elapsed_s": round(elapsed, 3),
            },
        )
        if summary.ignored:
            logger.warning(
                "ignored checkpoint tensors during model load",
                extra={
                    "model_class": model_cls.__name__,
                    "ignored_tensors": len(summary.ignored),
                    "ignored_preview": list(summary.ignored[:20]),
                },
            )
        return LoadResult(model=model, tokenizer=None, device=real_device or str(device))


def _first_parameter_dtype_device(model: UniModel) -> tuple[str | None, str | None]:
    params = getattr(model, "parameters", None)
    if not callable(params):
        return None, None
    for param in params():
        return str(getattr(param, "dtype", None)), str(getattr(param, "device", None))
    return None, None


register_loader("default", DefaultModelLoader())
