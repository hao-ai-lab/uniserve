"""Named H3 weight and activation representations at their numerical owners."""

from types import MappingProxyType

import torch

from uniserve.loading import weights
from uniserve.quantization import QuantizationConfig, Quantizer

from .config import TransformerConfig
from .video_vae import Config as VideoConfig

# This table is the single definition of each named numerical choice. Both
# complete presets and component overrides expand through weight_config.
_formats = {
    "default": ("bf16", "bf16", "bf16", "nvfp4"),
    "quality": ("bf16", "bf16", "bf16", "fp16"),
    "bf16": ("bf16", "bf16", "bf16", "fp16"),
    "balanced": ("bf16", "bf16", "bf16", "nvfp4"),
    "performance": ("bf16", "fp8", "bf16", "nvfp4"),
    "maximum": ("nvfp4", "mxfp8", "fp8", "nvfp4"),
    "fp8": ("fp8", "fp8", "bf16", "fp16"),
    "mxfp8": ("bf16", "mxfp8", "bf16", "fp16"),
    "nvfp4": ("nvfp4", "nvfp4", "nvfp4", "nvfp4"),
}


def weight_config(
    *,
    preset: str = "default",
    attention: str | None = None,
    mlp: str | None = None,
    text_encoder: str | None = None,
    video_vae: str | None = None,
) -> weights.Config:
    """Expand a preset and independent component choices into module paths."""

    if preset not in _formats:
        raise ValueError(f"unknown H3 precision {preset!r}; choose from {tuple(_formats)}")
    attention, mlp, text_encoder, video_vae = (
        base if override is None else override
        for base, override in zip(
            _formats[preset], (attention, mlp, text_encoder, video_vae), strict=True
        )
    )
    selections = (
        ("attention", attention, ("bf16", "fp8", "nvfp4")),
        ("mlp", mlp, ("bf16", "fp8", "mxfp8", "nvfp4")),
        ("text_encoder", text_encoder, ("bf16", "fp8", "nvfp4")),
        ("video_vae", video_vae, ("fp16", "bf16", "nvfp4")),
    )
    for name, value, supported in selections:
        if value not in supported:
            raise ValueError(f"H3 {name} requires one of {supported}")

    def encoded(value, *, tensorwise=False):
        if value in {"bf16", "fp16"}:
            return None
        quantizer = Quantizer(value, axis=0 if value == "fp8" and not tensorwise else None)
        return QuantizationConfig(quantizer, quantizer)

    dtypes = {
        "audio_decoder": torch.float32,
        "video_decoder": torch.float32,
        **{
            f"denoiser.transformer.{name}": torch.float32
            for name in ("video_input", "audio_input", "video_output", "audio_output")
        },
    }
    quantization = {}
    for index in range(TransformerConfig().num_hidden_layers):
        path = f"denoiser.transformer.layers.{index}"
        quantization[f"{path}.attention.projection"] = encoded(attention, tensorwise=True)
        quantization[f"{path}.attention.output"] = encoded(attention)
        quantization[f"{path}.mlp"] = encoded(mlp)
    quantization["text_encoder"] = encoded(text_encoder)
    path = "video_decoder.decoder.decoder"
    if video_vae in {"fp16", "bf16"}:
        dtype = torch.float16 if video_vae == "fp16" else torch.bfloat16
        dtypes[f"{path}.post_quant_conv"] = dtype
        dtypes[f"{path}.decoder.input"] = dtype
        dtypes[f"{path}.decoder.output"] = dtype
    else:
        dtype = torch.bfloat16
        quantization[f"{path}.decoder.output"] = encoded(video_vae)
        dtypes[f"{path}.decoder.output"] = dtype
    for index in range(VideoConfig().decoder_num_layers):
        for layer in ("qkv", "output", "mlp"):
            child = f"{path}.decoder.layers.{index}.{layer}"
            dtypes[child] = dtype
            quantization[child] = encoded(video_vae)
    return weights.Config(dtypes=dtypes, quantization=quantization)


precisions = MappingProxyType({name: weight_config(preset=name) for name in _formats})
