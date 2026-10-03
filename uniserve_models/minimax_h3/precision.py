"""Named H3 weight and activation representations at their numerical owners.

``weight_config`` expands a named preset, optionally with per-component
overrides, into a ``uniserve.loading.weights.Config`` whose ``dtypes`` and
``quantization`` mappings are keyed by module path and resolved by longest
prefix; the paths cover the denoisers the checkpoint holds. ``precisions``
lists every complete preset selectable by name at load time, and
``checkpoint_precision`` is the dense base for a calibrated ModelOpt
checkpoint, onto which ``uniserve_models.loading`` overlays the checkpoint's
own quantized modules.
"""

from collections.abc import Mapping
from types import MappingProxyType

import torch

from uniserve.loading import weights
from uniserve.quantization import QuantizationConfig, Quantizer

from .config import Config, TransformerConfig
from .video_vae import Config as VideoConfig

# This table is the single definition of each named numerical tier. Both
# complete presets and component overrides expand through weight_config.
# Each row names (attention, mlp, text_encoder, video_vae) representations, in
# weight_config's keyword order. "default" selects "quality": the dense
# checkpoint's own BF16 transformer and FP16 video decoder, which every
# supported GPU computes. Quantized tiers are explicit choices.
_formats = {
    "default": ("bf16", "bf16", "bf16", "fp16"),
    "quality": ("bf16", "bf16", "bf16", "fp16"),
    "balanced": ("bf16", "bf16", "bf16", "nvfp4"),
    "performance": ("bf16", "fp8", "bf16", "nvfp4"),
    "maximum": ("bf16", "nvfp4", "fp8", "nvfp4"),
}


def weight_config(
    config: Config,
    *,
    preset: str = "default",
    attention: str | None = None,
    mlp: str | None = None,
    text_encoder: str | None = None,
    video_vae: str | None = None,
) -> weights.Config:
    """Expand a preset and independent component choices into module paths.

    Args:
        config: The checkpoint's configuration; every denoiser it holds
            takes the same representation.
        preset: A key of the preset table; supplies every component the
            keyword overrides leave as None.
        attention: Denoiser attention projections: bf16, fp8 or nvfp4.
        mlp: Denoiser feed-forward projections: bf16, fp8, mxfp8 or nvfp4.
        text_encoder: Text encoder language-layer linear layers: bf16, fp8
            or nvfp4. The vision tower keeps the checkpoint's BF16.
        video_vae: Video VAE decoder projections: fp16, bf16 or nvfp4.

    Returns:
        Module-path dtypes and quantization for the default H3 layer counts
        of ``TransformerConfig`` and the video VAE ``Config``.

    Raises:
        ValueError: The preset is unknown or a component representation is
            unsupported for that component.
    """
    if preset not in _formats:
        raise ValueError(
            f"unknown H3 precision {preset!r}; choose from {tuple(_formats)}"
        )

    attention, mlp, text_encoder, video_vae = (
        base if override is None else override
        for base, override in zip(
            _formats[preset],
            (attention, mlp, text_encoder, video_vae),
            strict=True,
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
        # Unquantized precisions are expressed as a plain dtype, not a
        # Quantizer. One Quantizer encodes both the weight and the input
        # activation. FP8 keeps per-row statistics (axis 0) unless the caller
        # requests one tensor-wide scale; MXFP8 and NVFP4 use their block
        # formats.
        if value in {"bf16", "fp16"}:
            return None
        quantizer = Quantizer(
            value, axis=0 if value == "fp8" and not tensorwise else None
        )
        return QuantizationConfig(quantizer, quantizer)

    # The audio decoder, both condition encoders and each transformer's
    # latent input and output heads stay FP32 and unquantized, the VAEs'
    # native precision. Under the longest-prefix rule, every video decoder
    # parameter without a more specific entry below (norms, residual scales,
    # register tokens) also stays FP32. Every denoising component a
    # checkpoint may hold takes the same representation.
    dtypes = {
        "audio_decoder": torch.float32,
        "video_decoder": torch.float32,
        "video_encoder": torch.float32,
        "audio_encoder": torch.float32,
        **{
            f"{component}.transformer.{name}": torch.float32
            for component in config.denoisers
            for name in (
                "video_input",
                "audio_input",
                "video_output",
                "audio_output",
            )
        },
    }
    quantization = {}
    for component in config.denoisers:
        for index in range(TransformerConfig().num_hidden_layers):
            path = f"{component}.transformer.layers.{index}"
            # With FP8, the merged attention projection takes one tensor-wide
            # scale; the attention output and feed-forward take per-row
            # scales.
            quantization[f"{path}.attention.projection"] = encoded(
                attention, tensorwise=True
            )
            quantization[f"{path}.attention.output"] = encoded(attention)
            quantization[f"{path}.mlp"] = encoded(mlp)
    quantization["text_encoder"] = encoded(text_encoder)
    # The text encoder representations were chosen for its language layers;
    # its vision tower has no quantized representation and keeps BF16.
    quantization["text_encoder.vision"] = None

    # This path names video_vae.Decoder; its ``decoder`` child is the VAE
    # transformer whose ``input`` weight dtype video_vae.Model.compute_dtype
    # reads as the autocast dtype of the complete VAE invocation.
    path = "video_decoder.decoder.decoder"
    if video_vae in {"fp16", "bf16"}:
        dtype = torch.float16 if video_vae == "fp16" else torch.bfloat16
        dtypes[f"{path}.post_quant_conv"] = dtype
        dtypes[f"{path}.decoder.input"] = dtype
        dtypes[f"{path}.decoder.output"] = dtype
    else:
        dtype = torch.bfloat16
        quantization[f"{path}.decoder.output"] = encoded(video_vae)
        # NVFP4 quantizes only the output projection among the boundary
        # layers. The input projection and post_quant_conv stay unquantized in
        # BF16, which makes BF16 the autocast dtype of the VAE invocation.
        dtypes[f"{path}.post_quant_conv"] = dtype
        dtypes[f"{path}.decoder.input"] = dtype
        dtypes[f"{path}.decoder.output"] = dtype

    for index in range(VideoConfig().decoder_num_layers):
        for layer in ("qkv", "output", "mlp"):
            child = f"{path}.decoder.layers.{index}.{layer}"
            dtypes[child] = dtype
            quantization[child] = encoded(video_vae)
    return weights.Config(dtypes=dtypes, quantization=quantization)


def precisions(config: Config) -> Mapping[str, weights.Config]:
    """Complete presets ``uniserve_models.loading.load_model`` accepts by name.

    These apply to a dense checkpoint; a calibrated ModelOpt checkpoint offers
    none. ``read_config`` takes the "default" entry as a dense checkpoint's
    base weights, which a checkpoint quantization_config may refine.
    """
    return MappingProxyType(
        {name: weight_config(config, preset=name) for name in _formats}
    )


def checkpoint_precision(config: Config) -> weights.Config:
    """Dense base of a calibrated checkpoint's modules.

    Dense modules keep BF16, the video VAE projections included, so the VAE
    runs under BF16 autocast as with the NVFP4 video_vae choice. The audio
    decoder, the latent heads, and the untargeted video decoder parameters
    stay FP32.
    """
    return weight_config(config, preset="quality", video_vae="bf16")
