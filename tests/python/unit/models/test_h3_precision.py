"""H3 precision choices reject unsupported per-component formats."""

import pytest
import torch

from uniserve_models.minimax_h3 import weight_config

pytestmark = pytest.mark.unit


_TIERS = {
    "quality": ("bf16", "bf16", "bf16", "fp16"),
    "balanced": ("bf16", "bf16", "bf16", "nvfp4"),
    "performance": ("bf16", "fp8", "bf16", "nvfp4"),
    "maximum": ("bf16", "nvfp4", "fp8", "nvfp4"),
}


def _format(config, path):
    selected = config.quantization[path]
    if selected is not None:
        return selected.weight.format
    candidates = (
        (len(prefix), dtype)
        for prefix, dtype in config.dtypes.items()
        if path == prefix or path.startswith(prefix + ".")
    )
    dtype = max(candidates, default=(-1, config.dtype))[1]
    return "fp16" if dtype is torch.float16 else "bf16"


@pytest.mark.parametrize(("preset", "expected"), _TIERS.items())
def test_serving_tiers_expand_to_the_public_precision_contract(
    preset,
    expected,
):
    attention, mlp, text_encoder, video_vae = expected
    config = weight_config(preset=preset)
    layer = "denoiser.transformer.layers.0"
    decoder = "video_decoder.decoder.decoder"

    assert _format(config, f"{layer}.attention.projection") == attention
    assert _format(config, f"{layer}.mlp") == mlp
    assert _format(config, "text_encoder") == text_encoder

    vae_dtype = torch.float16 if video_vae == "fp16" else torch.bfloat16
    assert config.dtypes[f"{decoder}.post_quant_conv"] is vae_dtype
    assert config.dtypes[f"{decoder}.decoder.input"] is vae_dtype
    assert config.dtypes[f"{decoder}.decoder.output"] is vae_dtype
    assert _format(config, f"{decoder}.decoder.layers.0.qkv") == video_vae


def test_default_precision_is_quality():
    assert weight_config() == weight_config(preset="quality")


@pytest.mark.parametrize(
    "choices",
    [
        {"attention": "mxfp8"},
        {"mlp": "fp16"},
        {"text_encoder": "mxfp8"},
        {"video_vae": "fp8"},
    ],
)
def test_unsupported_component_formats(choices):
    with pytest.raises(ValueError, match=next(iter(choices))):
        weight_config(**choices)
