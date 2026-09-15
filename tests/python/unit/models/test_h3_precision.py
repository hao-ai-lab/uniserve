"""Named H3 precision choices expose the supported numerical representations."""

import pytest
import torch

from uniserve.quantization import QuantizationConfig, Quantizer
from uniserve_models.minimax_h3 import precisions, weight_config

pytestmark = pytest.mark.unit


def _quantized(format, *, tensorwise=False):
    if format in {"bf16", "fp16"}:
        return None
    quantizer = Quantizer(
        format, axis=0 if format == "fp8" and not tensorwise else None
    )
    return QuantizationConfig(quantizer, quantizer)


@pytest.mark.parametrize(
    "name,formats",
    [
        ("default", ("bf16", "bf16", "bf16", "nvfp4")),
        ("quality", ("bf16", "bf16", "bf16", "fp16")),
        ("bf16", ("bf16", "bf16", "bf16", "fp16")),
        ("balanced", ("bf16", "bf16", "bf16", "nvfp4")),
        ("performance", ("bf16", "fp8", "bf16", "nvfp4")),
        ("maximum", ("nvfp4", "mxfp8", "fp8", "nvfp4")),
        ("fp8", ("fp8", "fp8", "bf16", "fp16")),
        ("mxfp8", ("bf16", "mxfp8", "bf16", "fp16")),
        ("nvfp4", ("nvfp4", "nvfp4", "nvfp4", "nvfp4")),
    ],
)
def test_named_representations(name, formats):
    attention, mlp, text, video = formats
    config = precisions[name]
    assert config.dtype == torch.bfloat16
    assert config.quantization[
        "denoiser.transformer.layers.0.attention.projection"
    ] == _quantized(attention, tensorwise=True)
    assert config.quantization[
        "denoiser.transformer.layers.0.attention.output"
    ] == _quantized(attention)
    assert config.quantization[
        "denoiser.transformer.layers.0.mlp"
    ] == _quantized(mlp)
    assert config.quantization["text_encoder"] == _quantized(text)
    assert config.quantization[
        "video_decoder.decoder.decoder.decoder.layers.0.qkv"
    ] == _quantized(video)
    assert config.dtypes[
        "video_decoder.decoder.decoder.decoder.layers.0.qkv"
    ] == (torch.float16 if video == "fp16" else torch.bfloat16)
    for component in (
        "audio_decoder",
        "video_decoder",
        "denoiser.transformer.video_input",
        "denoiser.transformer.audio_output",
    ):
        assert config.dtypes[component] == torch.float32


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
