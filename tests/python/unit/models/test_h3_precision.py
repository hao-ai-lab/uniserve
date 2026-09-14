"""H3 load configuration preserves preset formats and component restrictions."""

import pytest

from uniserve.nn.quant.config import resolve_component_precisions
from uniserve_models.minimax_h3.config import (
    PRECISION_PRESETS,
    PRECISION_SHORTHANDS,
    SUPPORTED_PRECISIONS,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, ("bf16", "bf16", "bf16", "nvfp4")),
        ({"mode": "quality"}, ("bf16", "bf16", "bf16", "fp16")),
        ({"mode": "balanced"}, ("bf16", "bf16", "bf16", "nvfp4")),
        ({"mode": "performance"}, ("bf16", "fp8", "bf16", "nvfp4")),
        ({"mode": "maximum"}, ("nvfp4", "mxfp8", "fp8", "nvfp4")),
        ({"quant_method": "bf16"}, ("bf16", "bf16", "bf16", "fp16")),
        ({"quant_method": "fp8"}, ("fp8", "fp8", "bf16", "fp16")),
        ({"quant_method": "mxfp8"}, ("bf16", "mxfp8", "bf16", "fp16")),
        ({"quant_method": "nvfp4"}, ("nvfp4", "nvfp4", "nvfp4", "nvfp4")),
        (
            {
                "mode": "balanced",
                "components": {
                    "transformer.mlp": "fp8",
                    "text_encoder": "fp8",
                    "video_vae": "fp16",
                },
            },
            ("bf16", "fp8", "fp8", "fp16"),
        ),
        (
            {
                "quant_method": "nvfp4",
                "components": {
                    "transformer.attention": "fp8",
                    "text_encoder": "bf16",
                    "video_vae": "bf16",
                },
            },
            ("fp8", "nvfp4", "bf16", "bf16"),
        ),
    ],
)
def test_component_formats(config, expected):
    resolved = resolve_component_precisions(
        config,
        supported=SUPPORTED_PRECISIONS,
        presets=PRECISION_PRESETS,
        shorthands=PRECISION_SHORTHANDS,
        default_mode="balanced",
    )
    assert resolved == dict(
        zip(
            ("transformer.attention", "transformer.mlp", "text_encoder", "video_vae"),
            expected,
        )
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"mode": "balanced", "quant_method": "fp8"}, "both mode and quant_method"),
        ({"mode": ""}, "mode"),
        ({"components": {"transformer.attention": "mxfp8"}}, "transformer.attention"),
        ({"components": {"video_vae": "fp8"}}, "video_vae"),
        ({"components": {"absent": "bf16"}}, "unknown entries"),
    ],
)
def test_invalid_component_formats(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_component_precisions(
            config,
            supported=SUPPORTED_PRECISIONS,
            presets=PRECISION_PRESETS,
            shorthands=PRECISION_SHORTHANDS,
            default_mode="balanced",
        )
