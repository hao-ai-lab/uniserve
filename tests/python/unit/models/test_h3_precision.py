"""MiniMax H3 component-precision policy behavior."""

from __future__ import annotations

import pytest

from uniserve_worker.models.minimax_h3.precision import H3LinearPrecisionPolicy

pytestmark = pytest.mark.unit


def test_h3_uses_balanced_mode_when_config_is_empty() -> None:
    assert H3LinearPrecisionPolicy.from_config({}) == H3LinearPrecisionPolicy(
        transformer_attention="bf16",
        transformer_mlp="bf16",
        text_encoder="bf16",
        video_vae="nvfp4",
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "quality",
            H3LinearPrecisionPolicy("bf16", "bf16", "bf16", "fp16"),
        ),
        (
            "balanced",
            H3LinearPrecisionPolicy("bf16", "bf16", "bf16", "nvfp4"),
        ),
        (
            "performance",
            H3LinearPrecisionPolicy("bf16", "fp8", "bf16", "nvfp4"),
        ),
        (
            "maximum",
            H3LinearPrecisionPolicy("nvfp4", "mxfp8", "bf16", "nvfp4"),
        ),
    ],
)
def test_h3_quantization_modes_resolve_component_policies(
    mode: str,
    expected: H3LinearPrecisionPolicy,
) -> None:
    assert H3LinearPrecisionPolicy.from_config({"mode": mode}) == expected


def test_h3_precision_shorthands_resolve_component_defaults() -> None:
    assert H3LinearPrecisionPolicy.resolve("bf16") == H3LinearPrecisionPolicy(
        transformer_attention="bf16",
        transformer_mlp="bf16",
        text_encoder="bf16",
        video_vae="fp16",
    )
    assert H3LinearPrecisionPolicy.resolve("fp8") == H3LinearPrecisionPolicy(
        transformer_attention="fp8",
        transformer_mlp="fp8",
        text_encoder="bf16",
        video_vae="fp16",
    )
    assert H3LinearPrecisionPolicy.resolve("mxfp8") == H3LinearPrecisionPolicy(
        transformer_attention="bf16",
        transformer_mlp="mxfp8",
        text_encoder="bf16",
        video_vae="fp16",
    )
    assert H3LinearPrecisionPolicy.resolve("nvfp4") == H3LinearPrecisionPolicy(
        transformer_attention="nvfp4",
        transformer_mlp="nvfp4",
        text_encoder="nvfp4",
        video_vae="nvfp4",
    )


def test_h3_precision_component_overrides_are_authoritative() -> None:
    policy = H3LinearPrecisionPolicy.resolve(
        "nvfp4",
        transformer_attention="fp8",
        transformer_mlp="nvfp4",
        text_encoder="bf16",
        video_vae="bf16",
    )

    assert policy == H3LinearPrecisionPolicy(
        transformer_attention="fp8",
        transformer_mlp="nvfp4",
        text_encoder="bf16",
        video_vae="bf16",
    )


def test_h3_component_quantization_config_resolves_policy() -> None:
    policy = H3LinearPrecisionPolicy.from_config(
        {
            "mode": "balanced",
            "components": {
                "transformer.mlp": "fp8",
                "video_vae": "fp16",
            },
        }
    )
    assert policy == H3LinearPrecisionPolicy(
        transformer_attention="bf16",
        transformer_mlp="fp8",
        text_encoder="bf16",
        video_vae="fp16",
    )


def test_h3_mode_and_quant_method_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="both mode and quant_method"):
        H3LinearPrecisionPolicy.from_config({"mode": "balanced", "quant_method": "fp8"})


def test_h3_mxfp8_is_scoped_to_transformer_mlp() -> None:
    policy = H3LinearPrecisionPolicy.from_config(
        {
            "quant_method": "bf16",
            "components": {"transformer.mlp": "mxfp8"},
        }
    )
    assert policy == H3LinearPrecisionPolicy(
        transformer_attention="bf16",
        transformer_mlp="mxfp8",
        text_encoder="bf16",
        video_vae="fp16",
    )

    with pytest.raises(ValueError, match="transformer.attention"):
        H3LinearPrecisionPolicy.from_config(
            {
                "quant_method": "bf16",
                "components": {"transformer.attention": "mxfp8"},
            }
        )
