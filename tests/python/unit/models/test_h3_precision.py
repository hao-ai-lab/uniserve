"""MiniMax H3 component-precision policy behavior."""

from __future__ import annotations

import pytest

from uniserve_worker.models.minimax_h3.precision import H3LinearPrecisionPolicy

pytestmark = pytest.mark.unit


def test_h3_precision_shorthands_resolve_component_defaults() -> None:
    assert H3LinearPrecisionPolicy.resolve("fp8") == H3LinearPrecisionPolicy(
        transformer_attention="fp8",
        transformer_mlp="fp8",
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
