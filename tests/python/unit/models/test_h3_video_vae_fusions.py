from __future__ import annotations

import pytest
import torch

from uniserve_worker.models.minimax_h3.video_vae_fusions import (
    qk_rmsnorm_partial_rope_,
    scaled_residual_layernorm,
    scaled_residual_layernorm_absmax,
    scaled_residual_rmsnorm_,
    scaled_residual_rmsnorm_absmax_,
    value_first_swiglu,
    value_first_swiglu_absmax,
    video_patch_output,
    video_rmsnorm,
    video_rmsnorm_absmax,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

_WIDTH = 2048


def test_video_vae_normalization_absmax_matches_bf16_boundaries() -> None:
    torch.manual_seed(41)
    rows = 32
    hidden = torch.randn((rows, _WIDTH), dtype=torch.bfloat16, device="cuda")
    update = torch.randn_like(hidden)
    scale = torch.randn((_WIDTH,), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((_WIDTH,), dtype=torch.bfloat16, device="cuda")
    update_bias = torch.randn((_WIDTH,), dtype=torch.bfloat16, device="cuda")
    eps = 1e-5

    expected_normalized = video_rmsnorm(hidden, weight, eps=eps)
    actual_normalized, actual_maximum = video_rmsnorm_absmax(
        hidden,
        weight,
        eps=eps,
    )
    assert torch.equal(actual_normalized, expected_normalized)
    assert torch.equal(actual_maximum, expected_normalized.abs().amax())

    expected_hidden, expected_residual_normalized = scaled_residual_rmsnorm_(
        hidden.clone(),
        update,
        scale,
        weight,
        update_bias=update_bias,
        eps=eps,
    )
    actual_hidden, actual_residual_normalized, actual_maximum = scaled_residual_rmsnorm_absmax_(
        hidden.clone(),
        update,
        scale,
        weight,
        update_bias=update_bias,
        eps=eps,
    )
    assert torch.equal(actual_hidden, expected_hidden)
    assert torch.equal(actual_residual_normalized, expected_residual_normalized)
    assert torch.equal(actual_maximum, expected_residual_normalized.abs().amax())


def test_video_vae_layernorm_absmax_matches_bf16_boundary() -> None:
    torch.manual_seed(43)
    rows = 32
    hidden = torch.randn((rows, _WIDTH), dtype=torch.bfloat16, device="cuda")
    update = torch.randn_like(hidden)
    scale = torch.randn((_WIDTH,), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((_WIDTH,), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((_WIDTH,), dtype=torch.bfloat16, device="cuda")
    update_bias = torch.randn((_WIDTH,), dtype=torch.bfloat16, device="cuda")
    eps = 1e-5

    expected = scaled_residual_layernorm(
        hidden,
        update,
        scale,
        weight,
        bias,
        update_bias=update_bias,
        eps=eps,
    )
    actual, maximum = scaled_residual_layernorm_absmax(
        hidden,
        update,
        scale,
        weight,
        bias,
        update_bias=update_bias,
        eps=eps,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(maximum, expected.abs().amax())


def test_video_vae_swiglu_absmax_matches_bf16_boundary() -> None:
    torch.manual_seed(47)
    rows, intermediate = 16, 8192
    value_gate = torch.randn(
        (rows, 2 * intermediate),
        dtype=torch.bfloat16,
        device="cuda",
    )
    bias = torch.randn((2 * intermediate,), dtype=torch.bfloat16, device="cuda")

    expected = value_first_swiglu(value_gate, bias)
    actual, maximum = value_first_swiglu_absmax(value_gate, bias)
    assert torch.equal(actual, expected)
    assert torch.equal(maximum, expected.abs().amax())


def test_video_vae_qkv_bias_fusion_preserves_outputs() -> None:
    torch.manual_seed(53)
    shape = (2, 17, 32, 64)
    query = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    query_bias = torch.randn((2048,), dtype=torch.bfloat16, device="cuda")
    key_bias = torch.randn((2048,), dtype=torch.bfloat16, device="cuda")
    value_bias = torch.randn((2048,), dtype=torch.bfloat16, device="cuda")
    cosine = torch.randn((2, 17, 1, 48), dtype=torch.bfloat16, device="cuda")
    sine = torch.randn_like(cosine)

    expected_query, expected_key = qk_rmsnorm_partial_rope_(
        query.clone(),
        key.clone(),
        cosine,
        sine,
        query_bias=query_bias,
        key_bias=key_bias,
    )
    expected_value = value + value_bias.view(32, 64)
    actual_value = value.clone()
    actual_query, actual_key = qk_rmsnorm_partial_rope_(
        query.clone(),
        key.clone(),
        cosine,
        sine,
        query_bias=query_bias,
        key_bias=key_bias,
        value=actual_value,
        value_bias=value_bias,
    )
    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)
    assert torch.equal(actual_value, expected_value)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_video_vae_patch_output_matches_bias_and_rearrangement(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(59)
    batch, frames, height, width = 2, 2, 3, 4
    patch_count = frames * height * width
    source = torch.randn(
        (batch, patch_count + 5, 3072),
        dtype=dtype,
        device="cuda",
    )
    bias = torch.randn((3072,), dtype=dtype, device="cuda")
    expected = (source + bias)[:, :patch_count].view(
        batch,
        frames,
        height,
        width,
        3,
        4,
        16,
        16,
    )
    expected = (
        expected.permute(0, 4, 1, 5, 2, 6, 3, 7)
        .contiguous()
        .reshape(batch, 3, frames * 4, height * 16, width * 16)
    )

    actual = video_patch_output(
        source,
        bias,
        frames=frames,
        height=height,
        width=width,
    )
    assert torch.equal(actual, expected)

    expected_without_bias = source[:, :patch_count].view(
        batch,
        frames,
        height,
        width,
        3,
        4,
        16,
        16,
    )
    expected_without_bias = (
        expected_without_bias.permute(0, 4, 1, 5, 2, 6, 3, 7)
        .contiguous()
        .reshape(batch, 3, frames * 4, height * 16, width * 16)
    )
    actual_without_bias = video_patch_output(
        source,
        None,
        frames=frames,
        height=height,
        width=width,
    )
    assert torch.equal(actual_without_bias, expected_without_bias)
