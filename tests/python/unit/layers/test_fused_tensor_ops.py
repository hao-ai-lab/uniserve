from __future__ import annotations

import pytest
import torch

from uniserve_worker.ops import value_first_swiglu, value_first_swiglu_absmax
from uniserve_worker.ops.patch import unpatchify_video_tokens
from uniserve_worker.ops.residual import (
    scaled_residual_layer_norm,
    scaled_residual_layer_norm_absmax,
    scaled_residual_rms_norm_,
    scaled_residual_rms_norm_absmax_,
    weighted_rms_norm,
    weighted_rms_norm_absmax,
)
from uniserve_worker.ops.rope import qk_rms_norm_partial_rope_

pytestmark = pytest.mark.unit


@pytest.fixture(params=("cpu", "cuda"))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return request.param


@pytest.mark.parametrize("width", (257, 2048))
def test_normalization_absmax_matches_bf16_boundaries(device, width) -> None:
    torch.manual_seed(41)
    rows = 32
    hidden = torch.randn((rows, width), dtype=torch.bfloat16, device=device)
    update = torch.randn_like(hidden)
    scale = torch.randn((width,), dtype=torch.bfloat16, device=device)
    weight = torch.randn((width,), dtype=torch.bfloat16, device=device)
    update_bias = torch.randn((width,), dtype=torch.bfloat16, device=device)
    eps = 1e-5

    expected_normalized = weighted_rms_norm(hidden, weight, eps=eps)
    actual_normalized, actual_maximum = weighted_rms_norm_absmax(
        hidden,
        weight,
        eps=eps,
    )
    assert torch.equal(actual_normalized, expected_normalized)
    assert torch.equal(actual_maximum, expected_normalized.abs().amax())

    expected_hidden, expected_residual_normalized = scaled_residual_rms_norm_(
        hidden.clone(),
        update,
        scale,
        weight,
        update_bias=update_bias,
        eps=eps,
    )
    actual_hidden, actual_residual_normalized, actual_maximum = scaled_residual_rms_norm_absmax_(
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


@pytest.mark.parametrize("width", (257, 2048))
def test_layernorm_absmax_matches_bf16_boundary(device, width) -> None:
    torch.manual_seed(43)
    rows = 32
    hidden = torch.randn((rows, width), dtype=torch.bfloat16, device=device)
    update = torch.randn_like(hidden)
    scale = torch.randn((width,), dtype=torch.bfloat16, device=device)
    weight = torch.randn((width,), dtype=torch.bfloat16, device=device)
    bias = torch.randn((width,), dtype=torch.bfloat16, device=device)
    update_bias = torch.randn((width,), dtype=torch.bfloat16, device=device)
    eps = 1e-5

    expected = scaled_residual_layer_norm(
        hidden,
        update,
        scale,
        weight,
        bias,
        update_bias=update_bias,
        eps=eps,
    )
    actual, maximum = scaled_residual_layer_norm_absmax(
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


def test_swiglu_absmax_matches_bf16_boundary(device) -> None:
    torch.manual_seed(47)
    rows, intermediate = 16, 8192
    value_gate = torch.randn(
        (rows, 2 * intermediate),
        dtype=torch.bfloat16,
        device=device,
    )
    bias = torch.randn((2 * intermediate,), dtype=torch.bfloat16, device=device)

    expected = value_first_swiglu(value_gate, bias)
    actual, maximum = value_first_swiglu_absmax(value_gate, bias)
    assert torch.equal(actual, expected)
    assert torch.equal(maximum, expected.abs().amax())


@pytest.mark.parametrize(("head_dim", "rotary_dim"), ((64, 48), (96, 64)))
def test_qkv_bias_fusion_preserves_outputs(device, head_dim, rotary_dim) -> None:
    torch.manual_seed(53)
    shape = (2, 17, 32, head_dim)
    query = torch.randn(shape, dtype=torch.bfloat16, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    query_bias = torch.randn((32 * head_dim,), dtype=torch.bfloat16, device=device)
    key_bias = torch.randn((32 * head_dim,), dtype=torch.bfloat16, device=device)
    value_bias = torch.randn((32 * head_dim,), dtype=torch.bfloat16, device=device)
    cosine = torch.randn((2, 17, 1, rotary_dim), dtype=torch.bfloat16, device=device)
    sine = torch.randn_like(cosine)

    expected = []
    for projection, bias in ((query, query_bias), (key, key_bias)):
        normalized = torch.nn.functional.rms_norm(
            projection + bias.view(32, head_dim), (head_dim,), eps=1e-5
        )
        left = normalized[..., : rotary_dim // 2].float()
        right = normalized[..., rotary_dim // 2 : rotary_dim].float()
        first = (
            left * cosine[..., : rotary_dim // 2].float()
            - right * sine[..., : rotary_dim // 2].float()
        )
        second = (
            right * cosine[..., rotary_dim // 2 :].float()
            + left * sine[..., rotary_dim // 2 :].float()
        )
        normalized[..., :rotary_dim] = torch.cat((first, second), dim=-1).to(normalized.dtype)
        expected.append(normalized)
    expected_query, expected_key = expected
    expected_value = value + value_bias.view(32, head_dim)
    actual_value = value.clone()
    actual_query, actual_key = qk_rms_norm_partial_rope_(
        query.clone(),
        key.clone(),
        cosine,
        sine,
        query_bias=query_bias,
        key_bias=key_bias,
        value=actual_value,
        value_bias=value_bias,
    )
    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
    assert torch.equal(actual_value, expected_value)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_patch_output_matches_bias_and_rearrangement(
    dtype: torch.dtype,
    device,
) -> None:
    torch.manual_seed(59)
    batch, frames, height, width = 2, 2, 3, 4
    patch_count = frames * height * width
    source = torch.randn(
        (batch, patch_count + 5, 3072),
        dtype=dtype,
        device=device,
    )
    bias = torch.randn((3072,), dtype=dtype, device=device)
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

    actual = unpatchify_video_tokens(
        source,
        bias,
        grid_shape=(frames, height, width),
        patch_shape=(4, 16, 16),
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
    actual_without_bias = unpatchify_video_tokens(
        source,
        None,
        grid_shape=(frames, height, width),
        patch_shape=(4, 16, 16),
    )
    assert torch.equal(actual_without_bias, expected_without_bias)


@pytest.mark.parametrize("width", (257, 2048))
def test_scaled_residual_normalizes_the_unrounded_fp32_sum(device, width):
    generator = torch.Generator(device=device).manual_seed(67)
    hidden = torch.randn((7, width), generator=generator, device=device, dtype=torch.float32)
    update = torch.randn((7, width), generator=generator, device=device, dtype=torch.bfloat16)
    scale = torch.randn((width,), generator=generator, device=device, dtype=torch.bfloat16)
    weight = torch.randn((width,), generator=generator, device=device, dtype=torch.bfloat16)
    update_bias = torch.randn((width,), generator=generator, device=device, dtype=torch.bfloat16)
    residual = hidden + (update + update_bias).float() * scale.float()
    normalized = residual * torch.rsqrt(residual.square().mean(-1, keepdim=True) + 1e-5)
    expected = (normalized * weight.float()).to(update.dtype)
    actual_hidden, actual, magnitude = scaled_residual_rms_norm_absmax_(
        hidden, update, scale, weight, update_bias, eps=1e-5
    )
    torch.testing.assert_close(actual_hidden, residual)
    torch.testing.assert_close(hidden, residual)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(magnitude, actual.abs().amax())


@pytest.mark.parametrize(("patch_shape", "channels"), (((2, 3, 5), 4), ((1, 2, 2), 1)))
def test_video_patch_geometry_preserves_logical_channel_coordinates(device, patch_shape, channels):
    frames, height, width = 2, 3, 4
    time_patch, row_patch, column_patch = patch_shape
    token_width = channels * time_patch * row_patch * column_patch
    source = torch.arange(2 * 25 * token_width, device=device, dtype=torch.float32).reshape(
        2, 25, token_width
    )
    expected = source[:, :24].reshape(2, frames, height, width, channels, *patch_shape)
    expected = expected.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        2, channels, frames * time_patch, height * row_patch, width * column_patch
    )
    actual = unpatchify_video_tokens(
        source, None, grid_shape=(frames, height, width), patch_shape=patch_shape
    )
    assert torch.equal(actual, expected)
