from __future__ import annotations

import pytest
import torch

from uniserve_worker.ops import value_first_swiglu_absmax
from uniserve_worker.ops.patch import unpatchify_video_tokens
from uniserve_worker.ops.residual import (
    scaled_residual_layer_norm_absmax,
    scaled_residual_rms_norm_absmax_,
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
def test_normalization_and_magnitude_preserve_rms_values(device, width) -> None:
    torch.manual_seed(41)
    rows = 32
    hidden = torch.randn((rows, width), dtype=torch.bfloat16, device=device)
    update = torch.randn_like(hidden)
    scale = torch.randn((width,), dtype=torch.bfloat16, device=device)
    weight = torch.randn((width,), dtype=torch.bfloat16, device=device)
    update_bias = torch.randn((width,), dtype=torch.bfloat16, device=device)
    eps = 1e-5

    def normalize(values):
        values = values.double()
        return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight.double()

    expected_normalized = normalize(hidden).to(hidden.dtype)
    actual_normalized, actual_maximum = weighted_rms_norm_absmax(
        hidden,
        weight,
        eps=eps,
    )
    torch.testing.assert_close(actual_normalized, expected_normalized, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_maximum, actual_normalized.abs().amax(), rtol=0, atol=0)

    residual = hidden.double() + (update.double() + update_bias.double()) * scale.double()
    expected_hidden = residual.to(hidden.dtype)
    expected_residual_normalized = normalize(residual).to(hidden.dtype)
    actual_hidden, actual_residual_normalized, actual_maximum = scaled_residual_rms_norm_absmax_(
        hidden.clone(),
        update,
        scale,
        weight,
        update_bias=update_bias,
        eps=eps,
    )
    torch.testing.assert_close(actual_hidden, expected_hidden, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        actual_residual_normalized, expected_residual_normalized, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_maximum, actual_residual_normalized.abs().amax(), rtol=0, atol=0
    )


@pytest.mark.parametrize("width", (257, 2048))
def test_layernorm_and_magnitude_preserve_affine_values(device, width) -> None:
    torch.manual_seed(43)
    rows = 32
    hidden = torch.randn((rows, width), dtype=torch.bfloat16, device=device)
    update = torch.randn_like(hidden)
    scale = torch.randn((width,), dtype=torch.bfloat16, device=device)
    weight = torch.randn((width,), dtype=torch.bfloat16, device=device)
    bias = torch.randn((width,), dtype=torch.bfloat16, device=device)
    update_bias = torch.randn((width,), dtype=torch.bfloat16, device=device)
    eps = 1e-5

    residual = hidden.double() + (update.double() + update_bias.double()) * scale.double()
    expected = torch.nn.functional.layer_norm(
        residual, (width,), weight.double(), bias.double(), eps
    ).to(hidden.dtype)
    actual, maximum = scaled_residual_layer_norm_absmax(
        hidden,
        update,
        scale,
        weight,
        bias,
        update_bias=update_bias,
        eps=eps,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(maximum, actual.abs().amax(), rtol=0, atol=0)


def test_swiglu_and_magnitude_preserve_gated_values(device) -> None:
    torch.manual_seed(47)
    rows, intermediate = 16, 8192
    value_gate = torch.randn(
        (rows, 2 * intermediate),
        dtype=torch.bfloat16,
        device=device,
    )
    bias = torch.randn((2 * intermediate,), dtype=torch.bfloat16, device=device)

    value, gate = (value_gate.double() + bias.double()).chunk(2, dim=-1)
    expected = (value * torch.nn.functional.silu(gate)).to(value_gate.dtype)
    actual, maximum = value_first_swiglu_absmax(value_gate, bias)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(maximum, actual.abs().amax(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(("head_dim", "rotary_dim"), ((64, 48), (96, 64)))
def test_qkv_bias_fusion_preserves_outputs(device, head_dim, rotary_dim, dtype) -> None:
    torch.manual_seed(53)
    shape = (2, 17, 32, head_dim)
    query = torch.randn(shape, dtype=dtype, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    query_bias = torch.randn((32 * head_dim,), dtype=dtype, device=device)
    key_bias = torch.randn((32 * head_dim,), dtype=dtype, device=device)
    value_bias = torch.randn((32 * head_dim,), dtype=dtype, device=device)
    cosine = torch.randn((2, 17, 1, rotary_dim), dtype=dtype, device=device)
    sine = torch.randn_like(cosine)

    expected = []
    for projection, bias in ((query, query_bias), (key, key_bias)):
        normalized = torch.nn.functional.rms_norm(
            projection.double() + bias.view(32, head_dim).double(), (head_dim,), eps=1e-5
        )
        left = normalized[..., : rotary_dim // 2].double()
        right = normalized[..., rotary_dim // 2 : rotary_dim].double()
        first = (
            left * cosine[..., : rotary_dim // 2].double()
            - right * sine[..., : rotary_dim // 2].double()
        )
        second = (
            right * cosine[..., rotary_dim // 2 :].double()
            + left * sine[..., rotary_dim // 2 :].double()
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
    tolerance = 2e-2 if dtype is torch.bfloat16 else 2e-3
    torch.testing.assert_close(
        actual_query.double(), expected_query, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(actual_key.double(), expected_key, rtol=tolerance, atol=tolerance)
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
    residual = hidden + (update.float() + update_bias.float()) * scale.float()
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


@pytest.mark.parametrize("axis_dims", ((6, 2, 2), (96, 48, 48), (128, 64, 64), (512, 256, 256)))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_multi_axis_qk_norm_rope_preserves_shared_normalization_groups(device, dtype, axis_dims):
    from uniserve_worker import ops

    generator = torch.Generator(device=device).manual_seed(73)
    width = sum(axis_dims)
    query = torch.randn((5, 8, width), generator=generator, device=device, dtype=dtype)
    key = torch.randn((5, 2, width), generator=generator, device=device, dtype=dtype)
    weights = tuple(
        tuple(
            torch.randn((size,), generator=generator, device=device, dtype=dtype)
            for size in (axis_dims[0], sum(axis_dims[1:]))
        )
        for _ in range(2)
    )
    angles = tuple(
        torch.randn((5, size // 2), generator=generator, device=device) for size in axis_dims
    )
    cosine, sine = tuple(value.cos() for value in angles), tuple(value.sin() for value in angles)
    with torch.inference_mode():
        actual = ops.qk_norm_rope(
            query,
            key,
            (weights[0][0], weights[0][1], weights[0][1]),
            (weights[1][0], weights[1][1], weights[1][1]),
            cosine,
            sine,
            1e-6,
            axis_dims=axis_dims,
        )
    tolerance = 2e-2 if dtype is torch.bfloat16 else 2e-3
    for source, groups, result in zip((query, key), weights, actual, strict=True):
        head, tail = source.double().split((axis_dims[0], sum(axis_dims[1:])), dim=-1)
        normalized = torch.cat(
            tuple(
                value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * weight.double()
                for value, weight in zip((head, tail), groups, strict=True)
            ),
            dim=-1,
        )
        rotated = []
        for value, cos, sin in zip(normalized.split(axis_dims, dim=-1), cosine, sine, strict=True):
            left, right = value.chunk(2, dim=-1)
            cos, sin = cos.double().unsqueeze(1), sin.double().unsqueeze(1)
            rotated.append(torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1))
        expected = torch.cat(rotated, dim=-1)
        torch.testing.assert_close(result.double(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    "shape", ((1, 3, 1, 128), (17, 64, 8, 128), (3, 5, 2, 96), (2, 3, 1, 1024))
)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_full_width_qk_norm_rope_preserves_strided_heads_and_tail_rows(device, dtype, shape):
    from uniserve_worker import ops

    tokens, query_heads, key_heads, width = shape
    generator = torch.Generator(device=device).manual_seed(83)
    packed = torch.randn(
        (tokens, query_heads + 2 * key_heads, width),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    query, key, _value = packed.split((query_heads, key_heads, key_heads), dim=1)
    original = packed.clone()
    # Query and key weights may have different storage dtypes. Both affine
    # transforms belong to the same FP32 normalization/rotation contract.
    weights = (
        torch.randn(width, generator=generator, device=device, dtype=torch.float32),
        torch.randn(width, generator=generator, device=device, dtype=dtype),
    )
    angles = torch.randn((tokens, width // 2), generator=generator, device=device)
    cosine, sine = angles.cos(), angles.sin()
    with torch.inference_mode():
        actual = ops.qk_norm_rope(query, key, *weights, cosine, sine, 1e-6)

    for source, weight, result in zip((query, key), weights, actual, strict=True):
        values = source.double()
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        left, right = (normalized * weight.double()).chunk(2, dim=-1)
        cos, sin = cosine.double().unsqueeze(1), sine.double().unsqueeze(1)
        expected = torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
        if dtype is torch.float32:
            torch.testing.assert_close(result, expected.float())
        else:
            tolerance = 2e-2 if dtype is torch.bfloat16 else 2e-3
            torch.testing.assert_close(result.double(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
