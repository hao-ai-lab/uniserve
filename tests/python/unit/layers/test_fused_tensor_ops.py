from __future__ import annotations

import pytest
import torch

from uniserve.nn.functional import (
    qk_bias_rms_norm_rope_,
    scaled_residual_layer_norm_absmax,
    scaled_residual_rms_norm_absmax_,
    swiglu,
    swiglu_absmax,
    unpatchify_video_tokens,
    value_first_swiglu_absmax,
    weighted_rms_norm_absmax,
)

pytestmark = pytest.mark.unit


@pytest.fixture(params=("cpu", pytest.param("cuda", marks=pytest.mark.gpu)))
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
        return (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
            * weight.double()
        )

    expected_normalized = normalize(hidden).to(hidden.dtype)
    actual_normalized, actual_maximum = weighted_rms_norm_absmax(
        hidden,
        weight,
        eps=eps,
    )
    torch.testing.assert_close(
        actual_normalized, expected_normalized, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_maximum, actual_normalized.float().abs().amax(), rtol=0, atol=0
    )
    assert actual_maximum.dtype is torch.float32

    residual = (
        hidden.double()
        + (update.double() + update_bias.double()) * scale.double()
    )
    expected_hidden = residual.to(hidden.dtype)
    expected_residual_normalized = normalize(residual).to(hidden.dtype)
    actual_hidden, actual_residual_normalized, actual_maximum = (
        scaled_residual_rms_norm_absmax_(
            hidden.clone(),
            update,
            scale,
            weight,
            update_bias=update_bias,
            eps=eps,
        )
    )
    torch.testing.assert_close(
        actual_hidden, expected_hidden, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        actual_residual_normalized,
        expected_residual_normalized,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        actual_maximum,
        actual_residual_normalized.float().abs().amax(),
        rtol=0,
        atol=0,
    )
    assert actual_maximum.dtype is torch.float32


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

    residual = (
        hidden.double()
        + (update.double() + update_bias.double()) * scale.double()
    )
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
    torch.testing.assert_close(
        maximum, actual.float().abs().amax(), rtol=0, atol=0
    )
    assert maximum.dtype is torch.float32


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
    torch.testing.assert_close(
        maximum, actual.float().abs().amax(), rtol=0, atol=0
    )
    assert maximum.dtype is torch.float32


@pytest.mark.parametrize("layout", ("separate", "reversed_views"))
def test_separate_swiglu_avoids_packed_layout_requirements(
    device, layout
) -> None:
    torch.manual_seed(49)
    rows, width = 17, 2048
    value = torch.randn((rows, width), dtype=torch.bfloat16, device=device)
    gate = torch.randn_like(value)
    if layout == "reversed_views":
        backing = torch.cat((gate, value), dim=-1)
        gate, value = backing.chunk(2, dim=-1)
    value_bias = torch.randn((width,), dtype=value.dtype, device=device)
    gate_bias = torch.randn_like(value_bias)

    expected = (
        (value.double() + value_bias.double())
        * torch.nn.functional.silu(gate.double() + gate_bias.double())
    ).to(value.dtype)
    options = {"value_bias": value_bias, "gate_bias": gate_bias}
    torch.testing.assert_close(
        swiglu(value, gate, **options), expected, rtol=2e-2, atol=2e-2
    )
    actual, maximum = swiglu_absmax(value, gate, **options)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        maximum, actual.float().abs().amax(), rtol=0, atol=0
    )
    assert maximum.dtype is torch.float32


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize(("head_dim", "rotary_dim"), ((64, 48), (96, 64)))
@pytest.mark.parametrize("layout", ("contiguous", "merged", "transposed"))
def test_qkv_bias_fusion_preserves_outputs(
    device, head_dim, rotary_dim, dtype, layout
) -> None:
    torch.manual_seed(53)
    shape = (2, 17, 32, head_dim)
    query = torch.randn(shape, dtype=dtype, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    if layout != "contiguous":
        packed = torch.cat((query, key, value), dim=-2)
        if layout == "transposed":
            packed = packed.transpose(0, 1).contiguous().transpose(0, 1)
        query, key, value = packed.chunk(3, dim=-2)
    query_bias = torch.randn((32 * head_dim,), dtype=dtype, device=device)
    key_bias = torch.randn((32 * head_dim,), dtype=dtype, device=device)
    value_bias = torch.randn((32 * head_dim,), dtype=dtype, device=device)
    # Compact factors: one phase per rotated pair of each token.
    cosine = torch.randn((2, 17, rotary_dim // 2), dtype=dtype, device=device)
    sine = torch.randn_like(cosine)

    expected = []
    for projection, bias in ((query, query_bias), (key, key_bias)):
        normalized = torch.nn.functional.rms_norm(
            projection.double() + bias.view(32, head_dim).double(),
            (head_dim,),
            eps=1e-5,
        )
        left = normalized[..., : rotary_dim // 2].double()
        right = normalized[..., rotary_dim // 2 : rotary_dim].double()
        cos, sin = cosine.double().unsqueeze(-2), sine.double().unsqueeze(-2)
        first = left * cos - right * sin
        second = right * cos + left * sin
        normalized[..., :rotary_dim] = torch.cat((first, second), dim=-1).to(
            normalized.dtype
        )
        expected.append(normalized)
    expected_query, expected_key = expected
    expected_value = value + value_bias.view(32, head_dim)
    actual_query, actual_key = qk_bias_rms_norm_rope_(
        query,
        key,
        cosine,
        sine,
        query_bias=query_bias,
        key_bias=key_bias,
        value=value,
        value_bias=value_bias,
    )
    tolerance = 2e-2 if dtype is torch.bfloat16 else 2e-3
    torch.testing.assert_close(
        actual_query.double(), expected_query, rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        actual_key.double(), expected_key, rtol=tolerance, atol=tolerance
    )
    assert torch.equal(value, expected_value)


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
    hidden = torch.randn(
        (7, width), generator=generator, device=device, dtype=torch.float32
    )
    update = torch.randn(
        (7, width), generator=generator, device=device, dtype=torch.bfloat16
    )
    scale = torch.randn(
        (width,), generator=generator, device=device, dtype=torch.bfloat16
    )
    weight = torch.randn(
        (width,), generator=generator, device=device, dtype=torch.bfloat16
    )
    update_bias = torch.randn(
        (width,), generator=generator, device=device, dtype=torch.bfloat16
    )
    residual = hidden + (update.float() + update_bias.float()) * scale.float()
    normalized = residual * torch.rsqrt(
        residual.square().mean(-1, keepdim=True) + 1e-5
    )
    expected = (normalized * weight.float()).to(update.dtype)
    actual_hidden, actual, magnitude = scaled_residual_rms_norm_absmax_(
        hidden, update, scale, weight, update_bias, eps=1e-5
    )
    torch.testing.assert_close(actual_hidden, residual)
    torch.testing.assert_close(hidden, residual)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(magnitude, actual.float().abs().amax())
    assert magnitude.dtype is torch.float32


@pytest.mark.parametrize(
    ("patch_shape", "channels"), (((2, 3, 5), 4), ((1, 2, 2), 1))
)
def test_video_patch_geometry_preserves_logical_channel_coordinates(
    device, patch_shape, channels
):
    frames, height, width = 2, 3, 4
    time_patch, row_patch, column_patch = patch_shape
    token_width = channels * time_patch * row_patch * column_patch
    source = torch.arange(
        2 * 25 * token_width, device=device, dtype=torch.float32
    ).reshape(2, 25, token_width)
    expected = source[:, :24].reshape(
        2, frames, height, width, channels, *patch_shape
    )
    expected = expected.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        2,
        channels,
        frames * time_patch,
        height * row_patch,
        width * column_patch,
    )
    actual = unpatchify_video_tokens(
        source,
        None,
        grid_shape=(frames, height, width),
        patch_shape=patch_shape,
    )
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_residual_norms_of_strided_rows_raise():
    rows = torch.randn((8, 2 * 256), device="cuda")[:, :256]
    weight = torch.ones(256, device="cuda")
    with pytest.raises(ValueError, match="weighted_rms_norm.*contiguous"):
        weighted_rms_norm_absmax(rows, weight, eps=1e-6)
    with pytest.raises(
        ValueError, match="scaled_residual_rms_norm_absmax_.*contiguous"
    ):
        scaled_residual_rms_norm_absmax_(
            rows, torch.zeros_like(rows), weight, weight, eps=1e-6
        )
