"""Public rotary and QKV calls preserve frequency and normalization domains."""

import pytest
import torch
from torch.nn import functional as F

from uniserve.nn import RMSNorm, RotaryEmbedding
from uniserve.nn.attention import (
    AxialQKVProjection,
    QKVProjection,
    RotaryQKVProjection,
)
from uniserve.nn.functional import Rounding, apply_rotary, qk_norm_rope
from uniserve.nn.linear import QKVParallelLinear
from uniserve.nn.rope import DynamicScaling, LinearScaling, LongRoPEScaling

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_partial_qk_rotation_normalizes_the_complete_head_in_place(device):
    generator = torch.Generator().manual_seed(182)
    projected = torch.randn(67, 3, 4, 128, generator=generator).to(
        device, torch.bfloat16
    )
    query, key, _, _ = projected.unbind(2)
    weights = tuple(
        torch.randn(128, generator=generator).to(device) for _ in range(2)
    )
    phase = torch.randn(67, 48, generator=generator).to(device)
    cosine, sine = phase.cos(), phase.sin()
    expected = []
    for value, weight in zip((query, key), weights, strict=True):
        normalized = value.double()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + 1e-5
        )
        normalized = normalized * weight.double()
        expected.append(
            torch.cat(
                (
                    _rotate(
                        normalized[..., :96],
                        cosine.double(),
                        sine.double(),
                        "split",
                    ),
                    normalized[..., 96:],
                ),
                dim=-1,
            ).to(value.dtype)
        )
    with torch.inference_mode():
        actual = qk_norm_rope(
            query,
            key,
            weights[:1],
            weights[1:],
            (cosine,),
            (sine,),
            eps=1e-5,
            axis_dims=(128,),
            out=(query, key),
        )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=2e-2, atol=2e-2)


def _rotate(value, cos, sin, rotation):
    if rotation == "split":
        left, right = value.chunk(2, dim=-1)
        turned = torch.cat((-right, left), dim=-1)
        cosine, sine = (
            torch.cat((cos, cos), dim=-1),
            torch.cat((sin, sin), dim=-1),
        )
    else:
        turned = torch.stack(
            (-value[..., 1::2], value[..., ::2]), dim=-1
        ).flatten(-2)
        cosine, sine = (
            cos.repeat_interleave(2, dim=-1),
            sin.repeat_interleave(2, dim=-1),
        )
    return value * cosine.unsqueeze(-2) + turned * sine.unsqueeze(-2)


@pytest.mark.parametrize("rotation", ["split", "interleaved"])
def test_partial_rotary_preserves_trailing_features_and_output(rotation):
    value = torch.arange(3 * 2 * 10).reshape(3, 2, 10).float() / 7
    positions = torch.tensor([0, 2, 5])
    rotary = RotaryEmbedding(8, theta=100)
    cosine, sine = rotary(positions, dtype=torch.float32, sequence_length=6)
    frequencies = 100 ** (-torch.arange(0, 8, 2).float() / 8)
    torch.testing.assert_close(
        cosine, torch.cos(positions[:, None] * frequencies), rtol=0, atol=0
    )
    expected = torch.cat(
        (_rotate(value[..., :8], cosine, sine, rotation), value[..., 8:]),
        dim=-1,
    )
    out = torch.empty(3, 2, 20)[..., ::2]
    assert apply_rotary(value, cosine, sine, rotation=rotation, out=out) is out
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("rotation", ["split", "interleaved"])
def test_rotation_rounds_once_from_fp32_products(device, rotation):
    # BF16 inputs whose products cancel expose any low-precision intermediate.
    value = torch.tensor(
        [[[0.77, 1.11, -0.31, 0.48]], [[-1.5, 0.25, 2.0, -0.75]]],
        dtype=torch.bfloat16,
        device=device,
    )
    cosine = torch.tensor(
        [[0.91, 0.83], [0.12, -0.99]], dtype=torch.bfloat16, device=device
    )
    sine = torch.tensor(
        [[0.37, -0.42], [0.99, 0.14]], dtype=torch.bfloat16, device=device
    )
    expected = _rotate(
        value.float(), cosine.float(), sine.float(), rotation
    ).to(torch.bfloat16)

    with torch.inference_mode():
        actual = apply_rotary(value, cosine, sine, rotation=rotation)

    if device == "cpu":
        assert torch.equal(actual, expected)
    else:
        # Fused multiply-adds may differ in the last FP32 bit before rounding.
        torch.testing.assert_close(actual, expected, rtol=2**-8, atol=2**-16)


@pytest.mark.parametrize("rotation", ["split", "interleaved"])
def test_rotary_factors_broadcast_over_every_head(rotation):
    value = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    identity = apply_rotary(
        value, torch.ones(2, 2), torch.zeros(2, 2), rotation=rotation
    )
    assert torch.equal(identity, value)

    phase = torch.tensor([[0.5, -1.25], [2.0, 0.75]])
    torch.testing.assert_close(
        apply_rotary(value, phase.cos(), phase.sin(), rotation=rotation),
        _rotate(value, phase.cos(), phase.sin(), rotation),
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_dynamic_rotary_calls_do_not_change_another_sequence_frequency_domain(
    device,
):
    positions = torch.tensor([0, 2, 4], device=device)
    rotary = RotaryEmbedding(
        8,
        theta=100,
        scaling=DynamicScaling(2),
        max_position_embeddings=8,
        device=device,
    )
    for length in (8, 32, 16, 8):
        base = 100 * (2 * max(length, 8) / 8 - 1) ** (8 / 6)
        phase = positions[:, None] * base ** (
            -torch.arange(0, 8, 2, device=device).float() / 8
        )
        actual = rotary(positions, dtype=torch.float32, sequence_length=length)
        torch.testing.assert_close(actual[0], phase.cos())
        torch.testing.assert_close(actual[1], phase.sin())


@pytest.mark.parametrize(
    ("position_dtype", "output_dtype", "shape"),
    [
        (torch.int64, torch.float32, (2, 3)),
        (torch.int32, torch.bfloat16, (6,)),
        (torch.float64, torch.float16, (2, 3)),
        (torch.float32, torch.float64, ()),
        (torch.float16, torch.float32, (6,)),
        (torch.bfloat16, torch.float32, (0, 3)),
    ],
)
@pytest.mark.gpu
def test_rotary_factors_preserve_position_views_and_fp32_phase(
    position_dtype, output_dtype, shape
):
    # Position coordinates need not be contiguous, positive or integral. Large
    # phases also exercise the full trigonometric range reduction domain.
    values = torch.tensor([-1000003, -3.25, 0, 2.5, 7, 100003], device="cuda")
    if position_dtype == torch.float16:
        values = values.clamp(-60000, 60000)
    storage = torch.empty(12, dtype=position_dtype, device="cuda")
    storage[::2] = values
    positions = storage[::2]
    if not shape:
        positions = positions[0]
    elif shape[0] == 0:
        positions = positions[:0].reshape(shape)
    else:
        positions = positions.reshape(shape)
        if len(shape) == 2:
            positions = positions.transpose(0, 1)
    rotary = RotaryEmbedding(
        12, theta=100, attention_scale=1.25, keep_freq_range=True
    ).cuda()
    # Frequencies are constructed on CPU and transferred with the module;
    # recomputing pow on CUDA can change their last bit before phase formation.
    frequency = (1 / (100 ** (torch.arange(0, 24, 2).float() / 24)))[::2].cuda()
    phase = positions.float().unsqueeze(-1) * frequency
    expected = (
        (phase.cos() * 1.25).to(output_dtype),
        (phase.sin() * 1.25).to(output_dtype),
    )
    actual = rotary(positions, dtype=output_dtype, sequence_length=1000004)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.gpu
def test_rotary_factors_read_changed_positions_on_graph_replay():
    rotary = RotaryEmbedding(128, theta=1000000).cuda()
    positions = torch.arange(40, device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rotary(positions, dtype=torch.bfloat16, sequence_length=4096)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = rotary(positions, dtype=torch.bfloat16, sequence_length=4096)
    frequencies = (
        1 / (1000000 ** (torch.arange(0, 128, 2).float() / 128))
    ).cuda()
    for offset in (31, 1024):
        positions.add_(offset)
        graph.replay()
        phase = positions.float()[:, None] * frequencies
        for value, reference in zip(
            actual, (phase.cos(), phase.sin()), strict=True
        ):
            torch.testing.assert_close(
                value, reference.to(torch.bfloat16), rtol=0, atol=0
            )


def test_longrope_switches_by_explicit_length_and_constants_have_the_same_values(  # noqa: E501
):
    scaling = LongRoPEScaling(2, 8, 1.25, (1, 2, 3, 4), (2, 4, 6, 8))
    rotary = RotaryEmbedding(
        8, theta=100, scaling=scaling, max_position_embeddings=16
    )
    positions = torch.tensor([0, 2, 4])
    base = 100 ** (-torch.arange(0, 8, 2).float() / 8)
    for length, factors in (
        (8, scaling.short_factor),
        (16, scaling.long_factor),
        (8, scaling.short_factor),
    ):
        phase = positions[:, None] * base / torch.tensor(factors)
        cosine, sine = rotary(
            positions, dtype=torch.float32, sequence_length=length
        )
        torch.testing.assert_close(cosine, phase.cos() * 1.25)
        torch.testing.assert_close(sine, phase.sin() * 1.25)
    storage = {
        name: torch.empty(item.shape, dtype=item.dtype)
        for name, item in rotary.constant_buffers(16).items()
    }
    rotary.prepare_constants(16, out=storage)
    expected = rotary(torch.arange(16), dtype=torch.float32, sequence_length=16)
    torch.testing.assert_close(storage["cos"], expected[0])
    torch.testing.assert_close(storage["sin"], expected[1])


@pytest.mark.parametrize("kind", ["unrotated", "rotary", "axial"])
def test_qkv_composition_preserves_heads_and_full_normalization(kind):
    projection = QKVParallelLinear(8, 4, 2, 8, dtype=torch.float32)
    hidden = torch.arange(24).reshape(3, 8).float() / 17
    query_norm, key_norm = RMSNorm(8, 1e-5), RMSNorm(8, 1e-5)
    positions = torch.tensor([1, 3, 6])
    if kind == "unrotated":
        network, dims, rotations, cos, sin = (
            QKVProjection(projection),
            (),
            (),
            (),
            (),
        )
    else:
        dims = (8,) if kind == "rotary" else (4, 4)
        rotations = ("split",) if kind == "rotary" else ("interleaved", "split")
        network = (
            RotaryQKVProjection(projection, query_norm, key_norm)
            if kind == "rotary"
            else AxialQKVProjection(
                projection,
                query_norm,
                key_norm,
                axis_dims=dims,
                rotations=rotations,
            )
        )
        pairs = tuple(
            RotaryEmbedding(width, scaling=LinearScaling(2))(
                positions, dtype=torch.float32, sequence_length=7
            )
            for width in dims
        )
        cos, sin = (
            tuple(pair[0] for pair in pairs),
            tuple(pair[1] for pair in pairs),
        )
    expected = []
    for name in ("q", "k", "v"):
        layer = projection.projections[name]
        value = F.linear(hidden, layer.weight, layer.bias).reshape(3, -1, 8)
        if name != "v" and kind != "unrotated":
            value = value * torch.rsqrt(
                value.square().mean(-1, keepdim=True) + 1e-5
            )
            value = torch.cat(
                tuple(
                    _rotate(part, cosine, sine, rotation)
                    for part, cosine, sine, rotation in zip(
                        value.split(dims, dim=-1),
                        cos,
                        sin,
                        rotations,
                        strict=True,
                    )
                ),
                dim=-1,
            )
        expected.append(value)
    for actual, reference in zip(
        network(hidden, cos, sin), expected, strict=True
    ):
        torch.testing.assert_close(actual, reference)


def test_axial_projection_preserves_independent_normalization_domains():
    projection = QKVParallelLinear(8, 2, 1, 8, dtype=torch.float32)
    hidden = torch.arange(16).reshape(2, 8).float() / 17
    query_norm = torch.nn.ModuleList((RMSNorm(4, 1e-5), RMSNorm(4, 1e-5)))
    key_norm = torch.nn.ModuleList((RMSNorm(4, 1e-5), RMSNorm(4, 1e-5)))
    network = AxialQKVProjection(
        projection,
        query_norm,
        key_norm,
        axis_dims=(4, 2, 2),
        rotations=("split",) * 3,
    )
    pairs = tuple(
        RotaryEmbedding(width)(
            torch.tensor([1, 2]), dtype=torch.float32, sequence_length=3
        )
        for width in (4, 2, 2)
    )
    cosine, sine = (
        tuple(pair[0] for pair in pairs),
        tuple(pair[1] for pair in pairs),
    )
    expected = []
    for name in ("q", "k", "v"):
        layer = projection.projections[name]
        value = F.linear(hidden, layer.weight, layer.bias).reshape(2, -1, 8)
        if name != "v":
            # The two spatial axes share one four-channel RMS denominator;
            # neither the temporal half nor the individual spatial axis does.
            domains = value.reshape(2, -1, 2, 4)
            normalized = (
                domains
                * torch.rsqrt(domains.square().mean(-1, keepdim=True) + 1e-5)
            ).reshape_as(value)
            value = torch.cat(
                tuple(
                    _rotate(part, cos, sin, "split")
                    for part, cos, sin in zip(
                        normalized.split((4, 2, 2), dim=-1),
                        cosine,
                        sine,
                        strict=True,
                    )
                ),
                dim=-1,
            )
        expected.append(value)
    for actual, target in zip(
        network(hidden, cosine, sine), expected, strict=True
    ):
        torch.testing.assert_close(actual, target)


def _normalize(value, weight, eps):
    value = value.double()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        * weight.double()
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize(
    "axis_dims", ((6, 2, 2), (96, 48, 48), (128, 64, 64), (512, 256, 256))
)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_shared_domain_normalizes_its_axes_with_one_denominator(
    device, axis_dims, dtype
):
    generator = torch.Generator(device=device).manual_seed(73)
    width = sum(axis_dims)
    query = torch.randn(
        (5, 8, width), generator=generator, device=device, dtype=dtype
    )
    key = torch.randn(
        (5, 2, width), generator=generator, device=device, dtype=dtype
    )
    # One weight per domain: the leading axis alone, then both tail axes.
    weights = tuple(
        tuple(
            torch.randn(
                (size,), generator=generator, device=device, dtype=dtype
            )
            for size in (axis_dims[0], sum(axis_dims[1:]))
        )
        for _ in range(2)
    )
    angles = tuple(
        torch.randn((5, size // 2), generator=generator, device=device)
        for size in axis_dims
    )
    cosine = tuple(value.cos() for value in angles)
    sine = tuple(value.sin() for value in angles)

    with torch.inference_mode():
        actual = qk_norm_rope(
            query,
            key,
            *weights,
            cosine,
            sine,
            eps=1e-6,
            axis_dims=axis_dims,
        )

    tolerance = 2e-2 if dtype is torch.bfloat16 else 2e-3
    for source, domains, result in zip(
        (query, key), weights, actual, strict=True
    ):
        head, tail = source.split((axis_dims[0], sum(axis_dims[1:])), dim=-1)
        normalized = torch.cat(
            (
                _normalize(head, domains[0], 1e-6),
                _normalize(tail, domains[1], 1e-6),
            ),
            dim=-1,
        )
        expected = torch.cat(
            tuple(
                _rotate(part, cos.double(), sin.double(), "split")
                for part, cos, sin in zip(
                    normalized.split(axis_dims, dim=-1),
                    cosine,
                    sine,
                    strict=True,
                )
            ),
            dim=-1,
        )
        torch.testing.assert_close(
            result.double(), expected, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_unrotated_domain_is_normalized_without_rotation(device, tokens, dtype):
    generator = torch.Generator(device=device).manual_seed(7)
    query = torch.randn(
        tokens, 8, 128, generator=generator, device=device, dtype=dtype
    )
    key = torch.randn(
        tokens, 2, 128, generator=generator, device=device, dtype=dtype
    )
    weights = tuple(
        tuple(
            torch.rand(width, generator=generator, device=device).to(dtype)
            + 0.5
            for width in (64, 64)
        )
        for _ in range(2)
    )
    angles = torch.rand(tokens, 32, generator=generator, device=device) * 6.0
    # A zero-width factor table leaves its axis unrotated.
    cosine = (angles.cos(), angles.new_empty(tokens, 0))
    sine = (angles.sin(), angles.new_empty(tokens, 0))

    with torch.inference_mode():
        actual = qk_norm_rope(
            query,
            key,
            *weights,
            cosine,
            sine,
            eps=1e-6,
            axis_dims=(64, 64),
        )

    tolerance = 2e-2 if dtype is torch.bfloat16 else 2e-3
    for source, domains, result in zip(
        (query, key), weights, actual, strict=True
    ):
        head, tail = source.split((64, 64), dim=-1)
        expected = torch.cat(
            (
                _rotate(
                    _normalize(head, domains[0], 1e-6),
                    cosine[0].double(),
                    sine[0].double(),
                    "split",
                ),
                _normalize(tail, domains[1], 1e-6),
            ),
            dim=-1,
        )
        torch.testing.assert_close(
            result.double(), expected, rtol=tolerance, atol=tolerance
        )


def test_normalization_domains_must_end_on_rotary_axis_boundaries():
    value = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    factors = (torch.ones(1, 1), torch.ones(1, 1))
    zeros = (torch.zeros(1, 1), torch.zeros(1, 1))
    weight = torch.ones(4)

    # One four-wide domain shares a denominator across both axes; two
    # two-wide domains normalize each axis separately.
    joint, _ = qk_norm_rope(
        value,
        value,
        (weight,),
        (weight,),
        factors,
        zeros,
        eps=1e-6,
        axis_dims=(2, 2),
    )
    separate, _ = qk_norm_rope(
        value,
        value,
        (weight[:2], weight[2:]),
        (weight[:2].clone(), weight[2:].clone()),
        factors,
        zeros,
        eps=1e-6,
        axis_dims=(2, 2),
    )
    torch.testing.assert_close(joint, _normalize(value, weight, 1e-6).float())
    torch.testing.assert_close(
        separate,
        torch.cat(
            tuple(
                _normalize(part, weight[:2], 1e-6)
                for part in value.split(2, dim=-1)
            ),
            dim=-1,
        ).float(),
    )

    for domains in ((weight, weight), (torch.ones(3), torch.ones(1))):
        with pytest.raises(ValueError, match="normalization domains"):
            qk_norm_rope(
                value,
                value,
                domains,
                domains,
                factors,
                zeros,
                eps=1e-6,
                axis_dims=(2, 2),
            )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize(
    "shape", ((1, 3, 1, 128), (17, 64, 8, 128), (3, 5, 2, 96), (2, 3, 1, 1024))
)
@pytest.mark.parametrize(
    "dtype", (torch.bfloat16, torch.float16, torch.float32)
)
@pytest.mark.parametrize("in_place", (False, True))
def test_full_head_rotation_preserves_strided_heads_and_value_rows(
    device, shape, dtype, in_place
):
    tokens, query_heads, key_heads, width = shape
    generator = torch.Generator(device=device).manual_seed(83)
    packed = torch.randn(
        (tokens, query_heads + 2 * key_heads, width),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    query, key, value = packed.split((query_heads, key_heads, key_heads), dim=1)
    original = packed.clone()
    # Query and key weights may have different storage dtypes. Both affine
    # transforms belong to the same FP32 normalization/rotation contract.
    weights = (
        torch.randn(
            width, generator=generator, device=device, dtype=torch.float32
        ),
        torch.randn(width, generator=generator, device=device, dtype=dtype),
    )
    angles = torch.randn(
        (tokens, width // 2), generator=generator, device=device
    )
    cosine, sine = angles.cos(), angles.sin()
    sources = (query.clone(), key.clone())
    with torch.inference_mode():
        actual = qk_norm_rope(
            query,
            key,
            weights[:1],
            weights[1:],
            (cosine,),
            (sine,),
            eps=1e-6,
            axis_dims=(width,),
            out=(query, key) if in_place else None,
        )

    for source, weight, result in zip(sources, weights, actual, strict=True):
        expected = _rotate(
            _normalize(source, weight, 1e-6),
            cosine.double(),
            sine.double(),
            "split",
        )
        if dtype is torch.float32:
            torch.testing.assert_close(result, expected.float())
        else:
            tolerance = 2e-2 if dtype is torch.bfloat16 else 2e-3
            torch.testing.assert_close(
                result.double(), expected, rtol=tolerance, atol=tolerance
            )
    if in_place:
        assert actual[0] is query and actual[1] is key
        assert torch.equal(value, original[:, query_heads + key_heads :])
    else:
        assert torch.equal(packed, original)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_interleaved_sections_match_qwen3_vl_multimodal_rotary(device):
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
    )

    # Qwen3-VL-32B's language rotary: 64 compact frequencies interleaved
    # over (temporal, height, width) coordinates.
    config = Qwen3VLTextConfig(
        head_dim=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
    )
    reference = Qwen3VLTextRotaryEmbedding(config).to(device)
    rotary = RotaryEmbedding(128, theta=5_000_000.0, sections=(24, 20, 20)).to(
        device
    )
    generator = torch.Generator().manual_seed(911)
    positions = torch.randint(0, 40_000, (3, 97), generator=generator).to(
        device
    )

    # The reference repeats the compact factors over both head halves.
    expected = reference(
        torch.empty((), dtype=torch.float32, device=device),
        positions[:, None],
    )
    actual = rotary(positions, dtype=torch.float32, sequence_length=97)
    for value, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, wanted[0, :, :64])


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
def test_sections_with_shared_coordinates_reproduce_one_axis_factors(device):
    # Text advances every axis together; its multimodal factors must equal
    # the one-dimensional recipe exactly, not merely closely.
    plain = RotaryEmbedding(128, theta=5_000_000.0).to(device)
    sectioned = RotaryEmbedding(
        128, theta=5_000_000.0, sections=(24, 20, 20)
    ).to(device)
    positions = torch.arange(1000, 1313, device=device)
    expected = plain(positions, dtype=torch.float32, sequence_length=313)
    actual = sectioned(
        positions.expand(3, -1), dtype=torch.float32, sequence_length=313
    )
    for value, wanted in zip(actual, expected, strict=True):
        assert torch.equal(value, wanted)


@pytest.mark.parametrize("sections", [(24, 20), (24, 20, 21), (8, 28, 28)])
def test_sections_must_partition_the_interleaved_frequencies(sections):
    with pytest.raises(ValueError, match="M-RoPE sections"):
        RotaryEmbedding(128, sections=sections)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("axes", ["one partial axis", "rotated and plain axes"])
@pytest.mark.parametrize("packed", [False, True])
def test_stepwise_qk_rotation_rounds_each_eager_operation(device, axes, packed):
    """Stepwise Q/K preparation equals eager BF16 PyTorch bit for bit.

    Heads of +-1 have an exact unit mean square, so each weighted
    normalization rounds to its weight in BF16 for any reduction order, and
    the rotation's BF16 products and sums are exact functions of the
    operands. 96 of the 128 channels rotate split-half, declared either as
    one axis whose factors cover a prefix or as a rotated axis followed by
    an unrotated one. ``packed`` Q/K are strided row views of one fused
    projection output, as attention layers produce them.
    """
    generator = torch.Generator().manual_seed(183)
    rows, heads, dim, rotated, eps = 37, 4, 128, 96, 1e-6

    signs = torch.randint(0, 2, (rows, 3, heads, dim), generator=generator)
    projection = (signs * 2 - 1).to(device=device, dtype=torch.bfloat16)
    q, k, _ = projection.unbind(1)
    if not packed:
        q, k = q.contiguous(), k.contiguous()
    q_weight, k_weight = (
        (torch.rand(dim, generator=generator) + 0.5).to(
            device=device, dtype=torch.bfloat16
        )
        for _ in range(2)
    )
    angles = torch.rand(rows, rotated // 2, generator=generator) * 40
    cos, sin = angles.cos().to(device), angles.sin().to(device)

    def eager(value, weight):
        normalized = F.rms_norm(value, (dim,), weight, eps)
        head, tail = normalized[..., :rotated], normalized[..., rotated:]
        first, second = head.chunk(2, dim=-1)
        cosine = torch.cat((cos, cos), -1).to(torch.bfloat16)[:, None]
        sine = torch.cat((sin, sin), -1).to(torch.bfloat16)[:, None]
        turned = torch.cat((-second, first), dim=-1)
        return torch.cat((head * cosine + turned * sine, tail), dim=-1)

    if axes == "one partial axis":
        factors, axis_dims = ((cos,), (sin,)), (dim,)
    else:
        factors = ((cos, cos[..., :0]), (sin, sin[..., :0]))
        axis_dims = (rotated, dim - rotated)
    query, key = qk_norm_rope(
        q,
        k,
        (q_weight,),
        (k_weight,),
        *factors,
        eps=eps,
        axis_dims=axis_dims,
        rounding=Rounding.STEPWISE,
    )

    torch.testing.assert_close(query, eager(q, q_weight), rtol=0, atol=0)
    torch.testing.assert_close(key, eager(k, k_weight), rtol=0, atol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize(
    ("heads", "axis_dims", "rotated"),
    (
        # One head-wide domain over two spatial axes: a 2-D vision RoPE.
        ((16, 16), (36, 36), (36, 36)),
        # A partially rotated head with different Q and K head counts.
        ((16, 2), (512,), (128,)),
        # A rotated axis beside a partially rotated axis in one domain.
        ((4, 2), (64, 64), (64, 32)),
    ),
)
def test_single_domain_rotates_every_axis_with_its_own_factors(
    device, heads, axis_dims, rotated
):
    generator = torch.Generator(device=device).manual_seed(97)
    width = sum(axis_dims)
    query, key = (
        torch.randn(
            (2, 7, count, width), generator=generator, device=device
        ).to(torch.bfloat16)
        for count in heads
    )
    weights = tuple(
        torch.rand(width, generator=generator, device=device) + 0.5
        for _ in range(2)
    )
    angles = tuple(
        torch.randn((2, 7, turned // 2), generator=generator, device=device)
        for turned in rotated
    )
    cosines = tuple(angle.cos() for angle in angles)
    sines = tuple(angle.sin() for angle in angles)

    with torch.inference_mode():
        actual = qk_norm_rope(
            query,
            key,
            weights[:1],
            weights[1:],
            cosines,
            sines,
            eps=1e-6,
            axis_dims=axis_dims,
        )

    for source, weight, result in zip(
        (query, key), weights, actual, strict=True
    ):
        normalized = _normalize(source, weight, 1e-6)
        parts = []
        for part, cosine, sine, turned in zip(
            normalized.split(axis_dims, dim=-1),
            cosines,
            sines,
            rotated,
            strict=True,
        ):
            parts.append(
                torch.cat(
                    (
                        _rotate(
                            part[..., :turned],
                            cosine.double(),
                            sine.double(),
                            "split",
                        ),
                        part[..., turned:],
                    ),
                    dim=-1,
                )
            )
        torch.testing.assert_close(
            result.double(), torch.cat(parts, dim=-1), rtol=2e-2, atol=2e-2
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rotation", ["split", "interleaved"])
@pytest.mark.parametrize("rotated", [512, 96])
def test_cuda_rotation_of_strided_rows_matches_the_portable_formula(
    rotation, rotated
):
    generator = torch.Generator(device="cuda").manual_seed(29)
    # One head per token, borrowed as a half of FP32 [tokens, 1024] features.
    features = torch.randn((37, 1024), generator=generator, device="cuda")
    value = features.chunk(2, dim=-1)[0].unsqueeze(1)
    angles = torch.randn((37, rotated // 2), generator=generator, device="cuda")
    cosine, sine = angles.cos(), angles.sin()
    expected = apply_rotary(
        value.cpu(), cosine.cpu(), sine.cpu(), rotation=rotation
    )

    with torch.inference_mode():
        actual = apply_rotary(value, cosine, sine, rotation=rotation)
        in_place = apply_rotary(
            value, cosine, sine, rotation=rotation, out=value
        )

    torch.testing.assert_close(actual.cpu(), expected)
    assert in_place is value
    torch.testing.assert_close(features[:, :512].cpu(), expected.squeeze(1))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("transposed_tokens", "token axes"),
        ("head_too_wide", "head width"),
        ("overlapping_output", "overlaps"),
        ("records_autograd", "autograd"),
    ),
)
def test_cuda_rotary_calls_without_a_kernel_raise(case, reason):
    width = 2048 if case == "head_too_wide" else 64
    value = torch.randn((3, 5, 2, width), device="cuda")
    angles = torch.randn((3, 5, width // 2), device="cuda")
    cosine, sine = angles.cos(), angles.sin()
    weight = torch.ones(width, device="cuda")
    out = None
    if case == "transposed_tokens":
        value = value.transpose(0, 1).contiguous().transpose(0, 1)
    elif case == "overlapping_output":
        storage = torch.empty((3 * 5 * 2 * width + 1,), device="cuda")
        out = storage[1:].view(value.shape)
        value = storage[:-1].view(value.shape).copy_(value)
    elif case == "records_autograd":
        value.requires_grad_()

    with pytest.raises(ValueError, match=f"apply_rotary.*{reason}"):
        apply_rotary(value, cosine, sine, rotation="split", out=out)
    with pytest.raises(ValueError, match=f"qk_norm_rope.*{reason}"):
        qk_norm_rope(
            value,
            value if out is None else out,
            (weight,),
            (weight,),
            (cosine,),
            (sine,),
            eps=1e-6,
            axis_dims=(width,),
            out=None if out is None else (out, value),
        )
