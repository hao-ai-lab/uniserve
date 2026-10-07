"""Encoded tensor values, borrowing, parameter identity and serialization."""

import io

import pytest
import torch

from uniserve.quantization import (
    QuantizedTensor,
    Quantizer,
    RowOrder,
    ScaleLayout,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("axis", [None, 0])
def test_fp8_quantization_preserves_scale_domains_and_output_storage(axis):
    source = torch.tensor([[224.0, 448.0], [112.0, -224.0]])
    converter = Quantizer("fp8", axis=axis)
    expected_scale = (
        torch.tensor(1.0) if axis is None else torch.tensor([[1.0], [0.5]])
    )
    target = converter.empty(
        tuple(source.shape), dtype=source.dtype, device=source.device
    )
    backing = dict(target.buffers())
    assert converter.quantize(source, out=target) is target
    torch.testing.assert_close(
        target.buffers()["scale"], expected_scale, atol=0, rtol=0
    )
    torch.testing.assert_close(target.dequantize(), source, atol=0, rtol=0)
    converter.quantize(-source, out=target)
    for name, value in backing.items():
        assert target.buffers()[name].data_ptr() == value.data_ptr()
    torch.testing.assert_close(target.dequantize(), -source, atol=0, rtol=0)


def test_fp8_source_encoding_is_borrowed_without_requantization():
    values = torch.tensor([[2.0, 4.0], [1.0, -2.0]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5], [4.0]])
    encoded = Quantizer("fp8", axis=0).from_tensors(
        {"values": values, "scale": scale}, shape=(2, 2), dtype=torch.bfloat16
    )
    torch.testing.assert_close(
        encoded.dequantize(),
        torch.tensor([[1.0, 2.0], [4.0, -8.0]], dtype=torch.bfloat16),
        atol=0,
        rtol=0,
    )
    scale.mul_(2)
    torch.testing.assert_close(
        encoded.dequantize(),
        torch.tensor([[2.0, 4.0], [8.0, -16.0]], dtype=torch.bfloat16),
        atol=0,
        rtol=0,
    )
    with pytest.raises(TypeError):
        encoded.buffers()["scale"] = torch.ones_like(scale)
    with pytest.raises(AttributeError):
        encoded.quantizer = Quantizer("fp8")


def test_quantized_parameters_keep_aliases_dtype_and_serialized_values():
    source = torch.tensor([[1.0, 2.0], [100.0, 200.0]])
    encoded = Quantizer("fp8", axis=0).quantize(source)
    module = torch.nn.Module()
    module.register_parameter(
        "left", torch.nn.Parameter(encoded, requires_grad=False)
    )
    module.register_parameter("right", module.left)
    module.to(dtype=torch.bfloat16)
    assert module.left is module.right
    assert isinstance(module.left, QuantizedTensor)
    torch.testing.assert_close(
        module.left.dequantize(), source.bfloat16(), atol=0, rtol=0
    )
    checkpoint = io.BytesIO()
    torch.save(module.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=True)
    torch.testing.assert_close(
        restored["left"].dequantize(), source.bfloat16(), atol=0, rtol=0
    )
    restored["left"].buffers()["scale"].mul_(2)
    torch.testing.assert_close(
        restored["right"].dequantize(), (source * 2).bfloat16(), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "shape,axis", [((0, 4), 0), ((3, 0), 0), ((0, 4), None), ((), None)]
)
def test_empty_and_scalar_fp8_statistics_are_defined(shape, axis):
    source = torch.zeros(shape)
    result = Quantizer("fp8", axis=axis).quantize(source)
    torch.testing.assert_close(result.dequantize(), source, atol=0, rtol=0)
    assert torch.isfinite(result.buffers()["scale"]).all()


@pytest.mark.parametrize(
    "format,axis",
    [("unknown", None), ("mxfp8", 0), ("nvfp4", 0), ("fp8", 1), ("fp8", False)],
)
def test_quantizer_rejects_unsupported_format_axis_combinations(format, axis):
    with pytest.raises(ValueError):
        Quantizer(format, axis=axis)


def test_calibrated_nvfp4_scale_is_validated_and_frozen():
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="calibrated scale"):
            Quantizer("nvfp4", calibrated_scale=value)
    with pytest.raises(ValueError, match="calibrated scale"):
        Quantizer("fp8", calibrated_scale=1.0)

    calibrated = Quantizer("nvfp4", calibrated_scale=24.0)
    assert calibrated.calibrated_scale == 24.0
    assert not calibrated.requires_complete_source
    assert Quantizer("nvfp4").requires_complete_source
    assert Quantizer("fp8").requires_complete_source
    assert not Quantizer("fp8", axis=0).requires_complete_source


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [0, 8])
def test_calibrated_nvfp4_scale_is_cuda_graph_capturable(rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (
        10,
        0,
    ):
        pytest.skip("NVFP4 graph capture requires an SM100-class CUDA device")
    source = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
    quantizer = Quantizer("nvfp4", calibrated_scale=24.0 / (448.0 * 6.0))

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        quantizer.quantize(source)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        encoded = quantizer.quantize(source)
    graph.replay()

    torch.testing.assert_close(
        encoded.buffers()["tensor_scale"],
        torch.tensor(24.0 / (448.0 * 6.0), device="cuda"),
        rtol=0,
        atol=0,
    )


def test_fp8_encoding_rejects_incomplete_or_incompatible_backing():
    converter = Quantizer("fp8", axis=0)
    fields = {
        "values": torch.zeros((2, 4), dtype=torch.float8_e4m3fn),
        "scale": torch.ones(2, 1),
    }
    invalid = [
        {"values": fields["values"]},
        {**fields, "tensor_scale": torch.ones(())},
        {**fields, "scale": torch.ones(())},
        {**fields, "scale": torch.ones(2, 1, dtype=torch.float64)},
        {**fields, "values": torch.zeros(2, 4)},
    ]
    for tensors in invalid:
        with pytest.raises(ValueError):
            converter.from_tensors(tensors, shape=(2, 4), dtype=torch.float32)
    with pytest.raises(ValueError):
        converter.from_tensors(
            fields,
            shape=(2, 4),
            dtype=torch.float32,
            scale_layout=ScaleLayout.SWIZZLED_128X4,
        )
    target = Quantizer("fp8").empty((2, 4), dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="output"):
        converter.quantize(torch.ones(2, 4), out=target)


def test_block_decoding_uses_encoded_values_and_both_nvfp4_scales():
    values = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
    )
    nvfp4 = Quantizer("nvfp4").from_tensors(
        {
            "values": values,
            "block_scale": torch.tensor(
                [[2.0]], dtype=torch.float8_e4m3fn
            ).view(torch.uint8),
            "tensor_scale": torch.tensor(0.25),
        },
        shape=(1, 16),
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [
            [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                3.0,
                0.0,
                -0.25,
                -0.5,
                -0.75,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
            ]
        ]
    )
    torch.testing.assert_close(nvfp4.dequantize(), expected, rtol=0, atol=0)
    mxfp8 = Quantizer("mxfp8").from_tensors(
        {
            "values": torch.ones((1, 32), dtype=torch.float8_e4m3fn),
            "scale": torch.tensor([[126]], dtype=torch.uint8),
        },
        shape=(1, 32),
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        mxfp8.dequantize(), torch.full((1, 32), 0.5), rtol=0, atol=0
    )


def _shuffled(rows):
    """Physical position of each row under the 32-row block shuffle."""
    row = torch.arange(rows)
    within = row % 32
    return row - within + (within % 4) * 8 + within // 4


def test_row_orders_permute_each_expert_without_changing_values():
    experts, rows, width = 2, 256, 32
    generator = torch.Generator().manual_seed(3)
    values = torch.randint(
        0,
        256,
        (experts, rows, width // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    block_scale = (
        (torch.rand(experts * rows, width // 16, generator=generator) + 0.5)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    encoded = Quantizer("nvfp4").from_tensors(
        {
            "values": values,
            "block_scale": block_scale,
            "tensor_scale": torch.tensor([0.5, 2.0]),
        },
        shape=(experts, rows, width),
        dtype=torch.float32,
    )
    logical = encoded.dequantize()

    # The interleaved order first puts row i of each half at 2i or 2i + 1.
    half = torch.arange(rows) % (rows // 2)
    second = (torch.arange(rows) >= rows // 2).long()
    interleaved = 2 * half + second
    positions = {
        RowOrder.SHUFFLED_128: _shuffled(rows),
        RowOrder.INTERLEAVED_SHUFFLED_128: _shuffled(rows)[interleaved],
        # Row i of either half goes to 128-row block i // 64, at offset
        # i % 64 in its first (first half) or second (second half) 64 rows.
        RowOrder.INTERLEAVED_64: 128 * (half // 64) + half % 64 + 64 * second,
        # Row i of either half goes to 32-row block i // 16, at offset
        # i % 16 in its second (first half) or first (second half) 16 rows.
        RowOrder.INTERLEAVED_16: 32 * (half // 16)
        + half % 16
        + 16 * (1 - second),
        RowOrder.INTERLEAVED_8: 16 * (half // 8) + half % 8 + 8 * (1 - second),
    }
    for order, position in positions.items():
        for layout in ScaleLayout:
            placed = encoded.repack(scale_layout=layout, row_order=order)
            assert placed.row_order is order
            assert torch.equal(placed.dequantize(), logical)
            # Physical row position[r] stores logical row r of every expert.
            assert torch.equal(placed.buffers()["values"][:, position], values)

            restored = placed.repack(
                scale_layout=ScaleLayout.LINEAR, row_order=RowOrder.LINEAR
            )
            for name in ("values", "block_scale"):
                assert torch.equal(
                    restored.buffers()[name], encoded.buffers()[name]
                )

            checkpoint = io.BytesIO()
            torch.save(placed, checkpoint)
            checkpoint.seek(0)
            loaded = torch.load(checkpoint, weights_only=True)
            assert loaded.row_order is order
            assert torch.equal(loaded.dequantize(), logical)


def test_row_orders_describe_only_stacked_block_scaled_rows():
    shuffled = RowOrder.SHUFFLED_128
    nvfp4 = Quantizer("nvfp4")
    # A rank-2 matrix, a stack of partial 32-row blocks, and a stack of
    # whole 32-row blocks that are not whole 128-row interleaving blocks.
    for shape, order in (
        ((64, 32), shuffled),
        ((2, 48, 32), shuffled),
        ((2, 64, 32), RowOrder.INTERLEAVED_64),
    ):
        rows = shape[-2] * (shape[0] if len(shape) == 3 else 1)
        fields = {
            "values": torch.zeros((*shape[:-1], 16), dtype=torch.uint8),
            "block_scale": torch.zeros((rows, 2), dtype=torch.uint8),
            "tensor_scale": torch.ones(()),
        }
        with pytest.raises(ValueError, match="row order"):
            nvfp4.from_tensors(
                fields, shape=shape, dtype=torch.float32, row_order=order
            )
    with pytest.raises(ValueError, match="row order"):
        Quantizer("fp8").from_tensors(
            {
                "values": torch.zeros((2, 32, 4), dtype=torch.float8_e4m3fn),
                "scale": torch.ones(()),
            },
            shape=(2, 32, 4),
            dtype=torch.float32,
            row_order=shuffled,
        )


@pytest.mark.parametrize("format", [None, "fp8"])
def test_linear_consumes_logical_operands_and_preserves_output_storage(format):
    from uniserve.nn.functional import linear

    x = torch.full((3, 32), 2.0)
    weight = torch.stack((torch.ones(32), torch.full((32,), -0.5)))
    if format is not None:
        x = Quantizer(format, axis=0).quantize(x)
        weight = Quantizer(format, axis=0).quantize(weight)
    out = torch.empty(2, 3).T
    assert linear(x, weight, torch.tensor([1.0, 3.0]), out=out) is out
    torch.testing.assert_close(
        out, torch.tensor([[65.0, -29.0]]).expand(3, 2), rtol=0, atol=0
    )
    torch.testing.assert_close(
        torch.nn.functional.linear(x, weight),
        torch.tensor([[64.0, -32.0]]).expand(3, 2),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    "device", ("cpu", pytest.param("cuda", marks=pytest.mark.gpu))
)
def test_merged_projections_preserve_independent_branch_scale_domains(
    quantized, device
):
    from uniserve.nn.functional import merged_linear

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    x = torch.full((3, 32), 2.0, device=device, dtype=dtype)
    weights = {
        "gate": torch.ones((16, 32), device=device, dtype=dtype),
        "up": torch.full((32, 32), 100.0, device=device, dtype=dtype),
    }
    if quantized:
        x = Quantizer("fp8", axis=0).quantize(x)
        weights = {
            name: Quantizer("fp8").quantize(weight)
            for name, weight in weights.items()
        }
    out = {
        "gate": torch.empty(16, 3, device=device, dtype=dtype).T,
        "up": torch.empty(32, 3, device=device, dtype=dtype).T,
    }
    result = merged_linear(
        x,
        weights,
        {"gate": torch.ones(16, device=device, dtype=dtype), "up": None},
        out=out,
    )
    for name, expected in (("gate", 65.0), ("up", 6400.0)):
        assert result[name] is out[name]
        torch.testing.assert_close(
            result[name],
            torch.full_like(result[name], expected),
            rtol=0,
            atol=0,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("axis", "shape"),
    ((0, (33, 5120)), (0, (3, 0)), (None, (33, 5120)), (None, (4, 9, 64))),
)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("supplied", (False, True))
def test_cuda_fp8_encoding_matches_the_portable_encoding(
    axis, shape, dtype, supplied
):
    generator = torch.Generator().manual_seed(59)
    source = (torch.randn(shape, generator=generator) * 5).to(dtype)
    quantizer = Quantizer("fp8", axis=axis)
    expected = quantizer.quantize(source).buffers()

    device = source.cuda()
    maximum = quantizer.amax(device)
    actual = quantizer.quantize(
        device, amax=maximum if supplied else None
    ).buffers()

    # Statistics and scales are exact on both devices.
    torch.testing.assert_close(
        maximum.cpu(), quantizer.amax(source), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["scale"].cpu(), expected["scale"], rtol=0, atol=0
    )
    # An FP32 quotient divided with at most 2 ulp of error can cross at most
    # one E4M3 rounding midpoint, so each code equals the portable code or
    # its same-sign neighbour, whose byte differs by one.
    actual_codes = actual["values"].cpu().view(torch.uint8).int()
    expected_codes = expected["values"].view(torch.uint8).int()
    assert ((actual_codes - expected_codes).abs() <= 1).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_row_fp8_of_single_rank_column_shards_matches_unsharded():
    from torch.distributed.tensor import Shard

    from uniserve.distributed import DeviceMesh, Distribution

    mesh = DeviceMesh(ranks=(0,), shape=(1,), axes=("tp",), rank=0)
    source = torch.randn((16, 4096), device="cuda").to(torch.bfloat16)
    quantizer = Quantizer("fp8", axis=0)

    sharded = quantizer.quantize(
        source, distribution=Distribution(mesh, (Shard(1),))
    ).buffers()
    unsharded = quantizer.quantize(source).buffers()

    for name in ("values", "scale"):
        assert torch.equal(
            sharded[name].view(torch.uint8), unsharded[name].view(torch.uint8)
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("axis", "shape", "reason"),
    (
        (0, (2, 3, 64), "rank-2"),
        (0, (2, 32769), "exceeds"),
        (None, (2, 32769), "exceeds"),
    ),
)
def test_cuda_fp8_encoding_without_a_kernel_raises(axis, shape, reason):
    source = torch.zeros(shape, device="cuda")
    with pytest.raises(ValueError, match=f"Quantizer.quantize.*{reason}"):
        Quantizer("fp8", axis=axis).quantize(source)
    strided = torch.zeros((4, 64, 2), device="cuda")[..., 0]
    with pytest.raises(ValueError, match="unit-strided"):
        Quantizer("fp8", axis=0).quantize(strided)
