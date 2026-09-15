"""Encoded tensor values, borrowing, parameter identity and serialization."""

import io

import pytest
import torch

from uniserve.quantization import QuantizedTensor, Quantizer, ScaleLayout

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
