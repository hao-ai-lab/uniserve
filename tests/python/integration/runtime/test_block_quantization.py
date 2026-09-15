"""Block encoding and repacking against the installed numerical provider."""

import pytest
import torch

from uniserve.quantization import Quantizer, ScaleLayout

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.parametrize("format", ["mxfp8", "nvfp4"])
def test_block_encoding_and_scale_repacking_preserve_native_values(format):
    import flashinfer

    generator = torch.Generator(device="cuda:0").manual_seed(401)
    source = torch.randn(
        (160, 256), device="cuda:0", dtype=torch.bfloat16, generator=generator
    )
    source[:8].zero_()
    converter = Quantizer(format)
    result = converter.quantize(source)
    packed = result.repack(scale_layout=ScaleLayout.SWIZZLED_128X4)
    restored = packed.repack(scale_layout=ScaleLayout.LINEAR)
    torch.testing.assert_close(
        restored.dequantize(), result.dequantize(), rtol=0, atol=0
    )
    if format == "mxfp8":
        values, scales = flashinfer.mxfp8_quantize(
            source, is_sf_swizzled_layout=True, backend="cuda"
        )
        native = converter.from_tensors(
            {"values": values, "scale": scales.reshape(-1)},
            shape=tuple(source.shape),
            dtype=source.dtype,
            scale_layout=ScaleLayout.SWIZZLED_128X4,
        )
    else:
        tensor_scale = source.float().abs().amax().clamp_min(1e-12) / (448 * 6)
        values, scales = flashinfer.nvfp4_quantize(
            source,
            1.0 / tensor_scale,
            sfLayout=flashinfer.SfLayout.layout_128x4,
            backend="cuda",
            enable_pdl=False,
        )
        native = converter.from_tensors(
            {
                "values": values,
                "block_scale": scales.reshape(-1),
                "tensor_scale": tensor_scale,
            },
            shape=tuple(source.shape),
            dtype=source.dtype,
            scale_layout=ScaleLayout.SWIZZLED_128X4,
        )
    torch.testing.assert_close(
        packed.dequantize(), native.dequantize(), rtol=0, atol=0
    )
    explicit = converter.quantize(source, amax=converter.amax(source))
    torch.testing.assert_close(
        explicit.dequantize(), native.dequantize(), rtol=0, atol=0
    )


@pytest.mark.parametrize("format", ["fp8", "mxfp8", "nvfp4"])
@torch.inference_mode()
def test_encoded_gemm_writes_exact_projection_to_borrowed_output(format):
    from uniserve.model import TextSize
    from uniserve.nn import Linear
    from uniserve.nn.functional import linear
    from uniserve.runtime import CUDAGraph, ExecutionContext

    x = torch.full((128, 256), 2.0, dtype=torch.bfloat16, device="cuda:0")
    weight = torch.ones((128, 256), dtype=torch.bfloat16, device="cuda:0")
    converter = Quantizer(format, axis=0 if format == "fp8" else None)
    left, right = converter.quantize(x), converter.quantize(weight)
    out = torch.full((128, 128), -1.0, device="cuda:0", dtype=torch.bfloat16)
    assert linear(left, right, out=out) is out
    expected = (
        left.dequantize(dtype=torch.float32)
        @ right.dequantize(dtype=torch.float32).T
    ).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    layer = Linear(256, 128, bias=False, dtype=x.dtype, device=x.device)
    layer.weight = torch.nn.Parameter(right, requires_grad=False)
    layer.input_quantizer = converter
    with ExecutionContext(layer) as context:
        context.prepare(TextSize(160, 1))
        assert layer(x, out=out) is out
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        with CUDAGraph(context=context) as graph:
            graph.capture(lambda: layer(x, out=out))
            x.mul_(2)
            graph.replay()
            encoded = converter.quantize(x)
            expected = (
                encoded.dequantize(dtype=torch.float32)
                @ right.dequantize(dtype=torch.float32).T
            ).bfloat16()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            torch.cuda.synchronize(x.device)


@pytest.mark.parametrize("format", [None, "fp8", "mxfp8", "nvfp4"])
@pytest.mark.parametrize("branch_width", [None, 4])
@torch.inference_mode()
def test_merged_projections_preserve_live_parameters_bias_and_strided_outputs(
    format, branch_width
):
    from uniserve.model import TextSize
    from uniserve.nn import MergedColumnParallelLinear
    from uniserve.runtime import CUDAGraph, ExecutionContext

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(718)
    layer = MergedColumnParallelLinear(
        128,
        {"first": 8, "second": 12},
        branch_width=branch_width,
        device=device,
        dtype=torch.bfloat16,
    )
    quantizer = None if format is None else Quantizer(format)
    # The offset and row stride both require alignment handling for native TMA.
    source = (
        torch.rand((129, 129), device=device, generator=generator) + 0.125
    ).bfloat16()[:, 1:]
    for index, branch in enumerate(layer.projections.values()):
        value = (
            torch.rand(branch.weight.shape, device=device, generator=generator)
            + 0.125
        ).bfloat16()
        value.mul_(0.25 if index == 0 else 8)
        if quantizer is None:
            branch.weight.copy_(value)
        else:
            converter = (
                Quantizer("fp8", axis=0)
                if format == "fp8" and index
                else quantizer
            )
            encoded = converter.quantize(value)
            if index and format in {"nvfp4", "mxfp8"}:
                encoded = encoded.repack(
                    scale_layout=ScaleLayout.SWIZZLED_128X4
                )
            branch.weight = torch.nn.Parameter(encoded, requires_grad=False)
            branch.input_quantizer = quantizer
        if index == 0:
            branch.bias.fill_(3)
        else:
            branch.register_parameter("bias", None)

    def expected(x):
        left = (
            x.float()
            if quantizer is None
            else quantizer.quantize(x).dequantize(dtype=torch.float32)
        )
        result = {}
        for name, branch in layer.projections.items():
            right = (
                branch.weight.float()
                if quantizer is None
                else branch.weight.dequantize(dtype=torch.float32)
            )
            value = left @ right.T
            if quantizer is not None:
                value = value.bfloat16()
            if branch.bias is not None:
                value = value + branch.bias
            result[name] = value.bfloat16()
        return result

    # FP32 accumulation over K=128 followed by a BF16 result and bias roundoff.
    unit = 2**-8
    bound = 2 * unit / (1 - 2 * unit) + 128 * 2**-24 / (1 - 128 * 2**-24)
    with ExecutionContext(layer) as context:
        context.prepare(TextSize(160, 1))
        initial = source[:3]
        if format in {"nvfp4", "mxfp8"}:
            initial = quantizer.quantize(initial).repack(
                scale_layout=ScaleLayout.SWIZZLED_128X4
            )
        retained = layer(initial)
        for name, value in expected(source[:3]).items():
            torch.testing.assert_close(
                retained[name], value, rtol=bound, atol=0
            )
        original = {name: value.clone() for name, value in retained.items()}
        out = {
            name: torch.empty(
                (branch.weight.shape[0], 129), device=device, dtype=source.dtype
            ).T
            for name, branch in layer.projections.items()
        }
        layer(source, out=out)
        with CUDAGraph(context=context) as graph:
            graph.capture(lambda: layer(source, out=out))
            source.mul_(2)
            for branch in layer.projections.values():
                if quantizer is None:
                    branch.weight.mul_(2)
                else:
                    replacement = branch.weight.quantizer.quantize(
                        branch.weight.dequantize() * 2
                    )
                    replacement = replacement.repack(
                        scale_layout=branch.weight.scale_layout
                    )
                    for name, buffer in branch.weight.buffers().items():
                        buffer.copy_(replacement.buffers()[name])
            graph.replay()
            for name, value in expected(source).items():
                torch.testing.assert_close(out[name], value, rtol=bound, atol=0)
            for name, value in retained.items():
                torch.testing.assert_close(
                    value, original[name], rtol=0, atol=0
                )
            torch.cuda.synchronize(device)
