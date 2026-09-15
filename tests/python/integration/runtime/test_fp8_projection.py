"""CUDA projection replay preserves encoded inputs and complete row statistics."""

import pytest
import torch
from torch import nn

from uniserve.nn.linear import Linear
from uniserve.quantization import Quantizer

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_projection_and_reused_encoding(dtype):
    device = torch.device("cuda", 0)
    quantizer = Quantizer("fp8", axis=0)
    generator = torch.Generator(device=device).manual_seed(71)
    x = torch.randn(32, 64, generator=generator, device=device, dtype=dtype)
    x[0].zero_()
    encoded = quantizer.empty(tuple(x.shape), dtype=dtype, device=device)
    layer = Linear(64, 32, dtype=dtype, device=device)
    layer.weight = nn.Parameter(quantizer.quantize(layer.weight), requires_grad=False)
    layer.input_quantizer = quantizer
    expected = quantizer.quantize(x, amax=x.float().abs().amax(1, keepdim=True))
    quantizer.quantize(x, out=encoded)
    torch.testing.assert_close(encoded.dequantize(), expected.dequantize(), rtol=0, atol=0)
    torch.testing.assert_close(layer(x), layer(encoded), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        quantizer.quantize(x, out=encoded)
    torch.cuda.current_stream(device).wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        quantizer.quantize(x, out=encoded)
    x.mul_(4)
    graph.replay()
    expected = quantizer.quantize(x, amax=x.float().abs().amax(1, keepdim=True))
    torch.testing.assert_close(encoded.dequantize(), expected.dequantize(), rtol=0, atol=0)
    graph.reset()


@torch.inference_mode()
@pytest.mark.parametrize("operand", ["input", "weight"])
def test_one_fp8_operand_preserves_the_other_operand_values(operand):
    from uniserve.nn.functional import linear

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(13)
    x = torch.randn(3, 64, dtype=torch.bfloat16, device=device, generator=generator)
    weight = torch.randn(32, 64, dtype=x.dtype, device=device, generator=generator)
    quantizer = Quantizer("fp8", axis=0)
    if operand == "input":
        x = quantizer.quantize(x)
        reference_x = x.dequantize(dtype=torch.float32)
        reference_weight = weight.float()
    else:
        weight = quantizer.quantize(weight)
        reference_x = x.float()
        reference_weight = weight.dequantize(dtype=torch.float32)
    expected = torch.mm(reference_x, reference_weight.T).to(torch.bfloat16)
    torch.testing.assert_close(linear(x, weight), expected, rtol=0, atol=0)
