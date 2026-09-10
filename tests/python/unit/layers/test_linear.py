"""Prepared projection values preserve logical scale domains and bias semantics."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.linear import (
    LinearBase,
    MergedColumnParallelLinear,
)
from uniserve_worker.nn.mesh import Communicator
from uniserve_worker.nn.mlp import GatedMLP
from uniserve_worker.nn.quant.base import PreparedLinearInput
from uniserve_worker.nn.quant.fp8 import DynamicW8A8Fp8LinearMethod

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("tensorwise", [False, True])
def test_prepared_fp8_values_preserve_shape_scale_and_deferred_bias(tensorwise):
    # Powers of two and small integers are exactly representable in E4M3.
    # Weight extrema of 448 give each logical output domain a unit scale.
    weights = torch.tensor([[448.0, 2.0], [-4.0, 448.0]])
    bias = torch.tensor([2.0, -4.0])
    linear = LinearBase(
        2,
        2,
        layer_config=LayerConfig(Communicator(), None),
        quant_method=DynamicW8A8Fp8LinearMethod(tensorwise=tensorwise),
    )
    linear.weight = nn.Parameter(weights.clone(), requires_grad=False)
    linear.bias = nn.Parameter(bias.clone(), requires_grad=False)
    linear.finalize_weights()
    values = torch.tensor([[[1.0, 2.0], [-2.0, 4.0]]]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5]]) if tensorwise else torch.tensor([[0.5], [2.0]])
    prepared = (
        PreparedLinearInput(values, tensor_scale=scale)
        if tensorwise
        else PreparedLinearInput(values, row_scales=scale)
    )
    expected = ((values.float().reshape(2, 2) * scale) @ weights.T).reshape(1, 2, 2)
    actual = linear.forward_prepared(prepared, output_dtype=torch.float32, include_bias=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        linear.forward_prepared(prepared, output_dtype=torch.float32),
        expected + bias,
        rtol=0,
        atol=0,
    )
    misplaced = (
        PreparedLinearInput(values, row_scales=scale)
        if tensorwise
        else PreparedLinearInput(values, tensor_scale=scale)
    )
    with pytest.raises(ValueError, match="activation scale"):
        linear.forward_prepared(misplaced)


def test_prepared_gated_mlp_preserves_row_scales_and_leading_dimensions():
    model = GatedMLP(
        2,
        2,
        layer_config=LayerConfig(Communicator(), None),
        quant_method=DynamicW8A8Fp8LinearMethod(),
    )
    weights = torch.eye(2) * 448
    with torch.no_grad():
        model.gate_up_proj.weight.copy_(weights.repeat(2, 1))
        model.down_proj.weight.copy_(weights)
    model.gate_up_proj.finalize_weights()
    model.down_proj.finalize_weights()
    values = torch.tensor([[[1, 2], [2, 1]]]).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.5], [2.0]])
    actual = model.forward_prepared(PreparedLinearInput(values, row_scales=scales))
    # The smallest gate is 224, where SiLU rounds exactly to its input.
    # All weights, activations, and FP8 scale ratios are representable here.
    dense = (values.float().reshape(2, 2) * scales).reshape(1, 2, 2)
    expected = ((dense * 448).square() * 448).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="activation scale"):
        model.forward_prepared(PreparedLinearInput(values))


@pytest.mark.parametrize("tensorwise", [False, True])
def test_reusable_activation_preparation_preserves_logical_values(tensorwise):
    linear = LinearBase(
        2,
        2,
        layer_config=LayerConfig(Communicator(), None),
        quant_method=DynamicW8A8Fp8LinearMethod(tensorwise=tensorwise),
        bias=False,
    )
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2) * 448)
    linear.finalize_weights()
    inputs = torch.tensor([[[224.0, 448.0], [-112.0, 224.0]]])
    for maximum in (None, inputs.abs().amax()):
        prepared = linear.prepare_input(inputs, absmax=maximum)
        actual = linear.forward_prepared(prepared, output_dtype=torch.float32)
        torch.testing.assert_close(actual, inputs * 448, rtol=0, atol=0)


def test_deferred_gated_mlp_preserves_dense_projection_bias():
    model = GatedMLP(
        4, 8, layer_config=LayerConfig(Communicator(), None), order="value_gate", bias=True
    )
    generator = torch.Generator().manual_seed(179)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    inputs = torch.randn((2, 3, 4), generator=generator)
    packed = F.linear(inputs, model.gate_up_proj.weight, model.gate_up_proj.bias)
    value, gate = packed.chunk(2, dim=-1)
    expected = F.linear(value * F.silu(gate), model.down_proj.weight, model.down_proj.bias)
    actual, bias = model.forward_deferred(inputs, input_absmax=inputs.abs().amax())
    if bias is not None:
        actual = actual + bias
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("component", ["mot", "vision"])
def test_multimodal_layers_keep_excluded_checkpoint_projections_dense(component):
    from uniserve_worker.nn.decoder.mot import MoTConfig, MoTModel
    from uniserve_worker.nn.quant.base import process_quantized_modules
    from uniserve_worker.nn.quant.config import QuantizationConfig
    from uniserve_worker.nn.vision.encoder import VisionEncoder, VisionEncoderConfig

    if component == "mot":
        prefix = "language_model.model"
        ignored = tuple(
            f"{prefix}.layers.0.self_attn.{name}_moe_gen"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ) + (f"{prefix}.layers.0.mlp_moe_gen",)
        model = MoTModel(
            MoTConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                vocab_size=32,
                rms_norm_eps=1e-6,
                rope_theta=10000,
                head_dim=4,
            ),
            layer_config=LayerConfig(
                Communicator(), QuantizationConfig(method="fp8", ignored_layers=ignored), prefix
            ),
        )
        dense_names = (
            "layers.0.qkv_proj_moe_gen.weight",
            "layers.0.o_proj_moe_gen.weight",
            "layers.0.mlp_moe_gen.gate_up_proj.weight",
            "layers.0.mlp_moe_gen.down_proj.weight",
        )
        quantized_names = ("layers.0.qkv_proj.weight", "layers.0.mlp.down_proj.weight")
    else:
        prefix = "vit_model.vision_model.encoder"
        ignored = (f"{prefix}.layers.0.self_attn.v_proj", f"{prefix}.layers.0.mlp.fc1")
        model = VisionEncoder(
            VisionEncoderConfig(8, 2, 16, 1),
            layer_config=LayerConfig(
                Communicator(), QuantizationConfig(method="fp8", ignored_layers=ignored), prefix
            ),
        )
        dense_names = ("layers.0.self_attn.v_proj.weight", "layers.0.mlp.0.weight")
        quantized_names = ("layers.0.self_attn.q_proj.weight", "layers.0.mlp.2.weight")
    model = model.bfloat16()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.5)
        process_quantized_modules(model.modules())
    state = model.state_dict()
    for name in dense_names:
        torch.testing.assert_close(
            state[name], torch.full(state[name].shape, 0.5, dtype=torch.bfloat16), rtol=0, atol=0
        )
    for name in quantized_names:
        assert state[name].dtype == torch.float8_e4m3fn


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("rows", [1, 3])
def test_packed_projection_branches_preserve_logical_weights_and_bias(quantized, rows):
    projection = MergedColumnParallelLinear(
        2,
        (2, 1, 1),
        layer_config=LayerConfig(Communicator(), None),
        quant_method=DynamicW8A8Fp8LinearMethod() if quantized else None,
        bias=True,
    )
    weights = torch.tensor([[448.0, 0.0], [0.0, 448.0], [448.0, 448.0], [448.0, -448.0]])
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
    with torch.no_grad():
        projection.weight.copy_(weights)
        projection.bias.copy_(bias)
    projection.finalize_weights()
    inputs = torch.tensor([[224.0, 448.0]]).expand(rows, 2)
    expected = F.linear(inputs, weights, bias)
    actual = torch.cat(projection.forward_branches(inputs), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("heads,kv_heads", [(4, 2), (4, 1), (8, 8)])
def test_query_owners_load_their_corresponding_grouped_key_value_heads(heads, kv_heads):
    from uniserve_worker.loader.handles import TensorWeightHandle
    from uniserve_worker.loader.weight_loaders import (
        attach_parameter_loaders,
        load_parameter_weight,
    )
    from uniserve_worker.nn.linear import QKVParallelLinear

    head_dim, width = 4, 8
    weights = tuple(
        torch.arange(count * head_dim * width, dtype=torch.float32).view(count * head_dim, width)
        for count in (heads, kv_heads, kv_heads)
    )
    biases = tuple(
        torch.arange(count * head_dim, dtype=torch.float32) for count in (heads, kv_heads, kv_heads)
    )
    values = torch.eye(width)[:3]
    ranks = (3, 1, 0, 2)
    for owner, physical in enumerate(ranks):
        linear = QKVParallelLinear(
            width,
            head_dim,
            heads,
            kv_heads,
            layer_config=LayerConfig(Communicator(ranks=ranks, rank=physical), None),
            bias=True,
        )
        attach_parameter_loaders(linear, device="cpu", dtype=torch.float32)
        for branch, weight, bias in zip(("q", "k", "v"), weights, biases, strict=True):
            load_parameter_weight(linear.weight, TensorWeightHandle("weight", weight), branch)
            load_parameter_weight(linear.bias, TensorWeightHandle("bias", bias), branch)
        query_heads = range(owner * heads // 4, (owner + 1) * heads // 4)
        key_heads = sorted({head // (heads // kv_heads) for head in query_heads})
        expected = []
        for selected, weight, bias in zip(
            (list(query_heads), key_heads, key_heads), weights, biases, strict=True
        ):
            columns = [head * head_dim + dim for head in selected for dim in range(head_dim)]
            expected.append(F.linear(values, weight, bias)[:, columns])
        torch.testing.assert_close(linear(values), torch.cat(expected, dim=-1), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 7), (0, 0), ()])
def test_empty_projection_keeps_the_declared_input_width(shape):
    linear = LinearBase(8, 4, layer_config=LayerConfig(Communicator(), None))
    with pytest.raises(ValueError, match="feature width"):
        linear(torch.empty(shape))
