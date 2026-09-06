"""Numerical behavior of checkpoint-streamed fixed-timestep preparation."""

import pytest
import torch
from torch.nn import functional as F

from uniserve_worker.nn.diffusion.modulation import ModulationPlan

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streamed_modulation_preserves_projection_batch_and_step_outputs(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(53)
    inputs = torch.randn((4, 2, 64), generator=generator)
    weights = tuple(torch.randn((96, 64), generator=generator).to(dtype) for _ in range(3))
    biases = tuple(torch.randn((96,), generator=generator).to(dtype) for _ in range(3))
    final_weight = torch.randn((32, 64), generator=generator).to(dtype)
    final_bias = torch.randn((32,), generator=generator).to(dtype)
    plan = ModulationPlan.materialize(
        inputs, zip(weights, biases, strict=True), (final_weight, final_bias), layer_count=3
    )
    expected = torch.stack(
        tuple(
            F.linear(inputs.flatten(0, 1).to(dtype), weight, bias).view(4, 2, 96)
            for weight, bias in zip(weights, biases, strict=True)
        ),
        dim=1,
    )
    final = F.linear(inputs.flatten(0, 1).to(dtype), final_weight, final_bias).view(4, 2, 32)
    block_output = torch.empty((3, 2, 96), dtype=dtype)
    final_output = torch.empty((2, 32), dtype=dtype)
    for step in (3, 0, 2, 1, 0):
        plan.copy_step(step, block_output, final_output)
        torch.testing.assert_close(block_output, expected[step], atol=0, rtol=0)
        torch.testing.assert_close(final_output, final[step], atol=0, rtol=0)
    with pytest.raises(ValueError, match="outside"):
        plan.copy_step(4, block_output, final_output)


@pytest.mark.parametrize("count", [1, 3])
def test_modulation_rejects_incomplete_checkpoint_layer_sets(count: int) -> None:
    projection = (torch.ones((8, 4)), None)
    with pytest.raises(ValueError, match="layer projections"):
        ModulationPlan.materialize(
            torch.ones((4, 2, 4)), iter([projection] * count), projection, layer_count=2
        )


def test_pipeline_layer_products_omit_final_projection():
    torch.manual_seed(103)
    inputs = torch.randn(4, 2, 8)
    weight = torch.randn(12, 8)
    plan = ModulationPlan.materialize(inputs, [(weight, None)], None, layer_count=1)
    output = torch.empty(1, 2, 12)
    for step in range(4):
        plan.copy_step(step, output, torch.empty(0))
        torch.testing.assert_close(
            output[0], F.linear(inputs.flatten(0, 1), weight).view(4, 2, 12)[step], rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="incompatible geometry"):
        plan.copy_step(0, output, torch.empty(2, 12))
