"""Numerical behavior of checkpoint-streamed fixed-timestep preparation."""

import pytest
import torch
from torch.nn import functional as F

from uniserve.nn import Modulation

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streamed_modulation_preserves_projection_batch_and_step_outputs(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(53)
    inputs = torch.randn((4, 2, 64), generator=generator)
    weights = tuple(
        torch.randn((96, 64), generator=generator).to(dtype) for _ in range(3)
    )
    biases = tuple(
        torch.randn((96,), generator=generator).to(dtype) for _ in range(3)
    )
    final_weight = torch.randn((32, 64), generator=generator).to(dtype)
    final_bias = torch.randn((32,), generator=generator).to(dtype)
    plan = Modulation.from_projections(
        inputs,
        zip(weights, biases, strict=True),
        (final_weight, final_bias),
        layer_count=3,
    )
    expected = torch.stack(
        tuple(
            F.linear(inputs.flatten(0, 1).to(dtype), weight, bias).view(
                4, 2, 96
            )
            for weight, bias in zip(weights, biases, strict=True)
        ),
        dim=1,
    )
    final = F.linear(
        inputs.flatten(0, 1).to(dtype), final_weight, final_bias
    ).view(4, 2, 32)
    # Steps are device indices, so one gather serves any evaluation order.
    for step in (3, 0, 2, 1, 0):
        index = torch.tensor([step])
        torch.testing.assert_close(plan(index), expected[step], atol=0, rtol=0)
        torch.testing.assert_close(
            plan.output(index), final[step], atol=0, rtol=0
        )
    with pytest.raises(ValueError, match="int64 index"):
        plan(torch.tensor([1], dtype=torch.int32))


@pytest.mark.parametrize("count", [1, 3])
def test_modulation_rejects_incomplete_checkpoint_layer_sets(
    count: int,
) -> None:
    projection = (torch.ones((8, 4)), None)
    with pytest.raises(ValueError, match="layer projections"):
        Modulation.from_projections(
            torch.ones((4, 2, 4)),
            iter([projection] * count),
            projection,
            layer_count=2,
        )


def test_pipeline_layer_products_omit_final_projection():
    torch.manual_seed(103)
    inputs = torch.randn(4, 2, 8)
    weight = torch.randn(12, 8)
    plan = Modulation.from_projections(
        inputs, [(weight, None)], None, layer_count=1
    )
    for step in range(4):
        torch.testing.assert_close(
            plan(torch.tensor([step]))[0],
            F.linear(inputs.flatten(0, 1), weight).view(4, 2, 12)[step],
            rtol=0,
            atol=0,
        )
    with pytest.raises(ValueError, match="no output projection"):
        plan.output(torch.tensor([0]))
