"""Shared schedule factories follow their closed forms."""

import pytest
import torch

from uniserve.diffusion import BlockGrid, UniformGrid, fuse_heads

pytestmark = pytest.mark.unit

# The OmniRef PDD contract: 32 fine intervals in blocks of four on the base
# clock [0, 0.999], video shift 12 and audio shift 3.
NODES = (0, 4, 8, 12, 16, 20, 24, 28, 32)


def test_uniform_grid_collapses_shift_collisions():
    # A large shift maps the leading FP32 grid points onto one value; the
    # schedule keeps each distinct sigma once and ends at the clean endpoint.
    grid = UniformGrid(1000, shift=1e6)
    schedule = grid.schedule(device="cpu")
    assert bool((schedule.sigmas[1:] < schedule.sigmas[:-1]).all())
    assert schedule.sigmas[0] == 1 and schedule.sigmas[-1] == 0
    assert schedule.num_steps == grid.num_steps < 999


def test_block_nodes_follow_the_fixed_point_shift():
    grid = BlockGrid(32, NODES, shift=12.0, max_t=0.999)
    schedule = grid.schedule(device="cpu")
    # The contract's published node sigmas, to their six printed decimals.
    expected = torch.tensor(
        [
            0.999,
            0.987247,
            0.972,
            0.951429,
            0.922154,
            0.877171,
            0.7992,
            0.630947,
            0,
        ]
    )
    torch.testing.assert_close(schedule.sigmas, expected, rtol=0, atol=5e-7)
    torch.testing.assert_close(
        schedule.timesteps, 1 - schedule.sigmas, rtol=0, atol=0
    )
    # The first node is the shift's fixed point for every modality.
    audio = BlockGrid(32, NODES, shift=3.0, max_t=0.999).schedule(device="cpu")
    assert float(audio.sigmas[0]) == pytest.approx(0.999, abs=1e-7)


@pytest.mark.parametrize("shift", [1.0, 3.0, 12.0])
def test_block_weights_integrate_to_node_increments(shift):
    grid = BlockGrid(32, NODES, shift=shift, max_t=0.999)
    fine = torch.linspace(0.999, 0.0, 33, dtype=torch.float64)
    shifted = shift * fine * 0.999 / (fine * (shift - 1) + 0.999)
    torch.testing.assert_close(
        grid.weights, shifted[1:] - shifted[:-1], rtol=1e-12, atol=1e-15
    )
    for block in range(len(NODES) - 1):
        weights = grid.block_weights(block)
        assert float(weights.sum()) == pytest.approx(1.0, abs=1e-15)
        assert bool((weights > 0).all())


def test_fused_heads_accumulate_in_fp32_and_round_once():
    generator = torch.Generator().manual_seed(7)
    heads, width = 8, 6
    weight = torch.randn((heads * width, 5), generator=generator).to(
        torch.bfloat16
    )
    mix = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    fused = fuse_heads(weight, mix, heads=heads, start=4)
    expected = (
        torch.einsum(
            "n,nij->ij",
            mix.float(),
            weight.unflatten(0, (heads, width))[4:8].float(),
        )
    ).to(torch.bfloat16)
    assert fused.dtype == torch.bfloat16
    torch.testing.assert_close(fused, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="whole parameter heads"):
        fuse_heads(weight, mix, heads=heads, start=6)
