"""BF16 grouped expert GEMMs store each route's expert output at its row.

Over FlashInfer's ``moe_sort`` metadata, ``gather_gemm`` (FC1) stores the
gated product of every sorted (token, route) row and ``route_gemm`` (FC2)
stores route ``k`` of token ``t``, unweighted, at row ``t * K + k``. Every
FC1 and FC2 tactic pair runs, since the measured tactic table may select
any of them. Every operand is positive, so each rounding step bounds the
relative error of every output independently of cancellation.
"""

import pytest
import torch
from torch.nn import functional as F

from uniserve_kernels import experts

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, TOKENS = 8, 256, 128, 2, 40
DEVICE = torch.device("cuda:0")
PAIRS = [
    (gather, route)
    for gather in experts.GATHER_TACTICS
    for route in experts.ROUTE_TACTICS
    if gather[0][0] == route[0][0]
]


def _gamma(roundings):
    """Relative bound of ``roundings`` BF16 unit roundoffs (Higham gamma)."""
    unit = 2**-8
    return roundings * unit / (1 - roundings * unit)


def _positive(generator, *shape, fan_in=1):
    """BF16 values in [0.125, 1.125) / fan_in, keeping products near one."""
    values = (torch.rand(shape, generator=generator) + 0.125) / fan_in
    return values.to(DEVICE, torch.bfloat16)


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("pair", PAIRS, ids=str)
@torch.inference_mode()
def test_route_rows_hold_each_routes_expert_output(pair, activation):
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort

    gather, route = pair
    tile = gather[0][0]
    generator = torch.Generator().manual_seed(53)
    hidden = _positive(generator, TOKENS, HIDDEN)
    # Each expert's up rows, then its gate rows.
    up_gate = _positive(
        generator, EXPERTS, 2 * INTERMEDIATE, HIDDEN, fan_in=HIDDEN
    )
    down = _positive(
        generator, EXPERTS, HIDDEN, INTERMEDIATE, fan_in=INTERMEDIATE
    )
    ids = torch.stack(
        [
            torch.randperm(EXPERTS, generator=generator)[:TOP_K]
            for _ in range(TOKENS)
        ]
    ).to(DEVICE, torch.int32)
    weights = torch.rand(TOKENS, TOP_K, generator=generator).to(DEVICE)

    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        _,
        permuted_idx_to_expanded_idx,
        _,
        num_non_exiting_tiles,
    ) = moe_sort(
        token_selected_experts=ids,
        token_final_scales=weights,
        num_experts=EXPERTS,
        top_k=TOP_K,
        tile_tokens_dim=tile,
    )
    metadata = (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
    )
    gated = torch.empty(
        (permuted_idx_to_expanded_idx.shape[0], INTERMEDIATE),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    experts.gather_gemm(
        hidden,
        up_gate,
        *metadata,
        gated,
        top_k=TOP_K,
        activation=activation,
        tactic=gather,
    )

    # FC1 leaves unspecified values in the rows past each tile's limit and
    # in tiles past the non-exiting count; no route row may depend on them.
    rows = torch.arange(gated.shape[0], device=DEVICE)
    tiles = rows // tile
    limits = tile_idx_to_mn_limit[
        tiles.clamp(max=tile_idx_to_mn_limit.numel() - 1)
    ]
    gated[(tiles >= num_non_exiting_tiles) | (rows >= limits)] = float("nan")
    routes = torch.full(
        (TOKENS * TOP_K, HIDDEN),
        float("nan"),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    experts.route_gemm(gated, down, *metadata, routes, tactic=route)

    projected = torch.einsum(
        "th,tkoh->tko", hidden.double(), up_gate.double()[ids.long()]
    )
    up, gate = projected.split(INTERMEDIATE, dim=-1)
    activated = (
        F.silu(gate)
        if activation == "silu"
        else F.gelu(gate, approximate="tanh")
    ) * up
    expected = torch.einsum(
        "tki,tkhi->tkh", activated, down.double()[ids.long()]
    )

    # BF16 roundings of the gated product and of each route row, plus the
    # kernel's approximate activation, whose relative error is below one
    # BF16 unit on positive inputs. Every row is written: none stays NaN.
    torch.testing.assert_close(
        routes.double().view(TOKENS, TOP_K, HIDDEN),
        expected,
        rtol=_gamma(3),
        atol=0,
    )
