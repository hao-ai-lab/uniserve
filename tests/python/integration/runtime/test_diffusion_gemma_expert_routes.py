"""DiffusionGemma's expert routes enter its feed-forward sandwich exactly.

Layer 0's MoE block of each published checkpoint is loaded through the
public model loader. For every native expert provider that serves the
checkpoint's representation, the post-feed-forward sandwich normalization
of the experts' routes (``FusedMoE(..., combine=False)``) equals, bit for
bit, the normalization of their combined output, eager and replayed, so
evaluating the combination inside the sandwich launch changes no stream
value. Each provider prepares a freshly loaded block, since preparation
places the weights in the provider's physical row order.

The checkpoint directories come from ``UNISERVE_DIFFUSION_GEMMA_MODEL``
(BF16) and ``UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL`` (NVFP4).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from uniserve.model import TextSize
from uniserve.nn.functional import sandwich_rms_norm
from uniserve.runtime import CUDAStream, ExecutionContext
from uniserve.runtime.backends import moe as backends
from uniserve.runtime.cuda_graph import CUDAGraph
from uniserve_models import loading as models

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("diffusion_gemma"),
    pytest.mark.slow,
]

CHECKPOINTS = {
    "bf16": "UNISERVE_DIFFUSION_GEMMA_MODEL",
    "nvfp4": "UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL",
}
BLOCK = "text.backbone.layers.0.moe"
EPS = 1e-6
TOKENS = 512
DEVICE = torch.device("cuda:0")


def _moe_block(precision: str):
    value = os.environ.get(CHECKPOINTS[precision], "")
    if not value or not Path(value).is_dir():
        pytest.fail(
            f"{CHECKPOINTS[precision]} must name the {precision} checkpoint"
        )
    config = models.read_config(value, modules=frozenset({BLOCK}))
    return models.load_model(config, device=DEVICE).model.get_submodule(BLOCK)


def _providers(experts):
    """Every native provider that serves the experts' representation."""
    served = []
    for name in ("cutlass", "cutedsl", "trtllm"):
        try:
            backends.resolve(name, module=experts, device=DEVICE)
        except ValueError:
            continue
        served.append(name)
    return served


@pytest.mark.parametrize("precision", sorted(CHECKPOINTS))
@torch.inference_mode()
def test_sandwich_of_expert_routes_equals_sandwich_of_their_sum(precision):
    providers = _providers(_moe_block(precision).experts)
    assert providers, "no native expert provider serves the checkpoint"
    for provider in providers:
        _check_provider(_moe_block(precision), provider)


def _check_provider(block, provider):
    experts = block.experts
    generator = torch.Generator(device=DEVICE).manual_seed(29)

    def normal(*shape, scale=1.0):
        return (
            torch.randn(shape, generator=generator, device=DEVICE) * scale
        ).to(torch.bfloat16)

    hidden = normal(TOKENS, experts.hidden_size)
    routed = normal(
        TOKENS, experts.hidden_size, scale=experts.hidden_size**-0.5
    )
    stream = normal(TOKENS, experts.hidden_size, scale=4.0)
    dense = normal(TOKENS, experts.hidden_size)
    dense_weight = normal(experts.hidden_size, scale=0.1) + 1
    weight = normal(experts.hidden_size, scale=0.1) + 1
    scale = torch.tensor([0.5], dtype=torch.bfloat16, device=DEVICE)

    def feed_forward(expert_output):
        result, _ = sandwich_rms_norm(
            stream,
            (
                (dense, dense_weight),
                (expert_output, block.output_norm.weight),
            ),
            weight,
            eps=EPS,
            scale=scale,
        )
        return result

    cuda_stream = CUDAStream.external(torch.cuda.Stream(device=DEVICE))
    cuda_stream.wait(torch.cuda.current_stream(DEVICE))
    with (
        cuda_stream,
        ExecutionContext(block, stream=cuda_stream, moe=provider) as context,
    ):
        context.prepare(TextSize(TOKENS, 1))
        with context.activate():
            ids, weights = block.router(routed)
            expected = feed_forward(experts(hidden, ids, weights))
            routes = experts(hidden, ids, weights, combine=False)
            assert torch.equal(feed_forward(routes), expected), provider
        with CUDAGraph(context=context) as graph:
            graph.capture(
                lambda: feed_forward(
                    experts(hidden, ids, weights, combine=False)
                )
            )
            assert torch.equal(graph.replay(), expected), provider
    torch.cuda.synchronize(DEVICE)
