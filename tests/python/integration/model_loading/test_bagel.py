"""BAGEL public text and image capabilities share checkpoint-backed experts."""

import pytest
import torch
from torch.nn import functional as F

from tests.python.fixtures.checkpoints import bagel_checkpoint, load_bagel
from uniserve.media import image
from uniserve.model import TextInput, TextSize
from uniserve.nn.attention import PagedInput, SequenceLengths, VarlenInput
from uniserve.runtime import ExecutionContext, PrefixCache

pytestmark = pytest.mark.integration


def test_text_prefill_decode_and_zero_query_match_qwen_equations(tmp_path):
    reference, _, config = bagel_checkpoint(tmp_path)
    model = load_bagel(tmp_path, config)
    assert model.text.backbone is model.denoiser.backbone
    tokens = torch.tensor([1, 4, 8, 3])
    with torch.no_grad():
        expected = reference(tokens[None]).logits[0]
    with PrefixCache(
        model.text.cache_config, num_blocks=1, block_size=4, device="cpu"
    ) as cache:
        with ExecutionContext(
            model.text, cache=cache, attention="torch"
        ) as context:
            context.prepare(TextSize(4, 1))
            for start, stop in ((0, 3), (3, 4), (4, 4)):
                batch = PagedInput.from_blocks(
                    blocks=((0,),),
                    query_lengths=(stop - start,),
                    prefix_lengths=(start,),
                    block_size=4,
                    causal=True,
                    device="cpu",
                )
                context.bind_attention(batch)
                hidden = model.text(
                    TextInput(
                        tokens[start:stop], torch.arange(start, stop), batch
                    )
                )
                actual = model.text.compute_logits(
                    hidden, token_indices=torch.arange(stop - start)
                ).gather()
                torch.testing.assert_close(
                    actual, expected[start:stop], rtol=2e-2, atol=2e-3
                )


def test_image_markers_use_text_expert_and_flow_preserves_residual(tmp_path):
    _, state, config = bagel_checkpoint(tmp_path)
    model = load_bagel(tmp_path, config)
    size = image.Config(8, 8)
    from uniserve_worker.bootstrap.inputs import image_builder

    factory = image_builder(model)
    sample = torch.empty((4, 8), dtype=torch.bfloat16)
    factory.initialize(size, seed=71, out=sample)
    expected_noise = torch.randn(
        (4, 8),
        generator=torch.Generator().manual_seed(71),
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(sample, expected_noise, rtol=0, atol=0)
    timestep = torch.tensor(0.5)
    lengths = SequenceLengths.from_lengths((6,), device="cpu")
    inputs = factory.bind(
        samples=(sample,),
        sizes=(size,),
        timesteps=(timestep,),
        positions=(factory.positions(size, 9, device="cpu"),),
        attention=VarlenInput(lengths, lengths, (False,)),
        step_index=0,
    )
    before = sample.clone()
    # GLIDE time features and the checkpoint's zero-update flow residual give
    # an independent closed form for the denoiser prediction, including casts.
    frequencies = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(128) / 128
    )
    angles = timestep * frequencies
    time = torch.cat((angles.cos(), angles.sin())).bfloat16().expand(4, -1)
    time = F.linear(
        time,
        state["time_embedder.mlp.0.weight"],
        state["time_embedder.mlp.0.bias"],
    )
    time = F.linear(
        F.silu(time),
        state["time_embedder.mlp.2.weight"],
        state["time_embedder.mlp.2.bias"],
    )
    hidden = F.linear(sample, state["vae2llm.weight"], state["vae2llm.bias"])
    hidden = hidden + time + state["latent_pos_embed.pos_embed"][[0, 1, 4, 5]]
    normalized = (
        hidden.float()
        * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    expected = F.linear(
        normalized, state["llm2vae.weight"], state["llm2vae.bias"]
    )
    with torch.no_grad():
        prediction = model.denoiser(
            inputs, state={}, constants={}, workspace={}
        )["image"][0]
    torch.testing.assert_close(prediction.tensor, expected, rtol=0, atol=0)
    torch.testing.assert_close(sample, before, rtol=0, atol=0)
    assert prediction.layout.shape == sample.shape
