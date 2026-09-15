"""BAGEL public text and image capabilities share checkpoint-backed experts."""

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.media import image
from uniserve.model import TextInput, TextSize
from uniserve.nn.attention import PagedInput, SequenceLengths, VarlenInput
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve_models import bagel, siglip
from uniserve_models.bagel import vae

pytestmark = pytest.mark.integration


def _checkpoint(root):
    torch.manual_seed(662)
    config = Qwen3Config(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
        attention_bias=True,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).bfloat16().eval()
    # BAGEL has Q/K/V biases and an unbiased attention output contraction.
    for layer in reference.model.layers:
        layer.self_attn.o_proj.register_parameter("bias", None)
    state = {}
    for name, value in reference.state_dict().items():
        state["language_model." + name] = value
        if name == "model.norm.weight":
            state["language_model.model.norm_moe_gen.weight"] = value.clone()
        elif name.startswith("model.layers."):
            parts = name.split(".")
            if parts[3] == "self_attn":
                parts[4] += "_moe_gen"
            else:
                parts[3] += "_moe_gen"
            # A zero-update flow expert makes its residual identity observable
            # while text markers still traverse a nonzero independent expert.
            state["language_model." + ".".join(parts)] = torch.zeros_like(value)
    for prefix, out_features, in_features in (
        ("vae2llm", 32, 8),
        ("llm2vae", 8, 32),
        ("time_embedder.mlp.0", 32, 256),
        ("time_embedder.mlp.2", 32, 32),
    ):
        state[prefix + ".weight"] = (torch.randn(out_features, in_features) * 0.02).bfloat16()
        state[prefix + ".bias"] = (torch.randn(out_features) * 0.01).bfloat16()
    state["latent_pos_embed.pos_embed"] = (torch.randn(16, 32) * 0.02).bfloat16()
    # Declared, unselected vision fields can coexist in the same primary file.
    state["vit_pos_embed.pos_embed"] = torch.randn(16, 32).bfloat16()
    save_file(state, root / "ema.safetensors")
    config = bagel.Config(
        bagel.TransformerConfig(32, 48, 2, 4, 2, 37, 1e-6, 1_000_000.0, 8, True, 64),
        siglip.Config(2, 8, 3, siglip.TransformerConfig(32, 4, 48, 1, 1e-6)),
        vae.Config(8, 3, 2, 32, 3, (1, 1), 1, 2, 0.5, 0.25),
        35,
        36,
        2,
        4,
        1.0,
        "gelu_pytorch_tanh",
    )
    return reference, state, config


def _load(root, config):
    return loading.load_model(
        bagel.Model,
        config,
        checkpoint=(
            checkpoint.Config("primary", filenames=("ema.safetensors",)).resolve(
                root, io=loading.Config()
            ),
        ),
        mapping=bagel.checkpoint_mappings,
        device="cpu",
        weights=weights.Config(),
        modules=frozenset(("text", "denoiser")),
    ).model


def test_text_prefill_decode_and_zero_query_match_qwen_equations(tmp_path):
    reference, _, config = _checkpoint(tmp_path)
    model = _load(tmp_path, config)
    assert model.text.backbone is model.denoiser.backbone
    tokens = torch.tensor([1, 4, 8, 3])
    with torch.no_grad():
        expected = reference(tokens[None]).logits[0]
    with PrefixCache(model.text.cache_config, num_blocks=1, block_size=4, device="cpu") as cache:
        with ExecutionContext(model.text, cache=cache, attention="torch") as context:
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
                hidden = model.text(TextInput(tokens[start:stop], torch.arange(start, stop), batch))
                actual = model.text.compute_logits(
                    hidden, token_indices=torch.arange(stop - start)
                ).gather()
                torch.testing.assert_close(actual, expected[start:stop], rtol=2e-2, atol=2e-3)


def test_image_markers_use_text_expert_and_flow_preserves_residual(tmp_path):
    _, state, config = _checkpoint(tmp_path)
    model = _load(tmp_path, config)
    size = image.Config(8, 8)
    from uniserve_worker.bootstrap.inputs import image_inputs

    factory = image_inputs(model)
    sample = torch.empty((4, 8), dtype=torch.bfloat16)
    factory.initialize(size, seed=71, out=sample)
    expected_noise = torch.randn(
        (4, 8), generator=torch.Generator().manual_seed(71), dtype=torch.bfloat16
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
    frequencies = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(128) / 128)
    angles = timestep * frequencies
    time = torch.cat((angles.cos(), angles.sin())).bfloat16().expand(4, -1)
    time = F.linear(time, state["time_embedder.mlp.0.weight"], state["time_embedder.mlp.0.bias"])
    time = F.linear(
        F.silu(time), state["time_embedder.mlp.2.weight"], state["time_embedder.mlp.2.bias"]
    )
    hidden = F.linear(sample, state["vae2llm.weight"], state["vae2llm.bias"])
    hidden = hidden + time + state["latent_pos_embed.pos_embed"][[0, 1, 4, 5]]
    normalized = (
        hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    expected = F.linear(normalized, state["llm2vae.weight"], state["llm2vae.bias"])
    with torch.no_grad():
        prediction = model.denoiser(inputs, state={}, constants={}, workspace={})["image"][0]
    torch.testing.assert_close(prediction.tensor, expected, rtol=0, atol=0)
    torch.testing.assert_close(sample, before, rtol=0, atol=0)
    assert prediction.layout.shape == sample.shape
