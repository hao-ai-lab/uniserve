"""Public U1 checkpoint loading preserves axial attention and image velocity."""

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from tests.python.fixtures.checkpoints import sensenova_checkpoint
from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.media import image
from uniserve.model import LatentInput, TextInput, TextSize, VisionInput
from uniserve.nn.attention import (
    AttentionBatch,
    PagedInput,
    SequenceLengths,
    VarlenInput,
)
from uniserve.runtime import ExecutionContext, PrefixCache
from uniserve_models import sensenova_u1 as u1

pytestmark = pytest.mark.integration


def _rms(x, weight):
    return (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(x.dtype) * weight


def _linear(x, state, name):
    return F.linear(x, state[name + ".weight"], state.get(name + ".bias"))


def _text(tokens, state):
    hidden = F.embedding(
        tokens, state["language_model.model.embed_tokens.weight"]
    )
    for layer in range(2):
        prefix = f"language_model.model.layers.{layer}."
        normalized = _rms(hidden, state[prefix + "input_layernorm.weight"])
        projections = []
        for name, heads in (("q", 4), ("k", 2), ("v", 2)):
            value = _linear(
                normalized, state, prefix + f"self_attn.{name}_proj"
            ).reshape(-1, heads, 8)
            if name != "v":
                # Temporal and both spatial axes have separate four-value RMS
                # reductions. Text spatial coordinates are zero.
                value = torch.cat(
                    tuple(
                        _rms(
                            part,
                            state[
                                prefix + f"self_attn.{name}_norm{suffix}.weight"
                            ],
                        )
                        for part, suffix in zip(
                            value.chunk(2, -1), ("", "_hw"), strict=True
                        )
                    ),
                    -1,
                )
                temporal, spatial = value.split(4, -1)
                angles = torch.arange(tokens.numel()).float()[
                    :, None, None
                ] * torch.tensor([1.0, 0.01])
                left, right = temporal.float().chunk(2, -1)
                temporal = torch.cat(
                    (
                        left * angles.cos() - right * angles.sin(),
                        right * angles.cos() + left * angles.sin(),
                    ),
                    -1,
                ).to(value.dtype)
                value = torch.cat((temporal, spatial), -1)
            projections.append(value.transpose(0, 1))
        q, k, v = projections
        logits = (
            q.float()
            @ k.float().repeat_interleave(2, 0).transpose(-1, -2)
            / 8**0.5
        )
        mask = torch.ones(
            tokens.numel(), tokens.numel(), dtype=torch.bool
        ).triu(1)
        probability = logits.masked_fill(mask, -torch.inf).softmax(-1)
        attended = (probability @ v.float().repeat_interleave(2, 0)).to(
            hidden.dtype
        )
        hidden = hidden + _linear(
            attended.transpose(0, 1).reshape(-1, 32),
            state,
            prefix + "self_attn.o_proj",
        )
        normalized = _rms(
            hidden, state[prefix + "post_attention_layernorm.weight"]
        )
        gated = F.silu(
            _linear(normalized, state, prefix + "mlp.gate_proj")
        ) * _linear(normalized, state, prefix + "mlp.up_proj")
        hidden = hidden + _linear(gated, state, prefix + "mlp.down_proj")
    return _linear(
        _rms(hidden, state["language_model.model.norm.weight"]),
        state,
        "language_model.lm_head",
    )


def _vision(pixels, state, prefix):
    prefix += ".embeddings."
    features = F.gelu(
        F.conv2d(
            pixels,
            state[prefix + "patch_embedding.weight"],
            state[prefix + "patch_embedding.bias"],
            stride=2,
        )
    )
    _, channels, height, width = features.shape
    values = features.permute(0, 2, 3, 1).float()
    y, x = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    rotated = []
    for part, positions in zip(values.chunk(2, -1), (x, y), strict=True):
        angles = positions[..., None].float() / (
            10000.0 ** (torch.arange(4).float() / 4)
        )
        pairs = part.reshape(1, height, width, 4, 2)
        first, second = pairs.unbind(-1)
        rotated.append(
            torch.stack(
                (
                    first * angles.cos() - second * angles.sin(),
                    second * angles.cos() + first * angles.sin(),
                ),
                -1,
            ).flatten(-2)
        )
    features = torch.cat(rotated, -1).to(pixels.dtype).permute(0, 3, 1, 2)
    return (
        F.conv2d(
            features,
            state[prefix + "dense_embedding.weight"],
            state[prefix + "dense_embedding.bias"],
            stride=2,
        )
        .permute(0, 2, 3, 1)
        .reshape(-1, 32)
    )


def _time(value, state, prefix):
    angles = value.float().reshape(-1, 1) * torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(128) / 128
    )
    features = torch.cat((angles.cos(), angles.sin()), -1).to(
        state[prefix + ".mlp.0.weight"].dtype
    )
    return _linear(
        F.silu(_linear(features, state, prefix + ".mlp.0")),
        state,
        prefix + ".mlp.2",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_text_matches_axial_attention_equations(tmp_path, dtype):
    model, state = sensenova_checkpoint(tmp_path, dtype)
    tokens = torch.tensor([1, 5, 2, 8])
    expected = _text(tokens, state)
    with PrefixCache(
        model.text.cache_config, num_units=1, block_size=4, device="cpu"
    ) as cache:
        with ExecutionContext(
            model.text, cache=cache, attention="torch"
        ) as context:
            context.prepare(TextSize(4, 1))
            for start, stop in ((0, 3), (3, 4), (4, 4)):
                attention = AttentionBatch.single(
                    PagedInput.from_blocks(
                        blocks=((0,),),
                        query_lengths=(stop - start,),
                        prefix_lengths=(start,),
                        block_size=4,
                        causal=True,
                        device="cpu",
                    )
                )
                context.bind_attention(attention)
                hidden = model.text(
                    TextInput(
                        tokens[start:stop], torch.arange(start, stop), attention
                    )
                )
                actual = model.text.compute_logits(
                    hidden, token_indices=torch.arange(stop - start)
                ).gather()
                torch.testing.assert_close(
                    actual,
                    expected[start:stop],
                    rtol=2e-2 if dtype == torch.bfloat16 else 1e-5,
                    atol=2e-3 if dtype == torch.bfloat16 else 1e-6,
                )


@pytest.mark.parametrize("worker_inputs", (False, True))
def test_images_and_velocity_follow_independent_equations(
    tmp_path, worker_inputs
):
    model, state = sensenova_checkpoint(tmp_path, torch.bfloat16)
    pixels = torch.linspace(-1, 1, 3 * 8 * 8).reshape(1, 3, 8, 8).bfloat16()
    grid = torch.tensor([[4, 4]])
    actual = model.vision_encoder.encode(
        VisionInput((pixels[0],), (None,), (None,))
    )[0]
    torch.testing.assert_close(
        actual, _vision(pixels, state, "vision_model"), rtol=0, atol=0
    )
    # The numerical conditioning and current solver sample are independent
    # inputs. Their difference catches subtracting the conditioning by mistake.
    sample = torch.linspace(0.5, -0.5, 4 * 48).reshape(4, 48).bfloat16()
    timestep = torch.tensor(0.99)
    scale = torch.tensor(2.0)
    size = image.Config(8, 8)
    if worker_inputs:
        from uniserve_worker.bootstrap.inputs import image_builder

        factory = image_builder(model)
        factory.initialize(size, seed=71, out=sample)
        pixels = torch.randn(
            (1, 3, 8, 8),
            generator=torch.Generator().manual_seed(71),
            dtype=torch.bfloat16,
        )
        patches = (
            pixels.reshape(1, 3, 2, 4, 2, 4)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(4, 48)
        )
        torch.testing.assert_close(sample, patches, rtol=0, atol=0)
        scale = torch.tensor(1.0)
    hidden = _vision(pixels, state, "fm_modules.vision_model_mot_gen")
    hidden = (
        hidden
        + _time(timestep.expand(4), state, "fm_modules.timestep_embedder")
        + _time((scale / 8).expand(4), state, "fm_modules.noise_scale_embedder")
    )
    hidden = _rms(hidden, state["language_model.model.norm_mot_gen.weight"])
    predicted = _linear(
        F.gelu(_linear(hidden, state, "fm_modules.fm_head.0")),
        state,
        "fm_modules.fm_head.2",
    )
    expected = (predicted - sample) / (1 - timestep.reshape(1)).clamp_min(0.02)
    lengths = SequenceLengths.from_lengths((4,), device="cpu")
    inputs = u1.DenoiserInput(
        {"image": (LatentInput(sample, timestep),)},
        (image.Config(8, 8),),
        torch.zeros(1, dtype=torch.int64),
        (u1.ImageConditioning(pixels, grid, scale),),
        (torch.tensor([[9, 9, 9, 9], [0, 0, 1, 1], [0, 1, 0, 1]]),),
        (4,),
        AttentionBatch.single(VarlenInput(lengths, lengths, (False,))),
    )
    if worker_inputs:
        inputs = factory.bind(
            samples=(sample,),
            sizes=(size,),
            timesteps=(timestep,),
            positions=(factory.positions(size, 9, device="cpu"),),
            attention=AttentionBatch.single(
                VarlenInput(lengths, lengths, (False,))
            ),
            step=torch.zeros(1, dtype=torch.int64),
        )
    with torch.no_grad():
        prediction = model.denoiser(
            inputs, state={}, constants={}, workspace={}
        )["image"][0]
    torch.testing.assert_close(prediction.tensor, expected, rtol=0, atol=0)
    assert prediction.tensor.dtype == torch.float32


@pytest.mark.parametrize("rank", (None, 0, 1))
@torch.inference_mode()
def test_tied_vocabulary_uses_the_canonical_embedding_at_pipeline_endpoints(
    tmp_path, rank
):
    from dataclasses import replace

    from uniserve.distributed import DeviceMesh

    template, _ = sensenova_checkpoint(tmp_path, torch.float32)
    config = replace(
        template.config,
        text=replace(template.config.text, tie_word_embeddings=True),
    )
    values = torch.arange(37 * 32, dtype=torch.float32).reshape(37, 32) / 128
    save_file(
        {"language_model.model.embed_tokens.weight": values},
        tmp_path / "model.safetensors",
    )
    selected = "text.lm_head" if rank == 1 else "text.backbone.embedding"
    result = loading.load_model(
        u1.Model,
        config,
        checkpoint=(
            checkpoint.Config("primary").resolve(tmp_path, io=loading.Config()),
        ),
        mapping=u1.checkpoint_mappings,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
        meshes=None
        if rank is None
        else {
            "text": DeviceMesh(
                ranks=(0, 1), shape=(2,), axes=("pp",), rank=rank
            )
        },
        modules=frozenset({selected}),
    )
    model = result.model
    if rank in (None, 0):
        tokens = torch.tensor([1, 3, 36])
        torch.testing.assert_close(
            model.text.embed_input_ids(tokens), values[tokens], rtol=0, atol=0
        )
    if rank in (None, 1):
        hidden = torch.arange(64, dtype=torch.float32).reshape(2, 32) / 64
        logits = model.text.compute_logits(
            hidden, token_indices=torch.tensor([1, 0])
        ).gather()
        torch.testing.assert_close(
            logits, hidden[[1, 0]] @ values.T, rtol=0, atol=0
        )
