"""H3 checkpoint conditioning preserves independent document equations.

It also preserves independent timestep equations.
"""

import pytest
import torch
from diffusers.models.embeddings import (
    TimestepEmbedding as ReferenceTimestepEmbedding,
)
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3TokenRefiner,
)
from safetensors.torch import save_file

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve_models.minimax_h3 import TransformerConfig
from uniserve_models.minimax_h3.conditioning import Conditioner, assignments
from uniserve_models.minimax_h3.modulation import TimestepEmbedding

pytestmark = pytest.mark.integration


def _config():
    return TransformerConfig(
        hidden_size=64,
        num_attention_heads=2,
        head_dim=32,
        num_hidden_layers=2,
        num_refiner_layers=2,
        intermediate_size=128,
        text_dim=40,
        frequency_dim=16,
        time_hidden_dim=64,
        time_dim=32,
        rope_frequency_dim=4,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_conditioning_loader_preserves_variable_documents(tmp_path, dtype):
    config = _config()
    torch.manual_seed(108)
    projection = (
        torch.nn.Linear(config.text_dim, config.hidden_size).to(dtype).eval()
    )
    reference = (
        MiniMaxH3TokenRefiner(
            config.hidden_size,
            config.num_attention_heads,
            config.head_dim,
            config.intermediate_size,
            config.num_refiner_layers,
            config.norm_eps,
            config.qk_norm_eps,
            config.norm_eps,
        )
        .to(dtype)
        .eval()
    )
    values = {
        **{
            f"context_embedder.{name}": value
            for name, value in projection.state_dict().items()
        },
        **{
            f"token_refiner.{name}": value
            for name, value in reference.state_dict().items()
        },
    }
    save_file(values, tmp_path / "model.safetensors")

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: assignments(model, reader),
                frozenset(name for name, _ in model.named_parameters()),
            ),
        )

    model = loading.load_model(
        Conditioner,
        config,
        checkpoint=(
            checkpoint.Config().resolve(tmp_path, io=loading.Config()),
        ),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(dtype=dtype),
    ).model
    features = tuple(
        torch.randn(tokens, config.text_dim, dtype=dtype)
        for tokens in (7, 3, 7)
    )
    with torch.no_grad():
        expected = tuple(
            reference(projection(value.unsqueeze(0)))[0] for value in features
        )
        actual = model.encode(features)
    tolerance = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    for value, result in zip(expected, actual, strict=True):
        torch.testing.assert_close(
            result, value, rtol=tolerance[0], atol=tolerance[1]
        )

    # Documents padded to one row count, with large values in their padding
    # rows, refine their text rows as the unpadded documents do.
    padded = torch.randn(len(features), 10, config.text_dim, dtype=dtype) * 100
    for index, value in enumerate(features):
        padded[index, : value.shape[0]] = value
    lengths = torch.tensor(
        [value.shape[0] for value in features], dtype=torch.int32
    )
    with torch.no_grad():
        refined = model.encode(tuple(padded.unbind()), lengths=lengths)
    for value, result in zip(expected, refined, strict=True):
        torch.testing.assert_close(
            result[: value.shape[0]],
            value,
            rtol=tolerance[0],
            atol=tolerance[1],
        )


def test_timestep_projection_preserves_both_modality_coordinates():
    config = _config()
    torch.manual_seed(302)
    reference = ReferenceTimestepEmbedding(
        config.frequency_dim, config.time_hidden_dim, out_dim=config.time_dim
    ).eval()
    model = TimestepEmbedding(config)
    with torch.no_grad():
        model.video_projection[0].weight.copy_(reference.linear_1.weight)
        model.video_projection[0].bias.copy_(reference.linear_1.bias)
        model.video_projection[2].weight.copy_(reference.linear_2.weight)
        model.video_projection[2].bias.copy_(reference.linear_2.bias)
        times = torch.tensor([[0.0, 0.0], [0.125, 0.4], [0.7, 0.95]])
        features = get_timestep_embedding(
            times.flatten(),
            config.frequency_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        expected = reference(features).reshape(*times.shape, config.time_dim)
        torch.testing.assert_close(model(times), expected, rtol=1e-5, atol=1e-6)
