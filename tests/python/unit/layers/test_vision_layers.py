"""Conformance for shared vision building blocks."""
from __future__ import annotations

import pytest
import torch

from uniserve_worker.nn.vision import (
    MLPConnector,
    NeoVitConfig,
    NeoVitEncoder,
    PatchEmbed,
    PositionEmbedding,
    get_2d_sincos_pos_embed,
    get_flattened_position_ids_extrapolate,
    patchify_batch,
    unpatchify_batch,
)

pytestmark = pytest.mark.unit


def test_mlp_connector_is_shared_vision_infrastructure():
    connector = MLPConnector(3, 5)

    assert connector.fc1.in_features == 3
    assert connector.fc2.out_features == 5


def test_mlp_connector_forward_is_stable_for_bagel_activation():
    torch.manual_seed(7)
    shared = MLPConnector(3, 5, activation="gelu_pytorch_tanh")
    x = torch.randn(2, 4, 3)
    out = shared(x)
    assert out.shape == (2, 4, 5)
    torch.testing.assert_close(out, shared.fc2(shared.act(shared.fc1(x))))


def test_patch_embed_conv_flatten_shape_and_values():
    patch = PatchEmbed(3, 2, patch_size=2, bias=False)
    with torch.no_grad():
        patch.proj.weight.fill_(1.0)
    x = torch.ones(1, 3, 4, 4)
    out = patch(x)
    assert out.shape == (1, 4, 2)
    torch.testing.assert_close(out, torch.full((1, 4, 2), 12.0))


def test_batch_patchify_unpatchify_round_trips_sensenova_layout():
    images = torch.arange(1 * 3 * 4 * 6, dtype=torch.float32).reshape(1, 3, 4, 6)
    patches = patchify_batch(images, 2)
    assert patches.shape == (1, 6, 12)
    torch.testing.assert_close(unpatchify_batch(patches, 2, height=4, width=6), images)

    channel_first = patchify_batch(images, 2, channel_first=True)
    assert channel_first.shape == (1, 6, 12)
    assert not torch.equal(channel_first, patches)


def test_sensenova_vision_model_is_native_module_not_hf_pretrained_model():
    from transformers.modeling_utils import PreTrainedModel

    model = NeoVitEncoder(NeoVitConfig(hidden_size=8, llm_hidden_size=16, downsample_ratio=1.0))
    assert isinstance(model, torch.nn.Module)
    assert not isinstance(model, PreTrainedModel)


def test_position_embedding_supports_bagel_and_sensenova_initialization_modes():
    zeros = PositionEmbedding(2, 4, init_sincos=False)
    torch.testing.assert_close(zeros.pos_embed, torch.zeros(4, 4))

    sincos = PositionEmbedding(2, 4)
    expected = torch.from_numpy(get_2d_sincos_pos_embed(4, 2)).float()
    torch.testing.assert_close(sincos.pos_embed, expected)

    ids = get_flattened_position_ids_extrapolate(4, 6, 2, 8)
    torch.testing.assert_close(ids, torch.tensor([0, 1, 2, 8, 9, 10]))
