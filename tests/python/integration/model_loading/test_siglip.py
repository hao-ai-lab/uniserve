"""Packed SigLIP image boundaries and checkpoint projections agree with HF."""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.model import PatchEncoder, VisionInput
from uniserve.nn.functional import patchify
from uniserve_models import siglip

pytestmark = pytest.mark.integration


def test_variable_image_grids_match_independent_encoder(tmp_path):
    torch.manual_seed(474)
    config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=8,
        patch_size=2,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
    )
    reference = SiglipVisionModel(config).eval()
    state = {}
    for name, value in reference.state_dict().items():
        name = name.removeprefix("vision_model.")
        if name.startswith("head."):
            continue
        if name == "embeddings.patch_embedding.weight":
            value = value.permute(0, 2, 3, 1).flatten(1).contiguous()
        state[name] = value
    save_file(state, tmp_path / "model.safetensors")
    config = siglip.Config(2, 8, 3, siglip.TransformerConfig(32, 4, 48, 2, 1e-6))
    model = loading.load_model(
        siglip.Encoder,
        config,
        checkpoint=(checkpoint.Config("vision").resolve(tmp_path, io=loading.Config()),),
        mapping=lambda model: (
            weights.ModuleMapping(
                model,
                "vision",
                lambda reader: siglip.assignments(model, reader),
                frozenset(dict(model.named_parameters())),
            ),
        ),
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model
    encoder = PatchEncoder(
        model, nn.Identity(), patch_size=2, downsample=1, output_size=32, output_dtype=torch.float32
    )
    pixels = (torch.randn(3, 8, 8), torch.randn(3, 8, 4), torch.randn(3, 4, 8))
    expected = []
    with torch.no_grad():
        # HF's ordinary encoder consumes the selected learned grid positions;
        # no interpolation is part of the NaViT checkpoint's grid convention.
        for value in pixels:
            rows, columns = value.shape[-2] // 2, value.shape[-1] // 2
            features = reference.embeddings.patch_embedding(value[None]).flatten(2).transpose(1, 2)
            positions = (torch.arange(rows)[:, None] * 4 + torch.arange(columns)).flatten()
            features = features + reference.embeddings.position_embedding(positions)[None]
            hidden = reference.encoder(inputs_embeds=features).last_hidden_state
            expected.append(reference.post_layernorm(hidden)[0])
        actual = encoder.encode(VisionInput(pixels, (None,) * 3, (None,) * 3))
        shapes = tuple((value.shape[-2] // 2, value.shape[-1] // 2) for value in pixels)
        packed = encoder.encode(
            VisionInput(
                tuple(patchify(value, patch_size=2) for value in pixels),
                tuple(torch.tensor((shape,)) for shape in shapes),
                shapes,
            )
        )
        for output, patch_output, target in zip(actual, packed, expected, strict=True):
            torch.testing.assert_close(output, target, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(patch_output, target, rtol=1e-5, atol=1e-6)
    assert encoder.encode(VisionInput((), (), ())) == ()


@pytest.mark.gpu
@torch.inference_mode()
def test_image_encoding_replay_reads_updated_pixels():
    from uniserve.model import TextSize
    from uniserve.runtime import CUDAGraph, ExecutionContext

    torch.manual_seed(987)
    config = siglip.Config(2, 8, 3, siglip.TransformerConfig(32, 4, 48, 1, 1e-6))
    encoder = (
        PatchEncoder(
            siglip.Encoder(config),
            nn.Identity(),
            patch_size=2,
            downsample=1,
            output_size=32,
            output_dtype=torch.bfloat16,
        )
        .to(device="cuda:0", dtype=torch.bfloat16)
        .eval()
    )
    pixels = (
        torch.randn(3, 4, 4, device="cuda:0", dtype=torch.bfloat16),
        torch.randn(3, 4, 4, device="cuda:0", dtype=torch.bfloat16),
    )
    inputs = VisionInput(pixels, (None, None), (None, None))
    with ExecutionContext(encoder) as context:
        context.prepare(TextSize(8, 2))
        encoder.encode(inputs)
        with CUDAGraph(context=context) as graph:
            graph.capture(lambda: encoder.encode(inputs))
            pixels[0].mul_(0.5)
            pixels[1].add_(0.25)
            result = graph.replay()
            expected = encoder.encode(inputs)
            for actual, target in zip(result, expected, strict=True):
                torch.testing.assert_close(actual, target, rtol=0, atol=0)
            torch.cuda.synchronize()
