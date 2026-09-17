"""Image denoisers assemble their own typed input from borrowed views.

Execution calls `bind_inputs` on whichever denoiser a model exposes and passes
the same borrowed numerical views to each. These run on CPU so an implementation
that reads a field the contract does not carry is caught without a GPU.
"""

from __future__ import annotations

import json

import pytest
import torch

from tests.python.fixtures.model_metadata import neo_metadata
from uniserve.loading import Config as IOConfig
from uniserve.media import image
from uniserve.model import ImageDenoiser, LatentInput
from uniserve.nn.attention import DenseInput
from uniserve.processing import BranchSource

pytestmark = pytest.mark.unit


def _sensenova_denoiser(tmp_path):
    from uniserve_models.sensenova_u1 import read_config
    from uniserve_models.sensenova_u1.denoiser import Denoiser
    from uniserve_models.sensenova_u1.transformer import Transformer

    (tmp_path / "config.json").write_text(json.dumps(neo_metadata()))
    config = read_config(tmp_path, IOConfig())
    with torch.device("meta"):
        return Denoiser(config, Transformer(config.text))


def _stub_denoiser(tmp_path):
    from uniserve_models.stub import Config, Model

    del tmp_path
    return Model(Config()).denoiser


@pytest.mark.parametrize(
    "build", [_sensenova_denoiser, _stub_denoiser], ids=["sensenova_u1", "stub"]
)
def test_image_denoisers_bind_their_own_typed_input(build, tmp_path):
    denoiser = build(tmp_path)
    assert isinstance(denoiser, ImageDenoiser)
    assert isinstance(denoiser.framing_tokens, int)
    assert isinstance(denoiser.image_unconditional, BranchSource)
    assert denoiser.max_sequence_tokens > 0

    size = image.Config(
        height=denoiser.downsample * 2, width=denoiser.downsample * 2
    )
    rows, width = denoiser.latent_shape("image", size)
    sample = torch.zeros((rows, width), dtype=torch.float32)
    length = rows + denoiser.framing_tokens

    bound = denoiser.bind_inputs(
        latents={"image": (LatentInput(sample, torch.zeros(())),)},
        sizes=(size,),
        step_index=0,
        positions=(torch.zeros((3, length), dtype=torch.int64),),
        sequence_lengths=(length,),
        attention=DenseInput(causal=False, mask=None),
    )

    assert bound.latents["image"][0].tensor is sample
    assert bound.sizes == (size,)
    assert bound.step_index == 0
