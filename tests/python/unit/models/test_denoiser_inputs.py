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
from uniserve.model import LatentInput
from uniserve.nn.attention import SequenceLengths, VarlenInput

pytestmark = pytest.mark.unit


def _sensenova_denoiser(tmp_path):
    from uniserve_models.sensenova_u1 import read_config
    from uniserve_models.sensenova_u1.denoiser import Denoiser
    from uniserve_models.sensenova_u1.transformer import Transformer

    (tmp_path / "config.json").write_text(json.dumps(neo_metadata()))
    config = read_config(tmp_path, IOConfig(), sources={})
    with torch.device("meta"):
        return Denoiser(config, Transformer(config.text))


def _stub_denoiser(tmp_path):
    from uniserve_models.stub import Config, Model

    del tmp_path
    return Model(Config()).denoiser


def _bind(denoiser):
    """Bind one two-by-two patch image and its framed attention sequence."""
    size = image.Config(
        height=denoiser.downsample * 2, width=denoiser.downsample * 2
    )
    rows, width = denoiser.latent_shape("image", size)
    sample = torch.zeros((rows, width), dtype=torch.float32)
    length = rows + denoiser.framing_tokens
    lengths = SequenceLengths.from_lengths((length,), device="cpu")

    step = torch.zeros(1, dtype=torch.int64)
    bound = denoiser.bind_inputs(
        latents={"image": (LatentInput(sample, torch.zeros(())),)},
        sizes=(size,),
        step=step,
        positions=(torch.zeros((3, length), dtype=torch.int64),),
        sequence_lengths=(length,),
        attention=VarlenInput(lengths, lengths, (False,)),
    )
    return bound, sample, size, step


@pytest.mark.parametrize(
    "build", [_sensenova_denoiser, _stub_denoiser], ids=["sensenova_u1", "stub"]
)
def test_image_denoisers_bind_borrowed_latents(build, tmp_path):
    bound, sample, size, step = _bind(build(tmp_path))

    # Solver updates write the resident latent in place, and a captured step
    # reads the step index as device data, so the bound input must borrow
    # both rather than hold copies.
    assert bound.latents["image"][0].tensor is sample
    assert bound.sizes == (size,)
    assert bound.step is step


def test_bound_input_drives_one_prediction_per_latent(tmp_path):
    denoiser = _stub_denoiser(tmp_path)
    bound, sample, _, _ = _bind(denoiser)

    prediction = denoiser(bound, state={}, constants={}, workspace={})

    (output,) = prediction["image"]
    assert output.tensor.shape == sample.shape
    assert output.tensor.dtype == denoiser.prediction_dtype
    assert output.layout.shape == tuple(sample.shape)
