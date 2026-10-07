"""Public loading accepts nonresident source records.

It rejects unknown names.
"""

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)
from safetensors.torch import save_file

from tests.python.fixtures.checkpoints import bagel_checkpoint
from tests.python.fixtures.h3 import dmd_denoiser
from uniserve import loading
from uniserve.distributed import DeviceMesh
from uniserve.loading import checkpoint, weights
from uniserve_models import bagel
from uniserve_models.minimax_h3 import Transformer, TransformerConfig
from uniserve_models.minimax_h3.config import TRANSFORMER_FIELDS
from uniserve_models.minimax_h3.weights import transformer_component

pytestmark = pytest.mark.unit


def _pipeline(rank):
    return DeviceMesh(ranks=(0, 1), rank=rank, shape=(2, 1), axes=("pp", "tp"))


@pytest.mark.parametrize("rank", [0, 1])
def test_bagel_loading_rejects_unknown_records_on_each_pipeline_stage(
    tmp_path, rank
):
    _, state, config = bagel_checkpoint(tmp_path)

    def load():
        return loading.load_model(
            bagel.Model,
            config,
            checkpoint=(
                checkpoint.Config(
                    "primary", filenames=("ema.safetensors",)
                ).resolve(tmp_path, io=loading.Config()),
            ),
            mapping=bagel.checkpoint_mappings,
            device="cpu",
            modules=frozenset(("text", "denoiser")),
            meshes={"text": _pipeline(rank), "denoiser": _pipeline(rank)},
        )

    loaded = load()
    skipped = {name for report in loaded.reports for name in report.skipped}
    nonresident = (
        "language_model.model.layers.1.input_layernorm.weight"
        if rank == 0
        else "language_model.model.embed_tokens.weight"
    )
    assert nonresident in skipped
    state.update(
        {
            "language_model.model.layers.1.unknown.weight": torch.zeros(1),
            "unknown.weight": torch.zeros(1),
        }
    )
    save_file(state, tmp_path / "ema.safetensors")
    with pytest.raises(
        RuntimeError,
        match="unexpected=.*language_model.model.layers.1.unknown.weight.*unknown.weight",
    ):
        load()


@pytest.mark.parametrize("rank", [0, 1])
def test_h3_loading_rejects_unknown_records_on_each_pipeline_stage(
    tmp_path, rank
):
    config = TransformerConfig(
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_refiner_layers=1,
        intermediate_size=64,
        text_dim=24,
        frequency_dim=16,
        time_hidden_dim=32,
        time_dim=16,
        rope_frequency_dim=4,
    )
    native = MiniMaxH3Transformer3DModel(
        **{
            source: getattr(config, target)
            for source, target in TRANSFORMER_FIELDS.items()
        },
        patch_size=(1, 2, 2),
        final_norm_eps=config.norm_eps,
    )
    source = native.state_dict()
    for index in range(config.num_hidden_layers):
        source[f"transformer_blocks.{index}.attn.to_gate_compress.weight"] = (
            torch.zeros(
                config.num_attention_heads * config.head_dim, config.hidden_size
            )
        )
    save_file(source, tmp_path / "model.safetensors")

    denoiser = dmd_denoiser(config)

    def load():
        return loading.load_model(
            lambda value: Transformer(
                value,
                attention=denoiser.attention,
                entries=8,
                steps=4,
                groups=2,
            ),
            # The FastH3 student's transformer, which rounds once.
            denoiser.transformer,
            checkpoint=(
                checkpoint.Config("denoiser").resolve(
                    tmp_path, io=loading.Config()
                ),
            ),
            mapping=lambda model: (
                transformer_component(model, denoiser, "denoiser"),
            ),
            device="cpu",
            meshes={"": _pipeline(rank)},
            weights=weights.Config(),
        )

    loaded = load()
    skipped = {name for report in loaded.reports for name in report.skipped}
    assert f"transformer_blocks.{1 - rank}.attn.to_q.weight" in skipped
    unknown = (
        "time_embedder.unknown.weight",
        "transformer_blocks.50.attn.to_q.weight",
        "transformer_blocks.0.adaln_proj.unknown.weight",
        "token_refiner.unknown.weight",
    )
    source.update(dict.fromkeys(unknown, torch.zeros(1)))
    # safetensors requires independently owned values for distinct records.
    save_file(
        {name: value.clone() for name, value in source.items()},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(RuntimeError) as raised:
        load()
    assert "unexpected=" in str(raised.value)
    assert all(name in str(raised.value) for name in unknown)
