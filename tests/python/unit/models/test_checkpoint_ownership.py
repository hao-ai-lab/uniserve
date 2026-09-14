"""Checkpoint loading distinguishes nonresident weights from invalid records."""

import pytest
import torch

from tests.python.fixtures.model_execution import model_arguments
from uniserve.distributed.mesh import Communicator
from uniserve.loading.handles import TensorWeightHandle
from uniserve.model.limits import ModelLimits
from uniserve.nn.decoder.mot import MoTConfig
from uniserve.nn.layer import LayerConfig
from uniserve_models.bagel import BagelConfig, BagelForConditionalGeneration
from uniserve_models.minimax_h3.config import H3Config


@pytest.mark.parametrize(
    ("rank", "nonresident"),
    [
        (0, "language_model.model.layers.2.input_layernorm.weight"),
        (1, "language_model.model.embed_tokens.weight"),
    ],
)
def test_bagel_loading_reports_unknown_records_on_each_pipeline_stage(rank, nonresident):
    with torch.device("meta"):
        model = BagelForConditionalGeneration(
            BagelConfig(text=MoTConfig(num_hidden_layers=3)),
            **model_arguments(
                LayerConfig(Communicator(), None, pipeline=Communicator(ranks=(0, 1), rank=rank))
            ),
        )
    unknown = ("language_model.model.layers.2.unknown.weight", "unknown.weight")

    report = model.load_weights(
        tuple(TensorWeightHandle(name, torch.zeros(1)) for name in (nonresident, *unknown))
    )

    assert report.skipped == [nonresident]
    assert report.unexpected == list(unknown)


@pytest.mark.parametrize("rank", [0, 1])
def test_h3_checkpoint_reports_reject_unknown_records_on_each_pipeline_stage(rank):
    from tests.python.fixtures.model_execution import h3_arguments
    from uniserve.distributed.mesh import DeviceMesh
    from uniserve.distributed.parallel import ParallelConfig
    from uniserve_models.catalog import MINIMAX_H3_ENTRY
    from uniserve_models.minimax_h3.model import MiniMaxH3Model

    parallel = ParallelConfig(pipeline_parallel_size=2)
    denoiser = DeviceMesh(
        (0, 1),
        rank,
        parallel,
        groups={"pp": Communicator(ranks=(0, 1), rank=rank, name="pp")},
    )
    local = DeviceMesh((rank,), rank, ParallelConfig(), groups={})
    arguments = h3_arguments(
        parallel={"denoiser": parallel},
        meshes={"denoiser": denoiser, "text_encoder": local, "video_decoder": local},
        limits=ModelLimits(text_tokens=64, video_frames=22),
        precisions=MINIMAX_H3_ENTRY.component_precisions({"mode": "quality"}),
    )
    with torch.device("meta"):
        model = MiniMaxH3Model(H3Config(), **arguments)
    unknown = {
        "denoiser": (
            "time_embedder.unknown.weight",
            "transformer_blocks.50.attn.to_q.weight",
            "transformer_blocks.0.adaln_proj.unknown.weight",
            "token_refiner.unknown.weight",
        ),
        "text_encoder": (
            "model.visual.blocks.27.attn.qkv.weight",
            "model.language_model.layers.64.self_attn.q_proj.weight",
            "model.visual.unknown.weight",
        ),
        "video_decoder": (
            "encoder.unknown.weight",
            "quant_conv.unknown",
            "decoder.unknown.weight",
        ),
    }
    for component in model.checkpoint_components():
        assert component.map_weights is not None
        names = unknown[component.source]
        report = component.map_weights(
            tuple(TensorWeightHandle(name, torch.zeros(1)) for name in names)
        )
        assert report.unexpected == list(names)
