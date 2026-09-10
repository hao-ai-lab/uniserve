"""Checkpoint loading distinguishes nonresident weights from invalid records."""

import pytest
import torch

from uniserve_worker.loader.handles import TensorWeightHandle
from uniserve_worker.models.bagel import BagelConfig, BagelForConditionalGeneration, LLMConfig
from uniserve_worker.nn.layer import LayerConfig
from uniserve_worker.nn.mesh import Communicator


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
            BagelConfig(llm=LLMConfig(num_hidden_layers=3)),
            layer_config=LayerConfig(
                Communicator(), None, pipeline=Communicator(ranks=(0, 1), rank=rank)
            ),
        )
    unknown = ("language_model.model.layers.2.unknown.weight", "unknown.weight")

    report = model.load_weights(
        tuple(TensorWeightHandle(name, torch.zeros(1)) for name in (nonresident, *unknown))
    )

    assert report.skipped == [nonresident]
    assert report.unexpected == list(unknown)
