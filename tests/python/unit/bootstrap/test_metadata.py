"""Checkpoint metadata supplies numerical construction values without model allocation."""

import json

import pytest
import torch
from safetensors.torch import save_file

from uniserve_worker.bootstrap.metadata import bagel_config
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.loader.source import WeightSourceSet

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("inline", [False, True])
def test_bagel_metadata_resolves_towers_and_checkpoint_position_extent(tmp_path, inline):
    towers = {
        "llm_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 64,
        },
        "vit_config": {"hidden_size": 24, "patch_size": 2, "image_size": 16},
        "vae_config": {"z_channels": 4, "downsample": 2},
    }
    if not inline:
        for name, values in towers.items():
            (tmp_path / f"{name}.json").write_text(json.dumps(values))
    path = tmp_path / "ema.safetensors"
    save_file({"latent_pos_embed.pos_embed": torch.zeros(9, 16)}, path)
    source = WeightSourceSet(tmp_path, (path,), (path.name,))

    config = bagel_config(towers if inline else {}, tmp_path, (source,))

    assert config.llm.hidden_size == 16
    assert config.llm.num_attention_heads == 4
    assert config.vit_hidden_size == 24
    assert config.vit_token_capacity == 64
    assert config.latent_channel == 4
    assert config.latent_downsample == 4
    assert config.max_latent_size == 3
    assert config.latent_token_capacity == 9

    # Positional vectors must form the square grid used by latent patch indexing.
    save_file({"latent_pos_embed.pos_embed": torch.zeros(10, 16)}, path)
    with pytest.raises(WorkerError, match="position count 10 is not square"):
        bagel_config(towers if inline else {}, tmp_path, (source,))


def test_h3_worker_advertises_numerical_components_and_media_delivery():
    from uniserve_worker.bootstrap.catalog import MINIMAX_H3_ENTRY
    from uniserve_worker.bootstrap.worker_info_builder import build_worker_layout
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.modeling.context import BuildContext
    from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
    from uniserve_worker.nn.mesh import DeviceMesh
    from uniserve_worker.nn.parallel import ParallelConfig
    from uniserve_worker.protocol.batch import VIDEO_STAGES, PipelineStage, TransferMode

    parallel = ParallelConfig()
    # Output metadata has no learned parameters. Its real component context
    # retains the numerical declarations and postprocessing geometry.
    context = BuildContext(
        parallel={"output": parallel},
        meshes={"output": DeviceMesh((0,), 0, parallel)},
        layers={},
        limits={"text_tokens": 64, "video_seconds": 22 / 24},
        component_precisions=MINIMAX_H3_ENTRY.component_precisions({"mode": "quality"}),
        schedule=MINIMAX_H3_ENTRY.create_schedule(torch.device("cpu")),
    )
    model = MiniMaxH3Model({}, context)
    config = WorkerConfig(
        device="cpu",
        max_batch_operations=2,
        max_batch_tokens=2,
        max_request_pool_size=2,
    )

    info = build_worker_layout(model, config, queue_depth=6).info

    assert info.kv_cache is None
    assert info.num_inference_steps == 4
    assert set(info.supported_ops) == {*VIDEO_STAGES, TransferMode.TENSOR}
    assert info.pipeline_components == {
        PipelineStage.TEXT_ENCODING: "text_encoder",
        PipelineStage.LATENT_PREPARATION: "denoiser",
        PipelineStage.DENOISING: "denoiser",
        PipelineStage.VIDEO_DECODING: "video_decoder",
        PipelineStage.AUDIO_DECODING: "audio_decoder",
        PipelineStage.VIDEO_ENCODING: "output",
        PipelineStage.AUDIO_ENCODING: "output",
        PipelineStage.MUXING: "output",
    }
