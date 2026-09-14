"""Checkpoint metadata supplies numerical construction values without model allocation."""

import json

import pytest
import torch
from safetensors.torch import save_file

from uniserve.loading.source import WeightSourceSet
from uniserve.model.limits import ModelLimits
from uniserve_models.metadata import bagel_config
from uniserve_models.minimax_h3.config import H3Config

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
        "vit_config": {
            "hidden_size": 24,
            "num_attention_heads": 4,
            "num_hidden_layers": 3,
            "patch_size": 2,
            "image_size": 16,
        },
        "vae_config": {"z_channels": 4, "downsample": 2, "ch_mult": [1, 2], "scale_factor": 0.5},
    }
    if not inline:
        for name, values in towers.items():
            (tmp_path / f"{name}.json").write_text(json.dumps(values))
    path = tmp_path / "ema.safetensors"
    save_file({"latent_pos_embed.pos_embed": torch.zeros(9, 16)}, path)
    source = WeightSourceSet(tmp_path, (path,), (path.name,))

    raw = {**(towers if inline else {}), "start_of_image_id": 62, "end_of_image_id": 63}
    config = bagel_config(raw, tmp_path, (source,))

    assert config.text.hidden_size == 16
    assert config.text.num_attention_heads == 4
    assert config.vision.hidden_size == 24
    assert config.vision.num_hidden_layers == 2
    assert config.vae.scale_factor == 0.5
    assert config.vit_token_capacity == 64
    assert config.latent_channel == 4
    assert config.latent_downsample == 4
    assert config.max_latent_size == 3
    assert config.latent_token_capacity == 9

    # Released checkpoints may serialize a constructor default; actual learned
    # positions determine both numerical packing and allocation bounds.
    normalized = bagel_config({**raw, "max_latent_size": 4}, tmp_path, (source,))
    assert normalized.max_latent_size == 3
    assert normalized.latent_token_capacity == 9

    save_file({"latent_pos_embed.pos_embed": torch.zeros(9, 12)}, path)
    with pytest.raises(ValueError, match="text hidden size"):
        bagel_config(raw, tmp_path, (source,))

    # Positional vectors must form the square grid used by latent patch indexing.
    save_file({"latent_pos_embed.pos_embed": torch.zeros(10, 16)}, path)
    with pytest.raises(ValueError, match="position count 10 is not square"):
        bagel_config(raw, tmp_path, (source,))


def test_h3_worker_advertises_numerical_components_and_media_delivery():
    from tests.python.fixtures.model_execution import h3_arguments
    from uniserve.distributed.mesh import DeviceMesh
    from uniserve.distributed.parallel import ParallelConfig
    from uniserve_models.catalog import MINIMAX_H3_ENTRY
    from uniserve_models.minimax_h3.model import MiniMaxH3Model
    from uniserve_worker.bootstrap.worker_info_builder import build_worker_layout
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.protocol.batch import VIDEO_STAGES, PipelineStage, TransferMode

    parallel = ParallelConfig()
    # Output metadata has no learned parameters. Its real component context
    # retains the numerical declarations and postprocessing geometry.
    arguments = h3_arguments(
        parallel={"video_output": parallel},
        meshes={"video_output": DeviceMesh((0,), 0, parallel)},
        limits=ModelLimits(text_tokens=64, video_frames=22),
        precisions=MINIMAX_H3_ENTRY.component_precisions({"mode": "quality"}),
    )
    model = MiniMaxH3Model(H3Config(), **arguments)
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


def _neo_metadata():
    return {
        "llm_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "vocab_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rope_theta": 10000.0,
        },
        "vision_config": {"hidden_size": 8, "llm_hidden_size": [16], "downsample_ratio": [0.5]},
        "pad_token_id": 3,
    }


def test_sensenova_reader_resolves_aliases_and_numerical_layer_modes():
    from uniserve_models.sensenova.config import read_config

    raw = _neo_metadata()
    raw["llm_config"].update(use_sliding_window=False, sliding_window=64, max_window_layers=1)
    config = read_config(raw)
    assert config.text.layer_types == ("full_attention",) * 3
    assert config.text.sliding_window == 64
    assert config.text.pad_token_id == 3
    assert config.vision.llm_hidden_size == 16
    assert config.vision.downsample_ratio == 0.5
    raw["llm_config"]["hidden_size"] = 32
    raw["vision_config"]["llm_hidden_size"][0] = 32
    assert config.text.hidden_size == config.vision.llm_hidden_size == 16


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("rope_parameters", {"rope_theta": 20000.0}, "conflicting aliases for rope_theta"),
        ("layer_types", ["full_attention"], "every decoder layer"),
        ("num_key_value_heads", 3, "divisible by KV heads"),
        ("num_experts", 8, "sparse MoE is not supported"),
    ],
)
def test_sensenova_reader_rejects_inconsistent_checkpoint_math(field, value, error):
    from uniserve_models.sensenova.config import read_config

    raw = _neo_metadata()
    raw["llm_config"][field] = value
    with pytest.raises(ValueError, match=error):
        read_config(raw)


def test_sensenova_direct_config_rejects_mismatched_vision_features():
    from dataclasses import replace

    from uniserve_models.sensenova.config import read_config

    config = read_config(_neo_metadata())
    with pytest.raises(ValueError, match="vision output must match"):
        replace(config, vision=replace(config.vision, llm_hidden_size=32))


def test_sensenova_reader_rejects_unimplemented_sliding_attention():
    from uniserve_models.sensenova.config import read_config

    raw = _neo_metadata()
    raw["llm_config"].update(use_sliding_window=True, sliding_window=64, max_window_layers=1)
    with pytest.raises(ValueError, match="sliding_attention is not supported"):
        read_config(raw)
