"""Checkpoint metadata supplies numerical construction values without model allocation."""

import json

import pytest
import torch
from safetensors.torch import save_file

from uniserve.loading import Config as IOConfig
from uniserve_models.bagel import read_config as bagel_config

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

    raw = {**(towers if inline else {}), "start_of_image_id": 62, "end_of_image_id": 63}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = bagel_config(tmp_path, IOConfig())

    assert config.text.hidden_size == 16
    assert config.text.num_attention_heads == 4
    assert config.vision.encoder.hidden_size == 24
    assert config.vision.encoder.num_hidden_layers == 2
    assert config.vae.scale_factor == 0.5
    assert config.vae.latent_channels == 4
    assert config.vae.downsample * config.latent_patch_size == 4
    assert config.max_latent_size == 3

    # Released checkpoints may serialize a constructor default; actual learned
    # positions determine both numerical packing and allocation bounds.
    (tmp_path / "config.json").write_text(json.dumps({**raw, "max_latent_size": 4}))
    normalized = bagel_config(tmp_path, IOConfig())
    assert normalized.max_latent_size == 3

    save_file({"latent_pos_embed.pos_embed": torch.zeros(9, 12)}, path)
    with pytest.raises(ValueError, match="square grid at text width"):
        bagel_config(tmp_path, IOConfig())

    # Positional vectors must form the square grid used by latent patch indexing.
    save_file({"latent_pos_embed.pos_embed": torch.zeros(10, 16)}, path)
    with pytest.raises(ValueError, match="square grid at text width"):
        bagel_config(tmp_path, IOConfig())


def test_h3_worker_advertises_bounded_media_products():
    from uniserve_models.minimax_h3 import Config, Model
    from uniserve_worker.bootstrap.components import media_components, supported_operations
    from uniserve_worker.config import WorkerConfig
    from uniserve_worker.protocol.batch import VIDEO_STAGES, PipelineStage, TransferMode
    from uniserve_worker.runtime.results import resolve_outputs

    with torch.device("meta"):
        model = Model(Config())
    config = WorkerConfig(
        device="cpu",
        max_sequence_tokens=65,
        max_video_seconds=1.0,
        max_request_pool_size=2,
        min_request_pool_size=2,
    )
    outputs = resolve_outputs(model, config)
    assert set(supported_operations(model)) == {*VIDEO_STAGES, TransferMode.TENSOR}
    assert media_components(model) == {
        PipelineStage.TEXT_ENCODING: "text_encoder",
        PipelineStage.LATENT_PREPARATION: "denoiser",
        PipelineStage.DENOISING: "denoiser",
        PipelineStage.VIDEO_DECODING: "video_decoder",
        PipelineStage.AUDIO_DECODING: "audio_decoder",
        PipelineStage.VIDEO_ENCODING: "output",
        PipelineStage.AUDIO_ENCODING: "output",
        PipelineStage.MUXING: "output",
    }
    products = {value.name: value for values in outputs.values() for value in values}
    # One second is covered by two native 17-frame windows and the final
    # five-frame overlap. Each video latent frame contains 24x42 patch tokens.
    expected = {
        "conditioning": (65, 5120),
        "video_latents": (12 * 24 * 42, 96),
        "audio_latents": (2 * 65, 32),
        "video_segments": (2, 1, 3, 25, 768, 1344),
        "audio_samples": (52_000, 2),
    }
    assert products.keys() == expected.keys()
    for name, shape in expected.items():
        assert products[name].shape_bound.max_elements == torch.Size(shape).numel()

    from uniserve_worker.bootstrap.worker_info_builder import build_worker_info

    info = build_worker_info(model, config, queue_depth=6)
    assert info.num_inference_steps == 4
    assert info.request_slots == 2
    assert info.kv_cache is None
    assert info.max_batch_ops == 2


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


def test_sensenova_reader_resolves_aliases_and_numerical_layer_modes(tmp_path):
    from uniserve_models.sensenova_u1 import read_config

    raw = _neo_metadata()
    raw["llm_config"].update(use_sliding_window=False, sliding_window=64, max_window_layers=1)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = read_config(tmp_path, IOConfig())
    assert config.text.layer_types == ("full_attention",) * 3
    assert config.text.sliding_window == 64
    assert config.text.pad_token_id == 3
    assert config.vision.output_size == 16
    assert config.vision.downsample_ratio == 0.5
    raw["llm_config"]["hidden_size"] = 32
    raw["vision_config"]["llm_hidden_size"][0] = 32
    assert config.text.hidden_size == config.vision.output_size == 16


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("rope_parameters", {"rope_theta": 20000.0}, "conflicting aliases for rope_theta"),
        ("layer_types", ["full_attention"], "every decoder layer"),
        ("num_key_value_heads", 3, "divisible by KV heads"),
        ("num_experts", 8, "sparse MoE is not supported"),
    ],
)
def test_sensenova_reader_rejects_inconsistent_checkpoint_math(tmp_path, field, value, error):
    from uniserve_models.sensenova_u1 import read_config

    raw = _neo_metadata()
    raw["llm_config"][field] = value
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=error):
        read_config(tmp_path, IOConfig())


def test_sensenova_direct_config_rejects_mismatched_vision_features(tmp_path):
    from dataclasses import replace

    from uniserve_models.sensenova_u1 import read_config

    (tmp_path / "config.json").write_text(json.dumps(_neo_metadata()))
    config = read_config(tmp_path, IOConfig())
    with pytest.raises(ValueError, match="vision output must match"):
        replace(config, vision=replace(config.vision, output_size=32))


def test_sensenova_reader_rejects_unimplemented_sliding_attention(tmp_path):
    from uniserve_models.sensenova_u1 import read_config

    raw = _neo_metadata()
    raw["llm_config"].update(use_sliding_window=True, sliding_window=64, max_window_layers=1)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="sliding_attention is not supported"):
        read_config(tmp_path, IOConfig())


@pytest.mark.parametrize("storage", ("bfloat16", "float8_e4m3fn"))
def test_text_worker_reports_exact_cache_capacity(storage):
    from uniserve_models.qwen3 import Config, Model
    from uniserve_worker.bootstrap.worker_info_builder import build_worker_info
    from uniserve_worker.config import WorkerConfig

    with torch.device("meta"):
        model = Model(
            Config(
                vocab_size=65,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                hidden_act="silu",
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                max_position_embeddings=128,
                attention_bias=False,
                tie_word_embeddings=False,
                num_experts=0,
                num_experts_per_tok=1,
                moe_intermediate_size=48,
            )
        ).to(dtype=torch.bfloat16)
    config = WorkerConfig(
        device="cpu",
        block_size=64,
        kv_token_capacity=256,
        kv_cache_dtype=storage,
        max_sequence_tokens=128,
    )
    info = build_worker_info(model, config)
    cache = info.kv_cache
    assert cache.num_blocks == 4
    assert cache.total_layers == cache.num_layers == 2
    assert cache.total_kv_heads == cache.num_kv_heads == 2
    assert cache.layer_offset == cache.kv_head_offset == 0
    assert cache.head_dim == 8
    assert cache.dtype == storage
    payload = 2 * 2 * 2 * 8 * (1 if storage == "float8_e4m3fn" else 2) * 64
    scales = 2 * 2 * 4 if storage == "float8_e4m3fn" else 0
    initialization = 2 * 2
    assert cache.bytes_per_token == (payload + scales + initialization + 63) // 64
