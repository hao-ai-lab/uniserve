"""Checkpoint metadata supplies numerical construction values.

The values are supplied without model allocation.
"""

import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from tests.python.fixtures.model_metadata import neo_metadata
from uniserve.loading import Config as IOConfig
from uniserve_models import bagel

pytestmark = pytest.mark.unit


def _bagel_config(root):
    """Read BAGEL metadata with its primary source resolved in place."""
    io = IOConfig()
    primary = bagel.checkpoint_sources[0].resolve(root, io=io)
    return bagel.read_config(root, io, sources={primary.name: primary})


@pytest.mark.parametrize("inline", [False, True])
def test_bagel_metadata_resolves_towers_and_checkpoint_position_extent(
    tmp_path, inline
):
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
        "vae_config": {
            "z_channels": 4,
            "downsample": 2,
            "ch_mult": [1, 2],
            "scale_factor": 0.5,
        },
    }
    if not inline:
        for name, values in towers.items():
            (tmp_path / f"{name}.json").write_text(json.dumps(values))
    path = tmp_path / "ema.safetensors"
    save_file({"latent_pos_embed.pos_embed": torch.zeros(9, 16)}, path)

    raw = {
        **(towers if inline else {}),
        "start_of_image_id": 62,
        "end_of_image_id": 63,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = _bagel_config(tmp_path)

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
    (tmp_path / "config.json").write_text(
        json.dumps({**raw, "max_latent_size": 4})
    )
    normalized = _bagel_config(tmp_path)
    assert normalized.max_latent_size == 3

    save_file({"latent_pos_embed.pos_embed": torch.zeros(9, 12)}, path)
    with pytest.raises(ValueError, match="square grid at text width"):
        _bagel_config(tmp_path)

    # Positional vectors must form the square grid used by latent patch
    # indexing.
    save_file({"latent_pos_embed.pos_embed": torch.zeros(10, 16)}, path)
    with pytest.raises(ValueError, match="square grid at text width"):
        _bagel_config(tmp_path)


def test_h3_worker_advertises_bounded_media_products():
    from tests.python.fixtures.h3 import fasth3_config
    from uniserve_models.minimax_h3 import Model
    from uniserve_worker.bootstrap.components import (
        media_components,
        supported_calls,
    )
    from uniserve_worker.bootstrap.outputs import resolve_outputs
    from uniserve_worker.config.execution import WorkerConfig
    from uniserve_worker.protocol.call import MediaCall, TransferMode

    with torch.device("meta"):
        model = Model(fasth3_config())
    config = WorkerConfig(
        device="cpu",
        max_sequence_tokens=65,
        max_video_seconds=1.0,
        max_request_pool_size=2,
        min_request_pool_size=2,
    )
    outputs = resolve_outputs(model, config)
    assert set(supported_calls(model)) == {
        MediaCall.MEDIA_READING,
        MediaCall.VISION_ENCODING,
        MediaCall.LATENT_ENCODING,
        MediaCall.TEXT_ENCODING,
        MediaCall.LATENT_PREPARATION,
        MediaCall.DENOISING,
        MediaCall.VIDEO_DECODING,
        MediaCall.AUDIO_DECODING,
        MediaCall.VIDEO_ENCODING,
        MediaCall.AUDIO_ENCODING,
        MediaCall.MUXING,
        TransferMode.TENSOR,
    }
    assert media_components(model) == {
        # The conditioner's vision tower encodes the presentation's vision
        # blocks; both condition encoders form one component.
        MediaCall.VISION_ENCODING: "text_encoder",
        MediaCall.LATENT_ENCODING: "latent_encoder",
        MediaCall.TEXT_ENCODING: "text_encoder",
        MediaCall.LATENT_PREPARATION: "denoiser",
        MediaCall.DENOISING: "denoiser",
        MediaCall.VIDEO_DECODING: "video_decoder",
        MediaCall.AUDIO_DECODING: "audio_decoder",
        # The host components own no numerical method: the media reader
        # decodes condition media, the video codec encodes the decoded media
        # units and the muxer assembles them.
        MediaCall.MEDIA_READING: "media_reader",
        MediaCall.VIDEO_ENCODING: "video_codec",
        MediaCall.AUDIO_ENCODING: "muxer",
        MediaCall.MUXING: "muxer",
    }
    products = {
        value.name: value for values in outputs.values() for value in values
    }
    # One second is covered by two native 17-frame windows and the final
    # five-frame overlap. Each video latent frame contains 24x42 patch tokens.
    expected = {
        "conditioning": (65, 5120),
        "video_latents": (12 * 24 * 42, 96),
        "audio_latents": (2 * 65, 32),
        # A decoding round's product is its RGB media units, one row per
        # unit at the longest unit's frame count.
        "video_units": (2, 22, 768, 1344, 3),
        "audio_samples": (52_000, 2),
    }
    # A deployment without condition capacity declares no condition product.
    assert products.keys() == expected.keys() | {"encoded_units"}
    for name, shape in expected.items():
        assert (
            products[name].shape_bound.max_elements == torch.Size(shape).numel()
        )

    # Only conditioned tasks carry conditions: this deployment executes
    # text-to-video alone and provisions no condition product, whatever its
    # condition capacity.
    conditioned = replace(config, max_condition_rows=1000)
    assert {
        value.name
        for values in resolve_outputs(model, conditioned).values()
        for value in values
    } == products.keys()

    # With condition capacity, every condition product is bounded by the
    # condition rows or the presentation. A condition encodes one frame or a
    # generated frame count (22 or 39 here); at a 32-pixel raster each latent
    # frame is one row, and 39 frames encode to 12 latent frames, the most
    # pixels per row: 39 * 32 * 32 / 12 = 3328. An audio row is half a
    # stereo latent frame. Each presented token is 2x2 merged patches of
    # 3 x 2 x 16 x 16 values and one row of the embedding and three
    # DeepStack features.
    from uniserve_worker.bootstrap.components import describe_components
    from uniserve_worker.bootstrap.inputs import media_builder
    from uniserve_worker.model_executor.resources import (
        condition_media_layouts,
        output_layouts,
    )

    builder = media_builder(model, conditioned)
    layouts = dict(
        condition_media_layouts(
            conditioned,
            video_encoder=model.video_encoder,
            audio_encoder=model.audio_encoder,
            vision=model.text_encoder.vision,
            frame_counts=(1, 22, 39),
        )
    )
    calls = describe_components(model)
    for call in (*calls["text_encoder"], *calls["latent_encoder"]):
        layouts.update(output_layouts(conditioned, call, builder=builder))
    rate = model.audio_encoder.latent_rate
    for name, shape in {
        "condition_pixels": (1000 * 3328, 3),
        "condition_samples": (500 * rate, 2),
        "vision_pixels": (4 * 65, 1536),
        "features": (65, 4 * 5120),
        "video": (1000, 96),
        "audio": (1000, 32),
    }.items():
        assert layouts[name].shape == shape, name

    from uniserve_worker.bootstrap.report import build_worker_layout

    info = build_worker_layout(model, config, queue_depth=6).info
    assert info.num_inference_steps == 4
    assert info.request_slots == 2
    assert info.kv_cache is None
    assert info.max_batch_calls == 2


def test_sensenova_reader_resolves_aliases_and_numerical_layer_modes(tmp_path):
    from uniserve_models.sensenova_u1 import read_config

    raw = neo_metadata()
    raw["llm_config"].update(
        use_sliding_window=False, sliding_window=64, max_window_layers=1
    )
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = read_config(tmp_path, IOConfig(), sources={})
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
        (
            "rope_parameters",
            {"rope_theta": 20000.0},
            "conflicting aliases for rope_theta",
        ),
        ("layer_types", ["full_attention"], "every decoder layer"),
        ("num_key_value_heads", 3, "divisible by KV heads"),
        ("num_experts", 8, "sparse MoE is not supported"),
    ],
)
def test_sensenova_reader_rejects_inconsistent_checkpoint_math(
    tmp_path, field, value, error
):
    from uniserve_models.sensenova_u1 import read_config

    raw = neo_metadata()
    raw["llm_config"][field] = value
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=error):
        read_config(tmp_path, IOConfig(), sources={})


def test_sensenova_direct_config_rejects_mismatched_vision_features(tmp_path):
    from dataclasses import replace

    from uniserve_models.sensenova_u1 import read_config

    (tmp_path / "config.json").write_text(json.dumps(neo_metadata()))
    config = read_config(tmp_path, IOConfig(), sources={})
    with pytest.raises(ValueError, match="vision output must match"):
        replace(config, vision=replace(config.vision, output_size=32))


def test_sensenova_reader_rejects_unimplemented_sliding_attention(tmp_path):
    from uniserve_models.sensenova_u1 import read_config

    raw = neo_metadata()
    raw["llm_config"].update(
        use_sliding_window=True, sliding_window=64, max_window_layers=1
    )
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="sliding_attention is not supported"):
        read_config(tmp_path, IOConfig(), sources={})


@pytest.mark.parametrize("storage", ("bfloat16", "float8_e4m3fn"))
def test_text_worker_reports_exact_cache_capacity(storage):
    from uniserve_models.qwen3 import Config, Model
    from uniserve_worker.bootstrap.report import build_worker_layout
    from uniserve_worker.config.execution import WorkerConfig

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
                rope_scaling=None,
                norm_topk_prob=False,
                decoder_sparse_step=1,
                mlp_only_layers=(),
            )
        ).to(dtype=torch.bfloat16)
    config = WorkerConfig(
        device="cpu",
        block_size=64,
        kv_token_capacity=256,
        kv_cache_dtype=storage,
        max_sequence_tokens=128,
    )
    info = build_worker_layout(model, config).info
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
    assert (
        cache.bytes_per_token == (payload + scales + initialization + 63) // 64
    )


def test_h3_media_units_follow_each_request_canvas():
    """Decoded media units are laid out at each request's own canvas.

    A worker serving several canvases declares the decoded units at its
    largest height (9:16) and width (21:9) and names those raster axes,
    which the engine binds to each request's canvas; the worker lays out
    each request's units at that canvas. Encoded unit rows carry their own
    length, so every request uses the declared row.
    """
    from tests.python.fixtures.h3 import base_config
    from uniserve.distributed import Communicator, DeviceMesh
    from uniserve_models.minimax_h3 import Model
    from uniserve_worker.bootstrap.outputs import resolve_outputs
    from uniserve_worker.config.deployment import (
        ComponentConfig,
        ParallelConfig,
    )
    from uniserve_worker.config.execution import WorkerConfig
    from uniserve_worker.execution.model_executor import ModelExecutor
    from uniserve_worker.model_executor.component_binding import (
        ComponentBinding,
    )
    from uniserve_worker.protocol.batch import DecodeRange, DiffusionParams
    from uniserve_worker.protocol.identity import CallId, RequestKey
    from uniserve_worker.protocol.tensor import StaticDim

    with torch.device("meta"):
        model = Model(base_config())
    config = WorkerConfig(
        device="cpu",
        max_sequence_tokens=65,
        max_video_seconds=5.0,
        deployment_components=("denoiser", "video_decoder", "video_codec"),
    )
    products = {
        value.name: value
        for values in resolve_outputs(model, config).values()
        for value in values
    }
    units = products["video_units"]
    assert units.raster_axes == (2, 3)
    assert units.shape_bound.dims[1:] == (
        StaticDim(22),
        StaticDim(1344),
        StaticDim(1536),
        StaticDim(3),
    )
    (row,) = products["encoded_units"].shape_bound.dims[1:]

    # One rank decodes and encodes every unit of a round.
    dimensions = ParallelConfig().dimensions
    group = Communicator((0,), 0)
    bindings = {
        name: ComponentBinding(
            name,
            ComponentConfig((0,), distribution="temporal_units"),
            group,
            DeviceMesh(
                ranks=(0,),
                rank=0,
                shape=tuple(size for _, size in dimensions),
                axes=tuple(axis for axis, _ in dimensions),
            ),
            group.device,
        )
        for name in ("video_decoder", "video_codec")
    }
    runner = ModelExecutor(model, config, bindings=bindings)
    try:
        round_ = DecodeRange(
            RequestKey(1, 0, 0), CallId(1, 0), cursor=0, max_units=1
        )
        for width, height in ((1536, 672), (1344, 768), (768, 1344)):
            media = DiffusionParams(124, 7, 50, 1, width=width, height=height)
            assert runner.output_layout(
                "video_decoder", 0, media, round_, 10
            ).shape == (1, 22, height, width, 3)
            assert runner.output_layout(
                "video_codec", 0, media, round_, 10
            ).shape == (1, row.extent)
    finally:
        runner.close()


@pytest.mark.parametrize(
    "seconds,frames", [(5.0, 124), (10.0, 243), (15.0, 362)]
)
def test_h3_audio_product_holds_the_decoded_track(seconds, frames):
    """The advertised audio product is the whole track the decoder returns.

    The decoded track spans the whole latent frames generated with the video,
    so its length differs from the video duration in samples wherever that
    duration is a fractional number of latent frames: longer at 124 frames,
    equal at 243 and shorter at 362.
    """
    from tests.python.fixtures.h3 import fasth3_config
    from uniserve_models.minimax_h3 import Model
    from uniserve_models.minimax_h3.packing import audio_latent_frames
    from uniserve_worker.bootstrap.outputs import resolve_outputs
    from uniserve_worker.config.execution import WorkerConfig

    with torch.device("meta"):
        model = Model(fasth3_config())
    config = WorkerConfig(
        device="cpu",
        max_sequence_tokens=65,
        max_video_seconds=seconds,
        max_request_pool_size=2,
        min_request_pool_size=2,
    )
    products = {
        value.name: value
        for values in resolve_outputs(model, config).values()
        for value in values
    }
    track = audio_latent_frames(frames) * model.audio_decoder.latent_rate
    assert track == model.audio_decoder.track_samples(frames, 24)
    assert products["audio_samples"].shape_bound.max_elements == track * 2


def test_h3_video_capacity_rounds_half_frames_to_even():
    """A capacity sizes the frames its longest admitted request resolves to.

    5.1875 seconds is 124.5 frames at 24 fps. The server rounds that half to
    even, to 124 frames, which is already a complete temporal window;
    rounding away from zero would reach 125 and extend to 141 frames.
    """
    from tests.python.fixtures.h3 import fasth3_config
    from uniserve_models.minimax_h3 import Model
    from uniserve_models.minimax_h3.packing import video_latent_frames
    from uniserve_worker.bootstrap.outputs import resolve_outputs
    from uniserve_worker.config.execution import WorkerConfig

    with torch.device("meta"):
        model = Model(fasth3_config())
    config = WorkerConfig(
        device="cpu",
        max_sequence_tokens=65,
        max_video_seconds=5.1875,
        max_request_pool_size=2,
        min_request_pool_size=2,
    )
    products = {
        value.name: value
        for values in resolve_outputs(model, config).values()
        for value in values
    }
    assert products["video_latents"].shape_bound.max_elements == (
        video_latent_frames(124) * 24 * 42 * 96
    )
