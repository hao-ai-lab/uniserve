"""Qwen3-VL text encoding preserves the Transformers equations.

A small FP32 checkpoint saved by Transformers loads through the public
Qwen3-VL assignments. Vision tokens, their DeepStack features and the
retained layers' hidden state of multimodal and text prompts follow
``Qwen3VLForConditionalGeneration``.
"""

import pytest
import torch
from safetensors import safe_open
from torch import nn
from transformers import (
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLVideoProcessor,
)

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.model import VisionInput
from uniserve_models import qwen3, qwen3_vl

pytestmark = pytest.mark.integration

IMAGE, VIDEO, START, END = 30, 31, 32, 33
HIDDEN = 64

_VISION = {
    "depth": 4,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_heads": 2,
    "hidden_act": "gelu_pytorch_tanh",
    "in_channels": 3,
    "patch_size": 4,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": HIDDEN,
    "num_position_embeddings": 16,
    "deepstack_visual_indexes": [1, 2],
}
_TEXT = {
    "vocab_size": 40,
    "hidden_size": HIDDEN,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 512,
}
# The eight compact frequencies of a 16-wide head, interleaved over the
# (temporal, height, width) axes.
_SECTIONS = (4, 2, 2)

# The encoder runs the first three of the four checkpoint layers, so the
# DeepStack features (after layers 0 and 1) enter before its output.
_RETAINED = (0, 1, 2)


@pytest.mark.parametrize("num_frames", (1, 3))
def test_pixel_packing_matches_the_checkpoint_processor(num_frames):
    config = qwen3_vl.read_vision_config(_VISION)
    pixels = qwen3_vl.PixelConfig()
    with torch.device("meta"):
        encoder = qwen3_vl.VisionEncoder(config, pixels, dtype=torch.float32)
    frames = torch.randint(
        0,
        256,
        (num_frames, 15, 23, 3),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(127),
    )
    options = {
        "patch_size": config.patch_size,
        "temporal_patch_size": config.temporal_patch_size,
        "merge_size": config.spatial_merge_size,
        "size": {"shortest_edge": 384, "longest_edge": 384},
        "image_mean": pixels.mean,
        "image_std": pixels.std,
    }
    # Nonaligned spatial extents require resizing, and the odd video length
    # requires repeating its final frame to fill the last temporal patch.
    if num_frames == 1:
        reference = Qwen2VLImageProcessor(**options)(
            images=frames[0].permute(2, 0, 1), return_tensors="pt"
        )
        grid = reference["image_grid_thw"][0]
        expected = reference["pixel_values"]
    else:
        reference = Qwen3VLVideoProcessor(**options)(
            videos=[frames.permute(0, 3, 1, 2)],
            do_sample_frames=False,
            return_tensors="pt",
        )
        grid = reference["video_grid_thw"][0]
        expected = reference["pixel_values_videos"]
    actual = encoder.pack_pixels(frames, tuple(grid.tolist()))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _encoder(config):
    language, vision = config
    network = qwen3_vl.Transformer(language)
    # The output is the raw residual stream after the last retained layer.
    network.norm = nn.Identity()
    return qwen3_vl.Encoder(
        network,
        _RETAINED,
        vision,
        pixels=qwen3_vl.PixelConfig(),
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    config = Qwen3VLConfig(
        text_config={
            **_TEXT,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5_000_000.0,
                "mrope_section": list(_SECTIONS),
                "mrope_interleaved": True,
            },
        },
        vision_config=_VISION,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=START,
        vision_end_token_id=END,
    )
    for part in (config, config.text_config, config.vision_config):
        part._attn_implementation = "eager"
    torch.manual_seed(1203)
    reference = Qwen3VLForConditionalGeneration(config).eval()
    root = tmp_path_factory.mktemp("qwen3_vl")
    reference.save_pretrained(root)
    with safe_open(root / "model.safetensors", "pt") as handle:
        names = frozenset(handle.keys())

    language = qwen3.Config(
        **{
            name: _TEXT[name]
            for name in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "rms_norm_eps",
                "max_position_embeddings",
            )
        },
        num_hidden_layers=_TEXT["num_hidden_layers"],
        hidden_act="silu",
        rope_theta=5_000_000.0,
        rope_scaling=None,
        attention_bias=False,
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=1,
        moe_intermediate_size=_TEXT["intermediate_size"],
        mrope_sections=_SECTIONS,
        norm_topk_prob=False,
        decoder_sparse_step=1,
        mlp_only_layers=(),
    )

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "primary",
                lambda reader: qwen3_vl.weights.assignments(model, reader),
                frozenset(name for name, _ in model.named_parameters()),
                nonresident=names - qwen3_vl.weights.sources(model),
            ),
        )

    model = loading.load_model(
        _encoder,
        (language, qwen3_vl.read_vision_config(_VISION)),
        checkpoint=(checkpoint.Config().resolve(root, io=loading.Config()),),
        mapping=mapping,
        device="cpu",
        weights=weights.Config(dtype=torch.float32),
    ).model
    return reference, model


def _media():
    """Return an image and a two-step video as tubelet rows with grids."""
    generator = torch.Generator().manual_seed(58)
    image, video = (1, 4, 6), (2, 4, 4)
    return (
        (torch.randn(24, 96, generator=generator), image),
        (torch.randn(32, 96, generator=generator), video),
    )


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_vision_tokens_carry_embeddings_and_deepstack_features(models):
    reference, model = models
    media = _media()
    rows = model.vision.encode(
        VisionInput(
            tuple(pixels for pixels, _ in media),
            tuple(torch.tensor([grid]) for _, grid in media),
            tuple(grid for _, grid in media),
        )
    )
    for (pixels, grid), actual in zip(media, rows, strict=True):
        expected = reference.model.visual(pixels, grid_thw=torch.tensor([grid]))
        tokens = grid[0] * grid[1] * grid[2] // 4
        assert actual.shape == (tokens, 3 * HIDDEN)
        _close(actual[:, :HIDDEN], expected.pooler_output)
        for index, feature in enumerate(expected.deepstack_features):
            _close(
                actual[:, (index + 1) * HIDDEN : (index + 2) * HIDDEN], feature
            )


@torch.no_grad()
def test_multimodal_prompt_follows_the_reference_hidden_state(models):
    reference, model = models
    (image, image_grid), (video, video_grid) = _media()
    # A labelled image block, then a video whose two time steps are separate
    # blocks behind their own text, then the prompt.
    ids = [
        1,
        2,
        START,
        *(IMAGE,) * 6,
        END,
        3,
        4,
        START,
        *(VIDEO,) * 4,
        END,
        5,
        START,
        *(VIDEO,) * 4,
        END,
        6,
        7,
        8,
    ]
    input_ids = torch.tensor([ids])
    types = torch.zeros_like(input_ids)
    types[input_ids == IMAGE] = 1
    types[input_ids == VIDEO] = 2
    expected = reference.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=types,
        pixel_values=image,
        image_grid_thw=torch.tensor([image_grid]),
        pixel_values_videos=video,
        video_grid_thw=torch.tensor([video_grid]),
        use_cache=False,
        output_hidden_states=True,
    ).hidden_states[len(_RETAINED)][0]

    rows = model.vision.encode(
        VisionInput(
            (image, video),
            (torch.tensor([image_grid]), torch.tensor([video_grid])),
            (image_grid, video_grid),
        )
    )
    positions = model.positions(
        ids, image_grids=(image_grid,), video_grids=(video_grid,)
    )
    # A text-only prompt in the same call keeps its own rows.
    text = torch.tensor([9, 10, 11])
    actual, plain = model.encode(
        (torch.tensor(ids), text),
        positions=(positions, torch.arange(3).expand(3, -1)),
        visual=(torch.cat(rows), None),
    )
    _close(actual, expected)
    _close(
        plain,
        reference.model(
            input_ids=text[None],
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states[len(_RETAINED)][0],
    )


@torch.no_grad()
def test_text_prompt_follows_the_reference_hidden_state(models):
    reference, model = models
    ids = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
    expected = reference.model(
        input_ids=ids[None],
        attention_mask=torch.ones(1, 8, dtype=torch.long),
        use_cache=False,
        output_hidden_states=True,
    ).hidden_states[len(_RETAINED)][0]
    _close(model.encode((ids,))[0], expected)


@torch.no_grad()
def test_vision_rows_must_cover_the_placeholders(models):
    _, model = models
    ids = torch.tensor([1, START, IMAGE, IMAGE, END])
    with pytest.raises(ValueError, match="placeholders"):
        model.encode((ids,), visual=(torch.zeros(3, 3 * HIDDEN),))
    with pytest.raises(ValueError, match="placeholders"):
        model.encode((ids,), visual=(None,))


class _Conditioner(nn.Module):
    """A model whose conditioner encodes both prompts and vision blocks."""

    config = None

    def __init__(self, encoder):
        super().__init__()
        self.text_encoder = encoder


def entry_points(config):
    """Declare ``_Conditioner``'s one component, as a model package does.

    The component exposes ``encode`` on both the conditioner and its vision
    tower; the worker runs each by its own module.
    """
    from uniserve.model import ComponentEntry, EntryPoint

    return {
        "text_encoder": ComponentEntry(
            "text_encoder", (EntryPoint("encode"), EntryPoint("vision.encode"))
        )
    }


def test_worker_vision_and_text_path_follows_the_reference(models):
    from uniserve_worker.config.execution import WorkerConfig
    from uniserve_worker.execution.model_executor import ModelExecutor

    reference, model = models
    (image, image_grid), (video, video_grid) = _media()
    ids = [1, START, *(IMAGE,) * 6, END, 3, START, *(VIDEO,) * 4, END]
    ids += [5, START, *(VIDEO,) * 4, END, 6]
    input_ids = torch.tensor([ids])
    types = torch.zeros_like(input_ids)
    types[input_ids == IMAGE] = 1
    types[input_ids == VIDEO] = 2
    with torch.no_grad():
        expected = reference.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=types,
            pixel_values=image,
            image_grid_thw=torch.tensor([image_grid]),
            pixel_values_videos=video,
            video_grid_thw=torch.tensor([video_grid]),
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states[len(_RETAINED)][0]

    runner = ModelExecutor(_Conditioner(model), WorkerConfig(device="cpu"))
    try:
        rows = runner.encode_vision(
            VisionInput(
                (image, video),
                (torch.tensor([image_grid]), torch.tensor([video_grid])),
                (image_grid, video_grid),
            )
        ).values
        (actual,) = runner.encode_text(
            ids,
            visual=torch.cat(rows),
            image_grids=(image_grid,),
            video_grids=(video_grid,),
        ).values
    finally:
        runner.close()
    _close(actual, expected)
