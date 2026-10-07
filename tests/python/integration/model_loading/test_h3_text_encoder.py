"""H3's Qwen3-VL text encoder follows the released checkpoint's encoding.

The reference is the diffusers H3 prompt encoding
(``get_qwen3vl_prompt_embeds``) over Transformers'
``Qwen3VLForConditionalGeneration``: the unnormalized ``hidden_states[50]``
of an image prompt, a video prompt and a text prompt, and the vision tokens
and DeepStack features of their media.

Two BF16 evaluations of a 50-layer stack differ elementwise by their
rounding, so the contract is accuracy rather than identity. Both the
reference and this encoder run in BF16, and both are measured against the
reference evaluated in FP32 on the same BF16 weights. Each token's relative
error is measured, and three statistics must stay within twice the
reference's own BF16 value: the relative L2 error of the whole tensor, which
the few massive-activation tokens dominate, and the median and 90th
percentile token errors, which describe the ordinary tokens. Kernels with
different but valid accumulation orders stay inside that bound, while an
equation error (a misplaced position, a missing DeepStack term, a swapped
weight) moves the typical token by orders of magnitude more.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers.modular_pipelines.minimax_h3.encoders import (
    get_qwen3vl_prompt_embeds,
)
from PIL import Image
from transformers import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)

from uniserve import loading
from uniserve.loading import checkpoint, weights
from uniserve.model import TextSize, VisionInput
from uniserve.runtime import ExecutionContext
from uniserve_models import qwen3_vl
from uniserve_models.minimax_h3.config import TEXT_FIELDS
from uniserve_models.minimax_h3.encoder import TextEncoderConfig, text_encoder

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.model("minimax_h3"),
    pytest.mark.slow,
]

DEVICE = "cuda:0"
# H3 conditions on the output of the 50th decoder layer.
LAYER = 50
PROMPT = (
    "A clay fox walks across a sunlit wooden table, pauses beside a cup of "
    "tea, and looks up at the camera while soft piano music plays."
)


def _root() -> Path:
    root = os.environ.get("UNISERVE_H3_MODEL", "")
    if not root or not (Path(root) / "text_encoder").is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name an H3 checkpoint directory with "
            "text_encoder and processor"
        )
    return Path(root)


def _image() -> Image.Image:
    """A deterministic 16:9 keyframe: smooth gradients and solid shapes."""
    generator = np.random.default_rng(1344)
    rows, columns = np.mgrid[0:768, 0:1344].astype(np.float32)
    pixels = np.stack(
        (
            127 + 120 * np.sin(columns / 97 + rows / 211),
            127 + 120 * np.cos(rows / 53),
            (columns + rows) / (1344 + 768) * 255,
        ),
        axis=-1,
    )
    for _ in range(12):
        top, left = generator.integers(0, 700), generator.integers(0, 1280)
        height, width = generator.integers(16, 160, size=2)
        pixels[top : top + height, left : left + width] = generator.integers(
            0, 256, size=3
        )
    return Image.fromarray(pixels.clip(0, 255).astype(np.uint8))


def _video() -> np.ndarray:
    """Eight 2 fps frames of a pattern drifting across a 16:9 canvas."""
    rows, columns = np.mgrid[0:384, 0:672].astype(np.float32)
    frames = [
        np.stack(
            (
                127 + 120 * np.sin((columns - 24 * step) / 41),
                127 + 120 * np.cos((rows + 9 * step) / 29),
                np.full_like(rows, 32 * step),
            ),
            axis=-1,
        )
        for step in range(8)
    ]
    return np.stack(frames).clip(0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def processor():
    return Qwen3VLProcessor.from_pretrained(_root() / "processor")


@pytest.fixture(scope="module")
def cases(processor):
    """Return each prompt's token ids, vision inputs and patch grids.

    Presentations follow H3's rules (no chat template, no special tokens):
    ``"<Picture 1>: "`` and a vision block for a keyframe; ``"<Video 1>: "``
    and one timestamped block per merged frame pair for a reference video.
    """
    tokenizer = processor.tokenizer

    def text(value):
        return tokenizer(value, add_special_tokens=False)["input_ids"]

    def block(token, count):
        return [
            tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            *(tokenizer.convert_tokens_to_ids(token),) * count,
            tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        ]

    image = processor.image_processor(images=[_image()], return_tensors="pt")
    image_grid = tuple(image["image_grid_thw"][0].tolist())
    image_ids = (
        text("<Picture 1>: ")
        + block("<|image_pad|>", image_grid[1] * image_grid[2] // 4)
        + text(PROMPT)
    )

    video = processor.video_processor(
        videos=[_video()], do_sample_frames=False, return_tensors="pt"
    )
    video_grid = tuple(video["video_grid_thw"][0].tolist())
    video_ids = text("<Video 1>: ")
    for pair in range(video_grid[0]):
        # The block of 2 fps frames (2p, 2p + 1) carries their mean time.
        video_ids += text(f"<{pair + 0.25:.1f} seconds>")
        video_ids += block("<|video_pad|>", video_grid[1] * video_grid[2] // 4)
    video_ids += text(PROMPT)

    return {
        "image": (
            image_ids,
            {
                "pixel_values": image["pixel_values"],
                "image_grid_thw": image["image_grid_thw"],
            },
            (image_grid,),
            (),
        ),
        "video": (
            video_ids,
            {
                "pixel_values_videos": video["pixel_values_videos"],
                "video_grid_thw": video["video_grid_thw"],
            },
            (),
            (video_grid,),
        ),
        "text": (text(PROMPT), {}, (), ()),
    }


@torch.inference_mode()
def _reference_outputs(model, processor, cases, dtype):
    """Evaluate the reference encoding of every case at ``dtype``."""
    results = {}
    for name, (ids, vision, _, _) in cases.items():
        hidden = get_qwen3vl_prompt_embeds(
            model,
            processor,
            ids,
            vision,
            text_encoder_layer=LAYER,
            device=DEVICE,
            dtype=dtype,
        )[0]
        tokens = None
        if vision:
            pixels, grids = tuple(vision.values())
            pixels, grids = pixels.to(DEVICE, dtype), grids.to(DEVICE)
            features = (
                model.model.get_image_features(pixels, grids)
                if "pixel_values" in vision
                else model.model.get_video_features(pixels, grids)
            )
            tokens = (
                torch.cat(features.pooler_output),
                *features.deepstack_features,
            )
        results[name] = (hidden, tokens)
    return results


@pytest.fixture(scope="module")
def reference(processor, cases):
    """Reference results in BF16 and FP32, keyed by precision then case."""
    root = _root() / "text_encoder"
    # Layers past the 50th never reach hidden_states[50]; the 51st keeps
    # that state unnormalized, as the full stack's is.
    config = Qwen3VLConfig.from_pretrained(root)
    config.text_config.num_hidden_layers = LAYER + 1
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            root, config=config, dtype=torch.bfloat16
        )
        .to(DEVICE)
        .eval()
    )
    results = {"bf16": _reference_outputs(model, processor, cases, model.dtype)}
    model.float()
    results["fp32"] = _reference_outputs(model, processor, cases, model.dtype)
    del model
    torch.cuda.empty_cache()
    return results


def _encoder_config(root: Path) -> TextEncoderConfig:
    metadata = json.loads((root / "config.json").read_text())
    text = metadata["text_config"]
    return TextEncoderConfig(
        **{target: text[source] for source, target in TEXT_FIELDS.items()},
        mrope_sections=tuple(text["rope_scaling"]["mrope_section"]),
        image_token_id=metadata["image_token_id"],
        video_token_id=metadata["video_token_id"],
        vision=qwen3_vl.read_vision_config(metadata["vision_config"]),
    )


@pytest.fixture(scope="module")
def encoded(cases, reference):
    """This encoder's BF16 results through serving execution contexts.

    Depends on ``reference`` so the reference model is released first.
    """
    root = _root() / "text_encoder"
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = frozenset(index["weight_map"])

    def mapping(model):
        return (
            weights.ModuleMapping(
                model,
                "text_encoder",
                lambda reader: qwen3_vl.weights.assignments(model, reader),
                frozenset(name for name, _ in model.named_parameters()),
                nonresident=names - qwen3_vl.weights.sources(model),
            ),
        )

    config = _encoder_config(root)
    model = loading.load_model(
        text_encoder,
        config,
        checkpoint=(
            checkpoint.Config("text_encoder").resolve(
                root, io=loading.Config()
            ),
        ),
        mapping=mapping,
        device=DEVICE,
        weights=weights.Config(dtype=torch.bfloat16),
    ).model

    results = {}
    with torch.inference_mode():
        for name, (ids, vision, image_grids, video_grids) in cases.items():
            visual = None
            if vision:
                pixels, grids = tuple(vision.values())
                shapes = image_grids + video_grids
                with ExecutionContext(model.vision) as context:
                    context.prepare(None)
                    (visual,) = model.vision.encode(
                        VisionInput(
                            (pixels.to(DEVICE),),
                            (grids.to(DEVICE),),
                            shapes,
                        )
                    )
            positions = model.positions(
                ids, image_grids=image_grids, video_grids=video_grids
            ).to(DEVICE)
            tokens = torch.tensor(ids, device=DEVICE)
            with ExecutionContext(model) as context:
                context.prepare(TextSize(len(ids), 1))
                (hidden,) = model.encode(
                    (tokens,), positions=(positions,), visual=(visual,)
                )
                if name == "text":
                    # The text path without explicit coordinates.
                    (plain,) = model.encode((tokens,))
                    results["plain"] = plain.clone()
            # [tokens, embedding + DeepStack features, hidden] rows.
            layers = 1 + len(config.vision.deepstack_visual_indexes)
            tokens = (
                None
                if visual is None
                else tuple(visual.unflatten(-1, (layers, -1)).unbind(1))
            )
            results[name] = (hidden.clone(), tokens)
    del model
    torch.cuda.empty_cache()
    return results


def _errors(value: torch.Tensor, exact: torch.Tensor) -> dict[str, float]:
    """Relative L2 error of the tensor and quantiles of its token errors."""
    value, exact = value.float(), exact.float()
    tokens = (value - exact).norm(dim=-1) / exact.norm(dim=-1)
    return {
        "global": float((value - exact).norm() / exact.norm()),
        "median": float(tokens.quantile(0.5)),
        "p90": float(tokens.quantile(0.9)),
    }


def _check(record_property, label, value, reference, exact):
    ours, theirs = _errors(value, exact), _errors(reference, exact)
    for statistic in ours:
        record_property(f"{label}_{statistic}", ours[statistic])
        record_property(f"{label}_{statistic}_reference", theirs[statistic])
    exceeded = {
        statistic: (ours[statistic], theirs[statistic])
        for statistic in ours
        if ours[statistic] > 2 * theirs[statistic]
    }
    assert not exceeded, (
        f"{label}: errors (ours, reference BF16) exceed twice the "
        f"reference: {exceeded}"
    )


@pytest.mark.parametrize("case", ["image", "video"])
def test_vision_tokens_are_as_accurate_as_the_reference(
    reference, encoded, case, record_property
):
    actual = encoded[case][1]
    expected = reference["bf16"][case][1]
    exact = reference["fp32"][case][1]
    # The embedding, then the DeepStack features of blocks 8, 16 and 24.
    labels = ("embedding", "deepstack8", "deepstack16", "deepstack24")
    assert len(actual) == len(expected) == len(labels)
    for label, value, wanted, truth in zip(
        labels, actual, expected, exact, strict=True
    ):
        assert value.shape == wanted.shape
        _check(record_property, f"{case}_{label}", value, wanted, truth)


@pytest.mark.parametrize("case", ["image", "video", "text"])
def test_hidden_state_is_as_accurate_as_the_reference(
    reference, encoded, case, record_property
):
    actual = encoded[case][0]
    expected = reference["bf16"][case][0]
    assert actual.shape == expected.shape == (expected.shape[0], 5120)
    _check(
        record_property,
        f"{case}_hidden",
        actual,
        expected,
        reference["fp32"][case][0],
    )


def test_text_positions_leave_text_encoding_unchanged(encoded):
    # Text advances every M-RoPE axis together, which is exactly the
    # one-dimensional rotary path.
    assert torch.equal(encoded["plain"], encoded["text"][0])
