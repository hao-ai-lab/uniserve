"""MiniMax-H3 request planning matches the diffusers reference vectors.

The vectors in ``minimax_h3_plan.json`` are shared with the Rust planner's
tests (``crates/server/src/serving/video``), so both implementations are
checked against the same reference values.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from uniserve.media import image
from uniserve_models.minimax_h3 import processing
from uniserve_models.minimax_h3.processing import (
    AudioClip,
    AudioFacts,
    Condition,
    ConditionPlan,
    ConditionType,
    ImageFacts,
    ImageVision,
    KeyframeFit,
    PlanError,
    RequestPlan,
    Role,
    Target,
    Task,
    TextSegment,
    VideoClip,
    VideoFacts,
    VideoVision,
    VisionConfig,
    VisionSegment,
)

pytestmark = pytest.mark.unit

_FIXTURE = json.loads(
    (
        Path(__file__).parents[2] / "fixtures" / "minimax_h3_plan.json"
    ).read_text()
)
_VISION = VisionConfig(**_FIXTURE["vision"])
_ACCEPTED = [case for case in _FIXTURE["requests"] if "expected" in case]
_REJECTED = [case for case in _FIXTURE["requests"] if "error" in case]


def _size(width_height: list[int]) -> image.Config:
    return image.Config(height=width_height[1], width=width_height[0])


def _names(cases: list[dict[str, Any]]) -> list[str]:
    return [case["name"] for case in cases]


@pytest.mark.parametrize("case", _FIXTURE["canvas_cases"])
def test_canvas(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            processing.canvas(*case["aspect"])
        return
    assert processing.canvas(*case["aspect"]) == _size(case["canvas"])


@pytest.mark.parametrize("case", _FIXTURE["duration_cases"])
def test_duration(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            processing.frame_count(case["seconds"])
        return
    plan = processing.plan_request(
        Task.T2VA, Target(duration_seconds=case["seconds"]), [], _VISION
    )
    assert processing.frame_count(case["seconds"]) == case["num_frames"]
    assert plan.num_frames == case["num_frames"]
    assert plan.latent_frames == case["latent_frames"]
    assert plan.audio_latents == case["audio_latents"]


@pytest.mark.parametrize("case", _FIXTURE["reference_image_cases"])
def test_reference_image(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            processing.reference_image_size(*case["size"])
        return
    size = processing.reference_image_size(*case["size"])
    assert size == _size(case["resize"])
    grid = _VISION.image_grid(size.height, size.width)
    assert list(grid) == case["vision_grid"]
    assert _VISION.block_tokens(grid) == case["vision_tokens"]
    assert processing.rows_per_frame(size) == case["rows"]


@pytest.mark.parametrize("case", _FIXTURE["keyframe_cases"])
def test_keyframe_cover_crop(case):
    crop = processing.cover_crop(*case["size"], _size(case["canvas"]))
    assert {
        "width": crop.width,
        "height": crop.height,
        "left": crop.left,
        "top": crop.top,
    } == case["cover_crop"]


@pytest.mark.parametrize("case", _FIXTURE["video_reference_cases"])
def test_video_reference(case):
    media = VideoFacts(
        display_width=case["display"][0],
        display_height=case["display"][1],
        frame_rate=Fraction(*case["frame_rate"]),
        frames=case["frames"],
    )
    arguments = (media, case["start_seconds"], case["num_frames"], _VISION)
    if case.get("error"):
        with pytest.raises(ValueError):
            processing.video_reference(*arguments)
        return
    clip, seen = processing.video_reference(*arguments)
    assert clip.canvas == _size(case["canvas"])
    assert clip.start_frame == case["start_frame"]
    assert clip.frames == case["clip_frames"]
    assert clip.vae_frames == case["vae_frames"]
    assert clip.latent_frames == case["latent_frames"]
    assert list(seen.frame_indices) == case["frame_indices"]
    assert list(seen.block_timestamps) == case["block_timestamps"]
    assert list(seen.grid) == case["vision_grid"]
    assert seen.block_tokens == case["block_tokens"]
    rows = clip.latent_frames * processing.rows_per_frame(clip.canvas)
    assert rows == case["rows"]


@pytest.mark.parametrize("case", _FIXTURE["audio_cases"])
def test_audio_clip(case):
    clip = processing.audio_clip(
        AudioFacts(case["sample_rate"], case["samples"]),
        case["start_seconds"],
        case["num_frames"],
    )
    assert _audio(clip) == case["clip"]


def _media(entry: dict[str, Any]) -> ImageFacts | VideoFacts | AudioFacts:
    media = entry["media"]
    if "width" in media:
        return ImageFacts(media["width"], media["height"])
    if "display" in media:
        track = media.get("soundtrack")
        return VideoFacts(
            display_width=media["display"][0],
            display_height=media["display"][1],
            frame_rate=Fraction(*media["frame_rate"]),
            frames=media["frames"],
            soundtrack=None if track is None else AudioFacts(**track),
        )
    return AudioFacts(media["sample_rate"], media["samples"])


def _plan(case: dict[str, Any]) -> RequestPlan:
    target = case["target"]
    conditions = [
        Condition(
            type=ConditionType(entry["type"]),
            role=Role(entry["role"]),
            media=_media(entry),
            frame_index=entry.get("frame_index"),
            start_seconds=entry.get("start_time_seconds"),
        )
        for entry in case["conditions"]
    ]
    return processing.plan_request(
        Task(case["task"]),
        Target(
            aspect_ratio=target["aspect_ratio"],
            duration_seconds=target.get("duration_seconds"),
            short_edge=target["short_edge"],
        ),
        conditions,
        _VISION,
    )


def _audio(clip: AudioClip) -> dict[str, int]:
    return {
        "start_sample": clip.start_sample,
        "source_samples": clip.source_samples,
        "samples": clip.samples,
        "latents": clip.latents,
    }


def _condition(plan: ConditionPlan) -> dict[str, Any]:
    """Express a condition plan in the vectors' layout."""
    prepared, seen = plan.prepared, plan.vision
    entry: dict[str, Any] = {
        "index": plan.index,
        "video_rows": plan.video_rows,
        "audio_rows": plan.audio_rows,
        "vision": None,
    }
    if isinstance(seen, ImageVision):
        entry["vision"] = {"grid": list(seen.grid), "tokens": seen.tokens}
    elif isinstance(seen, VideoVision):
        entry["vision"] = {
            "grid": list(seen.grid),
            "block_tokens": seen.block_tokens,
            "frame_indices": list(seen.frame_indices),
            "block_timestamps": list(seen.block_timestamps),
        }

    if isinstance(prepared, KeyframeFit):
        crop = prepared.cover_crop
        entry |= {
            "kind": "keyframe",
            "position": prepared.position.value,
            "cover_crop": None
            if crop is None
            else {
                "width": crop.width,
                "height": crop.height,
                "left": crop.left,
                "top": crop.top,
            },
        }
    elif isinstance(prepared, image.Config):
        entry |= {"kind": "image", "resize": [prepared.width, prepared.height]}
    elif isinstance(prepared, AudioClip):
        entry |= {"kind": "audio", "clip": _audio(prepared)}
    else:
        assert isinstance(prepared, VideoClip)
        track = prepared.soundtrack
        entry |= {
            "kind": "video",
            "canvas": [prepared.canvas.width, prepared.canvas.height],
            "start_frame": prepared.start_frame,
            "clip_frames": prepared.frames,
            "vae_frames": prepared.vae_frames,
            "latent_frames": prepared.latent_frames,
            "soundtrack": None if track is None else _audio(track),
        }
    return entry


@pytest.mark.parametrize("case", _ACCEPTED, ids=_names(_ACCEPTED))
def test_request_plan(case):
    plan = _plan(case)
    expected = case["expected"]
    assert plan.canvas == _size(expected["canvas"])
    assert plan.num_frames == expected["num_frames"]
    assert plan.latent_frames == expected["latent_frames"]
    assert plan.audio_latents == expected["audio_latents"]
    assert plan.target_video_rows == expected["target_video_rows"]
    assert plan.target_audio_rows == expected["target_audio_rows"]
    conditions = [_condition(item) for item in plan.conditions]
    assert conditions == expected["conditions"]
    assert plan.condition_video_rows == expected["condition_video_rows"]
    assert plan.condition_audio_rows == expected["condition_audio_rows"]


@pytest.mark.parametrize("case", _ACCEPTED, ids=_names(_ACCEPTED))
def test_presentation_segments(case):
    segments = processing.presentation_segments(_plan(case), case["prompt"])
    expected = [
        VisionSegment(f"<|{item['vision']}_pad|>", item["tokens"])
        if "vision" in item
        else TextSegment(item["text"])
        for item in case["expected"]["segments"]
    ]
    assert segments == expected


class _RecordedTokenizer:
    """The checkpoint tokenizer's recorded output on the vectors' texts."""

    def __init__(self) -> None:
        self.ids = dict(_FIXTURE["tokens"])
        self.texts = {
            item["text"]: item["token_ids"]
            for case in _ACCEPTED
            for item in case["expected"]["segments"]
            if "text" in item
        }

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(self.texts[text])

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids[token]


def _joined(case: dict[str, Any]) -> tuple[list[int], list[int]]:
    """The expected token ids and tags, rebuilt from the segments."""
    tokens = _FIXTURE["tokens"]
    token_ids: list[int] = []
    tags: list[int] = []
    for item in case["expected"]["segments"]:
        if "vision" in item:
            pad = tokens[f"<|{item['vision']}_pad|>"]
            block = [
                tokens["<|vision_start|>"],
                *([pad] * item["tokens"]),
                tokens["<|vision_end|>"],
            ]
            token_ids += block
            tags += [0] * len(block)
        else:
            token_ids += item["token_ids"]
            tags += [1] * len(item["token_ids"])
    return token_ids, tags


@pytest.mark.parametrize("case", _ACCEPTED, ids=_names(_ACCEPTED))
def test_presentation_tokens(case):
    presentation = processing.present(
        _RecordedTokenizer(), _plan(case), case["prompt"]
    )
    tokens = (list(presentation.token_ids), list(presentation.tags))
    assert tokens == _joined(case)


@pytest.mark.parametrize("case", _ACCEPTED, ids=_names(_ACCEPTED))
def test_presentation_with_checkpoint_tokenizer(case):
    root = os.environ.get("UNISERVE_MINIMAX_H3_MODEL")
    if not root:
        pytest.skip("UNISERVE_MINIMAX_H3_MODEL names no MiniMax-H3 checkpoint")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        Path(root) / "tokenizer"
    )
    presentation = processing.present(tokenizer, _plan(case), case["prompt"])
    tokens = (list(presentation.token_ids), list(presentation.tags))
    assert tokens == _joined(case)


def test_prompt_must_be_plain_text():
    case = _ACCEPTED[0]
    tokenizer = _RecordedTokenizer()
    tokenizer.texts["<|image_pad|>"] = [_FIXTURE["tokens"]["<|image_pad|>"]]
    for prompt in ("<|image_pad|>", ""):
        with pytest.raises(PlanError) as error:
            processing.present(tokenizer, _plan(case), prompt)
        assert error.value.field == "prompt"


@pytest.mark.parametrize("case", _REJECTED, ids=_names(_REJECTED))
def test_rejected_request(case):
    with pytest.raises(PlanError) as error:
        _plan(case)
    assert error.value.field == case["error"]["field"]


def test_checkpoint_canvases_restrict_the_target():
    served = [image.Config(height=768, width=1344)]
    plan = processing.plan_request(
        Task.T2VA, Target(duration_seconds=5.0), [], _VISION, canvases=served
    )
    assert plan.canvas == served[0]
    with pytest.raises(PlanError) as error:
        processing.plan_request(
            Task.T2VA,
            Target(aspect_ratio="9:16", duration_seconds=5.0),
            [],
            _VISION,
            canvases=served,
        )
    assert error.value.field == "target.aspect_ratio"


def test_only_served_tasks_are_accepted():
    with pytest.raises(PlanError) as error:
        processing.plan_request(
            Task.T2VA,
            Target(duration_seconds=5.0),
            [],
            _VISION,
            tasks=(Task.REF2VA,),
        )
    assert error.value.field == "task"


def test_max_seconds_bounds_the_duration():
    with pytest.raises(PlanError) as error:
        processing.plan_request(
            Task.T2VA,
            Target(duration_seconds=10.0),
            [],
            _VISION,
            max_seconds=8.0,
        )
    assert error.value.field == "target.duration_seconds"
