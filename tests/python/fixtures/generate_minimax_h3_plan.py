"""Generate the MiniMax-H3 planning vectors from the diffusers reference.

The vectors in ``minimax_h3_plan.json`` are consumed by the Rust planner's
unit tests (``crates/server/src/serving/video``) and by the Python planner's
test (``tests/python/unit/models/test_h3_processing.py``). Every expected
value below comes from the diffusers MiniMax-H3 modular pipeline and the
checkpoint's own Qwen3-VL processor and tokenizer, run on synthetic media of
the probed sizes:

* canvases, frame counts and latent counts from the pipeline's
  ``resolve_canvas_size``, ``align_num_frames``, ``video_latent_num_frames``
  and ``audio_latent_num_frames``;
* keyframe fits from ``MiniMaxH3ResizeStep`` (the plan's cover crop is
  checked pixel for pixel against the step's output);
* reference sizes, 24 fps clips and truncated soundtracks from
  ``MiniMaxH3Ref2VASetupStep``;
* vision grids, 2 fps samples, block timestamps, token ids and tags from the
  three text-encoder steps, with the conditioner forward replaced by a
  function that records its inputs.

The reference pipeline leaves four rules open, which UniServe fixes (see
``uniserve_models/minimax_h3/processing.py``) and the generator composes
around the reference steps: the start offset of a video reference, the
22-frame minimum of a reference video, ``ref2va`` keyframes, and the
duration a lone soundtrack implies. Two reference internals are restated
because they run inside model forwards: a video reference keeps its leading
``17 * n + 5`` frames for the VAE (``MiniMaxH3Ref2VAReferenceEncoderStep``)
and the audio VAE pads a waveform to a whole 800-sample hop
(``AutoencoderKLMiniMaxH3Audio.encode``). torchaudio is not a repository
dependency, so a stand-in returns zeros of the length torchaudio's
``Resample`` produces, ``ceil(new * length / orig)`` for the gcd-reduced
rates; only lengths enter the vectors.

The request rules (counts, roles, fields, durations) are UniServe's serving
contract rather than the reference's; their rejection cases name the field
at fault.

Run from the repository root:

    .venv/bin/python tests/python/fixtures/generate_minimax_h3_plan.py \
        --model /path/to/MiniMax-H3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.minimax_h3 import encoders
from diffusers.modular_pipelines.minimax_h3.before_encoder import (
    MiniMaxH3Ref2VASetupStep,
    MiniMaxH3ResizeStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from diffusers.modular_pipelines.minimax_h3.references import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

OUTPUT = Path(__file__).with_name("minimax_h3_plan.json")

FPS = 24
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
MULTIPLE = 32
AUDIO_RATE = 32000
AUDIO_HOP = 800
MIN_REFERENCE_FRAMES = 22


def _install_resample_stand_in() -> None:
    """Make ``torchaudio.transforms.Resample`` return correctly sized zeros.

    torchaudio's ``_apply_sinc_resample_kernel`` truncates its output to
    ``ceil(new_freq * length / orig_freq)`` samples, with both rates divided
    by their gcd.
    """

    class Resample:
        def __init__(self, orig_freq: int, new_freq: int):
            divisor = math.gcd(orig_freq, new_freq)
            self.orig = orig_freq // divisor
            self.new = new_freq // divisor

        def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
            length = math.ceil(self.new * waveform.shape[-1] / self.orig)
            return torch.zeros(*waveform.shape[:-1], length)

    module = types.ModuleType("torchaudio")
    module.transforms = types.SimpleNamespace(Resample=Resample)  # type: ignore[attr-defined]
    sys.modules["torchaudio"] = module


class _Pipeline:
    """The pipeline attributes the preprocessing steps read."""

    canvas_multiple = MULTIPLE
    fps = FPS
    min_duration = 5.0
    max_duration = 15.0
    vae_frames_per_chunk = 17
    vae_latents_per_chunk = 5
    audio_sampling_rate = AUDIO_RATE
    text_tag = 1
    video_tag = 0
    text_encoder_layer = 50
    _execution_device = torch.device("cpu")
    text_encoder = types.SimpleNamespace(dtype=torch.float32)
    config = types.SimpleNamespace(
        canvas_short_edge=SHORT_EDGE,
        canvas_max_pixels=MAX_PIXELS,
        reference_image_short_edge=2048,
    )

    def __init__(self, tokenizer: Any, processor: Any):
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)
        self.tokenizer = tokenizer
        self.processor = processor


class _RecordingTokenizer:
    """Record every text segment the presentation tokenizes, in order."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.segments: list[tuple[str, list[int]]] = []

    def __call__(self, text: str, add_special_tokens: bool) -> dict:
        assert add_special_tokens is False
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        self.segments.append((text, list(ids)))
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)


def _run(step: Any, components: Any, **inputs: Any) -> PipelineState:
    state = PipelineState()
    for name, value in inputs.items():
        state.set(name, value)
    _, state = step(components, state)
    return state


def _encode(
    step: Any, components: _Pipeline, **inputs: Any
) -> tuple[list[int], list[int], dict, list[tuple[str, list[int]]]]:
    """Run a text-encoder step, recording the conditioner's inputs."""
    recorded: dict[str, Any] = {}

    def conditioner(_encoder, _processor, token_ids, vision_inputs, **_):
        recorded["token_ids"] = list(token_ids)
        recorded["vision"] = dict(vision_inputs or {})
        return torch.zeros(1, len(token_ids), 1)

    tokenizer = _RecordingTokenizer(components.tokenizer)
    original = encoders.get_qwen3vl_prompt_embeds
    encoders.get_qwen3vl_prompt_embeds = conditioner
    components_tokenizer = components.tokenizer
    components.tokenizer = tokenizer
    try:
        state = _run(step, components, **inputs)
    finally:
        encoders.get_qwen3vl_prompt_embeds = original
        components.tokenizer = components_tokenizer
    tags = state.get("text_token_tags").tolist()
    return recorded["token_ids"], tags, recorded["vision"], tokenizer.segments


def _segments(
    token_ids: list[int],
    texts: list[tuple[str, list[int]]],
    tokenizer: Any,
) -> list[dict]:
    """Split a presentation into its text segments and vision blocks."""
    start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    pads = {
        tokenizer.convert_tokens_to_ids("<|image_pad|>"): "image",
        tokenizer.convert_tokens_to_ids("<|video_pad|>"): "video",
    }
    segments: list[dict] = []
    position = 0
    pending = list(texts)
    while position < len(token_ids):
        if token_ids[position] == start:
            close = token_ids.index(end, position)
            block = token_ids[position + 1 : close]
            assert len(set(block)) == 1
            segments.append({"vision": pads[block[0]], "tokens": len(block)})
            position = close + 1
            continue
        text, ids = pending.pop(0)
        assert token_ids[position : position + len(ids)] == ids
        segments.append({"text": text, "token_ids": ids})
        position += len(ids)
    # A text segment that tokenizes to nothing still precedes nothing.
    assert all(not ids for _, ids in pending)
    return segments


def _canvas(width: float, height: float) -> list[int]:
    canvas_height, canvas_width = resolve_canvas_size(
        width, height, MULTIPLE, SHORT_EDGE, MAX_PIXELS
    )
    return [canvas_width, canvas_height]


def _vision_config(processor_dir: Path) -> dict:
    image = json.loads((processor_dir / "preprocessor_config.json").read_text())
    video = json.loads(
        (processor_dir / "video_preprocessor_config.json").read_text()
    )
    return {
        "patch_size": image["patch_size"],
        "temporal_patch_size": image["temporal_patch_size"],
        "merge_size": image["merge_size"],
        "image_min_pixels": image["size"]["shortest_edge"],
        "image_max_pixels": image["size"]["longest_edge"],
        "video_min_pixels": video["size"]["shortest_edge"],
        "video_max_pixels": video["size"]["longest_edge"],
    }


def canvas_cases() -> list[dict]:
    aspects = [
        (21, 9),
        (16, 9),
        (4, 3),
        (1, 1),
        (3, 4),
        (9, 16),
        (4, 1),
        (1, 4),
        (7, 4),
        (3, 2),
        (2, 3),
        (5, 4),
        (1920, 1080),
        (1080, 1920),
        (1919, 1080),
        (4000, 3000),
        (3024, 4032),
        (1000, 600),
        (37, 23),
        (853, 480),
        (640, 427),
        (2, 1),
        (1, 2),
        (12, 5),
        (5, 12),
        (400, 101),
        (101, 400),
        (1001, 250),
        (17, 9),
        (9, 17),
        (1200, 299),
    ]
    cases = []
    for width, height in aspects:
        try:
            cases.append(
                {"aspect": [width, height], "canvas": _canvas(width, height)}
            )
        except ValueError:
            cases.append({"aspect": [width, height], "error": True})
    return cases


def duration_cases() -> list[dict]:
    cases = []
    for seconds in (
        4.0,
        4.02,
        4.0625,
        4.5,
        5.0,
        5.1,
        6.0,
        7.3,
        8.0,
        9.99,
        10.0,
        12.0,
        12.5,
        14.479,
        15.0,
    ):
        # The serving range is [4, 15] on the requested duration; the
        # reference's own [5, 15] bound on the aligned one is not applied.
        num_frames = align_num_frames(round(seconds * FPS), 17, 5)
        cases.append(
            {
                "seconds": seconds,
                "num_frames": num_frames,
                "latent_frames": video_latent_num_frames(num_frames, 17, 5),
                "audio_latents": audio_latent_num_frames(num_frames),
            }
        )
    for seconds in (3.99, 15.01, 0.0, -1.0, 30.0):
        cases.append({"seconds": seconds, "error": True})
    return cases


def reference_image_cases(components: _Pipeline, vision: dict) -> list[dict]:
    sizes = [
        (1600, 900),
        (900, 1600),
        (4000, 3000),
        (512, 512),
        (2048, 2048),
        (8192, 2048),
        (2048, 8192),
        (2000, 501),
        (501, 2000),
        (333, 777),
        (3024, 4032),
        (100, 100),
        (1999, 1001),
        (4001, 1000),
    ]
    cases = []
    for width, height in sizes:
        reference = MiniMaxH3ImageReference(
            image=Image.new("RGB", (width, height))
        )
        try:
            state = _run(
                MiniMaxH3Ref2VASetupStep(),
                components,
                references=[reference],
                height=None,
                width=None,
                num_frames=124,
            )
        except ValueError:
            cases.append({"size": [width, height], "error": True})
            continue
        image = state.get("normalized_references")[0].image
        grid = components.processor.image_processor(
            images=[image], return_tensors="pt"
        )["image_grid_thw"][0].tolist()
        resized_width, resized_height = image.size
        cases.append(
            {
                "size": [width, height],
                "resize": [resized_width, resized_height],
                "vision_grid": grid,
                "vision_tokens": grid[1] * grid[2] // vision["merge_size"] ** 2,
                "rows": (resized_width // MULTIPLE)
                * (resized_height // MULTIPLE),
            }
        )
    return cases


def _cover_crop(width: int, height: int, canvas: list[int]) -> dict:
    canvas_width, canvas_height = canvas
    scale = max(canvas_width / width, canvas_height / height)
    resized_width = max(canvas_width, round(width * scale))
    resized_height = max(canvas_height, round(height * scale))
    return {
        "width": resized_width,
        "height": resized_height,
        "left": max(0, (resized_width - canvas_width) // 2),
        "top": max(0, (resized_height - canvas_height) // 2),
    }


def keyframe_cases(components: _Pipeline) -> list[dict]:
    rng = np.random.default_rng(0)
    pairs = [
        ((1344, 768), (1000, 600)),
        ((1344, 768), (640, 640)),
        ((768, 1344), (1920, 1080)),
        ((1024, 768), (1023, 767)),
        ((768, 768), (853, 480)),
        ((1536, 672), (333, 777)),
        ((1248, 800), (1600, 1067)),
        ((1344, 768), (1344, 768)),
        ((672, 1536), (101, 400)),
        ((1344, 768), (2000, 1125)),
    ]
    cases = []
    for canvas, (width, height) in pairs:
        follower = Image.fromarray(
            rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        )
        state = _run(
            MiniMaxH3ResizeStep(),
            components,
            image=Image.new("RGB", canvas),
            last_image=follower,
            height=canvas[1],
            width=canvas[0],
        )
        prepared = state.get("keyframes")[1]
        crop = _cover_crop(width, height, list(canvas))
        expected = follower.resize(
            (crop["width"], crop["height"]), Image.Resampling.LANCZOS
        ).crop(
            (
                crop["left"],
                crop["top"],
                crop["left"] + canvas[0],
                crop["top"] + canvas[1],
            )
        )
        if follower.size == canvas:
            expected = follower
        # The recorded crop reproduces the reference step pixel for pixel.
        assert np.array_equal(np.asarray(prepared), np.asarray(expected))
        cases.append(
            {
                "size": [width, height],
                "canvas": list(canvas),
                "cover_crop": crop,
            }
        )
    return cases


def _video_clip(
    display: tuple[int, int],
    frame_rate: Fraction,
    frames: int,
    start_seconds: float,
    num_frames: int,
) -> np.ndarray:
    """A reference video's 24 fps clip, as the setup step normalizes it.

    The start offset drops the first ``floor(s * 24 + 0.5)`` frames of the
    24 fps timeline the setup step resamples to.
    """
    start_frame = math.floor(start_seconds * FPS + 0.5)
    source = np.broadcast_to(
        np.zeros((1, display[1], display[0], 3), np.uint8),
        (frames, display[1], display[0], 3),
    )
    timeline = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
        source,
        float(frame_rate),
        start_frame + num_frames,
        MULTIPLE,
        SHORT_EDGE,
        MAX_PIXELS,
        float(FPS),
    )
    return timeline[start_frame:]


def video_reference_cases(components: _Pipeline, vision: dict) -> list[dict]:
    specs = [
        # display, frame rate, source frames, start seconds, target frames
        ((16, 9), Fraction(30000, 1001), 300, 0.0, 124),
        ((16, 9), Fraction(24), 120, 0.0, 124),
        ((9, 16), Fraction(30), 50, 0.0, 124),
        ((4, 3), Fraction(25), 375, 3.5, 243),
        ((1, 1), Fraction(60), 1800, 10.0, 362),
        ((37, 23), Fraction(24000, 1001), 100, 0.0, 124),
        ((21, 9), Fraction(50), 1000, 0.25, 175),
        ((3, 4), Fraction(30), 33, 0.0, 124),
        ((16, 9), Fraction(24), 40, 0.5, 124),
        ((16, 9), Fraction(2997, 125), 361, 0.0, 362),
        ((5, 1), Fraction(24), 120, 0.0, 124),
        ((16, 9), Fraction(30), 100, 3.0, 124),
    ]
    cases = []
    merge = vision["merge_size"] ** 2
    for display, frame_rate, frames, start_seconds, num_frames in specs:
        case: dict[str, Any] = {
            "display": list(display),
            "frame_rate": [frame_rate.numerator, frame_rate.denominator],
            "frames": frames,
            "start_seconds": start_seconds,
            "num_frames": num_frames,
        }
        try:
            clip = _video_clip(
                display, frame_rate, frames, start_seconds, num_frames
            )
        except ValueError:
            cases.append({**case, "error": True})
            continue
        if clip.shape[0] < MIN_REFERENCE_FRAMES:
            cases.append({**case, "error": True})
            continue
        clip_frames, canvas_height, canvas_width = clip.shape[:3]
        sampled, timestamps = (
            encoders.MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
                clip, float(FPS), 2.0, vision["temporal_patch_size"]
            )
        )
        grid = components.processor.video_processor(
            videos=[np.stack(sampled)],
            do_sample_frames=False,
            return_tensors="pt",
        )["video_grid_thw"][0].tolist()
        assert grid[0] == len(timestamps)
        # MiniMaxH3Ref2VAReferenceEncoderStep: the leading 17 * n + 5 frames.
        vae_frames = max(1, (clip_frames - 5) // 17) * 17 + 5
        latent_frames = video_latent_num_frames(vae_frames, 17, 5)
        rows_per_frame = (canvas_width // MULTIPLE) * (
            canvas_height // MULTIPLE
        )
        cases.append(
            {
                **case,
                "canvas": [canvas_width, canvas_height],
                "start_frame": math.floor(start_seconds * FPS + 0.5),
                "clip_frames": clip_frames,
                "vae_frames": vae_frames,
                "latent_frames": latent_frames,
                "frame_indices": _sampled_indices(clip_frames),
                "block_timestamps": timestamps,
                "vision_grid": grid,
                "block_tokens": grid[1] * grid[2] // merge,
                "rows": latent_frames * rows_per_frame,
            }
        )
        assert len(cases[-1]["frame_indices"]) == len(sampled)
    return cases


def _sampled_indices(num_frames: int) -> list[int]:
    """The frame indices the reference's 2 fps sampler picks.

    ``_sample_video_condition_frames`` returns the frames themselves; it is
    rerun on an index ramp to recover which ones.
    """
    ramp = np.arange(num_frames).reshape(num_frames, 1, 1, 1)
    frames, _ = (
        encoders.MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
            ramp, float(FPS), 2.0, 2
        )
    )
    return [int(frame[0, 0, 0]) for frame in frames]


def _audio_clip(
    samples: int, sample_rate: int, start_seconds: float, num_frames: int
) -> dict:
    start_sample = math.floor(start_seconds * sample_rate + 0.5)
    waveform = torch.zeros(1, samples)[:, start_sample:]
    truncated = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
        waveform, sample_rate, sample_rate, num_frames / FPS
    )
    resampled = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
        waveform, sample_rate, AUDIO_RATE, num_frames / FPS
    )
    return {
        "start_sample": start_sample,
        "source_samples": truncated.shape[-1],
        "samples": resampled.shape[-1],
        # AutoencoderKLMiniMaxH3Audio.encode pads to a whole 800-sample hop.
        "latents": math.ceil(resampled.shape[-1] / AUDIO_HOP),
    }


def audio_cases() -> list[dict]:
    specs = [
        (32000, 32000 * 20, 0.0, 124),
        (44100, 145530, 0.0, 124),
        (48000, 158400, 0.0, 362),
        (44100, 44100 * 30, 0.0, 124),
        (22050, 22050 * 6, 1.25, 124),
        (8000, 8000 * 4, 0.0, 107),
        (44100, 110592, 0.5, 124),
        (16000, 1, 0.0, 124),
        (11025, 11025 * 15 + 7, 0.0, 362),
        (96000, 96000 * 5, 0.1, 124),
        (32000, 799, 0.0, 124),
        (44100, 2 * 44100, 1.999, 124),
    ]
    cases = []
    for sample_rate, samples, start_seconds, num_frames in specs:
        cases.append(
            {
                "sample_rate": sample_rate,
                "samples": samples,
                "start_seconds": start_seconds,
                "num_frames": num_frames,
                "clip": _audio_clip(
                    samples, sample_rate, start_seconds, num_frames
                ),
            }
        )
    return cases


PROMPT = (
    "integrated_multimodal_description: A red fox trots across fresh snow "
    "at dawn while its breath fogs the air; soft wind, distant birdsong."
)
PROMPT_UNICODE = (
    "一只猫在窗台上晒太阳，尾巴轻轻摆动。\nThe camera slowly pushes in."
)


def _reference_from(entry: dict, num_frames: int) -> Any:
    """A diffusers reference holding synthetic media of the probed facts."""
    media = entry["media"]
    if entry["type"] == "image":
        return MiniMaxH3ImageReference(
            image=Image.new("RGB", (media["width"], media["height"]))
        )
    if entry["type"] == "audio":
        return MiniMaxH3AudioReference(
            audio=torch.zeros(1, media["samples"]),
            sample_rate=media["sample_rate"],
        )
    start = entry.get("start_time_seconds") or 0.0
    frame_rate = Fraction(*media["frame_rate"])
    clip = _video_clip(
        tuple(media["display"]), frame_rate, media["frames"], start, num_frames
    )
    audio = sample_rate = None
    soundtrack = media.get("soundtrack")
    if soundtrack is not None:
        start_sample = math.floor(start * soundtrack["sample_rate"] + 0.5)
        audio = torch.zeros(1, soundtrack["samples"])[:, start_sample:]
        sample_rate = soundtrack["sample_rate"]
    return MiniMaxH3VideoReference(
        frames=np.ascontiguousarray(clip),
        fps=float(FPS),
        audio=audio,
        sample_rate=sample_rate,
    )


def _named_canvas(aspect_ratio: str, default: tuple[int, int]) -> list[int]:
    if aspect_ratio == "auto":
        return _canvas(*default)
    width, height = (int(part) for part in aspect_ratio.split(":"))
    return _canvas(width, height)


def _derived_frames(request: dict) -> int:
    target = request["target"]
    if target.get("duration_seconds") is not None:
        return align_num_frames(round(target["duration_seconds"] * FPS), 17, 5)
    # The one soundtrack sets the duration after its start offset.
    for entry in request["conditions"]:
        media = entry["media"]
        track = media if entry["type"] == "audio" else media.get("soundtrack")
        if entry["role"] == "reference" and track is not None:
            start = entry.get("start_time_seconds") or 0.0
            start_sample = math.floor(start * track["sample_rate"] + 0.5)
            seconds = (track["samples"] - start_sample) / track["sample_rate"]
            return align_num_frames(round(seconds * FPS), 17, 5)
    raise AssertionError("no soundtrack")


def _keyframe_entries(request: dict) -> list[dict]:
    return [
        entry for entry in request["conditions"] if entry["role"] == "keyframe"
    ]


def _keyframe_plans(
    components: _Pipeline, keyframes: list[dict], canvas: list[int]
) -> tuple[list[dict], list[Image.Image]]:
    images = [
        Image.new("RGB", (entry["media"]["width"], entry["media"]["height"]))
        for entry in keyframes
    ]
    plans = []
    for order, (entry, image) in enumerate(zip(keyframes, images)):
        plans.append(
            {
                "index": None,
                "kind": "keyframe",
                "position": "first" if entry["frame_index"] == 0 else "last",
                "cover_crop": None
                if order == 0
                else _cover_crop(*image.size, canvas),
                "video_rows": (canvas[0] // MULTIPLE) * (canvas[1] // MULTIPLE),
                "audio_rows": 0,
            }
        )
    state = _run(
        MiniMaxH3ResizeStep(),
        components,
        image=images[0] if keyframes[0]["frame_index"] == 0 else None,
        last_image=(images[-1] if keyframes[-1]["frame_index"] == -1 else None),
        height=canvas[1],
        width=canvas[0],
    )
    prepared = state.get("keyframes")
    assert [image.size for image in prepared] == [tuple(canvas)] * len(
        keyframes
    )
    return plans, prepared


def expected_request(
    components: _Pipeline, request: dict, vision: dict
) -> dict:
    task = request["task"]
    prompt = request["prompt"]
    target = request["target"]
    merge = vision["merge_size"] ** 2
    num_frames = _derived_frames(request)
    conditions: list[dict] = []
    if task == "t2va":
        canvas = _named_canvas(target["aspect_ratio"], (16, 9))
        token_ids, tags, _, texts = _encode(
            encoders.MiniMaxH3TextEncoderStep(), components, prompt=prompt
        )
    elif task == "fl2va":
        keyframes = _keyframe_entries(request)
        if target["aspect_ratio"] == "auto":
            first = keyframes[0]["media"]
            canvas = _canvas(first["width"], first["height"])
            # The step resolves the same canvas from the keyframe itself.
            images = [
                Image.new(
                    "RGB", (entry["media"]["width"], entry["media"]["height"])
                )
                for entry in keyframes
            ]
            state = _run(
                MiniMaxH3ResizeStep(),
                components,
                image=images[0] if keyframes[0]["frame_index"] == 0 else None,
                last_image=images[-1]
                if keyframes[-1]["frame_index"] == -1
                else None,
                height=None,
                width=None,
            )
            assert [state.get("width"), state.get("height")] == canvas
        else:
            canvas = _named_canvas(target["aspect_ratio"], (16, 9))
        plans, prepared = _keyframe_plans(components, keyframes, canvas)
        token_ids, tags, vision_inputs, texts = _encode(
            encoders.MiniMaxH3FL2VATextEncoderStep(),
            components,
            prompt=prompt,
            keyframes=prepared,
        )
        grids = vision_inputs["image_grid_thw"].tolist()
        for index, (plan, grid) in enumerate(zip(plans, grids)):
            plan["index"] = index
            plan["vision"] = {
                "grid": grid,
                "tokens": grid[1] * grid[2] // merge,
            }
        conditions = plans
    else:
        canvas = _named_canvas(target["aspect_ratio"], (16, 9))
        entries = request["conditions"]
        references = [
            entry for entry in entries if entry["role"] == "reference"
        ]
        state = _run(
            MiniMaxH3Ref2VASetupStep(),
            components,
            references=[
                _reference_from(entry, num_frames) for entry in references
            ],
            height=canvas[1],
            width=canvas[0],
            num_frames=num_frames,
        )
        assert state.get("num_frames") == num_frames
        normalized = state.get("normalized_references")
        token_ids, tags, vision_inputs, texts = _encode(
            encoders.MiniMaxH3Ref2VATextEncoderStep(),
            components,
            prompt=prompt,
            normalized_references=normalized,
        )
        image_grids = iter(
            vision_inputs.get("image_grid_thw", torch.zeros(0, 3)).tolist()
        )
        video_grids = iter(
            vision_inputs.get("video_grid_thw", torch.zeros(0, 3)).tolist()
        )
        keyframes = _keyframe_entries(request)
        keyframe_plans = []
        if keyframes:
            keyframe_plans, _ = _keyframe_plans(components, keyframes, canvas)
        keyframe_plans_iter = iter(keyframe_plans)
        normalized_iter = iter(normalized)
        for index, entry in enumerate(entries):
            if entry["role"] == "keyframe":
                plan = next(keyframe_plans_iter)
                plan["index"] = index
                plan["vision"] = None
                conditions.append(plan)
                continue
            reference = next(normalized_iter)
            media = entry["media"]
            if entry["type"] == "image":
                width, height = reference.image.size
                grid = next(image_grids)
                conditions.append(
                    {
                        "index": index,
                        "kind": "image",
                        "resize": [width, height],
                        "vision": {
                            "grid": grid,
                            "tokens": grid[1] * grid[2] // merge,
                        },
                        "video_rows": (width // MULTIPLE)
                        * (height // MULTIPLE),
                        "audio_rows": 0,
                    }
                )
            elif entry["type"] == "audio":
                clip = _audio_clip(
                    media["samples"], media["sample_rate"], 0.0, num_frames
                )
                assert reference.audio.shape[-1] == clip["samples"]
                conditions.append(
                    {
                        "index": index,
                        "kind": "audio",
                        "clip": clip,
                        "vision": None,
                        "video_rows": 0,
                        "audio_rows": 2 * clip["latents"],
                    }
                )
            else:
                frames = reference.frames
                clip_frames, canvas_height, canvas_width = frames.shape[:3]
                grid = next(video_grids)
                _, timestamps = (
                    encoders.MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
                        frames, float(FPS), 2.0, vision["temporal_patch_size"]
                    )
                )
                vae_frames = max(1, (clip_frames - 5) // 17) * 17 + 5
                latent_frames = video_latent_num_frames(vae_frames, 17, 5)
                start = entry.get("start_time_seconds") or 0.0
                soundtrack = None
                if media.get("soundtrack") is not None:
                    track = media["soundtrack"]
                    soundtrack = _audio_clip(
                        track["samples"],
                        track["sample_rate"],
                        start,
                        num_frames,
                    )
                    assert reference.audio.shape[-1] == soundtrack["samples"]
                conditions.append(
                    {
                        "index": index,
                        "kind": "video",
                        "canvas": [canvas_width, canvas_height],
                        "start_frame": math.floor(start * FPS + 0.5),
                        "clip_frames": clip_frames,
                        "vae_frames": vae_frames,
                        "latent_frames": latent_frames,
                        "soundtrack": soundtrack,
                        "vision": {
                            "grid": grid,
                            "block_tokens": grid[1] * grid[2] // merge,
                            "frame_indices": _sampled_indices(clip_frames),
                            "block_timestamps": timestamps,
                        },
                        "video_rows": latent_frames
                        * (canvas_width // MULTIPLE)
                        * (canvas_height // MULTIPLE),
                        "audio_rows": 0
                        if soundtrack is None
                        else 2 * soundtrack["latents"],
                    }
                )

    latent_frames = video_latent_num_frames(num_frames, 17, 5)
    audio_latents = audio_latent_num_frames(num_frames)
    segments = _segments(token_ids, texts, components.tokenizer)
    # The segments encode the presentation losslessly: the vectors keep them
    # rather than the (long, mostly vision-pad) token sequence.
    assert _join(segments, components.tokenizer) == (token_ids, tags)
    return {
        "canvas": canvas,
        "num_frames": num_frames,
        "latent_frames": latent_frames,
        "audio_latents": audio_latents,
        "target_video_rows": latent_frames
        * (canvas[0] // MULTIPLE)
        * (canvas[1] // MULTIPLE),
        "target_audio_rows": 2 * audio_latents,
        "conditions": conditions,
        "condition_video_rows": sum(plan["video_rows"] for plan in conditions),
        "condition_audio_rows": sum(plan["audio_rows"] for plan in conditions),
        "segments": segments,
    }


def _join(segments: list[dict], tokenizer: Any) -> tuple[list[int], list[int]]:
    """Rebuild the token ids and tags of a presentation from its segments."""
    token_ids: list[int] = []
    tags: list[int] = []
    for segment in segments:
        if "vision" in segment:
            pad = tokenizer.convert_tokens_to_ids(
                f"<|{segment['vision']}_pad|>"
            )
            block = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [pad] * segment["tokens"]
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += block
            tags += [0] * len(block)
        else:
            token_ids += segment["token_ids"]
            tags += [1] * len(segment["token_ids"])
    return token_ids, tags


def _image(width: int, height: int) -> dict:
    return {"width": width, "height": height}


def _video(display, frame_rate, frames, soundtrack=None) -> dict:
    return {
        "display": list(display),
        "frame_rate": list(frame_rate),
        "frames": frames,
        "soundtrack": soundtrack,
    }


def _audio(sample_rate: int, samples: int) -> dict:
    return {"sample_rate": sample_rate, "samples": samples}


def keyframe(media: dict, index: int) -> dict:
    return {
        "type": "image",
        "role": "keyframe",
        "frame_index": index,
        "media": media,
    }


def reference(kind: str, media: dict, start: float | None = None) -> dict:
    return {
        "type": kind,
        "role": "reference",
        "media": media,
        "start_time_seconds": start,
    }


def requests() -> list[dict]:
    """Requests whose plans the vectors record, then rejected requests."""
    accepted = [
        {
            "name": "t2va_default",
            "task": "t2va",
            "prompt": PROMPT,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": 5.0,
            },
            "conditions": [],
        },
        {
            "name": "t2va_portrait_unicode",
            "task": "t2va",
            "prompt": PROMPT_UNICODE,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "9:16",
                "duration_seconds": 15.0,
            },
            "conditions": [],
        },
        {
            "name": "fl2va_first_auto",
            "task": "fl2va",
            "prompt": PROMPT,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": 8.0,
            },
            "conditions": [keyframe(_image(1000, 600), 0)],
        },
        {
            "name": "fl2va_last_only",
            "task": "fl2va",
            "prompt": PROMPT,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": 5.0,
            },
            "conditions": [keyframe(_image(3024, 4032), -1)],
        },
        {
            "name": "fl2va_first_last_explicit",
            "task": "fl2va",
            "prompt": PROMPT_UNICODE,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "7:4",
                "duration_seconds": 6.0,
            },
            "conditions": [
                keyframe(_image(1920, 1080), 0),
                keyframe(_image(640, 640), -1),
            ],
        },
        {
            "name": "ref2va_image",
            "task": "ref2va",
            "prompt": PROMPT,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": 5.0,
            },
            "conditions": [reference("image", _image(1600, 900))],
        },
        {
            "name": "ref2va_image_audio",
            "task": "ref2va",
            "prompt": PROMPT,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "4:3",
                "duration_seconds": 5.0,
            },
            "conditions": [
                reference("image", _image(900, 1600)),
                reference("audio", _audio(44100, 44100 * 9)),
            ],
        },
        {
            "name": "ref2va_video_audio",
            "task": "ref2va",
            "prompt": PROMPT,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": 5.0,
            },
            "conditions": [
                reference(
                    "video_audio",
                    _video(
                        (16, 9), (30000, 1001), 300, _audio(44100, 44100 * 10)
                    ),
                    1.5,
                ),
                reference("audio", _audio(48000, 48000 * 3)),
            ],
        },
        {
            "name": "ref2va_two_videos",
            "task": "ref2va",
            "prompt": PROMPT_UNICODE,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 6.0,
            },
            "conditions": [
                reference("video", _video((9, 16), (25, 1), 75)),
                reference("image", _image(512, 512)),
                reference(
                    "video",
                    _video((4, 3), (60, 1), 900, _audio(32000, 32000 * 15)),
                ),
            ],
        },
        {
            "name": "ref2va_duration_from_audio",
            "task": "ref2va",
            "prompt": PROMPT,
            "target": {"short_edge": 768, "aspect_ratio": "1:1"},
            "conditions": [
                reference("image", _image(1080, 1350)),
                reference("audio", _audio(44100, 145530 * 2)),
            ],
        },
        {
            "name": "ref2va_mixed_keyframes",
            "task": "ref2va",
            "prompt": PROMPT,
            "target": {
                "short_edge": 768,
                "aspect_ratio": "auto",
                "duration_seconds": 5.0,
            },
            "conditions": [
                reference("image", _image(1600, 900)),
                keyframe(_image(1920, 1080), 0),
                keyframe(_image(1000, 1000), -1),
            ],
        },
    ]
    rejected = [
        (
            "t2va_with_condition",
            "t2va",
            {"duration_seconds": 5.0},
            [reference("image", _image(64, 64))],
            "conditions",
        ),
        (
            "short_edge",
            "t2va",
            {"short_edge": 720, "duration_seconds": 5.0},
            [],
            "target.short_edge",
        ),
        (
            "t2va_free_ratio",
            "t2va",
            {"aspect_ratio": "7:4", "duration_seconds": 5.0},
            [],
            "target.aspect_ratio",
        ),
        (
            "fl2va_extreme_ratio",
            "fl2va",
            {"aspect_ratio": "5:1", "duration_seconds": 5.0},
            [keyframe(_image(640, 480), 0)],
            "target.aspect_ratio",
        ),
        ("duration_missing", "t2va", {}, [], "target.duration_seconds"),
        (
            "duration_short",
            "t2va",
            {"duration_seconds": 3.9},
            [],
            "target.duration_seconds",
        ),
        (
            "duration_long",
            "t2va",
            {"duration_seconds": 15.5},
            [],
            "target.duration_seconds",
        ),
        (
            "fl2va_reference",
            "fl2va",
            {"duration_seconds": 5.0},
            [keyframe(_image(640, 480), 0), reference("image", _image(64, 64))],
            "conditions[1]",
        ),
        (
            "fl2va_reversed",
            "fl2va",
            {"duration_seconds": 5.0},
            [keyframe(_image(640, 480), -1), keyframe(_image(640, 480), 0)],
            "conditions",
        ),
        (
            "fl2va_middle_frame",
            "fl2va",
            {"duration_seconds": 5.0},
            [keyframe(_image(640, 480), 12)],
            "conditions[0]",
        ),
        (
            "fl2va_keyframe_ratio",
            "fl2va",
            {"duration_seconds": 5.0},
            [keyframe(_image(5000, 1000), 0)],
            "conditions[0]",
        ),
        (
            "keyframe_video",
            "ref2va",
            {"duration_seconds": 5.0},
            [
                reference("image", _image(64, 64)),
                {
                    "type": "video",
                    "role": "keyframe",
                    "frame_index": 0,
                    "media": _video((16, 9), (24, 1), 48),
                },
            ],
            "conditions[1]",
        ),
        (
            "start_on_image",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("image", _image(64, 64), 1.0)],
            "conditions[0]",
        ),
        (
            "too_many_images",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("image", _image(64, 64)) for _ in range(10)],
            "conditions",
        ),
        (
            "audio_only",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("audio", _audio(32000, 32000))],
            "conditions",
        ),
        (
            "reference_image_ratio",
            "ref2va",
            {"duration_seconds": 5.0},
            [
                reference("image", _image(64, 64)),
                reference("image", _image(4100, 1000)),
            ],
            "conditions[1]",
        ),
        (
            "short_video",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("video", _video((16, 9), (24, 1), 21))],
            "conditions[0]",
        ),
        (
            "start_past_video",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("video", _video((16, 9), (24, 1), 120), 4.5)],
            "conditions[0]",
        ),
        (
            "video_audio_silent",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("video_audio", _video((16, 9), (24, 1), 120))],
            "conditions[0]",
        ),
        (
            "duration_two_soundtracks",
            "ref2va",
            {},
            [
                reference("audio", _audio(32000, 32000 * 6)),
                reference(
                    "video",
                    _video((16, 9), (24, 1), 120, _audio(32000, 32000 * 6)),
                ),
            ],
            "target.duration_seconds",
        ),
        (
            "duration_no_soundtrack",
            "ref2va",
            {},
            [reference("video", _video((16, 9), (24, 1), 120))],
            "target.duration_seconds",
        ),
        (
            "duration_from_short_audio",
            "ref2va",
            {},
            [
                reference("image", _image(64, 64)),
                reference("audio", _audio(32000, 32000 * 3)),
            ],
            "conditions[1]",
        ),
        (
            "duration_images_only",
            "ref2va",
            {},
            [reference("image", _image(64, 64))],
            "target.duration_seconds",
        ),
    ]
    cases = list(accepted)
    for name, task, target, conditions, field in rejected:
        cases.append(
            {
                "name": name,
                "task": task,
                "prompt": PROMPT,
                "target": {"short_edge": 768, "aspect_ratio": "auto", **target},
                "conditions": conditions,
                "error": {"field": field},
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", type=Path, required=True, help="MiniMax-H3 checkpoint root"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    _install_resample_stand_in()
    tokenizer = AutoTokenizer.from_pretrained(args.model / "tokenizer")
    processor = AutoProcessor.from_pretrained(args.model / "processor")
    components = _Pipeline(tokenizer, processor)
    vision = _vision_config(args.model / "processor")

    cases = requests()
    for case in cases:
        if "error" not in case:
            case["expected"] = expected_request(components, case, vision)

    fixture = {
        "description": (
            "MiniMax-H3 request planning vectors shared by the Rust planner "
            "(crates/server/src/serving/video) and the Python planner "
            "(uniserve_models/minimax_h3/processing.py), generated from the "
            "diffusers MiniMax-H3 pipeline by "
            "tests/python/fixtures/generate_minimax_h3_plan.py, whose "
            "docstring lists each value's source. Sizes are [width, height]; "
            "vision grids are Qwen [t, h, w] patch grids; rows are denoiser "
            "rows. Media are described by probed facts: an image's displayed "
            "size, a video's display aspect, frame rate [num, den], decoded "
            "frame count and soundtrack, a soundtrack's sample rate and "
            "decoded sample count. A rejected request names the field at "
            "fault."
        ),
        "versions": {
            "diffusers": __import__("diffusers").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "vision": vision,
        "tokens": {
            name: tokenizer.convert_tokens_to_ids(name)
            for name in (
                "<|vision_start|>",
                "<|vision_end|>",
                "<|image_pad|>",
                "<|video_pad|>",
            )
        },
        "canvas_cases": canvas_cases(),
        "duration_cases": duration_cases(),
        "reference_image_cases": reference_image_cases(components, vision),
        "keyframe_cases": keyframe_cases(components),
        "video_reference_cases": video_reference_cases(components, vision),
        "audio_cases": audio_cases(),
        "requests": cases,
    }
    args.output.write_text(_format(fixture))


def _format(fixture: dict) -> str:
    """Lay the vectors out one case per line."""
    lines = ["{"]
    items = list(fixture.items())
    for position, (key, value) in enumerate(items):
        comma = "," if position < len(items) - 1 else ""
        if isinstance(value, list):
            lines.append(f"  {json.dumps(key)}: [")
            for index, case in enumerate(value):
                tail = "," if index < len(value) - 1 else ""
                lines.append(
                    f"    {json.dumps(case, ensure_ascii=False)}{tail}"
                )
            lines.append(f"  ]{comma}")
        else:
            rendered = json.dumps(value, ensure_ascii=False)
            lines.append(f"  {json.dumps(key)}: {rendered}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
