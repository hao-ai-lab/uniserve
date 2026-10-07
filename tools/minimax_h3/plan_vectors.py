r"""Generate the MiniMax-H3 request-planning vectors from diffusers.

Writes ``tests/python/fixtures/minimax_h3_plan.json``, the one set of vectors
both request planners are tested against: the Rust server planner
(``crates/server/src/serving/video/plan.rs`` and ``presentation.rs``) and the
Python model package (``uniserve_models/minimax_h3/processing.py``, tested by
``tests/python/unit/models/test_h3_processing.py``).

Every expected value comes from executing the diffusers 0.40 MiniMax-H3
integration with the checkpoint's own Qwen3-VL processor, tokenizer and audio
VAE, on synthetic media of the stated facts (zero pixels and samples, and
random pixels where a keyframe fit is compared):

* canvases, frame counts and latent counts: ``resolve_canvas_size``,
  ``align_num_frames``, ``video_latent_num_frames`` and
  ``audio_latent_num_frames``;
* keyframe fits: ``MiniMaxH3ResizeStep``; each recorded fit reproduces the
  step's output pixel for pixel;
* reference image sizes, 24 fps clips and 32 kHz soundtracks:
  ``MiniMaxH3Ref2VASetupStep``, torchaudio resampling included;
* vision grids, 2 fps samples, block timestamps, token ids and tags: the
  three text-encoder steps, with the Qwen3-VL forward replaced by a recorder
  of its inputs and the tokenizer wrapped to record every text it tokenizes;
  the frames the 2 fps sampler picks come from rerunning it on an index ramp;
* the frames the video VAE encodes, condition latent shapes and soundtrack
  latents: ``MiniMaxH3Ref2VAReferenceEncoderStep`` with the checkpoint's
  audio VAE and a video VAE stand-in that records the frames it is given and
  returns latents of the VAE's geometry (the VAE class's spatial compression
  and chunking, ``video_latent_num_frames``), since a CPU video VAE forward
  at canvas size is impractical and only shapes are recorded;
* text, condition, target and sequence rows: the packed layouts of
  ``MiniMaxH3PrepareLayoutStep.build_packed_sequence`` (``t2va``, ``fl2va``)
  and ``MiniMaxH3Ref2VAPrepareLayoutStep`` (``ref2va``).

The reference leaves four rules open. Both planners fix them identically
(see the top of ``plan.rs``) and the generator composes them around the
reference steps: a video reference's ``start_time_seconds`` drops the first
``floor(s * 24 + 0.5)`` frames of the 24 fps timeline the setup step
resamples to and the first ``floor(s * rate + 0.5)`` samples of its
soundtrack; a reference video keeps at least 22 frames (one ``17 * n + 5``
VAE window) after the offset; ``ref2va`` keyframes are fitted to the target
canvas like ``fl2va`` ones, add one target latent frame of rows each and do
not enter the conditioner; without ``target.duration_seconds`` the one
soundtrack of a ``ref2va`` request sets the duration after its offset. The
reference also checks the aligned duration against ``[5, 15]`` seconds,
whereas UniServe serves requested durations of 4 to 15 seconds: the
``t2va`` and ``fl2va`` cases outside the reference's range take their values
from the same helpers and layout builder, which carry no duration check, and
the ``ref2va`` cases stay inside it. The request rules (task, counts, roles,
fields, durations, prompt) are UniServe's serving contract; the rejected
requests name the field at fault.

Run on CPU from the repository root, with the reference environment's
accelerate and torchaudio overlay on ``PYTHONPATH`` (see
``diffusers_reference.py``)::

    PYTHONPATH=/workspace/envs/minimax_h3/diffusers_overlay \
        .venv/bin/python tools/minimax_h3/plan_vectors.py

Regenerating writes an identical file.

Layout of the vectors (sizes are ``[width, height]``, vision grids are Qwen
``[t, h, w]`` patch grids, rows are denoiser rows):

* ``description``, ``generator``, ``sources``: provenance (checkpoint
  revisions, diffusers, transformers and torchaudio versions).
* ``vision``: the Qwen3-VL processor geometry (``patch_size``,
  ``temporal_patch_size``, ``merge_size`` and the image and video pixel
  budgets); ``tokens``: the ids of the vision markers and placeholders.
* ``canvas_cases``: ``aspect`` and its ``canvas``, or ``error``.
* ``duration_cases``: ``seconds`` and its ``num_frames``, ``latent_frames``
  and ``audio_latents``, or ``error`` outside the served range.
* ``reference_image_cases``: ``size`` and its ``resize``, ``vision_grid``,
  ``vision_tokens`` and ``rows``, or ``error``.
* ``keyframe_cases``: a following keyframe's ``size``, the ``canvas`` and
  its ``cover_crop`` (resized ``width``/``height``, crop ``left``/``top``).
* ``video_reference_cases``: a video's ``display``, ``frame_rate``
  ``[num, den]``, decoded ``frames``, ``start_seconds`` and the generated
  ``num_frames``, and its ``canvas``, ``start_frame``, ``clip_frames`` (24 fps
  frames kept), ``vae_frames`` (the leading ``17 * n + 5`` the VAE encodes),
  ``latent_frames``, ``frame_indices`` (sampled at 2 fps),
  ``block_timestamps``, ``vision_grid``, ``block_tokens`` and ``rows``, or
  ``error``.
* ``audio_cases``: a soundtrack's ``sample_rate``, ``channels``, decoded
  ``samples`` per channel, ``start_seconds`` and the generated
  ``num_frames``, and its ``clip``: ``start_sample``, ``source_samples``
  (kept at the native rate), ``samples`` (at 32 kHz) and ``latents`` per
  channel.
* ``requests``: ``name``, ``task``, ``prompt``, ``target`` (``short_edge``,
  ``aspect_ratio``, ``duration_seconds`` when given) and ``conditions`` in
  request order (``type``, ``role``, ``frame_index`` of a keyframe,
  ``start_time_seconds`` of a reference, and ``media``: an image's displayed
  ``width``/``height``; a video's ``display`` size, ``frame_rate``,
  ``frames`` and ``soundtrack``; an audio stream's ``sample_rate``,
  ``channels`` and ``samples``). An accepted request has ``expected``:
  ``canvas``, ``num_frames``, ``latent_frames``, ``audio_latents``,
  ``target_video_rows``, ``target_audio_rows``, ``conditions`` (per
  condition ``index``, ``kind``, the preparation fields of the kind,
  ``vision`` or null, ``video_rows`` and ``audio_rows``: a ``keyframe`` has
  ``position`` and ``cover_crop``, null when it is stretched; an ``image``
  its ``resize``; an ``audio`` its ``clip``; a ``video`` its ``canvas``,
  ``start_frame``, ``clip_frames``, ``vae_frames``, ``latent_frames`` and
  ``soundtrack`` clip), ``condition_video_rows``, ``condition_audio_rows``,
  ``text_rows``, ``sequence_rows`` (text, conditions, target audio and target
  video, without padding) and ``segments``: the presentation as text
  segments (``text`` and the tokenizer's ``token_ids``, tagged text) and
  vision blocks (``vision`` ``image`` or ``video`` and the placeholder count
  ``tokens``, wrapped in the vision markers and tagged video). A rejected
  request has ``error``: the ``field`` at fault.

The media of the three official requests are the decoded facts of the
official CDN files (``fetch_inputs.py``) and their prompts are read from the
checkpoint's request scripts.
"""

import argparse
import json
import math
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import diffusers
import numpy as np
import torch
import torchaudio
import transformers
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.modular_pipelines import PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import encoders
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3Ref2VAPrepareLayoutStep,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import (
    MiniMaxH3Ref2VASetupStep,
    MiniMaxH3ResizeStep,
)
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3FL2VATextEncoderStep,
    MiniMaxH3Ref2VAReferenceEncoderStep,
    MiniMaxH3Ref2VATextEncoderStep,
    MiniMaxH3TextEncoderStep,
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
from fetch_inputs import _official_request
from PIL import Image

CHECKPOINT = Path("/workspace/models/MiniMax-H3")
REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "tests" / "python" / "fixtures" / "minimax_h3_plan.json"

# UniServe's fewest 24 fps frames of a reference video: one VAE window.
MIN_REFERENCE_FRAMES = 22

PROMPT = (
    "integrated_multimodal_description: A red fox trots across fresh snow "
    "at dawn while its breath fogs the air; soft wind, distant birdsong."
)
PROMPT_UNICODE = (
    "一只猫在窗台上晒太阳，尾巴轻轻摆动。\nThe camera slowly pushes in."
)
PROMPT_SHORT = "A short test prompt."


class _Presentation:
    """Records what the text-encoder steps present to Qwen3-VL.

    Registered as the tokenizer component, it records every text the steps
    tokenize; ``embed`` replaces ``get_qwen3vl_prompt_embeds`` and records the
    token ids and vision inputs of the conditioner call, whose output the
    vectors do not need.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.clear()

    def clear(self) -> None:
        self.texts: list[tuple[str, list[int]]] = []
        self.token_ids: list[int] = []
        self.vision: dict[str, torch.Tensor] = {}

    def __call__(self, text: str, add_special_tokens: bool) -> dict:
        ids = self.tokenizer(text, add_special_tokens=add_special_tokens)
        ids = list(ids["input_ids"])
        self.texts.append((text, ids))
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    def embed(self, _encoder, _processor, token_ids, vision_inputs=None, **_):
        self.token_ids = list(token_ids)
        self.vision = dict(vision_inputs or {})
        return torch.zeros(1, len(token_ids), 1)


class _Conditioner(torch.nn.Module):
    """The text-encoder component: only its device and dtype are read."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype


class _Posterior:
    """The video-encoder stand-in's posterior; a sample is its latents."""

    def __init__(self, latents: torch.Tensor):
        self.latents = latents

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.latents


class _VideoEncoder:
    """Stands in for the video VAE: records each encode and returns latents.

    The geometry is the checkpoint VAE's, read from the VAE class built
    without weights: a single frame encodes to one latent frame through the
    spatial encoder, ``17 * n + 5`` frames to ``5 * n + 2``, and both sides
    shrink by the spatial compression.
    """

    def __init__(self, checkpoint: Path):
        config = AutoencoderKLMiniMaxH3.load_config(checkpoint / "vae")
        with torch.device("meta"):
            vae = AutoencoderKLMiniMaxH3.from_config(config)
        self.config = vae.config
        self.spatial_compression_ratio = vae.spatial_compression_ratio
        self.tokens_chunk_size = vae.tokens_chunk_size
        self.encoded_frames: list[int] = []

    def encode(self, pixels: torch.Tensor, return_dict: bool = True):
        frames, height, width = pixels.shape[2:]
        self.encoded_frames.append(frames)
        latent_frames = 1
        if frames > 1:
            latent_frames = video_latent_num_frames(
                frames, self.config.clip_length, self.tokens_chunk_size
            )
        ratio = self.spatial_compression_ratio
        latents = torch.zeros(
            1,
            self.config.latent_channels,
            latent_frames,
            height // ratio,
            width // ratio,
        )
        return (_Posterior(latents),)


class _Reference:
    """The diffusers pipeline the vectors are computed with."""

    def __init__(self, checkpoint: Path):
        blocks = SequentialPipelineBlocks.from_blocks_dict(
            {
                "resize": MiniMaxH3ResizeStep(),
                "setup": MiniMaxH3Ref2VASetupStep(),
                "t2va_text": MiniMaxH3TextEncoderStep(),
                "fl2va_text": MiniMaxH3FL2VATextEncoderStep(),
                "ref2va_text": MiniMaxH3Ref2VATextEncoderStep(),
                "references": MiniMaxH3Ref2VAReferenceEncoderStep(),
                "ref2va_layout": MiniMaxH3Ref2VAPrepareLayoutStep(),
            }
        )
        pipe = blocks.init_pipeline(str(checkpoint))
        pipe.load_components(names=["tokenizer", "processor"])
        self.tokenizer = pipe.tokenizer
        self.presentation = _Presentation(self.tokenizer)
        self.video_encoder = _VideoEncoder(checkpoint)
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            str(checkpoint), subfolder="audio_vae", torch_dtype=torch.float32
        ).eval()
        pipe.update_components(
            tokenizer=self.presentation,
            text_encoder=_Conditioner(),
            vae=self.video_encoder,
            audio_vae=audio_vae,
        )
        encoders.get_qwen3vl_prompt_embeds = self.presentation.embed

        # Without the transformers loaded the pipeline falls back to the
        # (1, 2, 2) patch; it must be the checkpoint's.
        for name in ("transformer", "transformer_ref"):
            config = json.loads((checkpoint / name / "config.json").read_text())
            assert tuple(config["patch_size"]) == pipe.patch_size

        self.pipe = pipe
        self.processor = pipe.processor
        self.merge = self.processor.image_processor.merge_size**2
        self.sample_fps = MiniMaxH3Ref2VATextEncoderStep().video_sample_fps
        # Random keyframe pixels, drawn in case order.
        self.rng = np.random.default_rng(0)

    def run(self, steps: list[Any], **inputs: Any) -> PipelineState:
        """Run reference steps in order on a fresh pipeline state.

        The presentation and video-encoder records start empty, so they hold
        what this run's steps presented and encoded.
        """
        self.presentation.clear()
        self.video_encoder.encoded_frames.clear()
        state = PipelineState()
        for name, value in inputs.items():
            state.set(name, value)
        for step in steps:
            _, state = step(self.pipe, state)
        return state

    def canvas(self, width: float, height: float) -> list[int]:
        canvas_height, canvas_width = resolve_canvas_size(
            width,
            height,
            self.pipe.canvas_multiple,
            self.pipe.config.canvas_short_edge,
            self.pipe.config.canvas_max_pixels,
        )
        return [canvas_width, canvas_height]

    def frames(self, seconds: float) -> int:
        """The generated frame count of a requested duration."""
        return align_num_frames(
            round(seconds * self.pipe.fps),
            self.pipe.vae_frames_per_chunk,
            self.pipe.vae_latents_per_chunk,
        )

    def latent_frames(self, num_frames: int) -> int:
        return video_latent_num_frames(
            num_frames,
            self.pipe.vae_frames_per_chunk,
            self.pipe.vae_latents_per_chunk,
        )

    def noise_image(self, width: int, height: int) -> Image.Image:
        pixels = self.rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(pixels)

    def video_clip(
        self,
        display: list[int],
        frame_rate: Fraction,
        frames: int,
        start_seconds: float,
        num_frames: int,
    ) -> tuple[int, np.ndarray]:
        """A video reference's 24 fps clip after its start offset.

        The setup step's own normalization resamples the source onto the
        24 fps timeline at the canvas of its aspect; the offset then drops
        whole frames of that timeline. Returns the start frame and the clip,
        which the setup step passes through unchanged.
        """
        start_frame = _start_offset(start_seconds, self.pipe.fps)
        source = np.zeros((frames, display[1], display[0], 3), np.uint8)
        timeline = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
            source,
            float(frame_rate),
            start_frame + num_frames,
            self.pipe.canvas_multiple,
            self.pipe.config.canvas_short_edge,
            self.pipe.config.canvas_max_pixels,
            float(self.pipe.fps),
        )
        return start_frame, timeline[start_frame:]

    def sampled_indices(self, num_frames: int) -> list[int]:
        """The frames the conditioner's 2 fps sampler picks from a clip."""
        ramp = np.arange(num_frames).reshape(num_frames, 1, 1, 1)
        frames, _ = self._sample(ramp)
        return [int(frame[0, 0, 0]) for frame in frames]

    def block_timestamps(self, clip: np.ndarray) -> list[float]:
        return self._sample(clip)[1]

    def _sample(self, frames: np.ndarray) -> tuple[list, list[float]]:
        return MiniMaxH3Ref2VATextEncoderStep._sample_video_condition_frames(
            frames,
            float(self.pipe.fps),
            self.sample_fps,
            self.processor.video_processor.temporal_patch_size,
        )

    def audio_clip(
        self, track: dict, start_seconds: float, num_frames: int
    ) -> tuple[dict, torch.Tensor]:
        """A soundtrack's offset and native-rate truncation.

        Returns the clip's offset fields and the waveform after the offset,
        which the setup step truncates and resamples.
        """
        start_sample = _start_offset(start_seconds, track["sample_rate"])
        waveform = torch.zeros(track["channels"], track["samples"])
        waveform = waveform[:, start_sample:]
        kept = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
            waveform,
            track["sample_rate"],
            track["sample_rate"],
            num_frames / self.pipe.fps,
        )
        clip = {"start_sample": start_sample, "source_samples": kept.shape[-1]}
        return clip, waveform

    def audio_latents(self, rows: torch.Tensor) -> int:
        """Latents per channel of a soundtrack's channel-major rows."""
        assert rows.shape[0] % self.pipe.audio_channels == 0
        return rows.shape[0] // self.pipe.audio_channels

    def video_rows(self, latents: torch.Tensor) -> int:
        """Denoiser rows of a condition's latents, one per patch."""
        frames, height, width = latents.shape[2:]
        _, patch_height, patch_width = self.pipe.patch_size
        return frames * (height // patch_height) * (width // patch_width)


def _start_offset(seconds: float, rate: float) -> int:
    """Units of a ``rate`` timeline a start offset skips (UniServe's rule)."""
    return math.floor(seconds * rate + 0.5)


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


def canvas_cases(ref: _Reference) -> list[dict]:
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
            canvas = ref.canvas(width, height)
        except ValueError:
            cases.append({"aspect": [width, height], "error": True})
            continue
        cases.append({"aspect": [width, height], "canvas": canvas})
    return cases


def duration_cases(ref: _Reference) -> list[dict]:
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
        num_frames = ref.frames(seconds)
        cases.append(
            {
                "seconds": seconds,
                "num_frames": num_frames,
                "latent_frames": ref.latent_frames(num_frames),
                "audio_latents": audio_latent_num_frames(
                    num_frames, ref.pipe.fps
                ),
            }
        )
    # Outside the served range of requested durations.
    for seconds in (3.99, 15.01, 0.0, -1.0, 30.0):
        cases.append({"seconds": seconds, "error": True})
    return cases


def _reference_chain() -> list[Any]:
    """The ref2va steps from the setup to the packed layout."""
    return [
        MiniMaxH3Ref2VASetupStep(),
        MiniMaxH3Ref2VATextEncoderStep(),
        MiniMaxH3Ref2VAReferenceEncoderStep(),
        MiniMaxH3Ref2VAPrepareLayoutStep(),
    ]


def reference_image_cases(ref: _Reference) -> list[dict]:
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
            state = ref.run(
                _reference_chain(),
                prompt=PROMPT_SHORT,
                references=[reference],
                height=None,
                width=None,
                num_frames=ref.frames(5.0),
            )
        except ValueError:
            cases.append({"size": [width, height], "error": True})
            continue
        image = state.get("normalized_references")[0].image
        grid = ref.presentation.vision["image_grid_thw"][0].tolist()
        cases.append(
            {
                "size": [width, height],
                "resize": list(image.size),
                "vision_grid": grid,
                "vision_tokens": math.prod(grid) // ref.merge,
                "rows": state.get("num_condition_video_rows"),
            }
        )
    return cases


def _cover_crop(width: int, height: int, canvas: list[int]) -> dict:
    """The resize and centred crop the resize step gives a follower.

    The arithmetic of ``MiniMaxH3ResizeStep``; every use is checked pixel for
    pixel against the step's output.
    """
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


def _fitted(image: Image.Image, canvas: list[int], crop: dict | None):
    """Apply a recorded keyframe fit: a LANCZOS stretch or a cover crop."""
    if crop is None:
        return image.resize(tuple(canvas), Image.Resampling.LANCZOS)
    resized = image.resize(
        (crop["width"], crop["height"]), Image.Resampling.LANCZOS
    )
    return resized.crop(
        (
            crop["left"],
            crop["top"],
            crop["left"] + canvas[0],
            crop["top"] + canvas[1],
        )
    )


def _keyframe_inputs(ref: _Reference, entries: list[dict]) -> dict:
    """Random keyframes of the stated sizes as resize-step inputs."""
    images = [
        ref.noise_image(entry["media"]["width"], entry["media"]["height"])
        for entry in entries
    ]
    return {
        "image": images[0] if entries[0]["frame_index"] == 0 else None,
        "last_image": images[-1] if entries[-1]["frame_index"] == -1 else None,
    }


def _keyframe_fits(inputs: dict, state: PipelineState) -> list[dict]:
    """The fit of every keyframe the resize step put on the canvas.

    The first keyframe is stretched and a second one cover-cropped; each
    recorded fit reproduces the step's output pixel for pixel.
    """
    canvas = [state.get("width"), state.get("height")]
    images = [
        image
        for image in (inputs["image"], inputs["last_image"])
        if image is not None
    ]
    fits = []
    for order, (image, prepared, anchor) in enumerate(
        zip(
            images,
            state.get("keyframes"),
            state.get("keyframe_anchors"),
            strict=True,
        )
    ):
        crop = None if order == 0 else _cover_crop(*image.size, canvas)
        assert np.array_equal(
            np.asarray(prepared), np.asarray(_fitted(image, canvas, crop))
        )
        fits.append(
            {"kind": "keyframe", "position": anchor, "cover_crop": crop}
        )
    return fits


def keyframe_cases(ref: _Reference) -> list[dict]:
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
        entries = [
            keyframe(image(*canvas), 0),
            keyframe(image(width, height), -1),
        ]
        inputs = _keyframe_inputs(ref, entries)
        state = ref.run(
            [MiniMaxH3ResizeStep()],
            **inputs,
            height=canvas[1],
            width=canvas[0],
        )
        follower = _keyframe_fits(inputs, state)[1]
        cases.append(
            {
                "size": [width, height],
                "canvas": list(canvas),
                "cover_crop": follower["cover_crop"],
            }
        )
    return cases


def _video_vision(ref: _Reference, clip: np.ndarray, grid: list[int]) -> dict:
    """The conditioner's view of a video reference's clip."""
    timestamps = ref.block_timestamps(clip)
    assert grid[0] == len(timestamps)
    return {
        "grid": grid,
        "block_tokens": grid[1] * grid[2] // ref.merge,
        "frame_indices": ref.sampled_indices(clip.shape[0]),
        "block_timestamps": timestamps,
    }


def video_reference_cases(ref: _Reference) -> list[dict]:
    specs = [
        # display, frame rate, decoded frames, start seconds, target frames
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
    for display, frame_rate, frames, start_seconds, num_frames in specs:
        case: dict[str, Any] = {
            "display": list(display),
            "frame_rate": [frame_rate.numerator, frame_rate.denominator],
            "frames": frames,
            "start_seconds": start_seconds,
            "num_frames": num_frames,
        }
        try:
            start_frame, clip = ref.video_clip(
                list(display), frame_rate, frames, start_seconds, num_frames
            )
        except ValueError:
            cases.append({**case, "error": True})
            continue
        if clip.shape[0] < MIN_REFERENCE_FRAMES:
            cases.append({**case, "error": True})
            continue

        # The clip is what the setup step hands on; the text-encoder and
        # reference-encoder steps read it as it is.
        reference = MiniMaxH3VideoReference(
            frames=clip, fps=float(ref.pipe.fps)
        )
        state = ref.run(
            [
                MiniMaxH3Ref2VATextEncoderStep(),
                MiniMaxH3Ref2VAReferenceEncoderStep(),
            ],
            prompt=PROMPT_SHORT,
            normalized_references=[reference],
        )
        latents = state.get("condition_latents")[0]
        grid = ref.presentation.vision["video_grid_thw"][0].tolist()
        vision = _video_vision(ref, clip, grid)
        cases.append(
            {
                **case,
                "canvas": [clip.shape[2], clip.shape[1]],
                "start_frame": start_frame,
                "clip_frames": clip.shape[0],
                "vae_frames": ref.video_encoder.encoded_frames[0],
                "latent_frames": latents.shape[2],
                "frame_indices": vision["frame_indices"],
                "block_timestamps": vision["block_timestamps"],
                "vision_grid": grid,
                "block_tokens": vision["block_tokens"],
                "rows": ref.video_rows(latents),
            }
        )
    return cases


def audio_cases(ref: _Reference) -> list[dict]:
    specs = [
        # sample rate, decoded samples, start seconds, target frames
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
        track = {"sample_rate": sample_rate, "channels": 1, "samples": samples}
        clip, waveform = ref.audio_clip(track, start_seconds, num_frames)
        resampled = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
            waveform,
            sample_rate,
            ref.pipe.audio_sampling_rate,
            num_frames / ref.pipe.fps,
        )
        reference = MiniMaxH3AudioReference(
            audio=resampled, sample_rate=ref.pipe.audio_sampling_rate
        )
        state = ref.run(
            [MiniMaxH3Ref2VAReferenceEncoderStep()],
            normalized_references=[reference],
        )
        rows = state.get("audio_condition_latents")[0]
        cases.append(
            {
                **track,
                "start_seconds": start_seconds,
                "num_frames": num_frames,
                "clip": {
                    **clip,
                    "samples": resampled.shape[-1],
                    "latents": ref.audio_latents(rows),
                },
            }
        )
    return cases


def _segments(token_ids: list[int], ref: _Reference) -> list[dict]:
    """Split a presentation into its text segments and vision blocks."""
    tokenizer = ref.tokenizer
    start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    pads = {
        tokenizer.convert_tokens_to_ids("<|image_pad|>"): "image",
        tokenizer.convert_tokens_to_ids("<|video_pad|>"): "video",
    }
    segments: list[dict] = []
    position = 0
    texts = list(ref.presentation.texts)
    while position < len(token_ids):
        if token_ids[position] == start:
            close = token_ids.index(end, position)
            block = token_ids[position + 1 : close]
            assert len(set(block)) == 1
            segments.append({"vision": pads[block[0]], "tokens": len(block)})
            position = close + 1
            continue
        text, ids = texts.pop(0)
        assert ids and token_ids[position : position + len(ids)] == ids
        segments.append({"text": text, "token_ids": ids})
        position += len(ids)
    assert not texts
    return segments


def _join(segments: list[dict], ref: _Reference) -> tuple[list, list]:
    """Rebuild the token ids and tags of a presentation from its segments."""
    tokenizer = ref.tokenizer
    token_ids: list[int] = []
    tags: list[int] = []
    for segment in segments:
        if "vision" in segment:
            pad = f"<|{segment['vision']}_pad|>"
            block = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids(pad)] * segment["tokens"]
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += block
            tags += [ref.pipe.video_tag] * len(block)
        else:
            token_ids += segment["token_ids"]
            tags += [ref.pipe.text_tag] * len(segment["token_ids"])
    return token_ids, tags


def _layout_rows(layout: tuple, keyframe_rows: int = 0) -> dict:
    """The row counts of a packed layout and the keyframes added to it."""
    position_ids, _, video_indices, audio_indices, text_indices = layout[:5]
    condition_video_rows, condition_audio_rows = layout[5:]
    return {
        "target_video_rows": len(video_indices) - condition_video_rows,
        "target_audio_rows": len(audio_indices) - condition_audio_rows,
        "condition_video_rows": condition_video_rows + keyframe_rows,
        "condition_audio_rows": condition_audio_rows,
        "text_rows": len(text_indices),
        "sequence_rows": position_ids.shape[0] + keyframe_rows,
    }


def _target_duration(request: dict) -> float:
    """The requested duration, or the one soundtrack's after its offset."""
    seconds = request["target"].get("duration_seconds")
    if seconds is not None:
        return seconds
    tracks = []
    for entry in request["conditions"]:
        media = entry["media"]
        track = media if entry["type"] == "audio" else media.get("soundtrack")
        if entry["role"] == "reference" and track is not None:
            tracks.append((entry, track))
    ((entry, track),) = tracks
    start = _start_offset(
        entry.get("start_time_seconds") or 0.0, track["sample_rate"]
    )
    return (track["samples"] - start) / track["sample_rate"]


def _named_canvas(ref: _Reference, aspect_ratio: str) -> list[int]:
    """The canvas of ``W:H``; ``auto`` is 16:9 for ``t2va`` and ``ref2va``."""
    if aspect_ratio == "auto":
        return ref.canvas(16, 9)
    width, height = (int(part) for part in aspect_ratio.split(":"))
    return ref.canvas(width, height)


def _plan_layout(
    ref: _Reference,
    text_tags: torch.Tensor,
    canvas: list[int],
    num_frames: int,
    anchors: tuple,
) -> tuple[dict, int, int]:
    """Lay out a ``t2va`` or ``fl2va`` request with the reference builder.

    The builder is the layout step without its ``[5, 15]``-second check, so
    it also lays out the served 4-second requests.
    """
    ratio = ref.pipe.vae_spatial_compression_ratio
    latent_frames = ref.latent_frames(num_frames)
    audio_latents = audio_latent_num_frames(num_frames, ref.pipe.fps)
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_tags,
        latent_frames,
        canvas[1] // ratio,
        canvas[0] // ratio,
        audio_latents,
        ref.pipe.patch_size,
        ref.pipe.audio_channels,
        ref.pipe.audio_tag,
        ref.pipe.video_tag,
        anchors,
    )
    return _layout_rows(layout), latent_frames, audio_latents


def _plan_t2va(ref: _Reference, request: dict) -> dict:
    canvas = _named_canvas(ref, request["target"]["aspect_ratio"])
    num_frames = ref.frames(_target_duration(request))
    state = ref.run([MiniMaxH3TextEncoderStep()], prompt=request["prompt"])
    rows, latent_frames, audio_latents = _plan_layout(
        ref, state.get("text_token_tags"), canvas, num_frames, ()
    )
    return {
        "canvas": canvas,
        "num_frames": num_frames,
        "latent_frames": latent_frames,
        "audio_latents": audio_latents,
        "conditions": [],
        "rows": rows,
        "tags": state.get("text_token_tags").tolist(),
    }


def _plan_fl2va(ref: _Reference, request: dict) -> dict:
    aspect_ratio = request["target"]["aspect_ratio"]
    canvas = None
    if aspect_ratio != "auto":
        width, height = (int(part) for part in aspect_ratio.split(":"))
        canvas = ref.canvas(width, height)
    num_frames = ref.frames(_target_duration(request))
    entries = request["conditions"]
    inputs = _keyframe_inputs(ref, entries)
    state = ref.run(
        [MiniMaxH3ResizeStep(), MiniMaxH3FL2VATextEncoderStep()],
        prompt=request["prompt"],
        **inputs,
        height=None if canvas is None else canvas[1],
        width=None if canvas is None else canvas[0],
    )
    canvas = [state.get("width"), state.get("height")]
    anchors = state.get("keyframe_anchors")
    rows, latent_frames, audio_latents = _plan_layout(
        ref, state.get("text_token_tags"), canvas, num_frames, anchors
    )
    grids = ref.presentation.vision["image_grid_thw"].tolist()
    keyframe_rows, remainder = divmod(
        rows["condition_video_rows"], len(anchors)
    )
    assert remainder == 0
    conditions = []
    for index, (fit, grid) in enumerate(
        zip(_keyframe_fits(inputs, state), grids, strict=True)
    ):
        vision = {"grid": grid, "tokens": math.prod(grid) // ref.merge}
        conditions.append(
            {
                "index": index,
                **fit,
                "vision": vision,
                "video_rows": keyframe_rows,
                "audio_rows": 0,
            }
        )
    return {
        "canvas": canvas,
        "num_frames": num_frames,
        "latent_frames": latent_frames,
        "audio_latents": audio_latents,
        "conditions": conditions,
        "rows": rows,
        "tags": state.get("text_token_tags").tolist(),
    }


def _diffusers_reference(ref: _Reference, entry: dict, num_frames: int):
    """A diffusers reference holding synthetic media of the stated facts."""
    media = entry["media"]
    if entry["type"] == "image":
        return MiniMaxH3ImageReference(
            image=Image.new("RGB", (media["width"], media["height"]))
        )
    if entry["type"] == "audio":
        return MiniMaxH3AudioReference(
            audio=torch.zeros(media["channels"], media["samples"]),
            sample_rate=media["sample_rate"],
        )
    start = entry.get("start_time_seconds") or 0.0
    _, clip = ref.video_clip(
        media["display"],
        Fraction(*media["frame_rate"]),
        media["frames"],
        start,
        num_frames,
    )
    assert clip.shape[0] >= MIN_REFERENCE_FRAMES
    audio = sample_rate = None
    track = media.get("soundtrack")
    if track is not None:
        _, audio = ref.audio_clip(track, start, num_frames)
        sample_rate = track["sample_rate"]
    return MiniMaxH3VideoReference(
        frames=clip,
        fps=float(ref.pipe.fps),
        audio=audio,
        sample_rate=sample_rate,
    )


def _plan_ref2va(ref: _Reference, request: dict) -> dict:
    canvas = _named_canvas(ref, request["target"]["aspect_ratio"])
    num_frames = ref.frames(_target_duration(request))
    entries = request["conditions"]

    # Keyframes are UniServe's: fitted to the canvas like fl2va ones, one
    # target latent frame of rows each, and absent from the conditioner.
    keyframes = [entry for entry in entries if entry["role"] == "keyframe"]
    fits = []
    if keyframes:
        inputs = _keyframe_inputs(ref, keyframes)
        resized = ref.run(
            [MiniMaxH3ResizeStep()],
            **inputs,
            height=canvas[1],
            width=canvas[0],
        )
        fits = _keyframe_fits(inputs, resized)

    references = [entry for entry in entries if entry["role"] == "reference"]
    state = ref.run(
        _reference_chain(),
        prompt=request["prompt"],
        references=[
            _diffusers_reference(ref, entry, num_frames) for entry in references
        ],
        height=canvas[1],
        width=canvas[0],
        num_frames=num_frames,
    )
    assert state.get("num_frames") == num_frames
    layout = tuple(
        state.get(name)
        for name in (
            "position_ids",
            "token_tags",
            "video_indices",
            "audio_indices",
            "text_indices",
            "num_condition_video_rows",
            "num_condition_audio_rows",
        )
    )
    latent_frames = state.get("num_latent_frames")
    keyframe_rows, remainder = divmod(
        _layout_rows(layout)["target_video_rows"], latent_frames
    )
    assert remainder == 0
    rows = _layout_rows(layout, keyframe_rows * len(keyframes))

    # The reference encoder's outputs follow the references in packed order:
    # one visual latent per image or video, one block of channel-major rows
    # per soundtrack, and one vision grid per image or video.
    vision = ref.presentation.vision
    normalized = iter(state.get("normalized_references"))
    visual_latents = iter(state.get("condition_latents"))
    encoded_frames = iter(ref.video_encoder.encoded_frames)
    soundtrack_rows = iter(state.get("audio_condition_latents"))
    image_grids = iter(vision.get("image_grid_thw", torch.zeros(0, 3)).tolist())
    video_grids = iter(vision.get("video_grid_thw", torch.zeros(0, 3)).tolist())
    fitted = iter(fits)
    conditions = []
    for index, entry in enumerate(entries):
        condition: dict[str, Any] = {"index": index}
        conditions.append(condition)
        if entry["role"] == "keyframe":
            condition |= {
                **next(fitted),
                "vision": None,
                "video_rows": keyframe_rows,
                "audio_rows": 0,
            }
            continue

        reference = next(normalized)
        start = entry.get("start_time_seconds") or 0.0
        soundtrack = None
        audio_rows = 0
        if reference.has_audio:
            media = entry["media"]
            track = media if reference.kind == "audio" else media["soundtrack"]
            clip, _ = ref.audio_clip(track, start, num_frames)
            block = next(soundtrack_rows)
            audio_rows = block.shape[0]
            soundtrack = {
                **clip,
                "samples": reference.audio.shape[-1],
                "latents": ref.audio_latents(block),
            }

        if reference.kind == "audio":
            condition |= {
                "kind": "audio",
                "clip": soundtrack,
                "vision": None,
                "video_rows": 0,
                "audio_rows": audio_rows,
            }
        elif reference.kind == "image":
            latents = next(visual_latents)
            assert next(encoded_frames) == 1
            grid = next(image_grids)
            condition |= {
                "kind": "image",
                "resize": list(reference.image.size),
                "vision": {
                    "grid": grid,
                    "tokens": math.prod(grid) // ref.merge,
                },
                "video_rows": ref.video_rows(latents),
                "audio_rows": 0,
            }
        else:
            clip = reference.frames
            latents = next(visual_latents)
            condition |= {
                "kind": "video",
                "canvas": [clip.shape[2], clip.shape[1]],
                "start_frame": _start_offset(start, ref.pipe.fps),
                "clip_frames": clip.shape[0],
                "vae_frames": next(encoded_frames),
                "latent_frames": latents.shape[2],
                "soundtrack": soundtrack,
                "vision": _video_vision(ref, clip, next(video_grids)),
                "video_rows": ref.video_rows(latents),
                "audio_rows": audio_rows,
            }
    return {
        "canvas": canvas,
        "num_frames": num_frames,
        "latent_frames": latent_frames,
        "audio_latents": state.get("num_audio_latents"),
        "conditions": conditions,
        "rows": rows,
        "tags": state.get("text_token_tags").tolist(),
    }


def expected_plan(ref: _Reference, request: dict) -> dict:
    """The reference's plan of an accepted request."""
    planner = {
        "t2va": _plan_t2va,
        "fl2va": _plan_fl2va,
        "ref2va": _plan_ref2va,
    }[request["task"]]
    plan = planner(ref, request)
    token_ids = ref.presentation.token_ids
    rows = plan["rows"]
    conditions = plan["conditions"]

    # The per-condition rows add up to the layout's condition blocks.
    assert rows["text_rows"] == len(token_ids)
    assert rows["condition_video_rows"] == sum(
        condition["video_rows"] for condition in conditions
    )
    assert rows["condition_audio_rows"] == sum(
        condition["audio_rows"] for condition in conditions
    )
    # The segments encode the presentation losslessly; the vectors keep them
    # rather than the long, mostly placeholder, token sequence.
    segments = _segments(token_ids, ref)
    assert _join(segments, ref) == (token_ids, plan["tags"])
    return {
        "canvas": plan["canvas"],
        "num_frames": plan["num_frames"],
        "latent_frames": plan["latent_frames"],
        "audio_latents": plan["audio_latents"],
        "target_video_rows": rows["target_video_rows"],
        "target_audio_rows": rows["target_audio_rows"],
        "conditions": conditions,
        "condition_video_rows": rows["condition_video_rows"],
        "condition_audio_rows": rows["condition_audio_rows"],
        "text_rows": rows["text_rows"],
        "sequence_rows": rows["sequence_rows"],
        "segments": segments,
    }


def image(width: int, height: int) -> dict:
    return {"width": width, "height": height}


def video(
    display: tuple[int, int],
    frame_rate: tuple[int, int],
    frames: int,
    soundtrack: dict | None = None,
) -> dict:
    return {
        "display": list(display),
        "frame_rate": list(frame_rate),
        "frames": frames,
        "soundtrack": soundtrack,
    }


def audio(sample_rate: int, samples: int, channels: int = 1) -> dict:
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "samples": samples,
    }


def keyframe(media: dict, index: int) -> dict:
    return {
        "type": "image",
        "role": "keyframe",
        "frame_index": index,
        "media": media,
    }


def reference(kind: str, media: dict, start: float | None = None) -> dict:
    entry = {"type": kind, "role": "reference", "media": media}
    if start is not None:
        entry["start_time_seconds"] = start
    return entry


def request(
    name: str,
    task: str,
    conditions: Sequence[dict] = (),
    aspect_ratio: str = "auto",
    seconds: float | None = 5.0,
    prompt: str = PROMPT_SHORT,
) -> dict:
    target: dict[str, Any] = {"short_edge": 768, "aspect_ratio": aspect_ratio}
    if seconds is not None:
        target["duration_seconds"] = seconds
    return {
        "name": name,
        "task": task,
        "prompt": prompt,
        "target": target,
        "conditions": list(conditions),
    }


# The decoded facts of the official ref2va request's CDN media
# (fetch_inputs.py): a 1080p 24 fps video with a 44.1 kHz stereo soundtrack
# and a 44.1 kHz stereo voice reference.
OFFICIAL_VIDEO = video((1920, 1080), (24, 1), 145, audio(44100, 267264, 2))
OFFICIAL_AUDIO = audio(44100, 375552, 2)


def accepted_requests(prompts: dict[str, str]) -> list[dict]:
    """Requests the planners accept: every aspect, keyframe and reference."""
    cases = [
        request("t2va_default", "t2va", prompt=PROMPT),
        request(
            "t2va_portrait_unicode",
            "t2va",
            aspect_ratio="9:16",
            seconds=15.0,
            prompt=PROMPT_UNICODE,
        ),
    ]
    for aspect_ratio in ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "auto"):
        name = f"t2va_{aspect_ratio.replace(':', 'x')}_5s"
        cases.append(request(name, "t2va", aspect_ratio=aspect_ratio))
    for seconds in (4.0, 4.5, 6.0, 8.0, 10.0, 12.5, 15.0):
        cases.append(
            request(
                f"t2va_16x9_{seconds:g}s",
                "t2va",
                aspect_ratio="16:9",
                seconds=seconds,
            )
        )
    cases.append(
        request(
            "t2va_official",
            "t2va",
            aspect_ratio="16:9",
            seconds=10.0,
            prompt=prompts["t2va"],
        )
    )

    # fl2va: with `auto`, the first keyframe's aspect picks the canvas.
    cases += [
        request(
            "fl2va_first_auto",
            "fl2va",
            [keyframe(image(1000, 600), 0)],
            seconds=8.0,
            prompt=PROMPT,
        ),
        request(
            "fl2va_last_only",
            "fl2va",
            [keyframe(image(3024, 4032), -1)],
            prompt=PROMPT,
        ),
        request(
            "fl2va_first_last_explicit",
            "fl2va",
            [keyframe(image(1920, 1080), 0), keyframe(image(640, 640), -1)],
            aspect_ratio="7:4",
            seconds=6.0,
            prompt=PROMPT_UNICODE,
        ),
    ]
    for width, height in (
        (1920, 1080),
        (1080, 1920),
        (1000, 1000),
        (640, 480),
        (1376, 768),
        (333, 777),
        (2048, 512),
        (512, 2048),
        (4000, 1000),
        (1001, 4000),
    ):
        cases.append(
            request(
                f"fl2va_first_{width}x{height}",
                "fl2va",
                [keyframe(image(width, height), 0)],
            )
        )
    for width, height in ((1920, 1080), (512, 2048), (1240, 1754)):
        cases.append(
            request(
                f"fl2va_last_{width}x{height}",
                "fl2va",
                [keyframe(image(width, height), -1)],
            )
        )
    for first, last in (
        ((1920, 1080), (1376, 768)),
        ((1080, 1920), (1920, 1080)),
        ((2048, 512), (1000, 1000)),
        ((1000, 1000), (333, 777)),
    ):
        cases.append(
            request(
                "fl2va_first_{}x{}_last_{}x{}".format(*first, *last),
                "fl2va",
                [keyframe(image(*first), 0), keyframe(image(*last), -1)],
            )
        )
    cases += [
        request(
            "fl2va_first_1000x1000_16x9",
            "fl2va",
            [keyframe(image(1000, 1000), 0)],
            aspect_ratio="16:9",
        ),
        request(
            "fl2va_official",
            "fl2va",
            [keyframe(image(1920, 1080), 0)],
            seconds=8.0,
            prompt=prompts["fl2va"],
        ),
    ]

    # ref2va: images at their own 2048 short edge, soundtracks, videos.
    cases += [
        request(
            "ref2va_image",
            "ref2va",
            [reference("image", image(1600, 900))],
            prompt=PROMPT,
        ),
        request(
            "ref2va_image_audio",
            "ref2va",
            [
                reference("image", image(900, 1600)),
                reference("audio", audio(44100, 44100 * 9)),
            ],
            aspect_ratio="4:3",
            prompt=PROMPT,
        ),
        request(
            "ref2va_video_audio",
            "ref2va",
            [
                reference(
                    "video_audio",
                    video((16, 9), (30000, 1001), 300, audio(44100, 441000)),
                    1.5,
                ),
                reference("audio", audio(48000, 48000 * 3)),
            ],
            prompt=PROMPT,
        ),
        request(
            "ref2va_video_image_video",
            "ref2va",
            [
                reference("video", video((9, 16), (25, 1), 75)),
                reference("image", image(512, 512)),
                reference(
                    "video",
                    video((4, 3), (60, 1), 900, audio(32000, 32000 * 15)),
                ),
            ],
            aspect_ratio="16:9",
            seconds=6.0,
            prompt=PROMPT_UNICODE,
        ),
        request(
            "ref2va_duration_from_audio",
            "ref2va",
            [
                reference("image", image(1080, 1350)),
                reference("audio", audio(44100, 145530 * 2)),
            ],
            aspect_ratio="1:1",
            seconds=None,
            prompt=PROMPT,
        ),
        request(
            "ref2va_mixed_keyframes",
            "ref2va",
            [
                reference("image", image(1600, 900)),
                keyframe(image(1920, 1080), 0),
                keyframe(image(1000, 1000), -1),
            ],
            prompt=PROMPT,
        ),
    ]
    for width, height in (
        (2560, 1440),
        (1440, 2560),
        (1000, 1000),
        (300, 200),
        (4000, 1000),
        (1000, 4000),
        (1746, 1432),
    ):
        cases.append(
            request(
                f"ref2va_image_{width}x{height}",
                "ref2va",
                [reference("image", image(width, height))],
            )
        )
    for name, track in (
        ("44k_stereo_8.5s", audio(44100, 375552, 2)),
        ("16k_mono_3s", audio(16000, 16000 * 3)),
        ("48k_stereo_5s", audio(48000, 48000 * 5, 2)),
        ("32k_stereo_15s", audio(32000, 32000 * 15, 2)),
    ):
        cases.append(
            request(
                f"ref2va_image_audio_{name}",
                "ref2va",
                [
                    reference("image", image(2560, 1440)),
                    reference("audio", track),
                ],
            )
        )
    cases.append(
        request(
            "ref2va_audio_first_image",
            "ref2va",
            [
                reference("audio", audio(22050, 22050 * 4)),
                reference("image", image(1080, 1920)),
            ],
        )
    )
    for name, media, seconds in (
        ("1080p_24fps_6s_audio", OFFICIAL_VIDEO, 5.0),
        ("360p_30fps_3s", video((640, 360), (30, 1), 90), 5.0),
        (
            "portrait_25fps_10s_audio",
            video((360, 640), (25, 1), 250, audio(48000, 480000, 2)),
            10.0,
        ),
        ("square_60fps_2s", video((480, 480), (60, 1), 120), 5.0),
        (
            "ntsc_23.976fps_5s_mono",
            video((854, 480), (24000, 1001), 120, audio(32000, 160160)),
            5.0,
        ),
        ("1080p_24fps_1s", video((1920, 1080), (24, 1), 24), 5.0),
        (
            "1080p_24fps_15s_target_14s",
            video((1920, 1080), (24, 1), 360, audio(44100, 661500, 2)),
            14.0,
        ),
    ):
        cases.append(
            request(
                f"ref2va_video_{name}",
                "ref2va",
                [reference("video", media)],
                seconds=seconds,
            )
        )
    cases += [
        request(
            "ref2va_two_videos",
            "ref2va",
            [
                reference("video", OFFICIAL_VIDEO),
                reference(
                    "video",
                    video((2560, 1440), (24, 1), 158, audio(32000, 210944, 2)),
                ),
            ],
        ),
        request(
            "ref2va_three_videos_images_audio",
            "ref2va",
            [
                reference(
                    "video",
                    video((1920, 1080), (30, 1), 150, audio(44100, 220500, 2)),
                ),
                reference("image", image(1024, 1024)),
                reference("video", video((720, 1280), (24, 1), 96)),
                reference("audio", audio(48000, 48000 * 6, 2)),
                reference("video", video((1280, 720), (25, 1), 125)),
                reference("image", image(800, 1200)),
            ],
        ),
        request(
            "ref2va_official",
            "ref2va",
            [
                reference("video", OFFICIAL_VIDEO),
                reference("audio", OFFICIAL_AUDIO),
            ],
            prompt=prompts["ref2va"],
        ),
    ]
    return cases


def rejected_requests() -> list[dict]:
    """Requests the serving contract rejects, each naming the field at fault."""
    cases = [
        (
            "t2va_with_condition",
            "t2va",
            {"duration_seconds": 5.0},
            [reference("image", image(64, 64))],
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
            [keyframe(image(640, 480), 0)],
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
            [keyframe(image(640, 480), 0), reference("image", image(64, 64))],
            "conditions[1]",
        ),
        (
            "fl2va_reversed",
            "fl2va",
            {"duration_seconds": 5.0},
            [keyframe(image(640, 480), -1), keyframe(image(640, 480), 0)],
            "conditions",
        ),
        (
            "fl2va_middle_frame",
            "fl2va",
            {"duration_seconds": 5.0},
            [keyframe(image(640, 480), 12)],
            "conditions[0]",
        ),
        (
            "fl2va_keyframe_ratio",
            "fl2va",
            {"duration_seconds": 5.0},
            [keyframe(image(5000, 1000), 0)],
            "conditions[0]",
        ),
        (
            "keyframe_video",
            "ref2va",
            {"duration_seconds": 5.0},
            [
                reference("image", image(64, 64)),
                {
                    "type": "video",
                    "role": "keyframe",
                    "frame_index": 0,
                    "media": video((16, 9), (24, 1), 48),
                },
            ],
            "conditions[1]",
        ),
        (
            "start_on_image",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("image", image(64, 64), 1.0)],
            "conditions[0]",
        ),
        (
            "too_many_images",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("image", image(64, 64)) for _ in range(10)],
            "conditions",
        ),
        (
            "audio_only",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("audio", audio(32000, 32000))],
            "conditions",
        ),
        (
            "reference_image_ratio",
            "ref2va",
            {"duration_seconds": 5.0},
            [
                reference("image", image(64, 64)),
                reference("image", image(4100, 1000)),
            ],
            "conditions[1]",
        ),
        (
            "short_video",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("video", video((16, 9), (24, 1), 21))],
            "conditions[0]",
        ),
        (
            "start_past_video",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("video", video((16, 9), (24, 1), 120), 4.5)],
            "conditions[0]",
        ),
        (
            "video_audio_silent",
            "ref2va",
            {"duration_seconds": 5.0},
            [reference("video_audio", video((16, 9), (24, 1), 120))],
            "conditions[0]",
        ),
        (
            "duration_two_soundtracks",
            "ref2va",
            {},
            [
                reference("audio", audio(32000, 32000 * 6)),
                reference(
                    "video",
                    video((16, 9), (24, 1), 120, audio(32000, 32000 * 6)),
                ),
            ],
            "target.duration_seconds",
        ),
        (
            "duration_no_soundtrack",
            "ref2va",
            {},
            [reference("video", video((16, 9), (24, 1), 120))],
            "target.duration_seconds",
        ),
        (
            "duration_from_short_audio",
            "ref2va",
            {},
            [
                reference("image", image(64, 64)),
                reference("audio", audio(32000, 32000 * 3)),
            ],
            "conditions[1]",
        ),
        (
            "duration_images_only",
            "ref2va",
            {},
            [reference("image", image(64, 64))],
            "target.duration_seconds",
        ),
    ]
    rejected = []
    for name, task, target, conditions, field in cases:
        rejected.append(
            {
                "name": name,
                "task": task,
                "prompt": PROMPT,
                "target": {"short_edge": 768, "aspect_ratio": "auto", **target},
                "conditions": conditions,
                "error": {"field": field},
            }
        )
    # A blank prompt is rejected when the request is presented.
    rejected.append(
        {
            **request("prompt_blank", "t2va", prompt=" \n\t"),
            "error": {"field": "prompt"},
        }
    )
    return rejected


def _revision(checkpoint: Path, path: str) -> str:
    """The Hub revision a checkpoint file was downloaded at."""
    metadata = checkpoint / ".cache" / "huggingface" / "download"
    return (metadata / f"{path}.metadata").read_text().splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    ref = _Reference(args.checkpoint)
    prompts = {
        task: _official_request(task)["prompt"]
        for task in ("t2va", "fl2va", "ref2va")
    }

    fixture = {
        "description": (
            "MiniMax-H3 request-planning vectors shared by the Rust planner "
            "(crates/server/src/serving/video) and the Python planner "
            "(uniserve_models/minimax_h3/processing.py), generated from the "
            "diffusers MiniMax-H3 pipeline and the checkpoint's Qwen3-VL "
            "processor, tokenizer and audio VAE by "
            "tools/minimax_h3/plan_vectors.py, whose docstring documents the "
            "layout and the source of every value. Sizes are [width, "
            "height]; vision grids are Qwen [t, h, w] patch grids; rows are "
            "denoiser rows. Media are given by probed facts: an image's "
            "displayed size; a video's display size (only its aspect "
            "matters), frame rate [num, den], decoded frame count and "
            "soundtrack; an audio stream's sample rate, channel count and "
            "decoded samples per channel. A rejected request names the field "
            "at fault."
        ),
        "generator": "tools/minimax_h3/plan_vectors.py",
        "sources": {
            "checkpoint": "MiniMaxAI/MiniMax-H3",
            "components_revision": _revision(
                args.checkpoint, "tokenizer/tokenizer.json"
            ),
            "request_scripts_revision": _revision(
                args.checkpoint,
                "scripts/readme/reproducible-768p-t2va-request.sh",
            ),
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "torchaudio": torchaudio.__version__,
        },
        "vision": _vision_config(args.checkpoint / "processor"),
        "tokens": {
            name: ref.tokenizer.convert_tokens_to_ids(name)
            for name in (
                "<|vision_start|>",
                "<|vision_end|>",
                "<|image_pad|>",
                "<|video_pad|>",
            )
        },
        "canvas_cases": canvas_cases(ref),
        "duration_cases": duration_cases(ref),
        "reference_image_cases": reference_image_cases(ref),
        "keyframe_cases": keyframe_cases(ref),
        "video_reference_cases": video_reference_cases(ref),
        "audio_cases": audio_cases(ref),
    }
    cases = accepted_requests(prompts)
    for case in cases:
        case["expected"] = expected_plan(ref, case)
        expected = case["expected"]
        print(
            f"{case['name']}: {expected['canvas'][0]}x{expected['canvas'][1]}"
            f" {expected['num_frames']} frames,"
            f" {expected['sequence_rows']} rows",
            flush=True,
        )
    fixture["requests"] = cases + rejected_requests()
    args.output.write_text(_format(fixture))
    print(f"wrote {args.output}: {len(fixture['requests'])} requests")


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
