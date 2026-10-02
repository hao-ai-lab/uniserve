"""Plan a conditioned MiniMax-H3 request and present it to the conditioner.

A MiniMax-H3 request is sized before any pixel is decoded. The target canvas
and frame count, the size every condition is prepared at, the Qwen3-VL vision
grids, the denoiser's condition rows and the tokenized presentation all
follow from the request and a few probed facts per condition: the displayed
size of an image, the displayed aspect, frame rate and frame count of a
video, and the sample rate and sample count of a soundtrack.

``plan_request`` resolves those rules into a ``RequestPlan`` and ``present``
tokenizes the presentation the conditioner reads. The serving path applies
the same rules in Rust (``crates/server/src/serving/video``); both are checked
against ``tests/python/fixtures/minimax_h3_plan.json``, which is generated
from the diffusers reference pipeline.

Rules that the reference pipeline does not define are fixed here and in the
Rust planner alike:

* ``start_time_seconds`` drops the first ``floor(s * 24 + 0.5)`` frames of
  a reference video's 24 fps timeline and the first ``floor(s * rate + 0.5)``
  samples of its soundtrack.
* A reference video needs at least 22 frames (one ``17 * n + 5`` VAE window)
  on its 24 fps timeline after the start offset.
* ``ref2va`` keyframes are put on the target canvas like ``fl2va`` ones (the
  first stretched, a second one cover-cropped) and do not enter the
  conditioner.
* Without ``target.duration_seconds``, a ``ref2va`` request whose references
  carry exactly one soundtrack lasts as long as that soundtrack after its
  start offset.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from uniserve.media import image

from .packing import (
    AUDIO_CHANNELS,
    FPS,
    TEXT_TAG,
    VIDEO_TAG,
    audio_latent_frames,
    video_latent_frames,
)

__all__ = [
    "AudioClip",
    "AudioFacts",
    "Condition",
    "ConditionPlan",
    "ConditionType",
    "CoverCrop",
    "FramePosition",
    "ImageFacts",
    "ImageVision",
    "KeyframeFit",
    "PlanError",
    "Presentation",
    "RequestPlan",
    "Role",
    "Target",
    "Task",
    "Tokenizer",
    "TextSegment",
    "VideoClip",
    "VideoFacts",
    "VideoVision",
    "VisionConfig",
    "VisionSegment",
    "audio_clip",
    "canvas",
    "cover_crop",
    "frame_count",
    "plan_request",
    "present",
    "presentation_segments",
    "reference_image_size",
    "rows_per_frame",
    "sample_video_frames",
    "video_reference",
]

# Requested durations, in seconds. The aligned frame count may run past the
# upper bound: 15 seconds is 360 frames, aligned up to 362.
MIN_SECONDS = 4.0
MAX_SECONDS = 15.0

# The adapt_shape_v1 canvas rule, implemented by `canvas` alone: the short
# edge starts at 768 pixels, the area is capped at 768 * 1344 and each side
# rounds to the nearest multiple of 32 (the VAE's 16x spatial compression
# times the 2x2 patch). t2va and ref2va accept the named ratios; their `auto`
# is 16:9.
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4.0
NAMED_ASPECT_RATIOS = ((21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16))
DEFAULT_ASPECT_RATIO = (16, 9)

# An image reference is encoded at its own 2048-pixel short edge, upscaling
# included and without an area cap.
REFERENCE_IMAGE_SHORT_EDGE = 2048

# The video VAE encodes 17-frame windows plus a 5-frame overlap.
VAE_FRAMES_PER_CHUNK = 17
VAE_LATENTS_PER_CHUNK = 5
MIN_REFERENCE_FRAMES = VAE_FRAMES_PER_CHUNK + VAE_LATENTS_PER_CHUNK

# The audio VAE hops 800 samples at 32 kHz, i.e. 40 latents per second.
AUDIO_SAMPLE_RATE = 32000
AUDIO_HOP = 800

# The conditioner reads a reference video at 2 frames per second.
VIDEO_SAMPLE_FPS = 2.0

# Reference counts the released checkpoint documents for one request.
MAX_IMAGE_REFERENCES = 9
MAX_VIDEO_REFERENCES = 3
MAX_AUDIO_REFERENCES = 3
MAX_REFERENCES = 12

# Qwen3-VL vision placeholders. A presentation wraps every vision block in
# the start/end markers; the pads stand for the vision tokens.
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"


class Task(StrEnum):
    """A MiniMax-H3 task, by its request name."""

    T2VA = "t2va"
    FL2VA = "fl2va"
    REF2VA = "ref2va"


class ConditionType(StrEnum):
    """The media type of a condition, by its request name.

    ``VIDEO`` conditions on a soundtrack when the video has one;
    ``VIDEO_AUDIO`` requires one.
    """

    IMAGE = "image"
    VIDEO = "video"
    VIDEO_AUDIO = "video_audio"
    AUDIO = "audio"


class Role(StrEnum):
    """Whether a condition anchors a generated frame or is a reference."""

    KEYFRAME = "keyframe"
    REFERENCE = "reference"


class FramePosition(StrEnum):
    """The generated frame a keyframe anchors."""

    FIRST = "first"
    LAST = "last"


class PlanError(ValueError):
    """A request MiniMax-H3 cannot serve.

    Attributes:
        field: The request field at fault, such as ``"conditions[2]"`` or
            ``"target.aspect_ratio"``.
    """

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field


@dataclass(frozen=True, slots=True)
class ImageFacts:
    """An image as displayed, after its EXIF orientation."""

    width: int
    height: int


@dataclass(frozen=True, slots=True)
class AudioFacts:
    """A soundtrack's native sample rate and decoded sample count."""

    sample_rate: int
    samples: int


@dataclass(frozen=True, slots=True)
class VideoFacts:
    """A video's display aspect, frame rate, frame count and soundtrack.

    Attributes:
        display_width: Width of the displayed aspect. Only the ratio to
            ``display_height`` matters; a probe passes the coded size scaled
            by the sample aspect ratio and turned by the display rotation.
        display_height: Height of the displayed aspect.
        frame_rate: The container's average frame rate.
        frames: Decoded frames of the whole video.
        soundtrack: The first audio stream, when the video has one.
    """

    display_width: int
    display_height: int
    frame_rate: Fraction
    frames: int
    soundtrack: AudioFacts | None = None


@dataclass(frozen=True, slots=True)
class Condition:
    """One request condition with the facts probed from its media.

    Attributes:
        type: The condition's media type.
        role: Keyframe or reference.
        media: Facts of the media, matching ``type``.
        frame_index: ``0`` or ``-1`` for a keyframe; ``None`` otherwise.
        start_seconds: Offset into a video reference; ``None`` for zero.
    """

    type: ConditionType
    role: Role
    media: ImageFacts | VideoFacts | AudioFacts
    frame_index: int | None = None
    start_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class Target:
    """The requested output: short edge, aspect ratio and duration."""

    aspect_ratio: str = "auto"
    duration_seconds: float | None = None
    short_edge: int = CANVAS_SHORT_EDGE


@dataclass(frozen=True, slots=True)
class VisionConfig:
    """Qwen3-VL processor geometry: patching and the pixel budgets.

    Attributes:
        patch_size: Pixels per vision patch side.
        temporal_patch_size: Frames merged into one temporal patch.
        merge_size: Patches per merged vision token side.
        image_min_pixels: Lower pixel budget of an image.
        image_max_pixels: Upper pixel budget of an image.
        video_min_pixels: Lower budget of a video, frames times pixels.
        video_max_pixels: Upper budget of a video, frames times pixels.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    image_min_pixels: int
    image_max_pixels: int
    video_min_pixels: int
    video_max_pixels: int

    @classmethod
    def from_processor(cls, directory: str | Path) -> VisionConfig:
        """Read the image and video processor configs of a checkpoint.

        Args:
            directory: The checkpoint's ``processor`` directory, holding
                ``preprocessor_config.json`` and
                ``video_preprocessor_config.json``.

        Raises:
            ValueError: The two configs disagree on the patch geometry.
        """
        root = Path(directory)
        image = json.loads((root / "preprocessor_config.json").read_text())
        video = json.loads(
            (root / "video_preprocessor_config.json").read_text()
        )
        for name in ("patch_size", "temporal_patch_size", "merge_size"):
            if image[name] != video[name]:
                raise ValueError(
                    f"the image and video processors disagree on {name}: "
                    f"{image[name]} and {video[name]}"
                )
        return cls(
            patch_size=image["patch_size"],
            temporal_patch_size=image["temporal_patch_size"],
            merge_size=image["merge_size"],
            image_min_pixels=image["size"]["shortest_edge"],
            image_max_pixels=image["size"]["longest_edge"],
            video_min_pixels=video["size"]["shortest_edge"],
            video_max_pixels=video["size"]["longest_edge"],
        )

    @property
    def factor(self) -> int:
        """What a resized side is a multiple of: one merged token."""
        return self.patch_size * self.merge_size

    def image_grid(self, height: int, width: int) -> tuple[int, int, int]:
        """The ``(t, h, w)`` patch grid the image processor resizes to.

        This is Qwen2-VL's ``smart_resize``: each side rounds to the factor,
        and a size outside the pixel budget is rescaled with floor (above)
        or ceil (below) rounding.
        """
        factor = self.factor
        if max(height, width) / min(height, width) > 200:
            raise ValueError(f"image aspect too extreme: {width}x{height}")
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        if resized_height * resized_width > self.image_max_pixels:
            beta = math.sqrt((height * width) / self.image_max_pixels)
            resized_height = max(
                factor, math.floor(height / beta / factor) * factor
            )
            resized_width = max(
                factor, math.floor(width / beta / factor) * factor
            )
        elif resized_height * resized_width < self.image_min_pixels:
            beta = math.sqrt(self.image_min_pixels / (height * width))
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
        patch = self.patch_size
        return 1, resized_height // patch, resized_width // patch

    def video_grid(
        self, num_frames: int, height: int, width: int
    ) -> tuple[int, int, int]:
        """The ``(t, h, w)`` patch grid of ``num_frames`` sampled frames.

        This is Qwen3-VL's video ``smart_resize``: the budget covers all
        frames, and the frames are padded by repeating the last one up to a
        multiple of the temporal patch.
        """
        factor = self.factor
        temporal = self.temporal_patch_size
        if height < factor or width < factor:
            raise ValueError(f"video frames too small: {width}x{height}")
        if max(height, width) / min(height, width) > 200:
            raise ValueError(f"video aspect too extreme: {width}x{height}")
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        padded_frames = math.ceil(num_frames / temporal) * temporal
        budget = padded_frames * resized_height * resized_width
        if budget > self.video_max_pixels:
            beta = math.sqrt(
                (num_frames * height * width) / self.video_max_pixels
            )
            resized_height = max(
                factor, math.floor(height / beta / factor) * factor
            )
            resized_width = max(
                factor, math.floor(width / beta / factor) * factor
            )
        elif budget < self.video_min_pixels:
            beta = math.sqrt(
                self.video_min_pixels / (num_frames * height * width)
            )
            resized_height = math.ceil(height * beta / factor) * factor
            resized_width = math.ceil(width * beta / factor) * factor
        patch = self.patch_size
        return (
            padded_frames // temporal,
            resized_height // patch,
            resized_width // patch,
        )

    def block_tokens(self, grid: tuple[int, int, int]) -> int:
        """Vision tokens of one temporal block of ``grid``."""
        return grid[1] * grid[2] // self.merge_size**2


@dataclass(frozen=True, slots=True)
class CoverCrop:
    """An aspect-preserving resize followed by a centred crop.

    Attributes:
        width: Width the image is resized to before cropping.
        height: Height the image is resized to before cropping.
        left: Left edge of the crop in the resized image.
        top: Top edge of the crop in the resized image.
    """

    width: int
    height: int
    left: int
    top: int


@dataclass(frozen=True, slots=True)
class KeyframeFit:
    """How a keyframe is put on the target canvas.

    Attributes:
        position: The generated frame the keyframe anchors.
        cover_crop: ``None`` when the keyframe is stretched onto the canvas
            (the first keyframe); the crop otherwise.
    """

    position: FramePosition
    cover_crop: CoverCrop | None


@dataclass(frozen=True, slots=True)
class AudioClip:
    """The samples of a soundtrack the request conditions on.

    Attributes:
        sample_rate: The native sample rate.
        start_sample: Native samples skipped at the start.
        source_samples: Native samples kept after the start offset.
        samples: Samples after resampling to 32 kHz.
        latents: Audio latents per channel; the clip packs ``2 * latents``
            channel-major rows.
    """

    sample_rate: int
    start_sample: int
    source_samples: int
    samples: int
    latents: int

    @property
    def rows(self) -> int:
        """Denoiser rows of the clip: one per latent per stereo channel."""
        return AUDIO_CHANNELS * self.latents


@dataclass(frozen=True, slots=True)
class VideoClip:
    """The frames of a reference video the request conditions on.

    Attributes:
        canvas: The canvas of the video's own aspect the frames are put on.
        start_frame: Frames of the 24 fps timeline skipped at the start.
        frames: 24 fps frames kept after the start offset, at most the
            generated frame count; the conditioner samples these.
        vae_frames: The leading ``17 * n + 5`` frames the VAE encodes.
        latent_frames: Latent frames of the VAE encoding.
        soundtrack: The video's soundtrack, when it has one.
    """

    canvas: image.Config
    start_frame: int
    frames: int
    vae_frames: int
    latent_frames: int
    soundtrack: AudioClip | None


@dataclass(frozen=True, slots=True)
class ImageVision:
    """The conditioner's view of one image.

    Attributes:
        grid: Qwen ``(t, h, w)`` patch grid, ``t`` being 1.
        tokens: Vision tokens of the image.
    """

    grid: tuple[int, int, int]
    tokens: int


@dataclass(frozen=True, slots=True)
class VideoVision:
    """The conditioner's view of one reference video.

    Attributes:
        grid: Qwen ``(t, h, w)`` patch grid; ``t`` counts vision blocks.
        block_tokens: Vision tokens of each block.
        frame_indices: The 24 fps frames sampled at 2 fps.
        block_timestamps: The timestamp label of each block, in seconds.
    """

    grid: tuple[int, int, int]
    block_tokens: int
    frame_indices: tuple[int, ...]
    block_timestamps: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ConditionPlan:
    """How one condition is prepared and what it contributes.

    Attributes:
        index: Position of the condition in the request.
        type: The condition's media type.
        prepared: The keyframe fit, the size an image reference is resized
            to (LANCZOS), or the video or audio clip.
        vision: The conditioner's view, for conditions it reads.
        video_rows: Denoiser video condition rows.
        audio_rows: Denoiser audio condition rows.
    """

    index: int
    type: ConditionType
    prepared: KeyframeFit | image.Config | VideoClip | AudioClip
    vision: ImageVision | VideoVision | None
    video_rows: int
    audio_rows: int


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """The resolved size of a request and of every condition.

    Attributes:
        task: The request's task.
        canvas: The generated canvas.
        num_frames: Generated frames, of the form ``17 * n + 5``.
        latent_frames: Generated video latent frames.
        audio_latents: Generated audio latents per stereo channel.
        conditions: One plan per condition, in request order.
    """

    task: Task
    canvas: image.Config
    num_frames: int
    latent_frames: int
    audio_latents: int
    conditions: tuple[ConditionPlan, ...]

    @property
    def target_video_rows(self) -> int:
        """Denoiser rows of the generated video."""
        return self.latent_frames * rows_per_frame(self.canvas)

    @property
    def target_audio_rows(self) -> int:
        """Denoiser rows of the generated stereo audio."""
        return AUDIO_CHANNELS * self.audio_latents

    @property
    def condition_video_rows(self) -> int:
        """Denoiser video rows of all conditions."""
        return sum(condition.video_rows for condition in self.conditions)

    @property
    def condition_audio_rows(self) -> int:
        """Denoiser audio rows of all conditions."""
        return sum(condition.audio_rows for condition in self.conditions)


def canvas(aspect_width: float, aspect_height: float) -> image.Config:
    """Resolve a display aspect into a canvas with the adapt_shape_v1 rule.

    Only the ratio of the arguments matters. The short edge starts at 768
    pixels, an area above ``768 * 1344`` scales both sides down, and each
    side rounds to the nearest multiple of 32, so the final area may end up
    slightly above the cap.

    Raises:
        ValueError: The ratio lies outside 1:4 to 4:1.
    """
    if not (aspect_width > 0 and aspect_height > 0):
        raise ValueError(
            f"aspect must be positive, got {aspect_width}:{aspect_height}"
        )
    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"aspect {aspect_width}:{aspect_height} lies outside 1:4 to 4:1"
        )
    if ratio >= 1.0:
        width, height = CANVAS_SHORT_EDGE * ratio, float(CANVAS_SHORT_EDGE)
    else:
        width, height = float(CANVAS_SHORT_EDGE), CANVAS_SHORT_EDGE / ratio
    area = width * height
    if area > CANVAS_MAX_PIXELS:
        # `** 0.5` rather than `math.sqrt` reproduces the reference's
        # arithmetic.
        scale = (CANVAS_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    return image.Config(
        height=_nearest_multiple(height), width=_nearest_multiple(width)
    )


def rows_per_frame(size: image.Config) -> int:
    """Denoiser rows of one latent frame: one per 32x32 pixel block."""
    return (size.height // CANVAS_MULTIPLE) * (size.width // CANVAS_MULTIPLE)


def frame_count(seconds: float, max_seconds: float = MAX_SECONDS) -> int:
    """Return the generated frame count of a duration.

    The duration times 24 rounds half to even and aligns up to the next
    ``17 * n + 5``: 4 seconds give 107 frames, 15 seconds 362.

    Raises:
        ValueError: ``seconds`` lies outside ``[4, max_seconds]``.
    """
    if not math.isfinite(seconds) or not MIN_SECONDS <= seconds <= max_seconds:
        raise ValueError(
            f"duration must lie in [{MIN_SECONDS:g}, {max_seconds:g}] "
            f"seconds, got {seconds:g}"
        )
    requested = round(seconds * FPS)
    return (
        requested + (VAE_LATENTS_PER_CHUNK - requested) % VAE_FRAMES_PER_CHUNK
    )


def reference_image_size(width: int, height: int) -> image.Config:
    """Return the size an image reference is encoded at.

    The short edge scales to 2048 pixels, upscaling included, and each side
    rounds to the nearest multiple of 32. This is not the canvas rule: there
    is no area cap.

    Raises:
        ValueError: The image lies outside 1:4 to 4:1.
    """
    if width > 4 * height or height > 4 * width:
        raise ValueError(
            f"a reference image must lie within 1:4 and 4:1, got "
            f"{width}x{height}"
        )
    scale = REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    return image.Config(
        height=_nearest_multiple(height * scale),
        width=_nearest_multiple(width * scale),
    )


def cover_crop(width: int, height: int, size: image.Config) -> CoverCrop:
    """Return the resize and centred crop that cover a canvas of ``size``.

    The scale is the larger of the two side ratios; each resized side rounds
    half to even and never falls below the canvas, and the crop is centred
    with floor division.
    """
    scale = max(size.width / width, size.height / height)
    resized_width = max(size.width, round(width * scale))
    resized_height = max(size.height, round(height * scale))
    return CoverCrop(
        width=resized_width,
        height=resized_height,
        left=max(0, (resized_width - size.width) // 2),
        top=max(0, (resized_height - size.height) // 2),
    )


def sample_video_frames(
    num_frames: int, temporal_patch: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Sample a 24 fps clip at 2 fps and label its vision blocks.

    Every twelfth frame is sampled. The conditioner merges the samples in
    groups of ``temporal_patch``, repeating the last one to fill a group, and
    a group is labelled with the mean of its timestamps.

    Returns:
        The sampled frame indices and one timestamp per vision block.

    Raises:
        ValueError: Fewer frames are sampled than one group holds.
    """
    stride = FPS / VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < num_frames:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < temporal_patch:
        raise ValueError(
            f"a reference video must sample at least {temporal_patch} "
            f"frames at {VIDEO_SAMPLE_FPS:g} fps, got {len(indices)}"
        )
    timestamps = [index / VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    blocks = tuple(
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    )
    return tuple(indices), blocks


def video_reference(
    media: VideoFacts,
    start_seconds: float,
    num_frames: int,
    vision: VisionConfig,
) -> tuple[VideoClip, VideoVision]:
    """Size a reference video and the conditioner's view of it.

    The video is resampled to 24 fps, its first ``floor(s * 24 + 0.5)``
    frames are skipped and the rest is truncated to the generated frame
    count, on the canvas of its own aspect. Its soundtrack is offset by the
    same start and sized like an audio reference.

    Args:
        media: The probed video.
        start_seconds: The request's start offset into the video.
        num_frames: The generated frame count.
        vision: The conditioner's processor geometry.

    Raises:
        ValueError: The video's aspect lies outside 1:4 to 4:1, fewer than
            22 frames remain after the offset, or the offset lies past the
            end of its soundtrack.
    """
    try:
        size = canvas(media.display_width, media.display_height)
    except ValueError as error:
        raise ValueError(f"the video's {error}") from None
    frame_rate = float(media.frame_rate)
    if frame_rate == FPS:
        timeline = media.frames
    else:
        # The reference resamples to 24 fps by holding every frame until
        # the slot of the next one; the timeline ends at the slot the
        # stream's end rounds to.
        timeline = math.floor(media.frames * (FPS / frame_rate) + 0.5)
    start_frame = _start_offset(start_seconds, FPS)
    available = timeline - start_frame
    if available < MIN_REFERENCE_FRAMES:
        raise ValueError(
            f"a reference video needs at least {MIN_REFERENCE_FRAMES} "
            f"frames at {FPS} fps after start_time_seconds, got "
            f"{max(available, 0)}"
        )
    frames = min(available, num_frames)
    # The VAE encodes the leading complete `17 * n + 5` frames.
    vae_frames = (
        max(1, (frames - VAE_LATENTS_PER_CHUNK) // VAE_FRAMES_PER_CHUNK)
        * VAE_FRAMES_PER_CHUNK
        + VAE_LATENTS_PER_CHUNK
    )
    soundtrack = None
    if media.soundtrack is not None:
        soundtrack = audio_clip(media.soundtrack, start_seconds, num_frames)

    indices, timestamps = sample_video_frames(
        frames, vision.temporal_patch_size
    )
    grid = vision.video_grid(len(indices), size.height, size.width)
    clip = VideoClip(
        canvas=size,
        start_frame=start_frame,
        frames=frames,
        vae_frames=vae_frames,
        latent_frames=video_latent_frames(vae_frames),
        soundtrack=soundtrack,
    )
    seen = VideoVision(
        grid=grid,
        block_tokens=vision.block_tokens(grid),
        frame_indices=indices,
        block_timestamps=timestamps,
    )
    return clip, seen


def audio_clip(
    media: AudioFacts, start_seconds: float, num_frames: int
) -> AudioClip:
    """Size a soundtrack: offset, truncation, 32 kHz resampling, latents.

    The first ``floor(s * rate + 0.5)`` samples are skipped and the rest is
    truncated at the native rate to the generated duration, as the
    reference does before it resamples to 32 kHz.

    Raises:
        ValueError: The offset lies past the end of the soundtrack.
    """
    start_sample = _start_offset(start_seconds, media.sample_rate)
    available = media.samples - start_sample
    if available <= 0:
        raise ValueError(
            "start_time_seconds lies past the end of the audio track"
        )
    limit = int(num_frames / FPS * media.sample_rate)
    source = min(available, limit)
    if media.sample_rate == AUDIO_SAMPLE_RATE:
        samples = source
    else:
        # torchaudio's resampler returns ceil(new * length / orig) samples
        # for the gcd-reduced rates.
        divisor = math.gcd(media.sample_rate, AUDIO_SAMPLE_RATE)
        new = AUDIO_SAMPLE_RATE // divisor
        orig = media.sample_rate // divisor
        samples = -(-source * new // orig)
    return AudioClip(
        sample_rate=media.sample_rate,
        start_sample=start_sample,
        source_samples=source,
        samples=samples,
        # The audio VAE pads the waveform to a whole hop.
        latents=-(-samples // AUDIO_HOP),
    )


def plan_request(
    task: Task,
    target: Target,
    conditions: Sequence[Condition],
    vision: VisionConfig,
    *,
    tasks: Sequence[Task] = tuple(Task),
    max_seconds: float = MAX_SECONDS,
    canvases: Sequence[image.Config] | None = None,
) -> RequestPlan:
    """Validate a request and resolve every size it implies.

    Args:
        task: The request's task.
        target: The requested output.
        conditions: The conditions in request order, with probed facts.
        vision: The conditioner's processor geometry.
        tasks: The tasks of the denoiser that serves the request.
        max_seconds: The longest duration served.
        canvases: The only canvases the checkpoint serves, when restricted.

    Raises:
        PlanError: The request breaks a rule; ``field`` names the culprit.
    """
    if task not in tasks:
        served = ", ".join(served_task.value for served_task in tasks)
        raise PlanError(
            "task", f"{task.value} is not served; the denoiser serves {served}"
        )
    _check_request(task, target, conditions, max_seconds, canvases)

    # Media-dependent rules: soundtracks and the duration they may imply.
    soundtracks: list[tuple[int, AudioFacts, float]] = []
    for index, condition in enumerate(conditions):
        _check_media(index, condition)
        track = _soundtrack(condition)
        if track is not None:
            soundtracks.append((index, track, condition.start_seconds or 0.0))
    num_frames = _resolve_frames(target, soundtracks, max_seconds)

    size = _target_canvas(task, target, conditions)
    _check_served(size, canvases)

    plans = []
    keyframes = 0
    for index, condition in enumerate(conditions):
        if condition.role is Role.KEYFRAME:
            plans.append(
                _plan_keyframe(index, condition, size, keyframes, task, vision)
            )
            keyframes += 1
        else:
            plans.append(_plan_reference(index, condition, num_frames, vision))
    return RequestPlan(
        task=task,
        canvas=size,
        num_frames=num_frames,
        latent_frames=video_latent_frames(num_frames),
        audio_latents=audio_latent_frames(num_frames),
        conditions=tuple(plans),
    )


@dataclass(frozen=True, slots=True)
class TextSegment:
    """Presentation text, tokenized on its own."""

    text: str


@dataclass(frozen=True, slots=True)
class VisionSegment:
    """A vision block: start marker, ``tokens`` pads and end marker."""

    pad: str
    tokens: int


@dataclass(frozen=True, slots=True)
class Presentation:
    """The tokenized presentation and the modality tag of every token.

    Vision tokens, their markers included, carry the video tag; all other
    tokens carry the text tag.
    """

    token_ids: tuple[int, ...]
    tags: tuple[int, ...]


class Tokenizer(Protocol):
    """The tokenizer calls a presentation needs."""

    def encode(self, text: str, add_special_tokens: bool) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


def presentation_segments(
    plan: RequestPlan, prompt: str
) -> list[TextSegment | VisionSegment]:
    """List the presentation of a request, before tokenization.

    ``fl2va`` labels each keyframe ``"<Picture i>: "`` before its vision
    block. ``ref2va`` labels its references in request order, numbered per
    modality: a soundtrack ``"<Audio j>: "`` (a video's first, before its
    ``"<Video k>: "`` label), an image ``"<Picture i>: "`` before its block,
    and a video one ``"<{t:.1f} seconds>"`` label per vision block. Audio
    contributes labels only and ``ref2va`` keyframes nothing. The prompt
    follows verbatim.
    """
    segments: list[TextSegment | VisionSegment] = []
    pictures = videos = audios = 0
    for condition in plan.conditions:
        prepared, vision = condition.prepared, condition.vision
        if isinstance(prepared, AudioClip):
            audios += 1
            segments.append(TextSegment(f"<Audio {audios}>: "))
        elif isinstance(prepared, VideoClip):
            if prepared.soundtrack is not None:
                audios += 1
                segments.append(TextSegment(f"<Audio {audios}>: "))
            assert isinstance(vision, VideoVision)
            videos += 1
            segments.append(TextSegment(f"<Video {videos}>: "))
            for timestamp in vision.block_timestamps:
                segments.append(TextSegment(f"<{timestamp:.1f} seconds>"))
                segments.append(VisionSegment(VIDEO_PAD, vision.block_tokens))
        elif isinstance(vision, ImageVision):
            pictures += 1
            segments.append(TextSegment(f"<Picture {pictures}>: "))
            segments.append(VisionSegment(IMAGE_PAD, vision.tokens))
    segments.append(TextSegment(prompt))
    return segments


def present(
    tokenizer: Tokenizer, plan: RequestPlan, prompt: str
) -> Presentation:
    """Tokenize the presentation of a request.

    Every text segment is tokenized on its own without special tokens and
    without a chat template, and the pieces are concatenated.

    Raises:
        PlanError: The prompt is blank or contains a vision placeholder,
            which would misalign the conditioner's vision inputs.
    """
    if not prompt.strip():
        raise PlanError("prompt", "must not be blank")
    start = tokenizer.convert_tokens_to_ids(VISION_START)
    end = tokenizer.convert_tokens_to_ids(VISION_END)
    placeholders = {
        start,
        end,
        tokenizer.convert_tokens_to_ids(IMAGE_PAD),
        tokenizer.convert_tokens_to_ids(VIDEO_PAD),
    }

    token_ids: list[int] = []
    tags: list[int] = []
    segments = presentation_segments(plan, prompt)
    for position, segment in enumerate(segments):
        if isinstance(segment, VisionSegment):
            pad = tokenizer.convert_tokens_to_ids(segment.pad)
            block = [start, *([pad] * segment.tokens), end]
            token_ids += block
            tags += [VIDEO_TAG] * len(block)
            continue
        ids = tokenizer.encode(segment.text, add_special_tokens=False)
        if position == len(segments) - 1 and placeholders.intersection(ids):
            raise PlanError(
                "prompt", "must not contain vision placeholder tokens"
            )
        token_ids += ids
        tags += [TEXT_TAG] * len(ids)
    return Presentation(token_ids=tuple(token_ids), tags=tuple(tags))


def _nearest_multiple(value: float, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _parse_aspect_ratio(value: str) -> tuple[int, int] | None:
    """Parse ``"W:H"`` into positive integers; ``None`` if malformed.

    Each side must be written in canonical decimal form, without sign or
    leading zeros.
    """
    parts = value.split(":")
    if len(parts) != 2 or not all(
        part.isascii() and part.isdigit() and str(int(part)) == part
        for part in parts
    ):
        return None
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _check_served(
    size: image.Config, canvases: Sequence[image.Config] | None
) -> None:
    """Reject a canvas outside the checkpoint's restricted set."""
    if canvases is not None and size not in canvases:
        raise PlanError(
            "target.aspect_ratio",
            f"resolves to {size.width}x{size.height}, which the checkpoint "
            "does not serve",
        )


def _check_request(
    task: Task,
    target: Target,
    conditions: Sequence[Condition],
    max_seconds: float,
    canvases: Sequence[image.Config] | None,
) -> None:
    """Apply the rules that need no media facts, in a fixed order."""
    if target.short_edge != CANVAS_SHORT_EDGE:
        raise PlanError(
            "target.short_edge",
            f"must be {CANVAS_SHORT_EDGE}, got {target.short_edge}",
        )

    # The aspect ratio: named ratios for t2va and ref2va, any ratio within
    # 1:4 to 4:1 for fl2va, whose `auto` follows the first keyframe.
    if target.aspect_ratio != "auto":
        ratio = _parse_aspect_ratio(target.aspect_ratio)
        if task is Task.FL2VA:
            if ratio is None or not (
                MIN_ASPECT_RATIO <= ratio[0] / ratio[1] <= MAX_ASPECT_RATIO
            ):
                raise PlanError(
                    "target.aspect_ratio",
                    f"must be auto or W:H within 1:4 to 4:1, got "
                    f"{target.aspect_ratio!r}",
                )
        elif ratio not in NAMED_ASPECT_RATIOS:
            named = ", ".join(f"{w}:{h}" for w, h in NAMED_ASPECT_RATIOS)
            raise PlanError(
                "target.aspect_ratio",
                f"must be auto or one of {named}, got {target.aspect_ratio!r}",
            )
    if not (task is Task.FL2VA and target.aspect_ratio == "auto"):
        _check_served(_target_canvas(task, target, conditions), canvases)

    _check_conditions(task, conditions)

    if target.duration_seconds is not None:
        try:
            frame_count(target.duration_seconds, max_seconds)
        except ValueError as error:
            raise PlanError("target.duration_seconds", str(error)) from None
    elif task is not Task.REF2VA or not any(
        condition.type is not ConditionType.IMAGE for condition in conditions
    ):
        raise PlanError(
            "target.duration_seconds",
            "is required unless a ref2va request has exactly one reference "
            "with audio",
        )


def _check_conditions(task: Task, conditions: Sequence[Condition]) -> None:
    """Check each condition's fields, then the task's condition counts."""
    if task is Task.T2VA:
        if conditions:
            raise PlanError("conditions", "t2va takes no conditions")
        return

    for index, condition in enumerate(conditions):
        field = f"conditions[{index}]"
        if condition.role is Role.KEYFRAME:
            if condition.type is not ConditionType.IMAGE:
                raise PlanError(field, "a keyframe must be an image")
            if condition.frame_index is None:
                raise PlanError(field, "a keyframe needs frame_index")
            if condition.frame_index not in (0, -1):
                raise PlanError(
                    field,
                    f"frame_index must be 0 or -1, got {condition.frame_index}",
                )
        else:
            if task is Task.FL2VA:
                raise PlanError(field, "fl2va takes keyframes only")
            if condition.frame_index is not None:
                raise PlanError(field, "only a keyframe takes frame_index")
        if condition.start_seconds is not None:
            if condition.role is Role.KEYFRAME or condition.type not in (
                ConditionType.VIDEO,
                ConditionType.VIDEO_AUDIO,
            ):
                raise PlanError(
                    field,
                    "start_time_seconds applies to video references only",
                )
            if (
                not math.isfinite(condition.start_seconds)
                or condition.start_seconds < 0
            ):
                raise PlanError(
                    field, "start_time_seconds must be finite and >= 0"
                )

    signature = tuple(
        condition.frame_index
        for condition in conditions
        if condition.role is Role.KEYFRAME
    )
    if (task is Task.FL2VA or signature) and signature not in (
        (0,),
        (-1,),
        (0, -1),
    ):
        raise PlanError(
            "conditions",
            f"keyframe frame_index values must be [0], [-1] or [0, -1], "
            f"got {list(signature)}",
        )
    if task is Task.REF2VA:
        references = [
            condition.type
            for condition in conditions
            if condition.role is Role.REFERENCE
        ]
        images = references.count(ConditionType.IMAGE)
        audios = references.count(ConditionType.AUDIO)
        videos = len(references) - images - audios
        for kind, count, limit in (
            ("image", images, MAX_IMAGE_REFERENCES),
            ("video", videos, MAX_VIDEO_REFERENCES),
            ("audio", audios, MAX_AUDIO_REFERENCES),
            ("reference", len(references), MAX_REFERENCES),
        ):
            if count > limit:
                raise PlanError(
                    "conditions",
                    f"at most {limit} {kind} references, got {count}",
                )
        if images + videos == 0:
            raise PlanError(
                "conditions",
                "ref2va needs at least one image or video reference",
            )


def _check_media(index: int, condition: Condition) -> None:
    """Check that the probed media fits the condition's type."""
    field = f"conditions[{index}]"
    expected: type[ImageFacts | VideoFacts | AudioFacts] = {
        ConditionType.IMAGE: ImageFacts,
        ConditionType.VIDEO: VideoFacts,
        ConditionType.VIDEO_AUDIO: VideoFacts,
        ConditionType.AUDIO: AudioFacts,
    }[condition.type]
    if not isinstance(condition.media, expected):
        raise PlanError(field, f"is not {condition.type.value} media")
    if (
        condition.type is ConditionType.VIDEO_AUDIO
        and isinstance(condition.media, VideoFacts)
        and condition.media.soundtrack is None
    ):
        raise PlanError(field, "a video_audio reference needs an audio track")


def _soundtrack(condition: Condition) -> AudioFacts | None:
    """The soundtrack a reference conditions on, if any."""
    if condition.role is Role.KEYFRAME:
        return None
    if isinstance(condition.media, AudioFacts):
        return condition.media
    if isinstance(condition.media, VideoFacts):
        return condition.media.soundtrack
    return None


def _resolve_frames(
    target: Target,
    soundtracks: Sequence[tuple[int, AudioFacts, float]],
    max_seconds: float,
) -> int:
    """The generated frame count, from the target or the one soundtrack."""
    if target.duration_seconds is not None:
        return frame_count(target.duration_seconds, max_seconds)
    if len(soundtracks) != 1:
        raise PlanError(
            "target.duration_seconds",
            f"is required unless exactly one reference has audio, got "
            f"{len(soundtracks)}",
        )
    index, track, start_seconds = soundtracks[0]
    start_sample = _start_offset(start_seconds, track.sample_rate)
    if start_sample >= track.samples:
        raise PlanError(
            f"conditions[{index}]",
            "start_time_seconds lies past the end of the audio track",
        )
    seconds = (track.samples - start_sample) / track.sample_rate
    try:
        return frame_count(seconds, max_seconds)
    except ValueError as error:
        raise PlanError(
            f"conditions[{index}]",
            f"the audio track sets the duration, but {error}",
        ) from None


def _target_canvas(
    task: Task, target: Target, conditions: Sequence[Condition]
) -> image.Config:
    """The generated canvas: from the ratio, or the first keyframe."""
    if target.aspect_ratio != "auto":
        ratio = _parse_aspect_ratio(target.aspect_ratio)
        assert ratio is not None
        return canvas(*ratio)
    if task is not Task.FL2VA:
        return canvas(*DEFAULT_ASPECT_RATIO)
    first = conditions[0]
    assert isinstance(first.media, ImageFacts)
    try:
        return canvas(first.media.width, first.media.height)
    except ValueError as error:
        raise PlanError(
            "conditions[0]",
            f"the first keyframe sets the canvas, but its {error}",
        ) from None


def _start_offset(seconds: float, rate: float) -> int:
    """Units of a ``rate`` timeline skipped by a start offset."""
    return math.floor(seconds * rate + 0.5)


def _plan_keyframe(
    index: int,
    condition: Condition,
    size: image.Config,
    order: int,
    task: Task,
    vision: VisionConfig,
) -> ConditionPlan:
    """Fit a keyframe on the canvas; fl2va keyframes enter the conditioner."""
    media = condition.media
    assert isinstance(media, ImageFacts)
    position = (
        FramePosition.FIRST
        if condition.frame_index == 0
        else FramePosition.LAST
    )
    # The first keyframe anchors the canvas and is stretched onto it; a
    # second one follows and is cover-cropped.
    crop = None if order == 0 else cover_crop(media.width, media.height, size)
    seen = None
    if task is Task.FL2VA:
        grid = vision.image_grid(size.height, size.width)
        seen = ImageVision(grid=grid, tokens=vision.block_tokens(grid))
    return ConditionPlan(
        index=index,
        type=condition.type,
        prepared=KeyframeFit(position=position, cover_crop=crop),
        vision=seen,
        video_rows=rows_per_frame(size),
        audio_rows=0,
    )


def _plan_reference(
    index: int, condition: Condition, num_frames: int, vision: VisionConfig
) -> ConditionPlan:
    """Size an image, video or audio reference."""
    field = f"conditions[{index}]"
    media = condition.media
    if isinstance(media, ImageFacts):
        try:
            resize = reference_image_size(media.width, media.height)
        except ValueError as error:
            raise PlanError(field, str(error)) from None
        grid = vision.image_grid(resize.height, resize.width)
        # One latent frame at the reference's own size.
        return ConditionPlan(
            index=index,
            type=condition.type,
            prepared=resize,
            vision=ImageVision(grid=grid, tokens=vision.block_tokens(grid)),
            video_rows=rows_per_frame(resize),
            audio_rows=0,
        )
    try:
        if isinstance(media, AudioFacts):
            clip = audio_clip(media, 0.0, num_frames)
            return ConditionPlan(
                index=index,
                type=condition.type,
                prepared=clip,
                vision=None,
                video_rows=0,
                audio_rows=clip.rows,
            )
        video, seen = video_reference(
            media, condition.start_seconds or 0.0, num_frames, vision
        )
    except ValueError as error:
        raise PlanError(field, str(error)) from None
    soundtrack = video.soundtrack
    return ConditionPlan(
        index=index,
        type=condition.type,
        prepared=video,
        vision=seen,
        video_rows=video.latent_frames * rows_per_frame(video.canvas),
        audio_rows=0 if soundtrack is None else soundtrack.rows,
    )
