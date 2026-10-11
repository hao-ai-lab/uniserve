"""A video request's task, presentation tags and conditions.

The records mirror ``VideoAdmission`` in ``uniserve_worker_ipc`` and the
``uniserve_core`` condition types it carries, whose ``validate`` methods are
the authoritative checks. The server plans every condition before admission:
the shared-memory object holding its fetched bytes (``MediaLocator``), the
media the host media reader decodes it into (``ImageFit``, ``VideoClip``,
``AudioClip``), what the conditioner reads of it (``ConditionVision``) and
the denoiser rows its latent encoding yields. A worker sizes and checks the
condition products it reads and writes against these values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# ``VideoCondition``'s ``video`` field shadows the module name in its body.
from uniserve.media import image
from uniserve.media import video as media_video
from uniserve.model import ConditionRole


class VideoTask(StrEnum):
    """A video-and-audio generation task, by its request name."""

    T2VA = "t2va"
    FL2VA = "fl2va"
    REF2VA = "ref2va"


#: The presentation tag of a vision token or vision marker; every other
#: token is text.
VISION_TAG = 0


@dataclass(frozen=True, slots=True)
class MediaLocator:
    """A POSIX shared-memory object holding a condition's fetched bytes.

    The engine publishes it on the head host and keeps it until the request
    retires; a reader opens it by name and never unlinks it.
    """

    # Object name without its leading ``/``.
    name: str
    # Published byte count.
    bytes: int


@dataclass(frozen=True, slots=True)
class ImageFit:
    """A still image resized (LANCZOS) to ``resized``, then cropped.

    The kept window is ``size`` at (``left``, ``top``) of the resized image.
    """

    resized: image.Config
    left: int
    top: int
    size: image.Config


@dataclass(frozen=True, slots=True)
class VideoClip:
    """A reference video's frames on the output-rate timeline.

    The first ``start_frame`` frames are skipped and the next ``frames``
    kept, scaled onto ``canvas``; the video encoder encodes the leading
    ``vae_frames`` of them.
    """

    canvas: image.Config
    start_frame: int
    frames: int
    vae_frames: int


@dataclass(frozen=True, slots=True)
class AudioClip:
    """An audio track's kept native samples and their count at model rate.

    The first ``start_sample`` native samples are skipped and the next
    ``source_samples`` kept, then resampled once to ``samples`` samples.
    """

    sample_rate: int
    start_sample: int
    source_samples: int
    samples: int


@dataclass(frozen=True, slots=True)
class ConditionVision:
    """What the conditioner reads of a visual condition.

    ``grid`` is the ``(time, height, width)`` patch grid the condition's
    frames are resized to, ``tokens`` its vision placeholders in the
    presentation, and ``frame_indices`` the kept frames of a video the
    conditioner samples, empty for an image.
    """

    grid: tuple[int, int, int]
    tokens: int
    frame_indices: tuple[int, ...] = ()

    @property
    def patches(self) -> int:
        """Patch rows of the condition's processor input."""
        time, height, width = self.grid
        return time * height * width


@dataclass(frozen=True, slots=True)
class VideoCondition:
    """One condition of a video request.

    Its media is ``image`` alone (a keyframe or image reference), ``video``
    with its soundtrack in ``audio`` when the request conditions on one, or
    ``audio`` alone (an audio reference). ``latent_units`` lists the
    denoiser rows each temporal unit of its pixels encodes to, and
    ``audio_rows`` the rows of its audio track.
    """

    role: ConditionRole
    source: MediaLocator
    image: ImageFit | None
    video: VideoClip | None
    audio: AudioClip | None
    vision: ConditionVision | None
    latent_units: tuple[int, ...]
    audio_rows: int

    def __post_init__(self) -> None:
        # The transport states the role by its wire spelling.
        object.__setattr__(self, "role", ConditionRole(self.role))

    @property
    def pixels(self) -> media_video.Config | None:
        """The frames and raster the video encoder encodes, if any."""
        if self.image is not None:
            return media_video.Config(1, self.image.size)
        if self.video is not None:
            return media_video.Config(self.video.vae_frames, self.video.canvas)
        return None

    @property
    def pixel_bytes(self) -> int:
        """Bytes of the RGB24 pixels the video encoder encodes."""
        pixels = self.pixels
        if pixels is None:
            return 0
        return pixels.num_frames * pixels.frame.height * pixels.frame.width * 3

    @property
    def video_rows(self) -> int:
        """Denoiser video rows of the condition."""
        return sum(self.latent_units)


@dataclass(frozen=True, slots=True)
class VideoAdmission:
    """A video request's task, presentation tags and conditions."""

    task: VideoTask
    # The denoiser's AdaLN tag of each prompt token: ``VISION_TAG`` for a
    # vision token or vision marker, 1 for text.
    text_tags: tuple[int, ...]
    conditions: tuple[VideoCondition, ...] = ()

    def __post_init__(self) -> None:
        # The transport states the task by its wire spelling.
        object.__setattr__(self, "task", VideoTask(self.task))

    def vision_spans(self) -> tuple[tuple[int, int], ...]:
        """The presentation's vision blocks as ``(start, stop)`` token ranges.

        A block is a maximal run of ``VISION_TAG`` tokens, markers included.
        """
        spans = []
        start = None
        for index, tag in enumerate((*self.text_tags, None)):
            if tag == VISION_TAG and start is None:
                start = index
            elif tag != VISION_TAG and start is not None:
                spans.append((start, index))
                start = None
        return tuple(spans)


__all__ = [
    "VISION_TAG",
    "AudioClip",
    "ConditionVision",
    "ImageFit",
    "MediaLocator",
    "VideoAdmission",
    "VideoClip",
    "VideoCondition",
    "VideoTask",
]
