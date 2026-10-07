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
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.validation import _map, _seq, _str, _uint, _uints


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

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.bytes < 1:
            raise invalid_descriptor("a condition's media locator is invalid")


@dataclass(frozen=True, slots=True)
class ImageFit:
    """A still image resized (LANCZOS) to ``resized``, then cropped.

    The kept window is ``size`` at (``left``, ``top``) of the resized image.
    """

    resized: image.Config
    left: int
    top: int
    size: image.Config

    def __post_init__(self) -> None:
        if (
            self.left < 0
            or self.top < 0
            or self.left + self.size.width > self.resized.width
            or self.top + self.size.height > self.resized.height
        ):
            raise invalid_descriptor(
                "an image fit keeps a window outside its resized image"
            )


@dataclass(frozen=True, slots=True)
class VideoClip:
    """A reference video's frames on the 24 fps timeline.

    The first ``start_frame`` frames are skipped and the next ``frames``
    kept, scaled onto ``canvas``; the video encoder encodes the leading
    ``vae_frames`` of them.
    """

    canvas: image.Config
    start_frame: int
    frames: int
    vae_frames: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or not 1 <= self.vae_frames <= self.frames:
            raise invalid_descriptor(
                "a video clip encodes frames it does not keep"
            )


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

    def __post_init__(self) -> None:
        if (
            self.sample_rate < 1
            or self.start_sample < 0
            or self.source_samples < 1
            or self.samples < 1
        ):
            raise invalid_descriptor("an audio track keeps no samples")


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

    def __post_init__(self) -> None:
        if len(self.grid) != 3 or min(self.grid) < 1 or self.tokens < 1:
            raise invalid_descriptor("a conditioner view covers no patches")

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
        object.__setattr__(self, "role", ConditionRole(self.role))
        if (self.image is not None) + (self.video is not None) > 1 or (
            self.image is not None and self.audio is not None
        ):
            raise invalid_descriptor(
                "a video condition carries an invalid media combination"
            )
        if self.image is None and self.video is None and self.audio is None:
            raise invalid_descriptor("a video condition carries no media")
        if self.role is not ConditionRole.REFERENCE and self.image is None:
            raise invalid_descriptor("a keyframe is one still image")
        if (self.pixels is None) != (not self.latent_units) or any(
            rows < 1 for rows in self.latent_units
        ):
            raise invalid_descriptor(
                "a condition's latent units disagree with its pixels"
            )
        if (self.audio is None) != (self.audio_rows == 0):
            raise invalid_descriptor(
                "a condition's audio rows disagree with its audio track"
            )
        if self.vision is not None:
            if self.pixels is None:
                raise invalid_descriptor(
                    "the conditioner reads no audio reference"
                )
            frames = () if self.video is None else self.vision.frame_indices
            if self.video is None and (
                self.vision.frame_indices or self.vision.grid[0] != 1
            ):
                raise invalid_descriptor(
                    "an image's conditioner view samples one frame"
                )
            if self.video is not None and (
                not frames
                or any(b <= a for a, b in zip(frames, frames[1:]))
                or frames[-1] >= self.video.frames
            ):
                raise invalid_descriptor(
                    "a video's conditioner view samples frames it does not keep"
                )

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
        object.__setattr__(self, "task", VideoTask(self.task))
        if (self.task is VideoTask.T2VA) != (not self.conditions):
            raise invalid_descriptor(
                "only a conditioned video task carries conditions"
            )

    @classmethod
    def from_mapping(
        cls, value: object, where: str = "video admission"
    ) -> VideoAdmission:
        """Parse a video admission from its mapping form."""
        data = _map(value, where)
        return cls(
            task=VideoTask(_str(data.get("task"), f"{where}.task")),
            text_tags=_uints(data.get("text_tags", ()), f"{where}.text_tags"),
            conditions=tuple(
                _condition_from_mapping(item, f"{where}.conditions[{index}]")
                for index, item in enumerate(
                    _seq(data.get("conditions", ()), f"{where}.conditions")
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the admission; ``from_mapping`` reads it back."""
        return {
            "task": self.task.value,
            "text_tags": list(self.text_tags),
            "conditions": [
                _condition_to_mapping(condition)
                for condition in self.conditions
            ],
        }

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


def _raster(value: object, where: str) -> image.Config:
    data = _map(value, where)
    return image.Config(
        _uint(data.get("height"), f"{where}.height"),
        _uint(data.get("width"), f"{where}.width"),
    )


def _raster_mapping(value: image.Config) -> dict[str, int]:
    return {"height": value.height, "width": value.width}


def _audio_from_mapping(value: object, where: str) -> AudioClip:
    data = _map(value, where)
    return AudioClip(
        *(
            _uint(data.get(name), f"{where}.{name}")
            for name in (
                "sample_rate",
                "start_sample",
                "source_samples",
                "samples",
            )
        )
    )


def _condition_from_mapping(value: object, where: str) -> VideoCondition:
    """Parse one condition; absent media fields are None."""
    data = _map(value, where)
    source = _map(data.get("source"), f"{where}.source")
    fit = data.get("image")
    clip = data.get("video")
    track = data.get("audio")
    vision = data.get("vision")
    return VideoCondition(
        role=ConditionRole(_str(data.get("role"), f"{where}.role")),
        source=MediaLocator(
            _str(source.get("name"), f"{where}.source.name"),
            _uint(source.get("bytes"), f"{where}.source.bytes"),
        ),
        image=None
        if fit is None
        else ImageFit(
            _raster(_map(fit, f"{where}.image").get("resized"), where),
            _uint(fit.get("left"), f"{where}.image.left"),
            _uint(fit.get("top"), f"{where}.image.top"),
            _raster(fit.get("size"), f"{where}.image.size"),
        ),
        video=None
        if clip is None
        else VideoClip(
            _raster(_map(clip, f"{where}.video").get("canvas"), where),
            _uint(clip.get("start_frame"), f"{where}.video.start_frame"),
            _uint(clip.get("frames"), f"{where}.video.frames"),
            _uint(clip.get("vae_frames"), f"{where}.video.vae_frames"),
        ),
        audio=None
        if track is None
        else _audio_from_mapping(track, f"{where}.audio"),
        vision=None
        if vision is None
        else ConditionVision(
            grid=tuple(  # type: ignore[arg-type]
                _uints(_map(vision, where).get("grid"), f"{where}.grid")
            ),
            tokens=_uint(vision.get("tokens"), f"{where}.vision.tokens"),
            frame_indices=_uints(
                vision.get("frame_indices", ()), f"{where}.frame_indices"
            ),
        ),
        latent_units=_uints(
            data.get("latent_units", ()), f"{where}.latent_units"
        ),
        audio_rows=_uint(data.get("audio_rows", 0), f"{where}.audio_rows"),
    )


def _condition_to_mapping(condition: VideoCondition) -> dict[str, object]:
    fit, clip, track, vision = (
        condition.image,
        condition.video,
        condition.audio,
        condition.vision,
    )
    return {
        "role": condition.role.value,
        "source": {
            "name": condition.source.name,
            "bytes": condition.source.bytes,
        },
        "image": None
        if fit is None
        else {
            "resized": _raster_mapping(fit.resized),
            "left": fit.left,
            "top": fit.top,
            "size": _raster_mapping(fit.size),
        },
        "video": None
        if clip is None
        else {
            "canvas": _raster_mapping(clip.canvas),
            "start_frame": clip.start_frame,
            "frames": clip.frames,
            "vae_frames": clip.vae_frames,
        },
        "audio": None
        if track is None
        else {
            "sample_rate": track.sample_rate,
            "start_sample": track.start_sample,
            "source_samples": track.source_samples,
            "samples": track.samples,
        },
        "vision": None
        if vision is None
        else {
            "grid": list(vision.grid),
            "tokens": vision.tokens,
            "frame_indices": list(vision.frame_indices),
        },
        "latent_units": list(condition.latent_units),
        "audio_rows": condition.audio_rows,
    }


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
