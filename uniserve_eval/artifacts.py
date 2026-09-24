"""Writes durable benchmark metadata and validated media artifacts.

`ArtifactWriter` writes into one benchmark point's result directory: JSON and
JSON Lines records at its top level and generated media under `samples/`.
`pipeline.run.run_point` is its caller and decides which records exist and
when each is rewritten.

Every file this writer produces is written to a hidden temporary file in its
destination directory, flushed and fsynced, then moved into place with
`os.replace`. A reader therefore sees either the previous or the new complete
version of such a file, never a partial write. `run_point` writes
`summary.md` directly, so that file does not have this guarantee.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .transport.images import inspect_image_bytes
from .transport.video import inspect_video_bytes
from .types import DecodedImage, DecodedVideo


class ArtifactWriter:
    """Writes one benchmark result bundle with atomic metadata updates.

    Media samples are content-addressed: `DecodedImage.sample_filename` and
    `DecodedVideo.sample_filename` are the SHA-256 of the encoded bytes plus an
    extension, so identical outputs from different requests share one file.
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Create the result and media-sample directories."""
        self.output_dir = Path(output_dir)
        self.samples_dir = self.output_dir / "samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        """Serialize a value as atomically replaced, formatted JSON.

        `name` is relative to the output directory. Keys are sorted, so equal
        payloads produce identical files.
        """
        path = self.output_dir / name
        temporary_path: Path | None = None
        try:
            # The temporary file shares the destination directory so that
            # `os.replace` is a same-filesystem rename, and its bytes reach
            # disk before the rename publishes them.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            # After a successful rename the temporary name no longer exists
            # and this is a no-op; on failure it removes the partial file.
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def write_jsonl(self, name: str, payloads: list[Any]) -> Path:
        """Serialize values as an atomically replaced JSON Lines file."""
        path = self.output_dir / name
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                for payload in payloads:
                    json.dump(_jsonable(payload), handle, sort_keys=True)
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def write_image_sample(self, image: DecodedImage) -> Path:
        """Validate and store a content-addressed image sample.

        The bytes are decoded again and must reproduce the metadata recorded
        in `image`, so a persisted sample always matches the request record
        that references it.

        Returns:
            The sample path under `samples/`. An existing file with identical
            bytes is reused without rewriting.

        Raises:
            ImageOutputError: If `inspect_image_bytes` rejects the bytes or
                the declared MIME type.
            ValueError: If the decoded metadata differs from `image`, or an
                existing sample of the same name holds different bytes.
        """
        inspected = inspect_image_bytes(image.data, declared_mime=image.mime)
        if inspected.metadata_dict() != image.metadata_dict():
            raise ValueError(
                "generated image metadata does not match its response bytes"
            )

        path = self.samples_dir / image.sample_filename
        if path.exists():
            if path.read_bytes() != image.data:
                raise ValueError("content-addressed image sample collision")
            return path

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(image.data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def write_video_sample(self, video: DecodedVideo) -> Path:
        """Validate and store a content-addressed MP4 sample.

        Validation, reuse, and error behavior follow `write_image_sample`,
        with `VideoOutputError` raised when `inspect_video_bytes` rejects the
        bytes or the declared MIME type.
        """
        inspected = inspect_video_bytes(video.data, declared_mime=video.mime)
        if inspected.metadata_dict() != video.metadata_dict():
            raise ValueError(
                "generated video metadata does not match its response bytes"
            )

        path = self.samples_dir / video.sample_filename
        if path.exists():
            if path.read_bytes() != video.data:
                raise ValueError("content-addressed video sample collision")
            return path

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(video.data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path


def _jsonable(payload: Any) -> Any:
    """Convert supported structured values into JSON-compatible values.

    Dataclass instances become `dataclasses.asdict` output, which is not
    converted further. `Path` values become strings, and dicts and lists are
    converted element by element. Any other value is returned unchanged for
    `json.dump` to encode or reject.
    """
    if is_dataclass(payload) and not isinstance(payload, type):
        return asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_jsonable(value) for value in payload]
    return payload
