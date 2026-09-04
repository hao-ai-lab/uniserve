"""Writes durable benchmark metadata and validated media artifacts."""

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
    """Writes one benchmark result bundle with atomic metadata updates."""

    def __init__(self, output_dir: str | Path) -> None:
        """Create the result and media-sample directories."""

        self.output_dir = Path(output_dir)
        self.samples_dir = self.output_dir / "samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        """Serialize a value as atomically replaced, formatted JSON."""

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
                json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
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
        """Validate and store a content-addressed image sample."""

        inspected = inspect_image_bytes(image.data, declared_mime=image.mime)
        if inspected.metadata_dict() != image.metadata_dict():
            raise ValueError("generated image metadata does not match its response bytes")
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
        """Validate and store a content-addressed MP4 sample."""

        inspected = inspect_video_bytes(video.data, declared_mime=video.mime)
        if inspected.metadata_dict() != video.metadata_dict():
            raise ValueError("generated video metadata does not match its response bytes")
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
    """Convert supported structured values into JSON-compatible values."""

    if is_dataclass(payload) and not isinstance(payload, type):
        return asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_jsonable(value) for value in payload]
    return payload
