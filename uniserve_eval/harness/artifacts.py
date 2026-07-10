from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class ArtifactWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.samples_dir = self.output_dir / "samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
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

    def append_jsonl(self, name: str, payload: Any) -> Path:
        path = self.output_dir / name
        with path.open("a", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, sort_keys=True)
            handle.write("\n")
        return path

    def write_jsonl(self, name: str, payloads: list[Any]) -> Path:
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

    def write_sample(self, request_id: str, suffix: str, data: bytes | str) -> Path:
        path = self.samples_dir / f"{request_id}{suffix}"
        if isinstance(data, bytes):
            with path.open("wb") as handle:
                handle.write(data)
        else:
            with path.open("w", encoding="utf-8") as handle:
                handle.write(data)
        return path

    def write_png_sample(self, request_id: str, pixels_png_b64: str) -> Path:
        return self.write_sample(request_id, ".png", base64.b64decode(pixels_png_b64))


def _jsonable(payload: Any) -> Any:
    if is_dataclass(payload) and not isinstance(payload, type):
        return asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_jsonable(value) for value in payload]
    return payload
