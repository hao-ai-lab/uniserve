"""PIE-Bench source-image and edit-prompt loader."""

from __future__ import annotations

import base64
import io
import json
import random
import re
from pathlib import Path
from typing import Any, ClassVar

from ..types import Example
from .base import Dataset

PIE_BENCH_HF_REPO = "UB-CVML-Group/PIE_Bench_pp"
_MAPPING_CANDIDATES = ("mapping_file.json", "mapping_file_ti2i_benchmark.json")
_BRACKETS = re.compile(r"[\[\]]")


class PieBenchDataset(Dataset):
    name: ClassVar[str] = "pie-bench"

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        point = self.point
        mapping_path, images_root = _resolve_sources(point.dataset_path)
        with open(mapping_path, encoding="utf-8") as handle:
            mapping = json.load(handle)
        entries = [
            (key, value)
            for key, value in mapping.items()
            if isinstance(value, dict) and value.get("image_path")
        ]
        random.Random(point.load.seed).shuffle(entries)
        rows: list[Example] = []
        for key, value in entries:
            if len(rows) >= point.load.num_prompts:
                break
            image_file = _resolve_image_file(images_root, str(value["image_path"]))
            if image_file is None:
                continue
            instruction = _edit_instruction(value)
            if not instruction:
                continue
            rows.append(
                Example(
                    id=f"pie-{key}",
                    prompt=instruction,
                    input_image_b64=_png_b64(image_file),
                    input_image_mime="image/png",
                )
            )
        return rows


def _resolve_sources(dataset_path: str | None) -> tuple[Path, Path]:
    if dataset_path:
        base = Path(dataset_path)
        if base.is_file():
            return base, base.parent
        mapping = _find_mapping(base)
        if mapping is None:
            raise FileNotFoundError(f"no PIE-Bench mapping file found under {base}")
        return mapping, base
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=PIE_BENCH_HF_REPO, repo_type="dataset"))
    mapping = _find_mapping(snapshot)
    if mapping is None:
        raise FileNotFoundError(f"no PIE-Bench mapping file found in snapshot {snapshot}")
    return mapping, snapshot


def _find_mapping(root: Path) -> Path | None:
    for name in _MAPPING_CANDIDATES:
        direct = root / name
        if direct.is_file():
            return direct
    candidates = sorted(root.rglob("mapping_file*.json"))
    return candidates[0] if candidates else None


def _resolve_image_file(images_root: Path, rel_path: str) -> Path | None:
    candidates = [
        images_root / rel_path,
        images_root / "annotation_images" / rel_path,
        images_root.parent / rel_path,
        images_root.parent / "annotation_images" / rel_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(images_root.rglob(Path(rel_path).name))
    return matches[0] if matches else None


def _edit_instruction(entry: dict[str, Any]) -> str:
    instruction = entry.get("editing_instruction")
    if instruction:
        return str(instruction).strip()
    target = entry.get("editing_prompt") or entry.get("target_prompt")
    if target:
        return _BRACKETS.sub("", str(target)).strip()
    return ""


def _png_b64(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
