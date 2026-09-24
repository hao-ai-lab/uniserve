"""Loads PIE-Bench source images and image-editing instructions.

A PIE-Bench mapping file is a JSON object whose keys, prefixed, become example
ids; each entry names a source ``image_path`` relative to the image root and
carries the edit text. Each selected entry becomes one image-editing
``Example`` whose source image is embedded as base64 PNG, which image-input
tasks send inline as a data URL.
"""

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
# Mapping file names checked directly under the root, in preference order.
_MAPPING_CANDIDATES = ("mapping_file.json", "mapping_file_ti2i_benchmark.json")
_BRACKETS = re.compile(r"[\[\]]")


class PieBenchDataset(Dataset):
    """Adapts resolvable PIE-Bench pairs to embedded PNG examples."""

    name: ClassVar[str] = "pie-bench"

    def load(self, tokenizer: Any | None = None) -> list[Example]:
        """Select seeded entries that contain both an image and instruction.

        Entries are shuffled with a private generator seeded by the load seed
        before image files and instructions are checked, so an entry whose
        image cannot be resolved or whose instruction is empty is skipped
        and the next shuffled entry takes its place. Errors from reading the
        mapping, downloading the snapshot, and decoding images propagate.

        Raises:
            FileNotFoundError: If no mapping file can be found.
        """
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
            image_file = _resolve_image_file(
                images_root, str(value["image_path"])
            )
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
    """Resolve the mapping file and image root from local or hub storage.

    A ``dataset_path`` naming a file is the mapping itself, with its parent
    directory as the image root; a directory is searched for a mapping and
    serves as the image root. Without a path, the dataset repository is
    downloaded from the hub at its default revision; ``dataset_revision`` is
    not applied.

    Returns:
        The mapping file path and the image root directory.

    Raises:
        FileNotFoundError: If a directory or downloaded snapshot contains no
            mapping file.
    """
    if dataset_path:
        base = Path(dataset_path)
        if base.is_file():
            return base, base.parent
        mapping = _find_mapping(base)
        if mapping is None:
            raise FileNotFoundError(
                f"no PIE-Bench mapping file found under {base}"
            )
        return mapping, base

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(repo_id=PIE_BENCH_HF_REPO, repo_type="dataset")
    )
    mapping = _find_mapping(snapshot)
    if mapping is None:
        raise FileNotFoundError(
            f"no PIE-Bench mapping file found in snapshot {snapshot}"
        )
    return mapping, snapshot


def _find_mapping(root: Path) -> Path | None:
    """Find the preferred PIE-Bench mapping file beneath a root.

    Known names directly under ``root`` win in ``_MAPPING_CANDIDATES`` order;
    otherwise the lexicographically first ``mapping_file*.json`` anywhere
    below ``root`` is used.
    """
    for name in _MAPPING_CANDIDATES:
        direct = root / name
        if direct.is_file():
            return direct
    candidates = sorted(root.rglob("mapping_file*.json"))
    return candidates[0] if candidates else None


def _resolve_image_file(images_root: Path, rel_path: str) -> Path | None:
    """Resolve an image path across the supported snapshot layouts.

    The relative path is tried under ``images_root`` and its parent, each
    with and without an ``annotation_images`` directory. As a last resort,
    any path below ``images_root`` with the same base name matches; the
    first match in ``rglob`` order is used.

    Returns:
        The resolved path, or ``None`` when no candidate exists.
    """
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
    """Extract and normalize the usable edit instruction for an entry.

    A non-empty ``editing_instruction`` wins and is used without bracket
    removal. Otherwise ``editing_prompt``, or failing that
    ``target_prompt``, is used with its square bracket characters removed.

    Returns:
        The stripped instruction, which is empty when none of the fields is
        set or the chosen one has no text left after normalization.
    """
    instruction = entry.get("editing_instruction")
    if instruction:
        return str(instruction).strip()
    target = entry.get("editing_prompt") or entry.get("target_prompt")
    if target:
        return _BRACKETS.sub("", str(target)).strip()
    return ""


def _png_b64(path: Path) -> str:
    """Normalize an input image to RGB PNG and return base64 text.

    Every source format is re-encoded as PNG, matching the ``image/png``
    MIME type the examples declare. Conversion to RGB drops any alpha
    channel.
    """
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
