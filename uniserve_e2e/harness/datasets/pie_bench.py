"""PIE-Bench prompt+image loader for image-to-image (editing) benchmarking.

PIE-Bench (Prompt-driven Image Editing Benchmark, ~700 edit cases) ships an
``annotation_images/`` tree plus a ``mapping_file.json`` keyed by image id with
``image_path``, ``original_prompt``/``source_prompt``,
``editing_prompt``/``target_prompt``, and ``editing_instruction``.

For an i2i speed benchmark we send the *source* image (base64 PNG) plus an edit
instruction. ``dataset_path`` should point at a local PIE-Bench ``data`` directory
(or directly at the mapping JSON); otherwise an HF snapshot is downloaded and the
mapping file + image dir are auto-discovered within it.
"""
from __future__ import annotations

import base64
import io
import json
import random
import re
from pathlib import Path
from typing import Any

PIE_BENCH_HF_REPO = "UB-CVML-Group/PIE_Bench_pp"
_MAPPING_CANDIDATES = ("mapping_file.json", "mapping_file_ti2i_benchmark.json")
_BRACKETS = re.compile(r"[\[\]]")


def load_pie_bench(
    dataset_path: str | None,
    num_requests: int,
    *,
    seed: int = 42,
    hf_repo: str = PIE_BENCH_HF_REPO,
) -> list[dict[str, Any]]:
    mapping_path, images_root = _resolve_sources(dataset_path, hf_repo)
    with open(mapping_path, encoding="utf-8") as handle:
        mapping = json.load(handle)

    entries = [
        (key, value)
        for key, value in mapping.items()
        if isinstance(value, dict) and value.get("image_path")
    ]
    random.Random(seed).shuffle(entries)

    rows: list[dict[str, Any]] = []
    for key, value in entries:
        if len(rows) >= num_requests:
            break
        image_file = _resolve_image_file(images_root, str(value["image_path"]))
        if image_file is None:
            continue
        instruction = _edit_instruction(value)
        if not instruction:
            continue
        rows.append(
            {
                "id": f"pie-{key}",
                "task": "i2i",
                "prompt": instruction,
                "input_image_b64": _png_b64(image_file),
                "source_prompt": value.get("original_prompt") or value.get("source_prompt"),
                "target_prompt": value.get("editing_prompt") or value.get("target_prompt"),
                "editing_type_id": value.get("editing_type_id"),
            }
        )
    if not rows:
        raise ValueError(
            f"PIE-Bench source at {mapping_path} yielded no usable rows "
            "(could not resolve any source images)"
        )
    return rows


def _resolve_sources(dataset_path: str | None, hf_repo: str) -> tuple[Path, Path]:
    if dataset_path:
        base = Path(dataset_path)
        if base.is_file():
            return base, base.parent
        mapping = _find_mapping(base)
        if mapping is None:
            raise FileNotFoundError(
                f"no PIE-Bench mapping file found under {base} "
                f"(looked for {_MAPPING_CANDIDATES})"
            )
        return mapping, base
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=hf_repo, repo_type="dataset"))
    mapping = _find_mapping(snapshot)
    if mapping is None:
        raise FileNotFoundError(
            f"no PIE-Bench mapping file found in HF snapshot {snapshot}; "
            "pass --dataset-path to a local PIE-Bench data dir"
        )
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
    # Last resort: match by filename anywhere under the root.
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
