"""Recognize MiniMax-H3 checkpoints and the DiT partitions they hold.

The base release (``model_index.json`` naming ``MiniMaxH3ModularPipeline``)
holds every component and two DiT partitions: ``transformer`` serves t2va
and fl2va, and ``transformer_ref`` serves ref2va. FastH3 is its fast variant:
a FastVideo student of one partition, whose ``fastvideo_inference.json``
states the student's task, schedule and sparse attention. The task selects
the partition the student replaces. A FastH3 export that omits a component
directory reads it from the base revision its contract pins in
``base_model_revision``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

# The DiT partition of the base release that serves each task.
TASK_PARTITIONS: Mapping[str, str] = MappingProxyType(
    {"t2va": "transformer", "fl2va": "transformer", "ref2va": "transformer_ref"}
)

# FastVideo states a student's task as ``task`` (``t2av`` for
# text-to-video-and-audio) or as ``model_type``.
_FASTVIDEO_TASKS = MappingProxyType({"t2av": "t2va", "ref2va": "ref2va"})


@dataclass(frozen=True, slots=True)
class Layout:
    """One recognized checkpoint.

    Attributes:
        partitions: Each DiT partition the root holds, mapped to the tasks it
            serves.
        contract: A FastH3 export's ``fastvideo_inference.json``, or None for
            the base release.
    """

    partitions: Mapping[str, tuple[str, ...]]
    contract: Mapping[str, Any] | None


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return value


def detect(root: Path) -> Layout:
    """Classify ``root`` and list the DiT partitions it holds.

    Raises:
        FileNotFoundError: The root holds neither a FastH3 contract nor a
            pipeline index.
        ValueError: The contract names a task no partition serves, or the
            index is not a MiniMax-H3 release holding a DiT.
    """
    contract_path = root / "fastvideo_inference.json"
    if contract_path.is_file():
        contract = _json(contract_path)
        stated = contract.get("task", contract.get("model_type"))
        task = _FASTVIDEO_TASKS.get(stated) if isinstance(stated, str) else None
        if task is None:
            raise ValueError(
                "unsupported FastH3 checkpoint: its task must be one of "
                f"{sorted(_FASTVIDEO_TASKS)}, got {stated!r}"
            )
        return Layout(
            MappingProxyType({TASK_PARTITIONS[task]: (task,)}),
            MappingProxyType(contract),
        )

    index_path = root / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"{root} holds neither fastvideo_inference.json nor "
            "model_index.json"
        )
    index = _json(index_path)
    if index.get("_class_name") != "MiniMaxH3ModularPipeline":
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: model_index.json must name "
            "MiniMaxH3ModularPipeline"
        )
    partitions = {
        partition: tuple(
            task
            for task, owner in TASK_PARTITIONS.items()
            if owner == partition
        )
        for partition in dict.fromkeys(TASK_PARTITIONS.values())
        if partition in index and (root / partition / "config.json").is_file()
    }
    if not partitions:
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: the release holds neither "
            "transformer nor transformer_ref"
        )
    return Layout(MappingProxyType(partitions), None)


def base_checkpoint(root: Path) -> tuple[str, str] | None:
    """Return the ``(repository, revision)`` a FastH3 export pins, if any.

    The export states it as ``hf://<repository>@<revision>``; directories the
    export omits are read from that revision. The base release and an export
    that pins nothing return ``None``.

    Raises:
        FileNotFoundError: As ``detect``.
        ValueError: As ``detect``, or a pin of another form.
    """
    contract = detect(root).contract
    pinned = None if contract is None else contract.get("base_model_revision")
    if pinned is None:
        return None
    repository, separator, revision = (
        pinned.removeprefix("hf://").rpartition("@")
        if isinstance(pinned, str) and pinned.startswith("hf://")
        else ("", "", "")
    )
    if not separator or not repository or not revision:
        raise ValueError(
            "FastH3 base_model_revision must be hf://<repository>@<revision>, "
            f"got {pinned!r}"
        )
    return repository, revision
