"""Recognize the three MiniMax-H3 checkpoint layouts and their denoisers.

A MiniMax-H3 checkpoint is one of:

* a FastVideo full export (FastH3 V1/V2 and their NVFP4 exports): every
  component lives under the root, and ``fastvideo_inference.json`` carries
  a DMD contract (``dmd_denoising_steps``) for its ``transformer``;
* the MiniMax-H3 diffusers root (``model_index.json`` naming
  ``MiniMaxH3ModularPipeline``, no inference contract): every component
  lives under the root and both DiTs, ``transformer`` and
  ``transformer_ref``, use the full-step uniform schedule;
* a FastVideo component export (FastH3-OmniRef): the root holds one DiT and
  its schedulers, ``fastvideo_inference.json`` carries a PDD contract
  (``pdd_steps``) and ``base_model_revision`` pins the diffusers root that
  supplies every other component.

``Layout`` names the denoising components each layout holds: the
``transformer`` partition is the ``denoiser`` component and serves t2va and
fl2va, the ``transformer_ref`` partition is the ``reference_denoiser``
component and serves ref2va.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

# Inference contract schema every FastVideo MiniMax-H3 export publishes.
INFERENCE_SCHEMA = "fasth3-inference-contract-v1"

# Checkpoint subdirectory of each denoising component.
DENOISER_DIRECTORIES: Mapping[str, str] = MappingProxyType(
    {"denoiser": "transformer", "reference_denoiser": "transformer_ref"}
)

# Tasks each denoising component serves; the DiT partition fixes them.
DENOISER_TASKS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"denoiser": ("t2va", "fl2va"), "reference_denoiser": ("ref2va",)}
)


class Kind(enum.Enum):
    """The checkpoint layouts this package loads."""

    FASTVIDEO_EXPORT = "fastvideo_export"
    DIFFUSERS_ROOT = "diffusers_root"
    COMPONENT_EXPORT = "component_export"


@dataclass(frozen=True, slots=True)
class BaseRevision:
    """A Hugging Face repository pinned at one commit."""

    repository: str
    revision: str

    @classmethod
    def parse(cls, value: object) -> BaseRevision:
        """Parse ``hf://<repository>@<revision>``.

        Raises:
            ValueError: ``value`` is not of that form.
        """
        prefix = "hf://"
        if not isinstance(value, str) or not value.startswith(prefix):
            raise ValueError(
                "MiniMax-H3 base_model_revision must be "
                f"hf://<repository>@<revision>, got {value!r}"
            )
        repository, separator, revision = value[len(prefix) :].rpartition("@")
        if (
            not separator
            or repository.count("/") != 1
            or not all(repository.split("/"))
            or not revision
        ):
            raise ValueError(
                "MiniMax-H3 base_model_revision must be "
                f"hf://<repository>@<revision>, got {value!r}"
            )
        return cls(repository, revision)


@dataclass(frozen=True, slots=True)
class Layout:
    """One recognized checkpoint.

    Attributes:
        kind: Which of the three layouts the root holds.
        denoisers: Denoising components the root holds, keyed by component
            name, valued by their checkpoint subdirectory.
        contract: The FastVideo inference contract, or None for the
            diffusers root.
        base: The diffusers root a component export draws its other
            components from, or None when the root holds every component.
    """

    kind: Kind
    denoisers: Mapping[str, str]
    contract: Mapping[str, Any] | None
    base: BaseRevision | None


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return value


def detect(root: Path) -> Layout:
    """Classify ``root`` and list the denoising components it holds.

    Raises:
        FileNotFoundError: The root holds neither an inference contract nor a
            diffusers pipeline index.
        ValueError: The contract or index is not a MiniMax-H3 layout this
            package serves, naming the offending field.
    """
    contract_path = root / "fastvideo_inference.json"
    if contract_path.is_file():
        contract = _json(contract_path)
        if contract.get("schema_version") != INFERENCE_SCHEMA:
            raise ValueError(
                "unsupported MiniMax-H3 checkpoint: schema_version must be "
                f"{INFERENCE_SCHEMA!r}, got {contract.get('schema_version')!r}"
            )
        if "pdd_steps" in contract:
            component = contract.get("transformer_component")
            if component not in DENOISER_DIRECTORIES.values():
                raise ValueError(
                    "unsupported MiniMax-H3 checkpoint: transformer_component "
                    f"must name a DiT partition, got {component!r}"
                )
            name = next(
                key
                for key, value in DENOISER_DIRECTORIES.items()
                if value == component
            )
            return Layout(
                Kind.COMPONENT_EXPORT,
                MappingProxyType({name: component}),
                MappingProxyType(contract),
                BaseRevision.parse(contract.get("base_model_revision")),
            )
        if "dmd_denoising_steps" in contract:
            return Layout(
                Kind.FASTVIDEO_EXPORT,
                MappingProxyType({"denoiser": "transformer"}),
                MappingProxyType(contract),
                None,
            )
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: fastvideo_inference.json "
            "declares neither dmd_denoising_steps nor pdd_steps"
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
    denoisers = {
        name: directory
        for name, directory in DENOISER_DIRECTORIES.items()
        if directory in index and (root / directory / "config.json").is_file()
    }
    if not denoisers:
        raise ValueError(
            "unsupported MiniMax-H3 checkpoint: the diffusers root holds "
            "neither transformer nor transformer_ref"
        )
    return Layout(Kind.DIFFUSERS_ROOT, MappingProxyType(denoisers), None, None)
