"""System-owned immutable adapter weight versions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import torch
from safetensors.torch import load_file
from torch import nn

from ..foundation.errors import invalid_descriptor
from ..loader.weight_set import WeightSet
from ..loader.weight_utils import map_weight_name
from ..spec import WeightSpec

__all__ = ["AdapterSnapshot", "AdapterStore"]


@dataclass(frozen=True, slots=True)
class AdapterSnapshot:
    adapter_id: int | None
    version: int
    digest: str
    overrides: dict[str, torch.Tensor]


class AdapterStore:
    """Own immutable base and merged adapter weight sets.

    Loading constructs a new mapping and atomically selects it. The ready
    module's registered parameters remain the base weight set for its lifetime.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        weights: WeightSpec,
        base_digest: str | None = None,
    ) -> None:
        self._module = module
        self._weights = weights
        self._base = WeightSet.from_module(module, digest=base_digest)
        self._active = self._base
        self._overrides: dict[str, torch.Tensor] = {}
        self._target_count = 0
        self._version = 0
        self._lock = RLock()

    @property
    def active_id(self) -> int | None:
        with self._lock:
            return self._active.adapter_id

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def base(self) -> WeightSet:
        return self._base

    def view(self, adapter_id: int | None = None) -> WeightSet:
        """Return the immutable weight set selected for a request plan."""

        with self._lock:
            if adapter_id is not None and adapter_id != self._active.adapter_id:
                raise invalid_descriptor(
                    f"adapter {adapter_id!r} is not the active engine-wide version"
                )
            return self._active

    def loaded_count(self) -> int:
        """Return resident adapter versions in the protocol's adapter unit."""

        with self._lock:
            return int(self._active.adapter_id is not None)

    def snapshot(self) -> AdapterSnapshot:
        with self._lock:
            return AdapterSnapshot(
                adapter_id=self._active.adapter_id,
                version=self._version,
                digest=self._active.digest,
                overrides={
                    name: value.detach().cpu().contiguous()
                    for name, value in self._overrides.items()
                },
            )

    def restore(self, snapshot: AdapterSnapshot) -> None:
        if snapshot.version < 0:
            raise invalid_descriptor("adapter snapshot version must be non-negative")
        if snapshot.adapter_id is None:
            if snapshot.overrides or snapshot.digest != self._base.digest:
                raise invalid_descriptor("base adapter snapshot conflicts with the base weight set")
            active = WeightSet(
                digest=self._base.digest,
                version=snapshot.version,
                tensors=self._base.tensors,
            )
            overrides: dict[str, torch.Tensor] = {}
        else:
            if snapshot.adapter_id < 0 or not snapshot.overrides:
                raise invalid_descriptor("adapter snapshot identity or overrides are incomplete")
            overrides = {}
            for name, value in snapshot.overrides.items():
                base = self._base.tensors.get(name)
                if base is None:
                    raise invalid_descriptor(f"adapter snapshot targets unknown parameter {name!r}")
                if value.shape != base.shape or value.dtype != base.dtype:
                    raise invalid_descriptor(
                        f"adapter snapshot parameter {name!r} does not match base geometry"
                    )
                overrides[name] = value.to(device=base.device).contiguous()
            tensors = dict(self._base.tensors)
            tensors.update(overrides)
            active = WeightSet(
                digest=snapshot.digest,
                version=snapshot.version,
                tensors=tensors,
                adapter_id=snapshot.adapter_id,
            )
        with self._lock:
            self._active = active
            self._version = snapshot.version
            self._overrides = overrides
            self._target_count = len(overrides)

    def load(self, adapter_id: int, adapter_path: str) -> int:
        with self._lock:
            if self._active.adapter_id is not None:
                if self._active.adapter_id != adapter_id:
                    raise invalid_descriptor(
                        "a different LoRA adapter is already loaded; unload it first"
                    )
                merged = self._merged_parameters(adapter_path)
                digest = _adapter_weight_digest(
                    self._base.digest,
                    adapter_path,
                    merged,
                )
                if digest != self._active.digest:
                    raise invalid_descriptor(
                        "loaded adapter identity conflicts with the requested adapter"
                    )
                return self._target_count
            merged = self._merged_parameters(adapter_path)
            tensors = dict(self._base.tensors)
            tensors.update(merged)
            self._version += 1
            self._active = WeightSet(
                digest=_adapter_weight_digest(
                    self._base.digest,
                    adapter_path,
                    merged,
                ),
                version=self._version,
                tensors=tensors,
                adapter_id=adapter_id,
            )
            self._overrides = dict(merged)
            self._target_count = len(merged)
            return self._target_count

    def unload(self, adapter_id: int) -> int:
        with self._lock:
            if self._active.adapter_id is None:
                return 0
            if adapter_id != self._active.adapter_id:
                raise invalid_descriptor(
                    f"adapter {adapter_id!r} is not the active engine-wide version"
                )
            count = self._target_count
            self._version += 1
            self._active = WeightSet(
                digest=self._base.digest,
                version=self._version,
                tensors=self._base.tensors,
            )
            self._overrides = {}
            self._target_count = 0
            return count

    def _merged_parameters(self, adapter_path: str) -> dict[str, torch.Tensor]:
        root = Path(adapter_path)
        weights_path = root / "adapter_model.safetensors"
        config_path = root / "adapter_config.json"
        if not root.is_dir() or not weights_path.is_file() or not config_path.is_file():
            raise invalid_descriptor(
                "an adapter path must be a directory containing adapter_model.safetensors and adapter_config.json"
            )
        tensors = load_file(str(weights_path))
        rank, scale = self._rank_and_scale(config_path)
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in tensors.items():
            if name.endswith(".lora_A.weight"):
                target = name.removesuffix(".lora_A.weight")
                part = "a"
            elif name.endswith(".lora_B.weight"):
                target = name.removesuffix(".lora_B.weight")
                part = "b"
            else:
                raise invalid_descriptor(f"adapter tensor {name!r} is not a canonical LoRA factor")
            pair = pairs.setdefault(target, {})
            if part in pair:
                raise invalid_descriptor(
                    f"adapter target {target!r} repeats its LoRA {part.upper()} factor"
                )
            pair[part] = tensor
        if not pairs:
            raise invalid_descriptor("adapter contains no LoRA factors")
        params = dict(self._module.named_parameters())
        merged: dict[str, torch.Tensor] = {}
        for source, pair in pairs.items():
            if set(pair) != {"a", "b"}:
                raise invalid_descriptor(
                    f"adapter target {source!r} must contain one A and one B factor"
                )
            mapped = map_weight_name(self._weights, source, adapter=True)
            if mapped is None:
                raise invalid_descriptor(f"adapter target {source!r} is not declared by WeightSpec")
            pname = mapped + ".weight"
            if pname not in params:
                raise invalid_descriptor(
                    f"adapter target {source!r} maps to unknown parameter {pname!r}"
                )
            if pname in merged:
                raise invalid_descriptor(f"multiple adapter targets map to parameter {pname!r}")
            a = pair["a"]
            b = pair["b"]
            if a.ndim != 2 or b.ndim != 2 or int(a.shape[0]) != rank or int(b.shape[1]) != rank:
                raise invalid_descriptor(
                    f"adapter target {source!r} factors do not match declared rank {rank}"
                )
            weight = params[pname]
            delta = (
                b.to(weight.device, torch.float32) @ a.to(weight.device, torch.float32)
            ) * scale
            if delta.shape != weight.shape:
                raise invalid_descriptor(
                    f"LoRA delta shape {tuple(delta.shape)} does not match "
                    f"{pname} {tuple(weight.shape)}"
                )
            merged[pname] = weight.detach() + delta.to(weight.dtype)
        return merged

    @staticmethod
    def _rank_and_scale(config_path: Path) -> tuple[int, float]:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise invalid_descriptor(f"cannot read adapter configuration: {exc}") from exc
        if not isinstance(config, dict) or "r" not in config or "lora_alpha" not in config:
            raise invalid_descriptor("adapter_config.json must declare r and lora_alpha")
        rank = config["r"]
        alpha = config["lora_alpha"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise invalid_descriptor("adapter rank r must be a positive integer")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise invalid_descriptor("adapter lora_alpha must be numeric")
        scale = float(alpha) / float(rank)
        if not math.isfinite(scale):
            raise invalid_descriptor("adapter scale must be finite")
        return rank, scale


def _adapter_weight_digest(
    base_digest: str,
    adapter_path: str,
    merged: dict[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256(b"uniserve-adapter-weight-set\0")
    digest.update(base_digest.encode("ascii"))
    root = Path(adapter_path)
    files = sorted(path for path in root.iterdir() if path.is_file()) if root.is_dir() else [root]
    for path in files:
        encoded_name = path.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    for name in sorted(merged):
        digest.update(name.encode("utf-8"))
    return digest.hexdigest()
