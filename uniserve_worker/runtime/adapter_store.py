"""System-owned adapter residency for merged-representation models."""
from __future__ import annotations

import json
import os
import threading
from typing import Any

import torch
from safetensors.torch import load_file

from ..foundation.errors import invalid_descriptor

__all__ = [
    'AdapterStore',
]


class AdapterStore:
    """Owns adapter versions and the physical weight copies they displace.

    One adapter may be resident at a time. ``load`` computes the merged
    representation for every targeted parameter, saves a pre-merge copy of
    exactly those parameters, and writes the merged values; ``unload`` restores
    the saved originals bit-for-bit. Originals are kept only for the touched
    subset, so residency cost scales with the adapter's target set rather than
    the whole model. Every committed load/unload advances a monotonic
    ``version``; ``active_id`` names the resident adapter.
    """

    def __init__(self, module: Any) -> None:
        self._module = module
        self._originals: dict[str, torch.Tensor] = {}
        self._active_id: Any = None
        self._version = 0
        # Merge/unmerge rewrites shared model weights; serialize load/unload so
        # a concurrent control op cannot interleave a half-applied merge with
        # another adapter's commit.
        self._lock = threading.Lock()

    @property
    def active_id(self) -> Any:
        return self._active_id

    @property
    def version(self) -> int:
        return self._version

    def load(self, adapter_id: Any, adapter_path: str) -> int:
        with self._lock:
            if self._originals:
                raise invalid_descriptor("a LoRA adapter is already loaded; unload it first")
            merged = self._merged_parameters(adapter_path)
            params = dict(self._module.named_parameters())
            originals: dict[str, torch.Tensor] = {}
            for pname, value in merged.items():
                originals[pname] = params[pname].detach().clone()
                params[pname].data.copy_(value)
            self._originals = originals
            self._active_id = adapter_id
            self._version += 1
            return len(merged)

    def unload(self, adapter_id: Any) -> int:
        del adapter_id
        with self._lock:
            if not self._originals:
                return 0
            params = dict(self._module.named_parameters())
            for pname, original in self._originals.items():
                params[pname].data.copy_(original)
            count = len(self._originals)
            self._originals = {}
            self._active_id = None
            self._version += 1
            return count

    def _merged_parameters(self, adapter_path: str) -> dict[str, torch.Tensor]:
        """Compute the merged value of every parameter the adapter targets.

        Pure with respect to the module: validation failures leave the resident
        weights untouched because nothing is written until every target merges.
        """
        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = adapter_path
        tensors = load_file(weights_path)
        scale = self._scale(adapter_path)
        params = dict(self._module.named_parameters())
        merged: dict[str, torch.Tensor] = {}
        for a_name, a in tensors.items():
            if ".lora_A." not in a_name:
                continue
            b_name = a_name.replace(".lora_A.", ".lora_B.")
            b = tensors.get(b_name)
            if b is None:
                continue
            target = _strip_peft_prefix(a_name.split(".lora_A.")[0])
            pname = target + ".weight"
            w = params.get(pname)
            if w is None:
                continue
            delta = (b.to(w.device, torch.float32) @ a.to(w.device, torch.float32)) * scale
            if delta.shape != w.shape:
                raise invalid_descriptor(
                    f"LoRA delta shape {tuple(delta.shape)} does not match "
                    f"{pname} {tuple(w.shape)}"
                )
            merged[pname] = w.detach() + delta.to(w.dtype)
        if not merged:
            raise invalid_descriptor(f"adapter at {adapter_path} matched no model parameters")
        return merged

    @staticmethod
    def _scale(adapter_path: str) -> float:
        cfg_path = (
            os.path.join(adapter_path, "adapter_config.json")
            if os.path.isdir(adapter_path)
            else None
        )
        if not cfg_path or not os.path.exists(cfg_path):
            return 1.0
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        rank = float(cfg.get("r", 1) or 1)
        return float(cfg.get("lora_alpha", rank)) / rank


def _strip_peft_prefix(target: str) -> str:
    for prefix in ("base_model.model.", "base_model.", "model."):
        if target.startswith(prefix):
            return target[len(prefix):]
    return target
