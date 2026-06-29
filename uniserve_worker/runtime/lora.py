"""LoRA adapter utilities."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors.torch import load_file

from ..foundation.errors import invalid_descriptor

__all__ = [
    'MergeOnLoadLoRA',
]


@dataclass
class MergeOnLoadLoRA:
    """Engine-wide LoRA merge/unmerge helper.

    This preserves the current merge-on-load semantics: one adapter can be resident at a
    time, merged directly into model weights, and unload subtracts the exact
    saved fp32 deltas.
    """

    module: Any
    deltas: dict[str, torch.Tensor] = field(default_factory=dict)
    active_id: Any = None

    def load(self, lora_id: Any, lora_path: str) -> int:
        if self.deltas:
            raise invalid_descriptor("a LoRA adapter is already loaded; unload it first")
        weights_path = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = lora_path
        tensors = load_file(weights_path)
        scale = self._scale(lora_path)
        params = dict(self.module.named_parameters())
        loaded: dict[str, torch.Tensor] = {}
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
            w.data.add_(delta.to(w.dtype))
            loaded[pname] = delta
        if not loaded:
            raise invalid_descriptor(f"adapter at {lora_path} matched no model parameters")
        self.deltas = loaded
        self.active_id = lora_id
        return len(loaded)

    def unload(self, lora_id: Any) -> int:
        del lora_id
        if not self.deltas:
            return 0
        params = dict(self.module.named_parameters())
        for pname, delta in self.deltas.items():
            params[pname].data.sub_(delta.to(params[pname].dtype))
        count = len(self.deltas)
        self.deltas = {}
        self.active_id = None
        return count

    @staticmethod
    def _scale(lora_path: str) -> float:
        cfg_path = os.path.join(lora_path, "adapter_config.json") if os.path.isdir(lora_path) else None
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
