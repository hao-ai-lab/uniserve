"""Low-rank adapters a checkpoint source merges into its weights at load.

An adapter file holds, for each adapted module, a ``lora_A`` (down) and a
``lora_B`` (up) matrix and optionally its ``alpha``, the PEFT layout that
FastVideo's native exports also use. Merging adds
``strength * alpha / rank * B @ A`` to the module's checkpoint weight, as
PEFT and FastVideo merge an adapter at load; ``checkpoint.LoRAWeight`` adds
it as the weight is read.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

_KEY = re.compile(r"^(?P<module>.+)\.(?:(?P<factor>lora_[AB])\.weight|alpha)$")


@dataclass(frozen=True, slots=True)
class Adapter:
    """One adapter file and the strength it merges at.

    ``names`` renames the adapter's module paths to the checkpoint's, as
    ``(pattern, replacement)`` regular expressions of which the first that
    matches applies; a model package supplies them when its training code
    names modules differently from its checkpoint.
    """

    path: Path
    strength: float = 1.0
    names: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        if not math.isfinite(self.strength):
            raise ValueError("LoRA strength must be finite")


@dataclass(frozen=True, slots=True)
class LowRank:
    """One adapted weight's factors: it gains ``scale * up @ down``."""

    down: torch.Tensor
    up: torch.Tensor
    scale: float


def read(adapter: Adapter) -> Mapping[str, LowRank]:
    """Read an adapter's factors, keyed by the checkpoint weight they adapt.

    A module without ``alpha`` merges at ``alpha = rank``, as FastVideo
    reads it.

    Raises:
        ValueError: A key outside the adapter layout, a module missing one
            of its factors, factors whose ranks disagree, or two modules
            that rename to one checkpoint weight.
    """
    parts: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in load_file(adapter.path).items():
        match = _KEY.match(key)
        if match is None:
            raise ValueError(f"{adapter.path}: unexpected LoRA key {key!r}")
        parts.setdefault(match["module"], {})[match["factor"] or "alpha"] = (
            tensor
        )
    names = [(re.compile(pattern), new) for pattern, new in adapter.names]
    factors = {}
    for module, found in parts.items():
        if "lora_A" not in found or "lora_B" not in found:
            raise ValueError(f"{adapter.path}: {module} lacks lora_A or lora_B")
        down, up = found["lora_A"], found["lora_B"]
        if down.ndim != 2 or up.ndim != 2 or up.shape[1] != down.shape[0]:
            raise ValueError(
                f"{adapter.path}: {module} has lora_A {tuple(down.shape)} "
                f"and lora_B {tuple(up.shape)}"
            )
        rank = down.shape[0]
        alpha = float(found["alpha"].item()) if "alpha" in found else rank
        for pattern, new in names:
            if pattern.match(module):
                module = pattern.sub(new, module)
                break
        if f"{module}.weight" in factors:
            raise ValueError(
                f"{adapter.path}: two modules adapt {module}.weight"
            )
        factors[f"{module}.weight"] = LowRank(
            down, up, adapter.strength * alpha / rank
        )
    return factors
