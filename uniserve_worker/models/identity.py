"""Canonical model and checkpoint identities used across worker boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..foundation.errors import invalid_descriptor


def architecture_identity(architecture: str, config: Mapping[str, Any]) -> str:
    """Bind an architecture name to the exact load-time checkpoint configuration."""

    payload = {
        "architecture": str(architecture),
        "configuration": dict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    architecture: str
    architecture_digest: str
    weight_digest: str

    def __post_init__(self) -> None:
        if not self.architecture:
            raise invalid_descriptor("model architecture identity must be named")
        for name in ("architecture_digest", "weight_digest"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise invalid_descriptor(f"model {name} must be a lowercase SHA-256 digest")


__all__ = ["ModelIdentity", "architecture_identity"]
