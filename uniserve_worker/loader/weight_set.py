"""Live model-weight identity and digest construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import nn

from .config import LoadFormat
from .source import WeightSourceSet

__all__ = [
    "WeightSet",
    "dummy_weight_digest",
    "module_weight_digest",
    "source_weight_digest",
]


@dataclass(frozen=True, slots=True)
class WeightSet:
    """An immutable identity over a read-only view of the live module tensors."""

    digest: str
    version: int
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if len(self.digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.digest
        ):
            raise ValueError("weight digest must be a lowercase SHA-256 digest")
        if self.version < 0:
            raise ValueError("weight version cannot be negative")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, torch.Tensor)
            or tensor.is_meta
            for name, tensor in self.tensors.items()
        ):
            raise ValueError("live weight tensors must be named materialized tensors")
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        digest: str | None = None,
        version: int = 0,
    ) -> "WeightSet":
        return cls(
            digest=module_weight_digest(module) if digest is None else digest,
            version=int(version),
            tensors=_live_tensors(module),
        )


def source_weight_digest(
    architecture: str,
    scope: str,
    load_format: LoadFormat,
    sources: tuple[WeightSourceSet, ...],
) -> str:
    digest = hashlib.sha256(b"uniserve-weight-source\0")
    digest.update(architecture.encode("utf-8"))
    digest.update(b"\0")
    digest.update(scope.encode("utf-8"))
    digest.update(b"\0")
    digest.update(load_format.value.encode("ascii"))
    digest.update(b"\0")
    for source in sources:
        records = sorted(zip(source.relative_paths, source.weight_files), key=lambda item: item[0])
        for relative, path in records:
            encoded = relative.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            digest.update(path.stat().st_size.to_bytes(8, "little"))
            with path.open("rb") as checkpoint:
                while chunk := checkpoint.read(8 * 1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def dummy_weight_digest(
    architecture: str,
    scope: str,
    serving_dtype: str,
    module: nn.Module,
) -> str:
    digest = hashlib.sha256(b"uniserve-weight-dummy\0")
    digest.update(architecture.encode("utf-8"))
    digest.update(b"\0")
    digest.update(scope.encode("utf-8"))
    digest.update(b"\0")
    digest.update(serving_dtype.encode("ascii"))
    for name, parameter in sorted(module.named_parameters()):
        if parameter.is_meta:
            continue
        _update_metadata(digest, name, parameter)
    return digest.hexdigest()


def module_weight_digest(module: nn.Module) -> str:
    digest = hashlib.sha256(b"uniserve-weight-install\0")
    for name, tensor in sorted(_live_tensors(module).items()):
        _update_metadata(digest, name, tensor)
        raw = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
        digest.update(raw.tobytes())
    return digest.hexdigest()


def _live_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        **{
            name: parameter.detach()
            for name, parameter in module.named_parameters()
            if not parameter.is_meta
        },
        **{
            name: buffer.detach()
            for name, buffer in module.named_buffers()
            if not buffer.is_meta
        },
    }


def _update_metadata(digest: Any, name: str, tensor: torch.Tensor) -> None:
    encoded_name = name.encode("utf-8")
    digest.update(len(encoded_name).to_bytes(8, "little"))
    digest.update(encoded_name)
    dtype = str(tensor.dtype).encode("ascii")
    digest.update(len(dtype).to_bytes(2, "little"))
    digest.update(dtype)
    digest.update(len(tensor.shape).to_bytes(2, "little"))
    for dimension in tensor.shape:
        digest.update(int(dimension).to_bytes(8, "little"))
