"""Shared WorkerDriver base classes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Mapping

from ..contracts.caps import Caps, ExecutionConstraints
from ..foundation.sizing import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_BATCH_OPS

__all__ = ["BaseWorkerDriver"]

_MODEL_FREE_NUM_BLOCKS = 1
_MODEL_FREE_SCRATCH_TOKENS = 0
_MODEL_FREE_BYTES_PER_TOKEN = 1


class BaseWorkerDriver(ABC):
    """Common shell for IPC-facing worker drivers.

    Subclasses own execution semantics; this base standardizes ``block_size``,
    caps storage, and the default shallow-copy caps contract used by model-free
    workers.
    """

    block_size: int
    _caps: Caps

    def __init__(self, *, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        self.block_size = int(block_size)

    def caps(self) -> Caps:
        return replace(self._caps)

    def resource_pressure(self) -> list[dict[str, Any]]:
        return []

    def copy_blocks(self, copies: Any) -> None:
        pass

    def load_lora(self, lora_id: int, lora_path: str) -> None:
        pass

    def unload_lora(self, lora_id: int) -> None:
        pass

    def free_encoder(self, handles: Any) -> None:
        pass

    def reset_prefix_cache(self) -> None:
        pass

    def sleep(self) -> None:
        pass

    def wake_up(self) -> None:
        pass

    @abstractmethod
    def execute(self, batch: Mapping[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def drop_request(self, req_id: int) -> None: ...

    @classmethod
    def _build_model_free_caps(
        cls,
        *,
        block_size: int,
        supported_ops: tuple[str, ...],
        supported_controls: tuple[str, ...] = (),
        num_layers: int = 1,
        max_latent_size: int = 0,
        latent_downsample: int = 1,
        encoder_cache_budget: int | None = None,
        resource_classes: tuple[str, ...] = ("kv_block",),
        **overrides: Any,
    ) -> Caps:
        return Caps(
            block_size=int(block_size),
            num_blocks=int(overrides.pop("num_blocks", _MODEL_FREE_NUM_BLOCKS)),
            num_layers=int(num_layers),
            scratch_capacity_tokens=int(overrides.pop("scratch_capacity_tokens", _MODEL_FREE_SCRATCH_TOKENS)),
            supported_ops=tuple(supported_ops),
            max_latent_size=int(max_latent_size),
            latent_downsample=int(latent_downsample),
            bytes_per_token=int(overrides.pop("bytes_per_token", _MODEL_FREE_BYTES_PER_TOKEN)),
            supported_controls=tuple(supported_controls),
            adapter_mode=str(overrides.pop("adapter_mode", "none")),
            execution_constraints=ExecutionConstraints(
                max_batch_ops=int(overrides.pop("max_batch_ops", DEFAULT_MAX_BATCH_OPS))
            ),
            resource_classes=tuple(resource_classes),
            encoder_cache_budget=encoder_cache_budget,
            **overrides,
        )
