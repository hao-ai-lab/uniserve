"""Model bring-up contracts.

Model construction follows one of two contracts, selected by which one a model
class satisfies rather than by incidental attribute reflection:

* ``BaseModelLoader`` — config/format-driven loaders, each registered under a
  ``load_format`` and governed by this ABC. ``DefaultModelLoader`` streams a
  checkpoint into a ``UniModel`` built from ``(model_cls, config)``;
  ``DummyModelLoader`` random-initializes one; ``NativeTransformersLoader``
  performs the meta-init + per-tensor HF streaming materialization and hands the
  result to the model's ``from_native`` wrapper. Every loader returns a
  :class:`LoadResult` (model + tokenizer + device).
* ``ModelBringUp`` — model-owned bring-up for checkpoints whose construction
  needs heavyweight, model-specific wrappers (tokenizer, image processor,
  generation-device CUDA streams, HF attn-impl fallback) that the
  ``(model_cls, config)`` loader inputs cannot express. Such a model exposes a
  ``from_pretrained`` classmethod returning the fully-wrapped ``UniModel``; it
  may itself delegate the heavy materialization to a registered
  ``BaseModelLoader`` (e.g. sensenova routes through ``NativeTransformersLoader``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Type, runtime_checkable

from ..contracts.model_protocols import UniModel

__all__ = [
    "LoadResult",
    "BaseModelLoader",
    "ModelBringUp",
]


@dataclass(frozen=True)
class LoadResult:
    """The richer model-loading return contract: model + tokenizer + device.

    ``model`` is the fully-constructed ``UniModel``. ``tokenizer`` is the
    model's tokenizer when the loader materialized one (``NativeTransformersLoader``
    via the HF config), or ``None`` when the runner driver must still configure it
    from the model path (config-driven loaders). ``device`` is the resolved input
    device the model's parameters actually landed on.
    """

    model: UniModel
    tokenizer: Any | None = None
    device: str | None = None


class BaseModelLoader(ABC):
    @abstractmethod
    def load_model(
        self,
        model_cls: Type[UniModel],
        config: Any,
        *,
        device: str = "cpu",
        model_path: str | None = None,
        **kwargs: Any,
    ) -> LoadResult:
        """Construct and return a model under this loader's format.

        ``kwargs`` carries serving configuration (block_size, kv_token_capacity,
        attention_backend, and model-specific extras) that format loaders needing
        to build a heavyweight wrapper consume; config-driven loaders ignore it.
        """
        raise NotImplementedError


@runtime_checkable
class ModelBringUp(Protocol):
    """A model class that owns its bring-up via ``from_pretrained``.

    ``load_worker_model`` dispatches to this contract when a model class
    declares it, instead of reflecting an arbitrary ``from_pretrained``
    attribute. The classmethod returns the fully-constructed ``UniModel``
    (including its model-specific wrappers) for the requested serving
    configuration. An implementation may delegate the heavy weight
    materialization to a registered :class:`BaseModelLoader`.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str,
        block_size: int = ...,
        kv_token_capacity: int | None = ...,
        attention_backend: str | None = ...,
        **kwargs: Any,
    ) -> UniModel: ...
