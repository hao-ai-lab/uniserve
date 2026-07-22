"""Model loader contract.

Every model family is constructed by a registered ``BaseModelLoader``. The
loader owns checkpoint reading, validation, transform application, and weight
injection, driven by the family's declarative
:class:`~uniserve_worker.loader.weight_spec.WeightSpec`:

* ``DefaultModelLoader`` streams a checkpoint into a model built from
  ``(model_cls, config)``.
* ``NativeTransformersLoader`` meta-initializes a Hugging Face native module,
  streams it one tensor at a time, and wraps it in the serving model.
* ``CompositeCheckpointLoader`` eagerly builds a neural graph, streams a
  declared root file plus sidecar submodule files, and wraps it.
* ``DummyModelLoader`` random-initializes a model for bring-up without weights.

Every loader returns a :class:`LoadResult` (model + tokenizer + device).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Type

from ..contracts.model_protocols import UniModel

__all__ = [
    "LoadResult",
    "BaseModelLoader",
]


@dataclass(frozen=True)
class LoadResult:
    """The model-loading return contract: model + tokenizer + device.

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
        attention_backend, gen_snapshot_kv_capacity, tower_role) consumed by
        loaders that build a heavyweight serving wrapper; config-driven loaders
        ignore it.
        """
        raise NotImplementedError
