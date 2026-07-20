"""Model-family descriptors and their closed execution contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from ..foundation.errors import WorkerError, capability_mismatch
from .forward_mode import ForwardMode, mode_for_op

__all__ = [
    "FamilyExecutionContract",
    "ModelLoadScope",
    "ModelFamilyDescriptor",
    "ModelOperation",
    "ModelOperationSet",
]


class ModelLoadScope(StrEnum):
    """Semantic scope of model parameters materialized for a worker."""

    WHOLE = "whole"
    UNDERSTANDING = "understanding"
    GENERATION = "generation"

    @property
    def tower_role(self) -> str | None:
        if self is ModelLoadScope.UNDERSTANDING:
            return "und"
        if self is ModelLoadScope.GENERATION:
            return "gen"
        return None


@dataclass(frozen=True, slots=True)
class ModelOperation:
    """One declared operation and its canonical family-adapter method."""

    kind: str
    adapter_method: str

    @classmethod
    def from_kind(cls, kind: str) -> "ModelOperation":
        try:
            mode = mode_for_op(str(kind))
        except WorkerError as exc:
            raise capability_mismatch(f"model declares unknown op {kind!r}") from exc
        if mode in {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT}:
            return cls(str(kind), "forward")
        if mode is ForwardMode.DENOISE:
            return cls(str(kind), "predict_velocity")
        if mode is ForwardMode.COMMIT:
            return cls(str(kind), "decode_image")
        if mode is ForwardMode.ENCODE and kind == "vit_encode":
            return cls(str(kind), "encode_image")
        if mode is ForwardMode.ENCODE and kind == "vae_encode":
            return cls(str(kind), "encode_latents")
        raise capability_mismatch(f"model operation {kind!r} has no family-adapter method")


@dataclass(frozen=True, slots=True)
class ModelOperationSet:
    """Typed operation vocabulary for one model family."""

    operations: tuple[ModelOperation, ...]

    @classmethod
    def from_kinds(cls, kinds: tuple[str, ...]) -> "ModelOperationSet":
        return cls(tuple(ModelOperation.from_kind(str(kind)) for kind in kinds))

    @classmethod
    def from_model_class(cls, model_cls: type) -> "ModelOperationSet":
        return cls.from_kinds(
            tuple(str(op) for op in (getattr(model_cls, "supported_ops", ()) or ()))
        )

    def supported_ops(self) -> tuple[str, ...]:
        return tuple(operation.kind for operation in self.operations)

    def validate(self, model_cls: type) -> None:
        if not self.operations:
            return
        if not callable(getattr(model_cls, "forward", None)):
            raise capability_mismatch(
                f"{model_cls.__name__} declares model operations but has no forward()"
            )


@dataclass(frozen=True)
class FamilyExecutionContract:
    """Closed operation vocabulary for one model family."""

    operations: ModelOperationSet

    @classmethod
    def from_model_class(cls, model_cls: type) -> "FamilyExecutionContract":
        return cls(operations=ModelOperationSet.from_model_class(model_cls))


@dataclass(frozen=True)
class ModelFamilyDescriptor:
    """Resolved family-level behavior for model bring-up and dispatch."""

    family: str
    names: tuple[str, ...]
    model_class: type
    execution: FamilyExecutionContract
    processor_factory: Callable[[], Any] | None = None
    loader_name: str = "default"
    checkpoint_layout: Any | None = None
    image_pipeline_factory: Callable[[Any], Any] | None = None

    @classmethod
    def from_model_class(
        cls,
        model_cls: type,
        *,
        names: tuple[str, ...] | None = None,
    ) -> "ModelFamilyDescriptor":
        family_names = names or tuple(
            str(name) for name in getattr(model_cls, "architectures", (model_cls.__name__,))
        )
        return cls(
            family=str(getattr(model_cls, "family", model_cls.__name__)),
            names=family_names,
            model_class=model_cls,
            execution=FamilyExecutionContract.from_model_class(model_cls),
            loader_name=str(getattr(model_cls, "loader_name", "default")),
            checkpoint_layout=getattr(model_cls, "checkpoint_layout", None),
        )

    @property
    def operation_set(self) -> ModelOperationSet:
        return self.execution.operations

    def matches_model(self, model_cls: type) -> bool:
        names = {model_cls.__name__, *[str(v) for v in getattr(model_cls, "architectures", ())]}
        return bool(names.intersection(self.names))

    def with_processor(
        self,
        *,
        processor_factory: Callable[[], Any],
        image_pipeline_factory: Callable[[Any], Any] | None = None,
    ) -> "ModelFamilyDescriptor":
        return ModelFamilyDescriptor(
            family=self.family,
            names=self.names,
            model_class=self.model_class,
            execution=self.execution,
            processor_factory=processor_factory,
            loader_name=self.loader_name,
            checkpoint_layout=self.checkpoint_layout,
            image_pipeline_factory=image_pipeline_factory,
        )

    def processor(self) -> Any | None:
        if self.processor_factory is None:
            return None
        return self.processor_factory()

    def image_pipeline(self, processor: Any) -> Any | None:
        if self.image_pipeline_factory is None:
            return None
        return self.image_pipeline_factory(processor)
