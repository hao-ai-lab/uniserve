"""Model-family descriptors and their closed execution contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..foundation.errors import WorkerError, capability_mismatch
from .forward_mode import ForwardMode, mode_for_op
from .operations import OperationTag

if TYPE_CHECKING:
    from .cache_schema import FamilyCacheRegistration

__all__ = [
    "FamilyExecutionContract",
    "ModelFamilyDescriptor",
    "ModelOperation",
    "ModelOperationSet",
]


@dataclass(frozen=True, slots=True)
class ModelOperation:
    """One declared operation and its canonical family-adapter method."""

    kind: str
    tag: OperationTag
    adapter_method: str

    @classmethod
    def from_kind(cls, kind: str) -> "ModelOperation":
        try:
            mode = mode_for_op(str(kind))
        except WorkerError as exc:
            raise capability_mismatch(f"model declares unknown op {kind!r}") from exc
        if mode in {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.VERIFY_DRAFT}:
            return cls(str(kind), OperationTag.SEQUENCE_STEP, "forward")
        if mode is ForwardMode.DENOISE:
            return cls(str(kind), OperationTag.FLOW_STEP, "predict_velocity")
        if mode is ForwardMode.COMMIT:
            return cls(str(kind), OperationTag.MATERIALIZE_STEP, "decode_image")
        if mode is ForwardMode.ENCODE and kind == "vit_encode":
            return cls(str(kind), OperationTag.ENCODE_STEP, "encode_image")
        if mode is ForwardMode.ENCODE and kind == "vae_encode":
            return cls(str(kind), OperationTag.ENCODE_STEP, "encode_latents")
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

    def operation_tags(self) -> frozenset[OperationTag]:
        """Return the general operation algebra advertised by this family."""

        return frozenset(operation.tag for operation in self.operations)

    def validate(self, model_cls: type) -> None:
        if not self.operations:
            return
        if bool(getattr(model_cls, "whole_batch_forward", False)):
            if not callable(getattr(model_cls, "forward", None)):
                raise capability_mismatch(
                    f"{model_cls.__name__} declares a whole-batch forward but has no forward()"
                )
            return
        for operation in self.operations:
            if not callable(getattr(model_cls, operation.adapter_method, None)):
                raise capability_mismatch(
                    f"{model_cls.__name__} declares op {operation.kind!r} but does not implement "
                    f"{operation.adapter_method}()"
                )


@dataclass(frozen=True)
class FamilyExecutionContract:
    """Closed operation vocabulary and cache lowering for one model family."""

    operations: ModelOperationSet
    cache_registration_factory: Callable[..., FamilyCacheRegistration] | None

    @classmethod
    def from_model_class(cls, model_cls: type) -> "FamilyExecutionContract":
        factory = getattr(model_cls, "cache_registration_factory", None)
        if factory is not None and not callable(factory):
            raise capability_mismatch(
                f"{model_cls.__name__}.cache_registration_factory must be callable"
            )
        return cls(
            operations=ModelOperationSet.from_model_class(model_cls),
            cache_registration_factory=factory,
        )

    @property
    def operation_tags(self) -> frozenset[OperationTag]:
        return self.operations.operation_tags()

    def build_cache_registration(
        self,
        *,
        family: str,
        **geometry: int,
    ) -> FamilyCacheRegistration:
        """Build and prove the family's cache schema covers exactly its operations."""

        factory = self.cache_registration_factory
        if factory is None:
            raise capability_mismatch(f"model family {family!r} has no cache registration factory")
        registration = factory(**geometry)
        lowerable = frozenset(region.operation_tag for region in registration.schema.regions)
        if lowerable != self.operation_tags:
            raise capability_mismatch(
                f"model family {family!r} advertises "
                f"{sorted(tag.name for tag in self.operation_tags)} but its cache schema "
                f"lowers {sorted(tag.name for tag in lowerable)}"
            )
        return registration


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

    @property
    def operation_tags(self) -> frozenset[OperationTag]:
        return self.execution.operation_tags

    def build_cache_registration(self, **geometry: int) -> FamilyCacheRegistration:
        return self.execution.build_cache_registration(family=self.family, **geometry)

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
