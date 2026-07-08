"""Model-family descriptors and operation sets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..foundation.errors import WorkerError, capability_mismatch
from .forward_mode import ForwardMode, mode_for_op

__all__ = [
    "ModelFamilyDescriptor",
    "ModelOperationSet",
]


@dataclass(frozen=True)
class ModelOperationSet:
    """Operation vocabulary and hook resolution for one model family."""

    ops: tuple[str, ...]

    @classmethod
    def from_model_class(cls, model_cls: type) -> "ModelOperationSet":
        return cls(tuple(str(op) for op in (getattr(model_cls, "supported_ops", ()) or ())))

    def supported_ops(self) -> tuple[str, ...]:
        return self.ops

    def hook_for(self, op_kind: str) -> tuple[str, ...]:
        try:
            mode = mode_for_op(str(op_kind))
        except WorkerError as exc:
            raise capability_mismatch(f"model declares unknown op {op_kind!r}") from exc
        if mode in {ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY}:
            return ("forward",)
        if mode == ForwardMode.DENOISE:
            return ("predict_velocity",)
        if mode == ForwardMode.COMMIT:
            return ("decode_image",)
        if mode == ForwardMode.ENCODE and op_kind == "vit_encode":
            return ("encode_image",)
        if mode == ForwardMode.ENCODE and op_kind == "vae_encode":
            return ("encode_latents",)
        return ("forward",)

    def validate(self, model_cls: type) -> None:
        if not self.ops:
            return
        if bool(getattr(model_cls, "whole_batch_forward", False)):
            if not callable(getattr(model_cls, "forward", None)):
                raise capability_mismatch(
                    f"{model_cls.__name__} declares a whole-batch forward but has no forward()"
                )
            return
        for op in self.ops:
            hooks = self.hook_for(op)
            if not any(callable(getattr(model_cls, hook, None)) for hook in hooks):
                missing = " or ".join(f"{hook}()" for hook in hooks)
                raise capability_mismatch(
                    f"{model_cls.__name__} declares op {op!r} but does not implement {missing}"
                )


@dataclass(frozen=True)
class ModelFamilyDescriptor:
    """Resolved family-level behavior for model bring-up and dispatch."""

    names: tuple[str, ...]
    model_class: type
    operation_set: ModelOperationSet
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
            names=family_names,
            model_class=model_cls,
            operation_set=ModelOperationSet.from_model_class(model_cls),
            loader_name=str(getattr(model_cls, "loader_name", "default")),
            checkpoint_layout=getattr(model_cls, "checkpoint_layout", None),
        )

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
            names=self.names,
            model_class=self.model_class,
            operation_set=self.operation_set,
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
