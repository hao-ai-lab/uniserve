"""Typed worker capability declaration and wire projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar, cast

from .batch import (
    Operation,
    ProductKind,
    SamplingOwnership,
    SnapshotRef,
    WorkVariant,
    protocol_layout_digest,
    route_capability_digest,
)
from .foundation.errors import capability_mismatch, invalid_descriptor
from .spec import OperationType

_WORK_OPERATION_TYPES: dict[WorkVariant, OperationType] = {
    WorkVariant.TOKEN_EXTEND: OperationType.SEQUENCE_EXTEND,
    WorkVariant.TOKEN_DECODE: OperationType.SEQUENCE_DECODE,
    WorkVariant.TOKEN_VERIFY: OperationType.SEQUENCE_VERIFY,
    WorkVariant.ENCODE_VISION: OperationType.ENCODE_VISION,
    WorkVariant.ENCODE_LATENT: OperationType.ENCODE_LATENT,
    WorkVariant.TRANSFER_PRODUCT: OperationType.TRANSFER_PRODUCT,
    WorkVariant.TRANSFER_KV_PUBLISH: OperationType.TRANSFER_KV,
    WorkVariant.TRANSFER_KV_INSTALL: OperationType.TRANSFER_KV,
    WorkVariant.GEN_TRANSITION: OperationType.FLOW,
    WorkVariant.GEN_FLOW: OperationType.FLOW,
}


def operation_type(operation: Operation) -> OperationType:
    """Resolve one registered operation onto its internal execution type.

    Sampling is device postprocessing inside ``Token(*)`` and has no route of
    its own, so no ``Work`` variant maps to ``SEQUENCE_SAMPLE``. ``Draft`` has no
    depth-one route. Materialization is resolved from its exact input product:
    a latent input selects the model image-decode operation while transported or
    resident image inputs select the model-free frame operation.
    """

    variant = operation.work.variant
    if variant is WorkVariant.MATERIALIZE:
        return (
            OperationType.MATERIALIZE_IMAGE
            if any(reference.kind is ProductKind.LATENT for reference in operation.inputs)
            else OperationType.MATERIALIZE_FRAME
        )
    selected = _WORK_OPERATION_TYPES.get(variant)
    if selected is None:
        raise capability_mismatch(f"work variant {variant.value!r} has no route")
    return selected


# The route enum an operation runs under maps onto one or more closed work
# variants. Route selection stays keyed by ``OperationType`` inside the worker;
# the capability wire and its digests are keyed by ``WorkVariant``, so the two
# are bridged here. ``SEQUENCE_SAMPLE`` has no variant (sampling is device
# postprocessing inside the token modes) and contributes no work.
_OPERATION_TYPE_WORK_VARIANTS: dict[OperationType, tuple[WorkVariant, ...]] = {
    OperationType.SEQUENCE_EXTEND: (WorkVariant.TOKEN_EXTEND,),
    OperationType.SEQUENCE_DECODE: (WorkVariant.TOKEN_DECODE,),
    OperationType.SEQUENCE_VERIFY: (WorkVariant.TOKEN_VERIFY,),
    OperationType.SEQUENCE_SAMPLE: (),
    OperationType.FLOW: (WorkVariant.GEN_TRANSITION, WorkVariant.GEN_FLOW),
    OperationType.ENCODE_VISION: (WorkVariant.ENCODE_VISION,),
    OperationType.ENCODE_LATENT: (WorkVariant.ENCODE_LATENT,),
    OperationType.MATERIALIZE_IMAGE: (WorkVariant.MATERIALIZE,),
    OperationType.MATERIALIZE_FRAME: (WorkVariant.MATERIALIZE,),
    OperationType.TRANSFER_PRODUCT: (WorkVariant.TRANSFER_PRODUCT,),
    OperationType.TRANSFER_KV: (WorkVariant.TRANSFER_KV_PUBLISH, WorkVariant.TRANSFER_KV_INSTALL),
}

_UNCONFIGURED_WORK = frozenset({WorkVariant.TOKEN_VERIFY, WorkVariant.DRAFT})


def work_variants_for_operation_types(
    operation_types: Sequence[OperationType],
) -> tuple[WorkVariant, ...]:
    """The closed work variants a set of route operation types supports.

    The result is deduplicated and ordered by the canonical ``WorkVariant``
    position, matching the order the route-capability digest folds them in.
    """

    selected = {
        variant
        for operation_type in operation_types
        for variant in _OPERATION_TYPE_WORK_VARIANTS[operation_type]
    }
    return tuple(variant for variant in WorkVariant if variant in selected)


def configured_work_variants_for_operation_types(
    operation_types: Sequence[OperationType],
) -> tuple[WorkVariant, ...]:
    """Return the work leaves admitted by the configured serving routes."""

    return tuple(
        variant
        for variant in work_variants_for_operation_types(operation_types)
        if variant not in _UNCONFIGURED_WORK
    )


class RequestKind(StrEnum):
    GET_CAPABILITIES = "get_capabilities"
    EXECUTE = "execute"
    POLL_COMPLETIONS = "poll_completions"
    DROP_SESSION = "drop_session"
    SHUTDOWN = "shutdown"
    COPY_KV = "copy_kv"
    RELEASE_PRODUCTS = "release_products"
    GET_METRICS = "get_metrics"
    GET_PRESSURE = "get_pressure"
    SNAPSHOT_SESSION = "snapshot_session"
    RESTORE_SESSION = "restore_session"


class ResponseKind(StrEnum):
    CAPABILITIES = "capabilities"
    RESULT = "result"
    OK = "ok"
    ERROR = "error"
    METRICS = "metrics"
    PRESSURE = "pressure"
    SNAPSHOT = "snapshot"


class ResourceClass(StrEnum):
    KV_BLOCK = "kv_block"
    ENCODER_OUTPUT = "encoder_output"
    IMAGE_LATENT = "image_latent"
    SCRATCH = "scratch"


class KvGroupKind(StrEnum):
    FULL = "full"
    SLIDING_WINDOW = "sliding_window"


@dataclass(frozen=True, slots=True)
class KvGroupSpec:
    group_id: int
    block_offset: int
    num_blocks: int
    kind: KvGroupKind
    window: int
    sink: int

    @classmethod
    def from_wire(cls, value: object, where: str) -> KvGroupSpec:
        data = _map(value, where)
        return cls(
            group_id=_uint(data.get("group_id"), f"{where}.group_id"),
            block_offset=_uint(data.get("block_offset"), f"{where}.block_offset"),
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            kind=_enum(KvGroupKind, data.get("kind"), f"{where}.kind"),
            window=_uint(data.get("window"), f"{where}.window"),
            sink=_uint(data.get("sink"), f"{where}.sink"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "block_offset": self.block_offset,
            "num_blocks": self.num_blocks,
            "kind": self.kind.value,
            "window": self.window,
            "sink": self.sink,
        }


@dataclass(frozen=True, slots=True)
class RankInfo:
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1

    def __post_init__(self) -> None:
        for axis in ("tp", "pp", "dp"):
            rank = getattr(self, f"{axis}_rank")
            size = getattr(self, f"{axis}_size")
            if size < 1 or not 0 <= rank < size:
                raise invalid_descriptor(f"rank.{axis} must satisfy 0 <= rank < size")

    @classmethod
    def from_wire(cls, value: object, where: str = "rank") -> RankInfo:
        data = _map(value, where)
        return cls(
            tp_rank=_uint(data.get("tp_rank", 0), f"{where}.tp_rank"),
            tp_size=_uint(data.get("tp_size", 1), f"{where}.tp_size"),
            pp_rank=_uint(data.get("pp_rank", 0), f"{where}.pp_rank"),
            pp_size=_uint(data.get("pp_size", 1), f"{where}.pp_size"),
            dp_rank=_uint(data.get("dp_rank", 0), f"{where}.dp_rank"),
            dp_size=_uint(data.get("dp_size", 1), f"{where}.dp_size"),
        )

    def to_wire(self) -> dict[str, int]:
        return {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
        }


@dataclass(frozen=True, slots=True)
class CreditVector:
    registered_operations: int = 0
    execution_slots: int = 0
    completion_slots: int = 0
    device_products: int = 0
    kv_pages: int = 0
    rollback_deltas: int = 0
    latent_artifact_bytes: int = 0
    pinned_completion_staging_bytes: int = 0
    transfer_bytes: int = 0
    transfer_tickets: int = 0
    cpu_tasks: int = 0
    output_journal_bytes: int = 0

    _FIELDS = (
        "registered_operations",
        "execution_slots",
        "completion_slots",
        "device_products",
        "kv_pages",
        "rollback_deltas",
        "latent_artifact_bytes",
        "pinned_completion_staging_bytes",
        "transfer_bytes",
        "transfer_tickets",
        "cpu_tasks",
        "output_journal_bytes",
    )

    def __post_init__(self) -> None:
        if any(int(getattr(self, name)) < 0 for name in self._FIELDS):
            raise invalid_descriptor("credit vector values must not be negative")

    def contains(self, requested: CreditVector) -> bool:
        return all(getattr(requested, name) <= getattr(self, name) for name in self._FIELDS)

    def digest_values(self) -> tuple[int, ...]:
        return tuple(int(getattr(self, name)) for name in self._FIELDS)

    @classmethod
    def from_wire(cls, value: object, where: str) -> CreditVector:
        data = _map(value, where)
        return cls(**{name: _uint(data.get(name), f"{where}.{name}") for name in cls._FIELDS})

    def to_wire(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self._FIELDS}


@dataclass(frozen=True, slots=True)
class RouteCreditLimits:
    per_request: CreditVector
    worker: CreditVector

    def __post_init__(self) -> None:
        if not self.worker.contains(self.per_request):
            raise invalid_descriptor("route per-request credits exceed worker-wide credits")

    @classmethod
    def from_wire(cls, value: object, where: str) -> RouteCreditLimits:
        data = _map(value, where)
        return cls(
            per_request=CreditVector.from_wire(data.get("per_request"), f"{where}.per_request"),
            worker=CreditVector.from_wire(data.get("worker"), f"{where}.worker"),
        )

    def to_wire(self) -> dict[str, object]:
        return {"per_request": self.per_request.to_wire(), "worker": self.worker.to_wire()}


@dataclass(frozen=True, slots=True)
class RouteExecutionCapability:
    route: int
    supported_work: tuple[WorkVariant, ...]
    tensorized_mixed: bool
    sampling_ownership: SamplingOwnership
    preemptible: bool
    credits: RouteCreditLimits

    def __post_init__(self) -> None:
        if self.route < 0:
            raise invalid_descriptor("route capability id must not be negative")
        if not self.supported_work or len(set(self.supported_work)) != len(self.supported_work):
            raise invalid_descriptor("route capability work must be non-empty and unique")
        if (
            self.credits.per_request.registered_operations < 1
            or self.credits.per_request.execution_slots < 1
            or self.credits.per_request.completion_slots < 1
        ):
            raise invalid_descriptor(
                "route request credits must include registration and execution"
            )


@dataclass(frozen=True, slots=True)
class ExecutionConstraints:
    max_batch_operations: int
    max_speculative_points: int
    device_sequence_lengths: bool
    device_append_offsets: bool
    incremental_kv_publication: bool
    route_capabilities: tuple[RouteExecutionCapability, ...]

    def __post_init__(self) -> None:
        if self.max_batch_operations < 1:
            raise invalid_descriptor("max_batch_operations must be positive")
        if self.max_speculative_points < 1:
            raise invalid_descriptor("max_speculative_points must be positive")
        if (
            tuple(sorted(self.route_capabilities, key=lambda capability: capability.route))
            != self.route_capabilities
        ):
            raise invalid_descriptor("route capabilities must be canonical")
        if len({capability.route for capability in self.route_capabilities}) != len(
            self.route_capabilities
        ):
            raise invalid_descriptor("route capabilities repeat a route id")


@dataclass(frozen=True, slots=True)
class EngineCaps:
    block_size: int
    num_blocks: int
    num_layers: int
    scratch_capacity_tokens: int
    supported_work: tuple[WorkVariant, ...]
    max_latent_size: int
    latent_downsample: int
    max_vae_grid_tokens: int
    max_vit_grid_tokens: int
    max_latent_feature_bytes: int
    max_vision_feature_bytes: int
    commit_marker_tokens: int
    gen_rope_advance: int
    max_cfg_branches: int
    bytes_per_token: int
    groups: tuple[KvGroupSpec, ...]
    kv_dtype: str
    model_dtype: str
    attention_backend: str
    quantization: str | None
    rank: RankInfo
    pipeline_depth: int
    encoder_cache_budget: int
    supported_controls: tuple[RequestKind, ...]
    execution_constraints: ExecutionConstraints
    resource_classes: tuple[ResourceClass, ...]
    model_spec_digest: str
    weight_digest: str
    restored_snapshots: tuple[SnapshotRef, ...] = ()
    protocol_layout_digest: str = ""
    route_capability_digest: str = ""

    def __post_init__(self) -> None:
        if self.model_dtype not in {"float16", "bfloat16", "float32"}:
            raise invalid_descriptor("model_dtype must use the canonical runtime vocabulary")
        for name in (
            "block_size",
            "num_blocks",
            "num_layers",
            "latent_downsample",
            "bytes_per_token",
            "pipeline_depth",
            "commit_marker_tokens",
            "gen_rope_advance",
            "max_cfg_branches",
        ):
            if getattr(self, name) < 1:
                raise invalid_descriptor(f"capabilities.{name} must be positive")
        for name in (
            "scratch_capacity_tokens",
            "max_latent_size",
            "max_vae_grid_tokens",
            "max_vit_grid_tokens",
            "max_latent_feature_bytes",
            "max_vision_feature_bytes",
            "encoder_cache_budget",
        ):
            if getattr(self, name) < 0:
                raise invalid_descriptor(f"capabilities.{name} must not be negative")
        if not self.supported_work:
            raise invalid_descriptor("capabilities must support a work variant")
        if len(set(self.supported_work)) != len(self.supported_work):
            raise invalid_descriptor("capabilities repeat a work variant")
        routed_work = {
            variant
            for capability in self.execution_constraints.route_capabilities
            for variant in capability.supported_work
        }
        if routed_work != set(self.supported_work):
            raise invalid_descriptor("route capabilities do not cover the declared work")
        if (
            len(
                {
                    capability.credits.worker
                    for capability in self.execution_constraints.route_capabilities
                }
            )
            != 1
        ):
            raise invalid_descriptor("worker routes disagree on worker-wide credits")
        if len(set(self.supported_controls)) != len(self.supported_controls):
            raise invalid_descriptor("capabilities repeat a control")
        if len(set(self.resource_classes)) != len(self.resource_classes):
            raise invalid_descriptor("capabilities repeat a resource class")
        restored_session_ids = tuple(
            reference.version.request_key.session_id for reference in self.restored_snapshots
        )
        if restored_session_ids != tuple(sorted(set(restored_session_ids))):
            raise invalid_descriptor("capabilities restored snapshots are not canonical")
        # The two agreement digests are a pure function of this capability's own
        # fields, so they are computed here at the authoritative construction
        # point. Every construction path (declaration, wire decode, ``replace``)
        # therefore reports the exact digests the handshake validates.
        object.__setattr__(self, "protocol_layout_digest", protocol_layout_digest())
        object.__setattr__(
            self,
            "route_capability_digest",
            route_capability_digest(
                self.supported_work,
                self.max_cfg_branches,
                self.max_latent_size,
                self.max_vae_grid_tokens,
                self.max_vit_grid_tokens,
                self.max_latent_feature_bytes,
                self.max_vision_feature_bytes,
                self.execution_constraints.max_batch_operations,
                self.execution_constraints.max_speculative_points,
                self.execution_constraints.device_sequence_lengths,
                self.execution_constraints.device_append_offsets,
                self.execution_constraints.incremental_kv_publication,
                tuple(
                    (
                        capability.route,
                        capability.supported_work,
                        capability.tensorized_mixed,
                        capability.sampling_ownership,
                        capability.preemptible,
                        (
                            capability.credits.per_request.digest_values(),
                            capability.credits.worker.digest_values(),
                        ),
                    )
                    for capability in self.execution_constraints.route_capabilities
                ),
                self.kv_dtype,
                self.model_dtype,
                self.attention_backend,
            ),
        )

    @classmethod
    def from_wire(cls, value: object, where: str = "capabilities") -> EngineCaps:
        data = _map(value, where)
        return cls(
            block_size=_uint(data.get("block_size"), f"{where}.block_size"),
            num_blocks=_uint(data.get("num_blocks"), f"{where}.num_blocks"),
            num_layers=_uint(data.get("num_layers"), f"{where}.num_layers"),
            scratch_capacity_tokens=_uint(
                data.get("scratch_capacity_tokens"), f"{where}.scratch_capacity_tokens"
            ),
            supported_work=tuple(
                _enum(WorkVariant, item, f"{where}.supported_work[{index}]")
                for index, item in enumerate(
                    _seq(data.get("supported_work"), f"{where}.supported_work")
                )
            ),
            max_latent_size=_uint(data.get("max_latent_size"), f"{where}.max_latent_size"),
            latent_downsample=_uint(data.get("latent_downsample"), f"{where}.latent_downsample"),
            max_vae_grid_tokens=_uint(
                data.get("max_vae_grid_tokens"), f"{where}.max_vae_grid_tokens"
            ),
            max_vit_grid_tokens=_uint(
                data.get("max_vit_grid_tokens"), f"{where}.max_vit_grid_tokens"
            ),
            max_latent_feature_bytes=_uint(
                data.get("max_latent_feature_bytes"), f"{where}.max_latent_feature_bytes"
            ),
            max_vision_feature_bytes=_uint(
                data.get("max_vision_feature_bytes"), f"{where}.max_vision_feature_bytes"
            ),
            commit_marker_tokens=_uint(
                data.get("commit_marker_tokens"), f"{where}.commit_marker_tokens"
            ),
            gen_rope_advance=_uint(data.get("gen_rope_advance"), f"{where}.gen_rope_advance"),
            max_cfg_branches=_uint(data.get("max_cfg_branches"), f"{where}.max_cfg_branches"),
            bytes_per_token=_uint(data.get("bytes_per_token"), f"{where}.bytes_per_token"),
            groups=tuple(
                KvGroupSpec.from_wire(item, f"{where}.groups[{index}]")
                for index, item in enumerate(_seq(data.get("groups", ()), f"{where}.groups"))
            ),
            kv_dtype=_str(data.get("kv_dtype"), f"{where}.kv_dtype"),
            model_dtype=_str(data.get("model_dtype"), f"{where}.model_dtype"),
            attention_backend=_str(data.get("attention_backend"), f"{where}.attention_backend"),
            quantization=(
                None
                if data.get("quantization") is None
                else _str(data["quantization"], f"{where}.quantization")
            ),
            rank=RankInfo.from_wire(data.get("rank"), f"{where}.rank"),
            pipeline_depth=_uint(data.get("pipeline_depth"), f"{where}.pipeline_depth"),
            encoder_cache_budget=_uint(
                data.get("encoder_cache_budget"), f"{where}.encoder_cache_budget"
            ),
            supported_controls=tuple(
                _enum(RequestKind, item, f"{where}.supported_controls[{index}]")
                for index, item in enumerate(
                    _seq(data.get("supported_controls", ()), f"{where}.supported_controls")
                )
            ),
            execution_constraints=ExecutionConstraints(
                max_batch_operations=_uint(
                    _map(data.get("execution_constraints"), f"{where}.execution_constraints").get(
                        "max_batch_operations"
                    ),
                    f"{where}.execution_constraints.max_batch_operations",
                ),
                max_speculative_points=_uint(
                    _map(data.get("execution_constraints"), f"{where}.execution_constraints").get(
                        "max_speculative_points"
                    ),
                    f"{where}.execution_constraints.max_speculative_points",
                ),
                device_sequence_lengths=_bool(
                    _map(data.get("execution_constraints"), f"{where}.execution_constraints").get(
                        "device_sequence_lengths"
                    ),
                    f"{where}.execution_constraints.device_sequence_lengths",
                ),
                device_append_offsets=_bool(
                    _map(data.get("execution_constraints"), f"{where}.execution_constraints").get(
                        "device_append_offsets"
                    ),
                    f"{where}.execution_constraints.device_append_offsets",
                ),
                incremental_kv_publication=_bool(
                    _map(data.get("execution_constraints"), f"{where}.execution_constraints").get(
                        "incremental_kv_publication"
                    ),
                    f"{where}.execution_constraints.incremental_kv_publication",
                ),
                route_capabilities=tuple(
                    RouteExecutionCapability(
                        route=_uint(
                            _map(
                                item, f"{where}.execution_constraints.route_capabilities[{index}]"
                            ).get("route"),
                            f"{where}.execution_constraints.route_capabilities[{index}].route",
                        ),
                        supported_work=tuple(
                            _enum(
                                WorkVariant,
                                variant,
                                f"{where}.execution_constraints.route_capabilities[{index}].supported_work[{variant_index}]",
                            )
                            for variant_index, variant in enumerate(
                                _seq(
                                    _map(
                                        item,
                                        f"{where}.execution_constraints.route_capabilities[{index}]",
                                    ).get("supported_work", ()),
                                    f"{where}.execution_constraints.route_capabilities[{index}].supported_work",
                                )
                            )
                        ),
                        tensorized_mixed=_bool(
                            _map(
                                item, f"{where}.execution_constraints.route_capabilities[{index}]"
                            ).get("tensorized_mixed"),
                            f"{where}.execution_constraints.route_capabilities[{index}].tensorized_mixed",
                        ),
                        sampling_ownership=_enum(
                            SamplingOwnership,
                            _map(
                                item, f"{where}.execution_constraints.route_capabilities[{index}]"
                            ).get("sampling_ownership"),
                            f"{where}.execution_constraints.route_capabilities[{index}].sampling_ownership",
                        ),
                        preemptible=_bool(
                            _map(
                                item, f"{where}.execution_constraints.route_capabilities[{index}]"
                            ).get("preemptible"),
                            f"{where}.execution_constraints.route_capabilities[{index}].preemptible",
                        ),
                        credits=RouteCreditLimits.from_wire(
                            _map(
                                item, f"{where}.execution_constraints.route_capabilities[{index}]"
                            ).get("credits"),
                            f"{where}.execution_constraints.route_capabilities[{index}].credits",
                        ),
                    )
                    for index, item in enumerate(
                        _seq(
                            _map(
                                data.get("execution_constraints"), f"{where}.execution_constraints"
                            ).get("route_capabilities", ()),
                            f"{where}.execution_constraints.route_capabilities",
                        )
                    )
                ),
            ),
            resource_classes=tuple(
                _enum(ResourceClass, item, f"{where}.resource_classes[{index}]")
                for index, item in enumerate(
                    _seq(data.get("resource_classes", ()), f"{where}.resource_classes")
                )
            ),
            model_spec_digest=_str(data.get("model_spec_digest", ""), f"{where}.model_spec_digest"),
            weight_digest=_str(data.get("weight_digest", ""), f"{where}.weight_digest"),
            restored_snapshots=tuple(
                SnapshotRef.from_wire(item, f"{where}.restored_snapshots[{index}]")
                for index, item in enumerate(
                    _seq(data.get("restored_snapshots", ()), f"{where}.restored_snapshots")
                )
            ),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "num_layers": self.num_layers,
            "scratch_capacity_tokens": self.scratch_capacity_tokens,
            "supported_work": [value.value for value in self.supported_work],
            "max_latent_size": self.max_latent_size,
            "latent_downsample": self.latent_downsample,
            "max_vae_grid_tokens": self.max_vae_grid_tokens,
            "max_vit_grid_tokens": self.max_vit_grid_tokens,
            "max_latent_feature_bytes": self.max_latent_feature_bytes,
            "max_vision_feature_bytes": self.max_vision_feature_bytes,
            "commit_marker_tokens": self.commit_marker_tokens,
            "gen_rope_advance": self.gen_rope_advance,
            "max_cfg_branches": self.max_cfg_branches,
            "bytes_per_token": self.bytes_per_token,
            "groups": [value.to_wire() for value in self.groups],
            "kv_dtype": self.kv_dtype,
            "model_dtype": self.model_dtype,
            "attention_backend": self.attention_backend,
            "quantization": self.quantization,
            "rank": self.rank.to_wire(),
            "pipeline_depth": self.pipeline_depth,
            "encoder_cache_budget": self.encoder_cache_budget,
            "supported_controls": [value.value for value in self.supported_controls],
            "execution_constraints": {
                "max_batch_operations": self.execution_constraints.max_batch_operations,
                "max_speculative_points": self.execution_constraints.max_speculative_points,
                "device_sequence_lengths": self.execution_constraints.device_sequence_lengths,
                "device_append_offsets": self.execution_constraints.device_append_offsets,
                "incremental_kv_publication": self.execution_constraints.incremental_kv_publication,
                "route_capabilities": [
                    {
                        "route": capability.route,
                        "supported_work": [variant.value for variant in capability.supported_work],
                        "tensorized_mixed": capability.tensorized_mixed,
                        "sampling_ownership": capability.sampling_ownership.value,
                        "preemptible": capability.preemptible,
                        "credits": capability.credits.to_wire(),
                    }
                    for capability in self.execution_constraints.route_capabilities
                ],
            },
            "resource_classes": [value.value for value in self.resource_classes],
            "model_spec_digest": self.model_spec_digest,
            "weight_digest": self.weight_digest,
            "protocol_layout_digest": self.protocol_layout_digest,
            "route_capability_digest": self.route_capability_digest,
            "restored_snapshots": [reference.to_wire() for reference in self.restored_snapshots],
        }


_E = TypeVar("_E", bound=StrEnum)


def _enum(kind: type[_E], value: object, where: str) -> _E:
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    try:
        return kind(value)
    except ValueError:
        raise invalid_descriptor(f"{where} has unknown value {value!r}") from None


def _map(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be a map")
    return cast(Mapping[str, Any], value)


def _seq(value: object, where: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list")
    return value


def _uint(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"{where} must be a non-negative integer")
    return value


def _bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be a boolean")
    return value


def _str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    return value


__all__ = [name for name in globals() if not name.startswith("_")]
