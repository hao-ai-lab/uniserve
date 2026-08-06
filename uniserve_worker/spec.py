"""Immutable declarative model identity: ``ModelSpec`` plus its deployment overlay.

``ModelSpec`` is the flat composition a model declares once it is constructed:
architecture, checkpoint revision, physical routes, weights, inputs, cache
geometry, and (for flow families) flow semantics. Deployment placement and
capacity live in the separate :class:`DeploymentOverlay`; the pair resolves to
one :func:`resolved_digest` used for startup validation and graph-store
identity. Torch-free by design; dtypes are canonical string names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from typing import Any, TypeAlias

from .batch import WorkVariant
from .foundation.errors import invalid_descriptor

_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_KV_STORE_DTYPES = frozenset({*_FLOAT_DTYPES, "float8_e4m3fn"})

__all__ = [
    "CacheSpec",
    "DeploymentOverlay",
    "FlowSpec",
    "FlowPromptSpec",
    "FlowConditioningKind",
    "FlowBranchSource",
    "LatentLayout",
    "MaterializationKind",
    "NoiseScaleSpec",
    "NoiseScaleMode",
    "OperationSpec",
    "OperationStageCondition",
    "OperationStagePurpose",
    "OperationStageSpec",
    "PositionLayout",
    "ImageInputSpec",
    "ImagePatchSpec",
    "ImageTowerSpec",
    "InputSpec",
    "FeatureInjectionSpec",
    "FeatureLayout",
    "EncoderResourcePolicy",
    "KvBlockResourcePolicy",
    "LatentTokens",
    "ModelSpec",
    "ModelLoadScope",
    "PerBranch",
    "ResourcePlan",
    "RoutePlacement",
    "RouteRowKind",
    "RouteShape",
    "RouteShapeGrouping",
    "RouteSpec",
    "Cast",
    "Quantize",
    "Rename",
    "Reshape",
    "Shard",
    "Sidecar",
    "Slice",
    "Split",
    "Stack",
    "StrideResizeSpec",
    "Tie",
    "TowerSplit",
    "Transpose",
    "UnmatchedWeightPolicy",
    "WeightSpec",
    "WeightTarget",
    "WeightTransform",
    "active_latent_capacity_tokens",
    "resolved_digest",
]


class UnmatchedWeightPolicy(StrEnum):
    KEEP = "keep"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Rename:
    source: str
    target: str
    exact: bool = False
    substitutions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.source or (self.exact and not self.target):
            raise invalid_descriptor("weight rename endpoints must not be empty")


@dataclass(frozen=True, slots=True)
class Slice:
    source: str
    target: str
    axis: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.start < 0 or self.stop <= self.start:
            raise invalid_descriptor("weight slice declaration is invalid")


@dataclass(frozen=True, slots=True)
class Split:
    source: str
    targets: tuple[str, ...]
    axis: int
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.targets
            or len(self.targets) != len(self.sizes)
            or any(size < 1 for size in self.sizes)
        ):
            raise invalid_descriptor("weight split targets and positive sizes must align")


@dataclass(frozen=True, slots=True)
class Stack:
    target: str
    source: str
    part: str | int

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise invalid_descriptor("weight stack endpoints must not be empty")


@dataclass(frozen=True, slots=True)
class Transpose:
    source: str
    target: str
    axes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.axes:
            raise invalid_descriptor("weight transpose declaration is invalid")


@dataclass(frozen=True, slots=True)
class Reshape:
    source: str
    target: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.shape or any(
            dimension == 0 or dimension < -1 for dimension in self.shape
        ):
            raise invalid_descriptor("weight reshape declaration is invalid")


@dataclass(frozen=True, slots=True)
class Shard:
    source: str
    target: str
    axis: int
    topology_axis: str = "tp"

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.topology_axis:
            raise invalid_descriptor("weight shard declaration is invalid")


@dataclass(frozen=True, slots=True)
class Tie:
    source: str
    target: str

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise invalid_descriptor("weight tie declaration is invalid")


@dataclass(frozen=True, slots=True)
class Cast:
    source: str
    target: str
    dtype: str

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.dtype:
            raise invalid_descriptor("weight cast declaration is invalid")


@dataclass(frozen=True, slots=True)
class Quantize:
    source: str
    target: str
    scheme: str

    def __post_init__(self) -> None:
        if not self.source or not self.target or not self.scheme:
            raise invalid_descriptor("weight quantization declaration is invalid")


WeightTransform: TypeAlias = (
    Rename | Slice | Split | Stack | Transpose | Reshape | Shard | Tie | Cast | Quantize
)


@dataclass(frozen=True, slots=True)
class Sidecar:
    file: str
    module: str
    optional_substrings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.file or not self.module:
            raise invalid_descriptor("sidecar file and module must not be empty")


@dataclass(frozen=True, slots=True)
class TowerSplit:
    generation_prefixes: tuple[str, ...] = ()
    generation_infixes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WeightTarget:
    name: str
    shape: tuple[int, ...]
    dtype: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or any(dimension < 0 for dimension in self.shape) or not self.dtype:
            raise invalid_descriptor("weight target declaration is invalid")


@dataclass(frozen=True, slots=True)
class WeightSpec:
    """Closed checkpoint-to-parameter declaration consumed by the loader."""

    files: tuple[str, ...] = ()
    sidecars: tuple[Sidecar, ...] = ()
    transforms: tuple[WeightTransform, ...] = ()
    targets: tuple[WeightTarget, ...] = ()
    unmatched: UnmatchedWeightPolicy = UnmatchedWeightPolicy.KEEP
    tower: TowerSplit | None = None

    def __post_init__(self) -> None:
        names = tuple(target.name for target in self.targets)
        if len(set(names)) != len(names):
            raise invalid_descriptor("weight target names must be unique")
        if any(not file for file in self.files):
            raise invalid_descriptor("checkpoint file names must not be empty")
        stack_sources = tuple(
            transform.source for transform in self.transforms if isinstance(transform, Stack)
        )
        if len(set(stack_sources)) != len(stack_sources):
            raise invalid_descriptor("weight stack sources must be unique")


class KvBlockResourcePolicy(StrEnum):
    PER_BLOCK = "per_block"


class EncoderResourcePolicy(StrEnum):
    PER_HANDLE = "per_handle"


@dataclass(frozen=True, slots=True)
class LatentTokens:
    downsample: int = 16

    def __post_init__(self) -> None:
        if self.downsample < 1:
            raise invalid_descriptor("latent downsample must be positive")


@dataclass(frozen=True, slots=True)
class PerBranch:
    minimum: int = 1
    minimum_blocks: int = 1
    fixed_tokens: int = 0
    mirror_kv: bool = False
    latent_copies: int = 0
    tower_copy: bool = False

    def __post_init__(self) -> None:
        if self.minimum < 1 or self.minimum_blocks < 1:
            raise invalid_descriptor("scratch branch minima must be positive")
        if self.fixed_tokens < 0 or self.latent_copies < 0:
            raise invalid_descriptor("scratch token and latent counts must not be negative")


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    kv_block: KvBlockResourcePolicy | None = KvBlockResourcePolicy.PER_BLOCK
    image_latent: LatentTokens | None = None
    scratch: PerBranch | None = None
    encoder_output: EncoderResourcePolicy | None = None

    def classes(self) -> tuple[str, ...]:
        result: list[str] = []
        if self.kv_block is not None:
            result.append("kv_block")
        if self.encoder_output is not None:
            result.append("encoder_output")
        if self.image_latent is not None:
            result.append("image_latent")
        if self.scratch is not None:
            result.append("scratch")
        return tuple(result)


def active_latent_capacity_tokens(
    per_image_tokens: int,
    concurrency_token_budget: int | None,
) -> int:
    per_image = max(0, int(per_image_tokens))
    if per_image == 0:
        return 0
    if concurrency_token_budget is None:
        return per_image
    return max(per_image, int(concurrency_token_budget))


class ModelLoadScope(StrEnum):
    """Semantic parameter scope materialized for one worker."""

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


class RouteRowKind(StrEnum):
    TOKEN = "token"
    FLOW = "flow"
    ENCODE = "encode"
    DECODE = "decode"


class RoutePlacement(StrEnum):
    PRIMARY = "primary"
    GENERATION = "generation"
    MESH = "mesh"


class OperationStagePurpose(StrEnum):
    """Semantic role of one neural stage within a system operation."""

    PRIMARY = "primary"
    STATE = "state"


class OperationStageCondition(StrEnum):
    """Closed conditions under which a declared neural stage executes."""

    ALWAYS = "always"
    RETAIN_IMAGE = "retain_image"


class RouteShapeGrouping(StrEnum):
    """Declared hard shape key for rows sharing one physical forward."""

    FLEXIBLE = "flexible"
    INPUT_SHAPE = "input_shape"
    IMAGE_GEOMETRY = "image_geometry"


@dataclass(frozen=True, slots=True)
class OperationStageSpec:
    """One physical neural stage required by a typed system operation."""

    route: str
    row: RouteRowKind
    purpose: OperationStagePurpose = OperationStagePurpose.PRIMARY
    condition: OperationStageCondition = OperationStageCondition.ALWAYS

    def __post_init__(self) -> None:
        if not self.route:
            raise invalid_descriptor("operation stage route must not be empty")


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Declarative lowering recipe for one member of the operation union.

    An empty stage tuple denotes a system-only operation. Neural operations may
    have one primary stage and additional state-publication stages. The executor
    follows these declarations and never selects behavior from an architecture
    name.
    """

    kind: WorkVariant
    stages: tuple[OperationStageSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WorkVariant):
            raise invalid_descriptor(f"invalid operation kind {self.kind!r}")
        primary = sum(stage.purpose is OperationStagePurpose.PRIMARY for stage in self.stages)
        if primary > 1:
            raise invalid_descriptor(
                f"operation {self.kind.value!r} declares multiple primary stages"
            )
        identities = tuple(
            (stage.route, stage.row, stage.purpose, stage.condition) for stage in self.stages
        )
        if len(set(identities)) != len(identities):
            raise invalid_descriptor(f"operation {self.kind.value!r} repeats a neural stage")


@dataclass(frozen=True, slots=True)
class RouteShape:
    """Hard mathematical shape bounds for one route row."""

    max_tokens_per_row: int
    token_multiple: int
    grouping: RouteShapeGrouping = RouteShapeGrouping.FLEXIBLE

    def __post_init__(self) -> None:
        if self.max_tokens_per_row < 1 or self.token_multiple < 1:
            raise invalid_descriptor("route shape bounds must be positive")
        if self.max_tokens_per_row % self.token_multiple:
            raise invalid_descriptor("route token bound must be divisible by its alignment")


@dataclass(frozen=True)
class RouteSpec:
    """One physical forward route and the row vocabulary it accepts.

    ``mixed_combinations`` declares which accepted row kinds may combine in one
    ``ForwardBatch``; ``graph_eligible`` declares captured-graph execution
    eligibility for the route. Operation-to-route lowering belongs to
    :class:`OperationSpec`, keeping physical route compatibility independent
    from scheduler vocabulary.
    """

    name: str
    row_kinds: tuple[RouteRowKind, ...]
    mixed_combinations: tuple[tuple[RouteRowKind, ...], ...]
    dtype: str
    placement: RoutePlacement
    topology_axes: tuple[str, ...]
    shape: RouteShape
    graph_eligible: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise invalid_descriptor("RouteSpec.name must not be empty")
        if not self.row_kinds or len(set(self.row_kinds)) != len(self.row_kinds):
            raise invalid_descriptor(f"route {self.name!r} row kinds must be non-empty and unique")
        normalized_combinations = set()
        for combination in self.mixed_combinations:
            if len(combination) < 2 or len(set(combination)) != len(combination):
                raise invalid_descriptor(
                    f"route {self.name!r} mixed combinations require distinct row kinds"
                )
            if any(row not in self.row_kinds for row in combination):
                raise invalid_descriptor(f"route {self.name!r} mixes a row kind it does not accept")
            normalized = tuple(sorted(combination, key=str))
            if normalized in normalized_combinations:
                raise invalid_descriptor(f"route {self.name!r} repeats a mixed combination")
            normalized_combinations.add(normalized)
        if any(not axis for axis in self.topology_axes):
            raise invalid_descriptor(f"route {self.name!r} topology axes must not be empty")
        if self.dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor(
                f"route {self.name!r} dtype must be one of {sorted(_FLOAT_DTYPES)!r}"
            )


@dataclass(frozen=True)
class ImagePatchSpec:
    """Flattened conv-embedder patch rows from one understanding image.

    The transform chain is: grid-rounded resize (height/width divide
    ``patch_size / downsample_ratio`` with the pixel count scaled into
    ``[min_pixels, max_pixels]``, the per-image budget shrinking under
    ``multi_image_pixel_budget``), the named ``normalization``, then
    patchification into ``[grid_h * grid_w, channels * patch_size**2]`` rows
    plus a ``[1, 2]`` grid tensor. Reported image dimensions are the decoded
    source dimensions.
    """

    patch_size: int
    downsample_ratio: float
    min_pixels: int
    max_pixels: int
    multi_image_pixel_budget: int
    normalization: str = "imagenet"

    def __post_init__(self) -> None:
        if int(self.patch_size) < 1:
            raise invalid_descriptor("ImagePatchSpec.patch_size must be at least 1")
        if not 0 < float(self.downsample_ratio) <= 1:
            raise invalid_descriptor("ImagePatchSpec.downsample_ratio must be in (0, 1]")
        if not 0 < int(self.min_pixels) <= int(self.max_pixels):
            raise invalid_descriptor("ImagePatchSpec pixel bounds must satisfy 0 < min <= max")
        if not self.normalization:
            raise invalid_descriptor("ImagePatchSpec.normalization must not be empty")


@dataclass(frozen=True)
class StrideResizeSpec:
    """Stride-aligned bounded resize: scale into ``[min_size, max_size]``,
    round each side to ``stride``, and cap the pixel count at ``max_pixels``."""

    max_size: int
    min_size: int
    stride: int
    max_pixels: int

    def __post_init__(self) -> None:
        if int(self.stride) < 1:
            raise invalid_descriptor("StrideResizeSpec.stride must be at least 1")
        if not 0 < int(self.min_size) <= int(self.max_size):
            raise invalid_descriptor("StrideResizeSpec sizes must satisfy 0 < min <= max")
        if int(self.max_pixels) < 1:
            raise invalid_descriptor("StrideResizeSpec.max_pixels must be at least 1")


@dataclass(frozen=True)
class ImageTowerSpec:
    """Whole-image CHW tower input: one stride-aligned resize plus the named
    per-channel ``normalization`` (no patchification; the tower patchifies)."""

    resize: StrideResizeSpec
    normalization: str = "signed_unit"

    def __post_init__(self) -> None:
        if not self.normalization:
            raise invalid_descriptor("ImageTowerSpec.normalization must not be empty")


@dataclass(frozen=True)
class ImageInputSpec:
    """Declared image decode and transform requirements for encode routes.

    ``vit`` and ``vae`` declare the transform chain behind the matching encode
    op kind; an undeclared kind accepts no image payload. When ``vae`` is
    declared, every encode input is first resized onto that canvas — the
    canvas dimensions are the reported image dimensions — before the per-kind
    tower transform runs; without one, reported dimensions are the decoded
    source dimensions. ``staging_dtype`` names the device dtype pixels are
    staged to ahead of the neural encode (``None`` keeps the transform's
    dtype). Base64 payloads decode to RGB with transparency composited over
    white for every declared chain.
    """

    vit: ImagePatchSpec | ImageTowerSpec | None = None
    vae: ImageTowerSpec | None = None
    staging_dtype: str | None = None
    feature_injection: FeatureInjectionSpec | None = None

    def __post_init__(self) -> None:
        if self.vit is None and self.vae is None:
            raise invalid_descriptor("ImageInputSpec must declare at least one transform")
        if self.staging_dtype is not None and self.staging_dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor(
                f"ImageInputSpec.staging_dtype must be one of {sorted(_FLOAT_DTYPES)!r}"
            )


class FeatureLayout(StrEnum):
    """How encoder features become a sequence-state row."""

    DIRECT = "direct"
    FRAMED = "framed"


class PositionLayout(StrEnum):
    """Closed position-index layouts used by sequence and flow rows."""

    TEMPORAL = "temporal"
    TEMPORAL_SPATIAL = "temporal_spatial"


@dataclass(frozen=True, slots=True)
class FeatureInjectionSpec:
    """Declarative encoder-feature insertion into the sequence backbone."""

    layout: FeatureLayout
    positions: PositionLayout
    start_token: str | None = None
    end_token: str | None = None
    start_token_id: int | None = None
    end_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.layout is FeatureLayout.FRAMED:
            if self.start_token is None and self.start_token_id is None:
                raise invalid_descriptor("framed feature injection requires a start token")
            if self.end_token is None and self.end_token_id is None:
                raise invalid_descriptor("framed feature injection requires an end token")
        for name in ("start_token_id", "end_token_id"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise invalid_descriptor(f"FeatureInjectionSpec.{name} must not be negative")


@dataclass(frozen=True, slots=True)
class FlowPromptSpec:
    """Declarative chat framing used to provision flow-prefix KV branches."""

    user_prefix: str
    user_suffix: str
    assistant_suffix: str
    conditioned_append: str
    unconditional_append: str
    system_prefix: str = ""
    system_message: str = ""
    system_suffix: str = ""
    add_special_tokens: bool = True

    def __post_init__(self) -> None:
        for name in (
            "user_prefix",
            "user_suffix",
            "assistant_suffix",
            "conditioned_append",
            "unconditional_append",
        ):
            if not getattr(self, name):
                raise invalid_descriptor(f"FlowPromptSpec.{name} must not be empty")
        system_parts = (self.system_prefix, self.system_message, self.system_suffix)
        if any(system_parts) and not all(system_parts):
            raise invalid_descriptor("flow prompt system framing must be fully declared")


@dataclass(frozen=True)
class InputSpec:
    """Declarative input requirements.

    ``requires_worker_tokenizer`` drives worker-side tokenizer materialization
    and vocabulary configuration at bootstrap. ``images`` declares the image
    decode/transform chains the system input stage runs ahead of the model's
    encode entry points; models whose routes accept no encode work leave it
    ``None``.
    """

    requires_worker_tokenizer: bool = False
    images: ImageInputSpec | None = None
    flow_prompt: FlowPromptSpec | None = None
    encoder_cache_budget: int = 0
    max_vit_grid_tokens: int = 0

    def __post_init__(self) -> None:
        if int(self.encoder_cache_budget) < 0:
            raise invalid_descriptor("InputSpec.encoder_cache_budget must not be negative")
        if int(self.max_vit_grid_tokens) < 0:
            raise invalid_descriptor("InputSpec.max_vit_grid_tokens must not be negative")
        if self.flow_prompt is not None and not self.requires_worker_tokenizer:
            raise invalid_descriptor("flow prompt framing requires a worker tokenizer")


@dataclass(frozen=True)
class CacheSpec:
    """Declared KV geometry (the model's view; the system owns the pools)."""

    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    store_dtype: str | None = None
    position_layout: PositionLayout = PositionLayout.TEMPORAL

    def __post_init__(self) -> None:
        for field_name in ("num_layers", "num_attention_heads", "num_kv_heads", "head_dim"):
            if int(getattr(self, field_name)) < 1:
                raise invalid_descriptor(f"CacheSpec.{field_name} must be at least 1")
        if self.dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor(
                f"CacheSpec.dtype must be one of {sorted(_FLOAT_DTYPES)!r}"
            )
        if self.store_dtype is not None and self.store_dtype not in _KV_STORE_DTYPES:
            raise invalid_descriptor(
                f"CacheSpec.store_dtype must be one of {sorted(_KV_STORE_DTYPES)!r}"
            )


@dataclass(frozen=True)
class FlowSpec:
    """Declared flow-matching semantics for latent-generation families.

    ``cfg_recipe`` names how text/image guidance deltas combine
    (``nn.diffusion.cfg.CfgRecipe`` values). ``timestep_shift`` is the schedule
    shift applied when the request carries none; ``None`` requires the request
    to supply one.
    """

    latent_downsample: int
    prediction: str
    prediction_dtype: str
    schedule_direction: str
    schedule_shift_domain: str
    max_latent_tokens: int
    max_vae_grid_tokens: int
    commit_marker_tokens: int
    rope_advance: int
    max_cfg_branches: int
    latent_layout: LatentLayout
    latent_channels: int
    latent_patch_size: int
    positions: PositionLayout
    conditioning: FlowConditioningKind
    materialization: MaterializationKind
    noise_scale: NoiseScaleSpec
    text_unconditional: FlowBranchSource
    image_unconditional: FlowBranchSource
    cfg_recipe: str = "additive_deltas"
    timestep_shift: float | None = None

    def __post_init__(self) -> None:
        if int(self.latent_downsample) < 1:
            raise invalid_descriptor("FlowSpec.latent_downsample must be at least 1")
        for field_name in (
            "max_latent_tokens",
            "max_vae_grid_tokens",
            "commit_marker_tokens",
            "rope_advance",
            "max_cfg_branches",
            "latent_channels",
            "latent_patch_size",
        ):
            if int(getattr(self, field_name)) < 1:
                raise invalid_descriptor(f"FlowSpec.{field_name} must be at least 1")
        for field_name in (
            "prediction",
            "schedule_direction",
            "schedule_shift_domain",
            "cfg_recipe",
        ):
            if not getattr(self, field_name):
                raise invalid_descriptor(f"FlowSpec.{field_name} must not be empty")
        if self.prediction_dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor(
                f"FlowSpec.prediction_dtype must be one of {sorted(_FLOAT_DTYPES)!r}"
            )
        if self.timestep_shift is not None and float(self.timestep_shift) <= 0:
            raise invalid_descriptor("FlowSpec.timestep_shift must be positive")


class LatentLayout(StrEnum):
    """Authoritative latent representation in :class:`LatentStore`."""

    PATCH_TOKENS = "patch_tokens"
    IMAGE_NCHW = "image_nchw"


class FlowConditioningKind(StrEnum):
    """External neural conditioning supplied with a flow row."""

    NONE = "none"
    IMAGE_PATCHES = "image_patches"


class FlowBranchSource(StrEnum):
    """Declared KV source for an active classifier-free-guidance branch."""

    CONDITIONING = "conditioning"
    NEGATIVE_OR_START = "negative_or_start"
    START = "start"


class MaterializationKind(StrEnum):
    """How a committed latent becomes an RGB product."""

    DECODE_ROUTE = "decode_route"
    RGB_LATENT = "rgb_latent"


class NoiseScaleMode(StrEnum):
    CONSTANT = "constant"
    RESOLUTION = "resolution"
    DYNAMIC = "dynamic"
    DYNAMIC_SQRT = "dynamic_sqrt"


@dataclass(frozen=True, slots=True)
class NoiseScaleSpec:
    """Closed resolution-dependent noise-scale equation."""

    value: float = 1.0
    mode: NoiseScaleMode = NoiseScaleMode.CONSTANT
    base_image_tokens: float = 1.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if float(self.value) < 0 or float(self.base_image_tokens) <= 0 or float(self.maximum) <= 0:
            raise invalid_descriptor("noise-scale values must be non-negative with positive bounds")


@dataclass(frozen=True)
class ModelSpec:
    """Immutable flat composition of one loaded model's declarative specs.

    ``revision`` is checkpoint-derived and filled at load time by the
    bootstrap; models declare every other component themselves.
    """

    architecture: str
    routes: tuple[RouteSpec, ...]
    operations: tuple[OperationSpec, ...]
    weights: "WeightSpec"
    inputs: InputSpec
    cache: CacheSpec
    flow: FlowSpec | None = None
    revision: str = ""

    def __post_init__(self) -> None:
        if not self.architecture:
            raise invalid_descriptor("ModelSpec.architecture must not be empty")
        if not self.routes:
            raise invalid_descriptor("ModelSpec must declare at least one route")
        names = [route.name for route in self.routes]
        if len(set(names)) != len(names):
            raise invalid_descriptor("ModelSpec route names must be unique")
        operation_kinds = tuple(operation.kind for operation in self.operations)
        if not operation_kinds or len(set(operation_kinds)) != len(operation_kinds):
            raise invalid_descriptor("ModelSpec operation kinds must be non-empty and unique")
        routes = {route.name: route for route in self.routes}
        for operation in self.operations:
            for stage in operation.stages:
                route = routes.get(stage.route)
                if route is None:
                    raise invalid_descriptor(
                        f"operation {operation.kind.value!r} references unknown route {stage.route!r}"
                    )
                if stage.row not in route.row_kinds:
                    raise invalid_descriptor(
                        f"operation {operation.kind.value!r} maps {stage.row.value!r} onto route {stage.route!r}, which does not accept it"
                    )
        if self.flow is None and (
            WorkVariant.GEN_TRANSITION in operation_kinds
            or WorkVariant.GEN_FLOW in operation_kinds
        ):
            raise invalid_descriptor(
                "a route accepts flow work but the ModelSpec declares no FlowSpec"
            )

    def operation_variants(self) -> frozenset[WorkVariant]:
        """Union of the work variants accepted across every declared route."""
        return frozenset(operation.kind for operation in self.operations)

    def operation(self, variant: WorkVariant) -> OperationSpec:
        for operation in self.operations:
            if operation.kind is variant:
                return operation
        raise invalid_descriptor(f"model does not declare operation {variant.value!r}")


@dataclass(frozen=True)
class DeploymentOverlay:
    """Immutable deployment placement/capacity for one worker's loaded model.

    Environment-specific policy stays here so ``ModelSpec`` keeps checkpoint
    semantics only; the pair resolves to one digest.
    """

    device: str
    model_scope: str
    tp_rank: int
    tp_size: int
    block_size: int
    kv_token_capacity: int | None
    generation_kv_capacity_tokens: int | None
    attention_backend: str | None
    model_dtype: str
    kv_cache_dtype: str | None
    kv_memory_fraction: float
    resources: ResourcePlan
    max_batch_operations: int
    generation_device: str | None

    def __post_init__(self) -> None:
        if not self.device:
            raise invalid_descriptor("DeploymentOverlay.device must not be empty")
        if not self.model_scope:
            raise invalid_descriptor("DeploymentOverlay.model_scope must not be empty")
        if int(self.tp_size) < 1:
            raise invalid_descriptor("DeploymentOverlay.tp_size must be at least 1")
        if not 0 <= int(self.tp_rank) < int(self.tp_size):
            raise invalid_descriptor("DeploymentOverlay.tp_rank must be within tp_size")
        if int(self.block_size) < 1:
            raise invalid_descriptor("DeploymentOverlay.block_size must be at least 1")
        if not 0 < float(self.kv_memory_fraction) <= 1:
            raise invalid_descriptor("DeploymentOverlay.kv_memory_fraction must be in (0, 1]")
        if self.model_dtype not in _FLOAT_DTYPES:
            raise invalid_descriptor(
                f"DeploymentOverlay.model_dtype must be one of {sorted(_FLOAT_DTYPES)!r}"
            )
        if self.kv_cache_dtype is not None and self.kv_cache_dtype not in _KV_STORE_DTYPES:
            raise invalid_descriptor(
                f"DeploymentOverlay.kv_cache_dtype must be one of {sorted(_KV_STORE_DTYPES)!r}"
            )
        if int(self.max_batch_operations) < 1:
            raise invalid_descriptor("DeploymentOverlay.max_batch_operations must be at least 1")


def resolved_digest(spec: ModelSpec, overlay: DeploymentOverlay) -> str:
    """The sha256 identity of one resolved spec under one deployment overlay."""
    payload = {
        "model_spec": _canonical(spec),
        "deployment_overlay": _deployment_identity(overlay),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _deployment_identity(overlay: DeploymentOverlay) -> dict[str, Any]:
    """Project rank-local placement onto one tensor-parallel deployment identity."""

    value = _canonical(overlay)
    if not isinstance(value, dict):
        raise TypeError("deployment overlay did not canonicalize to a mapping")
    value.pop("tp_rank", None)
    value["device"] = str(overlay.device).partition(":")[0].lower()
    if overlay.generation_device is not None:
        value["generation_device"] = str(overlay.generation_device).partition(":")[0].lower()
    return value


def _canonical(value: Any) -> Any:
    """Deterministic JSON-encodable projection of a spec value tree.

    Dataclasses become type-tagged sorted mappings, enums become their values,
    so structurally equal declarations always digest equally. Runtime objects,
    classes, and callables are rejected rather than assigned unstable identities.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            **{f.name: _canonical(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    raise TypeError(f"model declaration contains non-declarative value {value!r}")
