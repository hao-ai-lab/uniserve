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
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..foundation.errors import invalid_descriptor
from .op_kinds import OP_KIND_TABLE, OP_KINDS

if TYPE_CHECKING:
    from ..loader.weight_spec import WeightSpec

__all__ = [
    "CacheSpec",
    "DeploymentOverlay",
    "FlowSpec",
    "ImageInputSpec",
    "ImagePatchSpec",
    "ImageTowerSpec",
    "InputSpec",
    "ModelSpec",
    "RouteSpec",
    "StrideResizeSpec",
    "resolved_digest",
]


@dataclass(frozen=True)
class RouteSpec:
    """One physical forward route and the row vocabulary it accepts.

    ``op_kinds`` is the accepted row-variant vocabulary (canonical forward op
    kinds). ``mixed`` declares whether the accepted kinds may combine in one
    ``ForwardBatch``; ``graph_eligible`` declares captured-graph execution
    eligibility for the route.
    """

    name: str
    op_kinds: tuple[str, ...]
    mixed: bool
    dtype: str
    graph_eligible: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise invalid_descriptor("RouteSpec.name must not be empty")
        if not self.op_kinds:
            raise invalid_descriptor(f"route {self.name!r} must accept at least one op kind")
        if len(set(self.op_kinds)) != len(self.op_kinds):
            raise invalid_descriptor(f"route {self.name!r} repeats an op kind")
        for kind in self.op_kinds:
            if kind not in OP_KINDS:
                raise invalid_descriptor(f"route {self.name!r} accepts unknown op kind {kind!r}")
        if not self.dtype:
            raise invalid_descriptor(f"route {self.name!r} must declare a dtype")


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

    def __post_init__(self) -> None:
        if self.vit is None and self.vae is None:
            raise invalid_descriptor("ImageInputSpec must declare at least one transform")


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


@dataclass(frozen=True)
class CacheSpec:
    """Declared KV geometry (the model's view; the system owns the pools)."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    store_dtype: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("num_layers", "num_kv_heads", "head_dim"):
            if int(getattr(self, field_name)) < 1:
                raise invalid_descriptor(f"CacheSpec.{field_name} must be at least 1")
        if not self.dtype:
            raise invalid_descriptor("CacheSpec.dtype must not be empty")

    @classmethod
    def from_kv_geometry(cls, kv: Any) -> "CacheSpec":
        """Declarative view of a runtime KV-geometry declaration (``KvCacheSpec``)."""
        store_dtype = getattr(kv, "store_dtype", None)
        return cls(
            num_layers=int(kv.num_layers),
            num_kv_heads=int(kv.num_kv_heads),
            head_dim=int(kv.head_dim),
            dtype=_dtype_name(kv.dtype),
            store_dtype=None if store_dtype is None else _dtype_name(store_dtype),
        )


@dataclass(frozen=True)
class FlowSpec:
    """Declared flow-matching semantics for latent-generation families."""

    latent_downsample: int
    prediction: str
    schedule_direction: str
    schedule_shift_domain: str

    def __post_init__(self) -> None:
        if int(self.latent_downsample) < 1:
            raise invalid_descriptor("FlowSpec.latent_downsample must be at least 1")
        for field_name in ("prediction", "schedule_direction", "schedule_shift_domain"):
            if not getattr(self, field_name):
                raise invalid_descriptor(f"FlowSpec.{field_name} must not be empty")


@dataclass(frozen=True)
class ModelSpec:
    """Immutable flat composition of one loaded model's declarative specs.

    ``revision`` is checkpoint-derived and filled at load time by the
    bootstrap; models declare every other component themselves.
    """

    architecture: str
    routes: tuple[RouteSpec, ...]
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
        seen: dict[str, str] = {}
        for route in self.routes:
            for kind in route.op_kinds:
                if kind in seen:
                    raise invalid_descriptor(
                        f"op kind {kind!r} is accepted by routes {seen[kind]!r} and {route.name!r};"
                        " an op kind binds exactly one physical route"
                    )
                seen[kind] = route.name
        if self.flow is None and any(
            OP_KIND_TABLE[kind].mode == "denoise" for kind in seen
        ):
            raise invalid_descriptor(
                "a route accepts denoise work but the ModelSpec declares no FlowSpec"
            )

    def op_kinds(self) -> frozenset[str]:
        """Union of the op kinds accepted across every declared route."""
        return frozenset(kind for route in self.routes for kind in route.op_kinds)


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


def resolved_digest(spec: ModelSpec, overlay: DeploymentOverlay) -> str:
    """The sha256 identity of one resolved spec under one deployment overlay."""
    payload = {
        "model_spec": _canonical(spec),
        "deployment_overlay": _canonical(overlay),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    """Deterministic JSON-encodable projection of a spec value tree.

    Dataclasses become type-tagged sorted mappings, enums become their values,
    and type/callable references (e.g. ``WeightSpec`` source bindings) become
    qualified names, so structurally equal declarations always digest equally.
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
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    return str(value)


def _dtype_name(value: Any) -> str:
    """Canonical dtype name for a torch dtype or dtype string."""
    return str(value).removeprefix("torch.")
