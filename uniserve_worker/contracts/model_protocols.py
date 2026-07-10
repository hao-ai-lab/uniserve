"""Uniform model contract Protocols consumed by the shared drivers.

Structural ``Protocol`` surfaces every registered model satisfies, plus the
:class:`DenoiseContext` value object. Concrete glue lives in
``execution.model_base``; resource-rule dataclasses live in
``contracts.resource_plan``. Higher-layer type references are annotation-only
(``from __future__ import annotations``).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol, runtime_checkable

from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import DEFAULT_MAX_BATCH_OPS
from .resource_plan import ResourcePlan

if TYPE_CHECKING:
    import torch

    from ..execution.denoise_driver import TextImageDenoiseStep
    from ..runtime.compile import CompileTarget
    from ..runtime.request_state import RequestStateTable
    from .batch_policy import BatchPolicy
    from .caps import Caps
    from .forward_batch import ForwardBatch
    from .resource_plan import ResourcePlan

__all__ = [
    "UniModel",
    "TextForwardCapable",
    "EncodeCapable",
    "DenoiseCapable",
    "ModelHooks",
    "DenoiseContext",
    "verify_model_conformance",
]


@dataclass(frozen=True)
class DenoiseContext:
    """Context built once by a model and consumed by the denoise driver.

    Frozen: the driver and models build a context per step and never reassign
    its fields (mutation, where needed, happens through the ``extra``/
    ``context_kv`` mappings, not by rebinding attributes).
    """

    image_embeds: "torch.Tensor | None" = None
    thw_index: "torch.Tensor | None" = None
    vae_mask: "torch.Tensor | None" = None
    context_kv: Mapping[str, Any] = field(default_factory=dict)
    image_token_num: int = 0
    state: Any = None
    op: Mapping[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class UniModel(Protocol):
    """Minimal contract every registered model satisfies.

    Which method each ``supported_ops`` entry requires is enforced at
    registration (``models.registry._missing_capability``). Capability
    Protocols below document method groups for type-checkers; a model implements
    exactly the ones its declared ops need.
    """

    architectures: tuple[str, ...]
    supported_ops: tuple[str, ...]
    supported_controls: tuple[str, ...]
    adapter_mode: str
    resource_plan: "ResourcePlan"

    def load_weights(self, weights: "Iterable[tuple[str, torch.Tensor]]") -> object: ...
    def caps(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> "Caps": ...
    def batch_policy(self) -> "BatchPolicy": ...
    def configure_runtime(self, **kwargs: Any) -> None: ...
    def bind_data_plane_handoff(self, handoff: Any) -> None: ...
    def kv_cache_spec(self) -> Any | None: ...
    def on_new_request(self, req_id: int, state: Any) -> None: ...
    def drop_request(self, req_id: int) -> None: ...
    def maybe_publish_conditioning(self, req_id: int, sampled_token_id: int) -> str | None: ...
    def copy_blocks(self, copies: Any) -> None: ...
    def load_lora(self, lora_id: int, lora_path: str) -> None: ...
    def unload_lora(self, lora_id: int) -> None: ...
    def free_encoder(self, handles: Any) -> None: ...
    def reset_prefix_cache(self) -> None: ...


@runtime_checkable
class TextForwardCapable(Protocol):
    """Text decoder surface required by text ops (prefill/decode/target_verify)."""

    def forward(
        self,
        input_ids: "torch.Tensor",
        positions: "torch.Tensor",
        *,
        mode: str | None = None,
        input_embeds: "torch.Tensor | None" = None,
        op: Mapping[str, Any] | None = None,
        request_state: Any = None,
    ) -> "torch.Tensor": ...

    def forward_batch(
        self,
        batch: "ForwardBatch",
        *,
        request_states: "RequestStateTable",
    ) -> list[Any]: ...

    def prepare_text_attention_metadata(
        self,
        batch: "ForwardBatch",
        *,
        request_states: "RequestStateTable",
        stager: Any | None = None,
    ) -> Any: ...

    def embed_tokens(self, input_ids: "torch.Tensor") -> "torch.Tensor": ...

    def compile_targets(self) -> "tuple[CompileTarget, ...]": ...


@runtime_checkable
class EncodeCapable(Protocol):
    """Vision/VAE encode surface required by ``vit_encode``/``vae_encode`` ops."""

    def encode_image(self, pixels: Any, grid: Any) -> Any: ...

    def encode_latents(self, pixels: Any, grid: Any) -> Any: ...

    def embed_multimodal(self, input_ids: "torch.Tensor", mm_features: Any) -> "torch.Tensor": ...


@runtime_checkable
class DenoiseCapable(Protocol):
    """Flow-matching surface required by ``denoise_gen``/``commit_gen`` ops."""

    def prepare_denoise(
        self, state: Any, op: Mapping[str, Any]
    ) -> DenoiseContext | TextImageDenoiseStep: ...

    def predict_velocity(
        self,
        ctx: Any,
        t: "torch.Tensor",
        latent: "torch.Tensor",
        branch: str,
    ) -> "torch.Tensor": ...

    def decode_image(
        self,
        latent: Any,
        *,
        req_id: int | None = None,
        state: Any = None,
        op: Mapping[str, Any] | None = None,
    ) -> Any: ...


class ModelHooks:
    """Default no-op lifecycle/control hooks for runner-backed models."""

    whole_batch_forward: bool = False
    supported_ops: tuple[str, ...] = ("prefill_und",)
    supported_controls: tuple[str, ...] = ()
    adapter_mode: str = "none"
    resource_plan: "ResourcePlan" = ResourcePlan()

    def caps(
        self,
        *,
        block_size: int | None = None,
        kv_token_capacity: int | None = None,
    ) -> "Caps":
        from ..foundation.sizing import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_BATCH_OPS
        from .caps import Caps, ExecutionConstraints
        from .resource_plan import ResourcePlan

        plan = getattr(self, "resource_plan", ResourcePlan())
        resolved_block_size = DEFAULT_BLOCK_SIZE if block_size is None else int(block_size)
        default_blocks = int(getattr(self, "num_blocks", 1))
        num_blocks = (
            max(1, int(kv_token_capacity) // resolved_block_size)
            if kv_token_capacity
            else default_blocks
        )
        return Caps(
            block_size=resolved_block_size,
            num_blocks=num_blocks,
            num_layers=int(getattr(self, "num_layers", 1)),
            scratch_capacity_tokens=int(getattr(self, "scratch_capacity_tokens", 0)),
            supported_ops=tuple(self.supported_ops),
            max_latent_size=int(getattr(self, "max_latent_size", 0)),
            latent_downsample=int(getattr(self, "latent_downsample", 1)),
            bytes_per_token=int(getattr(self, "bytes_per_token", 1)),
            supported_controls=tuple(self.supported_controls),
            adapter_mode=self.adapter_mode,
            execution_constraints=ExecutionConstraints(
                max_batch_ops=int(getattr(self, "max_batch_ops", DEFAULT_MAX_BATCH_OPS))
            ),
            resource_classes=plan.classes(),
            encoder_cache_budget=getattr(self, "encoder_cache_budget", None),
        )

    def on_new_request(self, req_id: int, state: Any) -> None:
        pass

    def drop_request(self, req_id: int) -> None:
        pass

    def configure_runtime(self, **kwargs: Any) -> None:
        pass

    def bind_data_plane_handoff(self, handoff: Any) -> None:
        pass

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

    def maybe_publish_conditioning(self, req_id: int, sampled_token_id: int) -> str | None:
        return None

    def prompt_predecessor_logits(self, req_id: int) -> Any | None:
        """Return worker-resident logits that predict the next prompt token."""
        return None

    def accept_denoise_update(self, ctx: Any, latent: Any) -> None:
        state = getattr(ctx, "state", None)
        if state is not None:
            state.latent = latent

    def velocity_parameterization(self) -> str:
        return "velocity"

    def encode_image(self, pixels: Any = None, grid: Any = None, *, op: Mapping[str, Any] | None = None) -> Any:
        raise invalid_descriptor("encode-capable model must implement encode_image()")

    def encode_latents(self, pixels: Any = None, grid: Any = None, *, op: Mapping[str, Any] | None = None) -> Any:
        raise invalid_descriptor("encode-capable model must implement encode_latents()")

    def run_text_logits_batch(self, ops: list[Mapping[str, Any]]) -> list[Any]:
        return [self.run_text_logits(op) for op in ops]

    def run_text_logits(self, op: Mapping[str, Any]) -> Any:
        raise invalid_descriptor(
            "TextDriver requires a system KV pool (thin model) or a self-managing "
            "model's run_text_logits[_batch]"
        )

    def prepare_denoise(
        self, state: Any, op: Mapping[str, Any]
    ) -> DenoiseContext | TextImageDenoiseStep:
        return DenoiseContext(state=state, op=op)

    def predict_velocity(self, ctx: Any, t: Any, latent: Any, branch: str) -> Any:
        raise invalid_descriptor("diffusion model must implement predict_velocity(ctx, t, latent, branch)")

    def predict_text_image_velocity_batch(self, steps: Any, branches_by_step: Any) -> Any:
        return None

    def decode_image(self, latent: Any, *, req_id: int | None = None, state: Any = None, op: Mapping[str, Any] | None = None) -> Any:
        raise invalid_descriptor("commit-capable model must implement decode_image()")

    def kv_cache_spec(self) -> Any | None:
        return None

    def batch_policy(self) -> Any:
        from ..contracts.batch_policy import BatchPolicy

        return BatchPolicy(max_batch_ops=DEFAULT_MAX_BATCH_OPS, supports_mixed_modes=True)

_CAPABILITY_PROTOCOLS: tuple[type, ...] = (
    UniModel,
    TextForwardCapable,
    EncodeCapable,
    DenoiseCapable,
)

_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _protocol_method_names(protocol: type) -> tuple[str, ...]:
    """Names of the callable members a Protocol declares, excluding data attrs.

    A ``runtime_checkable`` Protocol carries ``__protocol_attrs__`` covering both
    data members (annotations) and methods; only the latter live in the class
    namespace as functions, so the two sets intersect to the method surface.
    """
    declared = getattr(protocol, "__protocol_attrs__", None)
    if declared is None:
        declared = {
            name for name in vars(protocol) if callable(vars(protocol).get(name))
        }
    names: list[str] = []
    for name in declared:
        member = getattr(protocol, name, None)
        if callable(member) and not isinstance(member, type):
            names.append(name)
    return tuple(sorted(names))


def _split_parameters(sig: inspect.Signature) -> tuple[list[str], set[str], bool, bool]:
    """Return ``(positional_names, keyword_names, has_var_positional, has_var_keyword)``.

    ``self`` is dropped; ``positional_names`` keeps declaration order; the keyword
    set unions positional-or-keyword and keyword-only names (both addressable by
    name).
    """
    positional: list[str] = []
    keyword: set[str] = set()
    has_var_positional = False
    has_var_keyword = False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if param.kind in _POSITIONAL_KINDS:
            positional.append(name)
            keyword.add(name)
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            keyword.add(name)
    return positional, keyword, has_var_positional, has_var_keyword


def _method_violations(
    protocol_name: str, method_name: str, declared: Any, actual: Any
) -> list[str]:
    """Compare one protocol method's signature against a model's implementation.

    Lenient by design: it flags only gross mismatches (a model that cannot accept
    a parameter the protocol declares, by either position or name). The model may
    add extra parameters of any kind, supply defaults, or absorb arguments via
    ``*args``/``**kwargs``.
    """
    try:
        declared_sig = inspect.signature(declared)
        actual_sig = inspect.signature(actual)
    except (TypeError, ValueError):
        return []
    want_pos, want_kw, _, _ = _split_parameters(declared_sig)
    have_pos, have_kw, have_var_pos, have_var_kw = _split_parameters(actual_sig)
    violations: list[str] = []
    for index, name in enumerate(want_pos):
        accepts_positionally = index < len(have_pos) or have_var_pos
        accepts_by_name = name in have_kw or have_var_kw
        if not accepts_positionally and not accepts_by_name:
            violations.append(
                f"{protocol_name}.{method_name}: model does not accept parameter "
                f"{name!r} (declared at position {index})"
            )
    for name in want_kw:
        if name in want_pos:
            continue
        if name not in have_kw and not have_var_kw:
            violations.append(
                f"{protocol_name}.{method_name}: model does not accept "
                f"keyword parameter {name!r}"
            )
    return violations


def verify_model_conformance(model: Any) -> list[str]:
    """Signature-check the capability Protocols a model claims via ``isinstance``.

    ``runtime_checkable`` Protocols only assert member existence; this adds a
    lenient signature comparison for each capability the model structurally
    satisfies, returning human-readable violation strings (empty when conformant).
    """
    violations: list[str] = []
    for protocol in _CAPABILITY_PROTOCOLS:
        if not isinstance(model, protocol):
            continue
        for method_name in _protocol_method_names(protocol):
            actual = getattr(model, method_name, None)
            if not callable(actual):
                violations.append(
                    f"{protocol.__name__}.{method_name}: model is missing this method"
                )
                continue
            declared = getattr(protocol, method_name)
            violations.extend(
                _method_violations(protocol.__name__, method_name, declared, actual)
            )
    return violations
