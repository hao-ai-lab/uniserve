"""Single nominal contract for models executed by the shared runtime."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from ..foundation.errors import invalid_descriptor
from ..foundation.sizing import DEFAULT_MAX_BATCH_OPS
from .resource_plan import ResourcePlan

if TYPE_CHECKING:
    import torch

    from uniserve_worker.execution.flow import PreparedFlowStep
    from uniserve_worker.execution.segment import SegmentAdapter, SegmentExecutor
    from uniserve_worker.runtime.residency import ResidencyManager

    from .caps import Caps
    from .forward_batch import ForwardBatch

__all__ = [
    "UniModel",
    "FlowContext",
]


@dataclass(frozen=True)
class FlowContext:
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


class UniModel(ABC):
    """Nominal model contract with shared lifecycle and execution defaults."""

    architectures: tuple[str, ...] = ()
    supported_ops: tuple[str, ...] = ("prefill_und",)
    supported_controls: tuple[str, ...] = ()
    adapter_mode: str = "none"
    resource_plan: "ResourcePlan" = ResourcePlan()
    # System-provisioned segment execution; the worker builds the executor
    # over the model-declared family adapter and binds it here.
    segment_executor: "SegmentExecutor | None" = None
    # System-provisioned physical residency; the worker builds it from the
    # model-declared geometry and binds it here.
    residency: "ResidencyManager | None" = None

    def forward(self, batch: "ForwardBatch") -> Any:
        """Execute a complete homogeneous batch through runner-bound runtime services."""

        from .forward_context import get_forward_context

        ctx = get_forward_context()
        execute = ctx.default_model_forward
        if not callable(execute):
            raise invalid_descriptor("model forward requires a runner-bound execution context")
        return execute(self, batch)

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

    def configure_runtime(self, **kwargs: Any) -> None:
        pass

    def bind_data_plane_handoff(self, handoff: Any) -> None:
        pass

    def copy_blocks(self, copies: Any) -> None:
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

    def accept_flow_update(self, ctx: Any, latent: Any) -> None:
        state = getattr(ctx, "state", None)
        if state is not None:
            state.latent = latent

    def velocity_parameterization(self) -> str:
        return "velocity"

    def encode_image(
        self, pixels: Any = None, grid: Any = None, *, op: Mapping[str, Any] | None = None
    ) -> Any:
        raise invalid_descriptor("encode-capable model must implement encode_image()")

    def encode_latents(
        self, pixels: Any = None, grid: Any = None, *, op: Mapping[str, Any] | None = None
    ) -> Any:
        raise invalid_descriptor("encode-capable model must implement encode_latents()")

    def run_text_logits_batch(self, ops: list[Mapping[str, Any]]) -> list[Any]:
        return [self.run_text_logits(op) for op in ops]

    def run_text_logits(self, op: Mapping[str, Any]) -> Any:
        raise invalid_descriptor(
            "text execution requires a system KV pool or a self-managing model's "
            "run_text_logits[_batch]"
        )

    def prepare_flow(self, state: Any, op: Mapping[str, Any]) -> FlowContext | PreparedFlowStep:
        return FlowContext(state=state, op=op)

    def predict_velocity(self, ctx: Any, t: Any, latent: Any, branch: str) -> Any:
        raise invalid_descriptor(
            "diffusion model must implement predict_velocity(ctx, t, latent, branch)"
        )

    def predict_flow_velocity_batch(self, steps: Any, branches_by_step: Any) -> Any:
        return None

    def decode_image(
        self,
        latent: Any,
        *,
        req_id: int | None = None,
        state: Any = None,
        op: Mapping[str, Any] | None = None,
    ) -> Any:
        raise invalid_descriptor("commit-capable model must implement decode_image()")

    def kv_cache_spec(self) -> Any | None:
        return None

    def gen_residency_spec(self) -> Any | None:
        return None

    def segment_adapter(self) -> "SegmentAdapter | None":
        return None

    def batch_policy(self) -> Any:
        from .forward_batch import BatchPolicy

        return BatchPolicy(max_batch_ops=DEFAULT_MAX_BATCH_OPS, supports_mixed_modes=True)
