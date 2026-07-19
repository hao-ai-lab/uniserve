"""Zero-day diffusion probe model for conformance.

This model is intentionally tiny, but it exercises the real runner-owned
denoise path and the new ``cfg_zero_star`` guidance primitive as a model add
without being auto-discovered as a production model.
"""

from __future__ import annotations

from typing import Any

import torch

from uniserve_worker.contracts import BatchPolicy, UniModel
from uniserve_worker.contracts.caps import Caps, ExecutionConstraints
from uniserve_worker.contracts.resource_plan import LatentTokens, PerBranch, ResourcePlan


class UniServeZeroDayCfgZeroStarModel(UniModel):
    architectures = ("UniServeZeroDayCfgZeroStarModel",)
    supported_ops = ("denoise_gen",)
    supported_controls: tuple[str, ...] = ()
    adapter_mode = "none"
    resource_classes = ("kv_block", "image_latent", "scratch")
    resource_plan = ResourcePlan(
        kv_block="per_block",
        image_latent=LatentTokens(downsample=16),
        scratch=PerBranch(),
    )

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def caps(self, *, block_size: int = 256, kv_token_capacity: int | None = None) -> Caps:
        return Caps(
            block_size=block_size,
            num_blocks=max(1, (kv_token_capacity or block_size) // block_size),
            num_layers=1,
            scratch_capacity_tokens=block_size,
            supported_ops=tuple(self.supported_ops),
            max_latent_size=16,
            latent_downsample=16,
            bytes_per_token=1,
            supported_controls=tuple(self.supported_controls),
            adapter_mode=self.adapter_mode,
            execution_constraints=ExecutionConstraints(
                max_batch_ops=1024,
            ),
            resource_classes=tuple(self.resource_classes),
        )

    def batch_policy(self) -> BatchPolicy:
        return BatchPolicy(max_batch_ops=1024, supports_mixed_modes=True)

    def predict_velocity(self, ctx, t, latent, branch):
        del ctx, t
        if branch == "cond":
            value = 0.0
        elif branch == "uncond":
            value = 0.0
        elif branch.startswith("branch_"):
            value = float(branch.rsplit("_", 1)[-1])
        else:
            value = {"img_uncond": 0.0, "text_uncond": 1.0}.get(branch, 1.0)
        return torch.full_like(latent, value)


EntryClass = UniServeZeroDayCfgZeroStarModel
