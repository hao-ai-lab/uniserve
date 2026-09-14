"""Shared diffusion execution primitives."""

from uniserve.nn.diffusion.cfg import (
    Branch,
    CfgParams,
    CfgPlan,
    CfgRecipe,
    RenormKind,
    build_flow_cfg_plan,
    combine_cfg,
    combine_text_image_cfg,
)
from uniserve.nn.diffusion.config import DiffusionConfig
from uniserve.nn.diffusion.fm_modules import ConvDecoder, FlowMatchingHead
from uniserve.nn.diffusion.integrator import CleanSampleEulerSolver, EulerSolver, euler_step
from uniserve.nn.diffusion.noise import init_latent
from uniserve.nn.diffusion.schedule import (
    ScheduleDirection,
    ScheduleShiftDomain,
    flow_match_coordinate,
    x_pred_to_velocity,
)
from uniserve.nn.diffusion.timestep import TimestepEmbedder, timestep_embedding

__all__ = [
    "Branch",
    "CfgParams",
    "CfgPlan",
    "CfgRecipe",
    "ConvDecoder",
    "FlowMatchingHead",
    "RenormKind",
    "ScheduleDirection",
    "ScheduleShiftDomain",
    "TimestepEmbedder",
    "build_flow_cfg_plan",
    "combine_cfg",
    "combine_text_image_cfg",
    "euler_step",
    "DiffusionConfig",
    "EulerSolver",
    "CleanSampleEulerSolver",
    "flow_match_coordinate",
    "init_latent",
    "timestep_embedding",
    "x_pred_to_velocity",
]
