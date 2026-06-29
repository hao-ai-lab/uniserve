"""Classifier-free guidance combination."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

import torch

__all__ = [
    'approx',
    'RenormKind',
    'Branch',
    'PREV_RESULT',
    'CfgRecipe',
    'CfgParams',
    'combine_cfg',
    'CfgPlan',
    'build_text_image_cfg_plan',
    'combine_text_image_cfg',
]


def approx(a: float, b: float) -> bool:
    """Float-equality test for CFG-scale control flow.

    The plan builder branches on scales being exactly 1 (CFG disabled for that
    axis) or equal to each other; ``math.isclose`` tolerates the float noise that
    arises when scales arrive as parsed floats while keeping ``== 1.0`` decisions
    intact for literal unit scales.
    """

    return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)

# RESCALE renorm blend: final guidance is ``phi * norm_matched + (1 - phi) * raw_guided``.
# phi=0.7 follows Lin et al. guidance-rescale (avoids over-exposure after norm matching).
_RESCALE_BLEND_PHI = 0.7
_RESCALE_BLEND_COMPLEMENT = 0.3


class RenormKind(str, Enum):
    NONE = "none"
    GLOBAL = "global"
    CHANNEL = "channel"
    TEXT_CHANNEL = "text_channel"
    RESCALE = "rescale"
    CFG_ZERO_STAR = "cfg_zero_star"


class Branch(str, Enum):
    """Named CFG branch the model evaluates.

    Values match the wire strings ``"cond"``, ``"text_uncond"``, and ``"img_uncond"``.
    """

    COND = "cond"
    TEXT_UNCOND = "text_uncond"
    IMG_UNCOND = "img_uncond"


class _PrevResult(Enum):
    """Typed sentinel naming the running result of the previous combine op.

    Used as the first input of a chained op (the nested IMAGE_OVER_TEXT +
    text_channel recipe) in place of a magic branch-name string.
    """

    PREV = "prev"


PREV_RESULT = _PrevResult.PREV


class CfgRecipe(Enum):
    """How text and image CFG deltas are combined for a model.

    ``ADDITIVE_DELTAS`` adds text and image guidance deltas around the doubly-uncond
    base. ``IMAGE_OVER_TEXT`` applies image guidance on top of the text-guided
    prediction and enables the ``text_channel`` renorm special-case.
    """

    ADDITIVE_DELTAS = "additive_deltas"
    IMAGE_OVER_TEXT = "image_over_text"

    @classmethod
    def coerce(cls, value: "CfgRecipe | bool | str") -> "CfgRecipe":
        # Accept bool or ``image_scale_applies_to_text`` strings: True -> IMAGE_OVER_TEXT.
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.IMAGE_OVER_TEXT if value else cls.ADDITIVE_DELTAS
        return cls(str(value))


@dataclass(frozen=True)
class CfgParams:
    branch_count: int = 1
    scales: tuple[float, ...] = ()
    renorm: RenormKind = RenormKind.NONE
    renorm_min: float = 0.0

    def __post_init__(self) -> None:
        if self.branch_count < 1:
            raise ValueError("branch_count must be >= 1")
        if self.scales and len(self.scales) not in {self.branch_count - 1, self.branch_count}:
            raise ValueError("scales must be empty, branch_count-1, or branch_count long")

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "CfgParams":
        if raw is None:
            return CfgParams()
        # ``renorm_type`` is canonical; bare ``renorm`` is accepted as a synonym.
        renorm = raw.get("renorm_type", raw.get("renorm", RenormKind.NONE))
        if not isinstance(renorm, RenormKind):
            renorm = RenormKind(str(renorm))
        scales = raw.get("scales")
        if scales is None:
            branch_count = int(raw.get("branch_count", 1))
            if branch_count <= 1:
                scales = ()
            elif branch_count == 2:
                scales = (float(raw.get("text_scale", 1.0)),)
            else:
                scales = (
                    float(raw.get("text_scale", 1.0)),
                    float(raw.get("img_scale", 1.0)),
                )
        return CfgParams(
            branch_count=int(raw.get("branch_count", 1)),
            scales=tuple(float(s) for s in scales),
            renorm=renorm,
            renorm_min=float(raw.get("renorm_min", 0.0)),
        )


def combine_cfg(branch_velocities: torch.Tensor | list[torch.Tensor], params: CfgParams) -> torch.Tensor:
    """Combine branch velocities into one guided velocity.

    Branch 0 is the unconditional/base branch.  If ``scales`` is
    ``branch_count - 1`` long, each scale is applied as
    ``base + scale_i * (branch_i - base)``.  If it is ``branch_count`` long, the
    values are direct branch weights.
    """

    branches = torch.stack(branch_velocities, dim=0) if isinstance(branch_velocities, list) else branch_velocities
    if branches.shape[0] != params.branch_count:
        raise ValueError(f"got {branches.shape[0]} branches, expected {params.branch_count}")
    if params.branch_count == 1:
        return branches[0]
    if len(params.scales) == params.branch_count:
        weights = torch.tensor(params.scales, device=branches.device, dtype=branches.dtype)
        view_shape = (params.branch_count,) + (1,) * (branches.ndim - 1)
        guided = (branches * weights.view(view_shape)).sum(dim=0)
    else:
        scales = params.scales or (1.0,) * (params.branch_count - 1)
        guided = branches[0]
        for idx, scale in enumerate(scales, start=1):
            guided = guided + float(scale) * (branches[idx] - branches[0])
    return _renorm(guided, branches, params)


@dataclass(frozen=True)
class _CfgCombineOp:
    """One combine_cfg invocation in a plan, over named branch outputs.

    ``inputs`` names the branch tensors fed to ``combine_cfg`` in order (branch 0
    is the base). The first input may be the sentinel ``PREV_RESULT`` to chain
    the output of the previous op (used only by the nested IMAGE_OVER_TEXT +
    text_channel recipe).
    """

    inputs: tuple[Branch | _PrevResult, ...]
    scales: tuple[float, ...]
    renorm: RenormKind
    renorm_min: float


@dataclass(frozen=True)
class CfgPlan:
    """Unified branch selection + combination for text/image CFG.

    ``branches`` is the exact ordered set of branches the driver must evaluate.
    ``ops`` is the matching sequence of combine steps; an empty ``ops`` means the
    conditioned branch is returned unchanged. Selection and combination are
    derived together so a "selector said N branches, combiner wanted M"
    mismatch is unrepresentable.
    """

    branches: tuple[Branch, ...]
    ops: tuple[_CfgCombineOp, ...] = ()

    def combine(self, outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not self.ops:
            return outputs[Branch.COND]
        result: torch.Tensor | None = None
        for op in self.ops:
            tensors = []
            for name in op.inputs:
                if name is PREV_RESULT:
                    if result is None:
                        raise RuntimeError("cfg plan op references a missing prior result")
                    tensors.append(result)
                    continue
                value = outputs.get(name)
                if value is None:
                    raise RuntimeError(f"required CFG branch {name.value!r} is missing")
                tensors.append(value)
            result = combine_cfg(
                tensors,
                CfgParams(
                    branch_count=len(tensors),
                    scales=op.scales,
                    renorm=op.renorm,
                    renorm_min=op.renorm_min,
                ),
            )
        assert result is not None
        return result


def build_text_image_cfg_plan(
    *,
    cfg_text_scale: float,
    cfg_img_scale: float,
    recipe: CfgRecipe,
    renorm: str | RenormKind = RenormKind.NONE,
    renorm_min: float = 0.0,
    use_cfg: bool = True,
) -> CfgPlan:
    """Derive the branch set and combination plan for one text/image CFG step.

    The conditioned branch is always evaluated. ``recipe`` selects additive-delta vs.
    image-over-text combination; ``use_cfg`` gates cfg-interval application.
    """

    renorm_kind = renorm if isinstance(renorm, RenormKind) else RenormKind(str(renorm))
    renorm_min = float(renorm_min)
    applies_to_text = recipe is CfgRecipe.IMAGE_OVER_TEXT
    text_off = approx(cfg_text_scale, 1.0)
    img_off = approx(cfg_img_scale, 1.0)

    # cond is always evaluated; the cfg-interval gate and the both-scales==1
    # short-circuit collapse to the conditioned prediction unchanged.
    if not use_cfg or (text_off and img_off):
        return CfgPlan(branches=(Branch.COND,))

    # Image guidance disabled: two-branch text guidance only.
    if img_off:
        return _two_branch_text_plan(
            cfg_text_scale=cfg_text_scale,
            renorm_kind=renorm_kind,
            renorm_min=renorm_min,
        )

    # IMAGE_OVER_TEXT with active text guidance evaluates all three branches and
    # layers image guidance over the text-guided prediction. text_channel renorm
    # applies the channel match to the text stage, then a plain image stage.
    if applies_to_text and not text_off:
        if renorm_kind == RenormKind.TEXT_CHANNEL:
            return _image_over_text_channel_plan(
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                renorm_min=renorm_min,
            )
        return _three_branch_plan(cfg_text_scale, cfg_img_scale, applies_to_text, renorm_kind, renorm_min)

    # Two-branch image guidance when text guidance is off or scales coincide.
    # text_uncond is skipped; scale is cfg_img_scale for IMAGE_OVER_TEXT else cfg_text_scale.
    if approx(cfg_text_scale, cfg_img_scale) or text_off:
        scale = cfg_img_scale if applies_to_text else cfg_text_scale
        return _two_branch_image_plan(
            scale=scale,
            renorm_kind=renorm_kind,
            renorm_min=renorm_min,
        )

    return _three_branch_plan(cfg_text_scale, cfg_img_scale, applies_to_text, renorm_kind, renorm_min)


def _two_branch_text_plan(
    *,
    cfg_text_scale: float,
    renorm_kind: RenormKind,
    renorm_min: float,
) -> CfgPlan:
    return CfgPlan(
        branches=(Branch.COND, Branch.TEXT_UNCOND),
        ops=(
            _CfgCombineOp(
                inputs=(Branch.TEXT_UNCOND, Branch.COND),
                scales=(float(cfg_text_scale),),
                renorm=renorm_kind,
                renorm_min=renorm_min,
            ),
        ),
    )


def _two_branch_image_plan(
    *,
    scale: float,
    renorm_kind: RenormKind,
    renorm_min: float,
) -> CfgPlan:
    return CfgPlan(
        branches=(Branch.COND, Branch.IMG_UNCOND),
        ops=(
            _CfgCombineOp(
                inputs=(Branch.IMG_UNCOND, Branch.COND),
                scales=(float(scale),),
                renorm=renorm_kind,
                renorm_min=renorm_min,
            ),
        ),
    )


def _image_over_text_channel_plan(
    *,
    cfg_text_scale: float,
    cfg_img_scale: float,
    renorm_min: float,
) -> CfgPlan:
    ops = [
        _CfgCombineOp(
            inputs=(Branch.TEXT_UNCOND, Branch.COND),
            scales=(float(cfg_text_scale),),
            renorm=RenormKind.TEXT_CHANNEL,
            renorm_min=renorm_min,
        )
    ]
    if cfg_img_scale > 1.0:
        ops.append(
            _CfgCombineOp(
                inputs=(Branch.IMG_UNCOND, PREV_RESULT),
                scales=(float(cfg_img_scale),),
                renorm=RenormKind.NONE,
                renorm_min=0.0,
            )
        )
    return CfgPlan(branches=(Branch.COND, Branch.TEXT_UNCOND, Branch.IMG_UNCOND), ops=tuple(ops))


def _three_branch_plan(
    cfg_text_scale: float,
    cfg_img_scale: float,
    applies_to_text: bool,
    renorm_kind: RenormKind,
    renorm_min: float,
) -> CfgPlan:
    if applies_to_text:
        scales = (
            1.0 - float(cfg_img_scale),
            float(cfg_img_scale) * (1.0 - float(cfg_text_scale)),
            float(cfg_img_scale) * float(cfg_text_scale),
        )
    else:
        scales = (
            1.0 - float(cfg_img_scale),
            float(cfg_img_scale) - float(cfg_text_scale),
            float(cfg_text_scale),
        )
    return CfgPlan(
        branches=(Branch.COND, Branch.TEXT_UNCOND, Branch.IMG_UNCOND),
        ops=(
            _CfgCombineOp(
                inputs=(Branch.IMG_UNCOND, Branch.TEXT_UNCOND, Branch.COND),
                scales=scales,
                renorm=renorm_kind,
                renorm_min=renorm_min,
            ),
        ),
    )


def combine_text_image_cfg(
    out_cond: torch.Tensor,
    out_text_uncond: torch.Tensor | None,
    out_img_uncond: torch.Tensor | None,
    *,
    cfg_text_scale: float,
    cfg_img_scale: float,
    renorm: str | RenormKind = RenormKind.NONE,
    renorm_min: float = 0.0,
    image_scale_applies_to_text: "CfgRecipe | bool",
) -> torch.Tensor:
    """Combine three branch predictions using a :class:`CfgPlan`.

    ``image_scale_applies_to_text`` selects the recipe (bool values are coerced).
    """

    recipe = CfgRecipe.coerce(image_scale_applies_to_text)
    plan = build_text_image_cfg_plan(
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        recipe=recipe,
        renorm=renorm,
        renorm_min=renorm_min,
    )
    outputs: dict[str, torch.Tensor] = {Branch.COND: out_cond}
    if out_text_uncond is not None:
        outputs[Branch.TEXT_UNCOND] = out_text_uncond
    if out_img_uncond is not None:
        outputs[Branch.IMG_UNCOND] = out_img_uncond
    return plan.combine(outputs)


def _renorm_global(guided: torch.Tensor, ref: torch.Tensor, params: CfgParams, eps: float) -> torch.Tensor:
    dims = tuple(range(1, guided.ndim)) if guided.ndim >= 3 else tuple(range(guided.ndim))
    return _match_norm(guided, ref, dims=dims, minimum=params.renorm_min, eps=eps)


def _renorm_channel(guided: torch.Tensor, ref: torch.Tensor, params: CfgParams, eps: float) -> torch.Tensor:
    dims = (guided.ndim - 1,)
    return _match_norm(guided, ref, dims=dims, minimum=params.renorm_min, eps=eps)


def _renorm_rescale(guided: torch.Tensor, ref: torch.Tensor, params: CfgParams, eps: float) -> torch.Tensor:
    dims = (guided.ndim - 1,)
    matched = _match_norm(guided, ref, dims=dims, minimum=params.renorm_min, eps=eps)
    return _RESCALE_BLEND_PHI * matched + _RESCALE_BLEND_COMPLEMENT * guided


def _renorm_cfg_zero_star(guided: torch.Tensor, ref: torch.Tensor, params: CfgParams, eps: float) -> torch.Tensor:
    del ref, params, eps
    return guided - guided.mean(dim=tuple(range(1, guided.ndim)), keepdim=True)


# RenormKind -> handler. NONE is short-circuited before ``ref``/``eps`` are
# computed (it must not reference the base branch), so it is not in the table.
_RENORM_HANDLERS: dict[
    RenormKind, Callable[[torch.Tensor, torch.Tensor, "CfgParams", float], torch.Tensor]
] = {
    RenormKind.GLOBAL: _renorm_global,
    RenormKind.CHANNEL: _renorm_channel,
    RenormKind.TEXT_CHANNEL: _renorm_channel,
    RenormKind.RESCALE: _renorm_rescale,
    RenormKind.CFG_ZERO_STAR: _renorm_cfg_zero_star,
}


def _renorm(guided: torch.Tensor, branches: torch.Tensor, params: CfgParams) -> torch.Tensor:
    if params.renorm == RenormKind.NONE:
        return guided
    handler = _RENORM_HANDLERS.get(params.renorm)
    if handler is None:
        raise ValueError(f"unknown renorm kind {params.renorm!r}")
    ref = branches[-1]
    eps = torch.finfo(guided.dtype).eps if guided.is_floating_point() else 1e-6
    return handler(guided, ref, params, eps)


def _match_norm(
    guided: torch.Tensor,
    ref: torch.Tensor,
    *,
    dims: tuple[int, ...],
    minimum: float,
    eps: float,
) -> torch.Tensor:
    guided_norm = guided.float().norm(dim=dims, keepdim=True).clamp_min(eps)
    ref_norm = ref.float().norm(dim=dims, keepdim=True).clamp_min(minimum)
    scale = (ref_norm / guided_norm).clamp(max=1.0)
    return guided * scale.to(dtype=guided.dtype)
