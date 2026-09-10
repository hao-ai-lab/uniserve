"""Rotary embedding and fused QK normalization-plus-RoPE dispatch.

The module supports packed GPT-NeoX and interleaved rotary conventions,
HF-shaped query/key tensors, and multi-axis video layouts whose adjacent axes
may share one RMS normalization group. Specialized Triton strategies preserve
those normalization groups and rotary coordinates. Eager tensor execution handles
the general layout.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ..backends.triton import triton_available
from .core import Dispatcher, Operator
from .qk_plan import FusedStrategy, QKNormRopePlan
from .requests import (
    MultiAxisQKNormRopeReq,
    PackedRopeReq,
    QKNormRopeRequest,
)
from .rms import eager_rms_norm, run_qk_rms_norm
from .rope_kernels import (
    _EagerPackedRope,
    _TritonPackedRope,
    can_run_triton_qk_rms_norm_rope,
    can_run_triton_qk_rms_norm_rope_inplace,
    triton,
    triton_qk_rms_norm_rope_inplace,
    try_triton_qk_multi_axis_rms_norm_rope,
    try_triton_qk_rms_norm_rope,
    try_triton_qk_split_rms_norm_rope,
)

_triton_packed = _TritonPackedRope()
_eager_packed = _EagerPackedRope()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply the GPT-NeoX quarter-turn by exchanging and negating tensor halves."""

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotation: str = "neox",
) -> torch.Tensor:
    """Apply packed rotary embedding and return a newly allocated tensor.

    ``x`` has layout ``[seq, heads, dim]`` and ``cos``/``sin`` have layout
    ``[seq, dim/2]``. ``rotation="neox"`` pairs the two contiguous dimension
    halves; ``rotation="interleaved"`` pairs adjacent even and odd features.
    The input tensor is never modified.
    """

    if rotation == "interleaved":
        # Interleaved rotation uses adjacent feature pairs, so each factor row
        # broadcasts across heads against the even and odd slices.
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out

    if rotation != "neox":
        raise ValueError(f"unknown RoPE rotation convention {rotation!r}")

    # NeoX layout is delegated so the dispatcher can select a compatible
    # Triton implementation or the portable tensor provider.
    return rope_dispatcher().run(PackedRopeReq(x, cos, sin))


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply pre-indexed HF-style rotary factors to query and key tensors.

    ``cos`` and ``sin`` are expanded at ``unsqueeze_dim`` to broadcast over the
    query/key head layout. ``position_ids`` is accepted for HF call-site
    compatibility; the supplied factor tensors already encode those positions.
    """

    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return _apply_rotary_full_dim(q, cos, sin), _apply_rotary_full_dim(k, cos, sin)


def _apply_rotary_full_dim(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply full-width half-rotation factors, preserving an odd trailing feature."""

    dim = x.shape[-1]
    if cos.shape[-1] < dim or sin.shape[-1] < dim:
        raise ValueError(
            f"rotary cos/sin dim {cos.shape[-1]}/{sin.shape[-1]} is smaller than tensor dim {dim}"
        )
    cos = cos[..., :dim]
    sin = sin[..., :dim]
    rotary_dim = dim - (dim % 2)
    if rotary_dim == 0:
        return x

    # Pair the contiguous halves of the largest even prefix. Any odd final
    # feature is outside the rotary domain and copied unchanged.
    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x1 * cos[..., :half] - x2 * sin[..., :half]
    out[..., half:rotary_dim] = x2 * cos[..., half:rotary_dim] + x1 * sin[..., half:rotary_dim]
    if rotary_dim < dim:
        out[..., rotary_dim:] = x[..., rotary_dim:]
    return out


class TritonPackedRope(Operator):
    """Triton provider for packed NeoX rotary embedding."""

    def __init__(self) -> None:
        """Register the Triton packed-RoPE provider identity."""

        super().__init__("triton", "rope")

    def can_run(self, req: PackedRopeReq) -> bool:
        """Accept packed tensors matching the Triton launch contract."""

        return _triton_packed.is_eligible(req.x, req.cos, req.sin)

    def run(self, req: PackedRopeReq) -> torch.Tensor:
        """Launch the Triton packed-RoPE implementation."""

        return _triton_packed.run(req.x, req.cos, req.sin)


class EagerPackedRope(Operator):
    """Portable tensor provider for packed NeoX rotary embedding."""

    def __init__(self) -> None:
        """Register the eager packed-RoPE provider identity."""

        super().__init__("eager", "rope")

    def can_run(self, req: PackedRopeReq) -> bool:
        """Accept every request handled by the tensor implementation."""

        del req
        return True

    def run(self, req: PackedRopeReq) -> torch.Tensor:
        """Apply the eager packed-RoPE implementation."""

        return _eager_packed.run(req.x, req.cos, req.sin)


def _same_token_layout(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Check that Q and K expose the same batch and token axes."""

    if q.ndim != k.ndim:
        return False
    if q.ndim == 3:
        return int(q.shape[0]) == int(k.shape[0])
    if q.ndim == 4:
        return int(q.shape[0]) == int(k.shape[0]) and int(q.shape[2]) == int(k.shape[2])
    return False


def _flattened_token_count(x: torch.Tensor) -> int:
    """Return the token count after flattening an optional batch dimension."""

    if x.ndim == 3:
        return int(x.shape[0])
    if x.ndim == 4:
        return int(x.shape[0]) * int(x.shape[2])
    return 0


def _can_repeat_rope(cos: torch.Tensor, tokens: int) -> bool:
    """Check whether a rotary table can repeat or truncate evenly to ``tokens``."""

    return (
        cos.ndim == 2
        and int(cos.shape[0]) > 0
        and (tokens % int(cos.shape[0]) == 0 or int(cos.shape[0]) % tokens == 0)
    )


def _align_rope_table(table: torch.Tensor, tokens: int) -> torch.Tensor:
    """Make a contiguous rotary table with exactly ``tokens`` rows."""

    rows = int(table.shape[0])
    if rows == tokens:
        return table.contiguous()
    if rows < tokens and tokens % rows == 0:
        return table.repeat(tokens // rows, 1).contiguous()
    if rows > tokens and rows % tokens == 0:
        return table[:tokens].contiguous()
    raise RuntimeError(f"cannot align RoPE table length {rows} to token count {tokens}")


def _flatten_heads(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], bool]:
    """Convert BHSD input to packed token/head rows while recording its shape."""

    if x.ndim == 3:
        return x, tuple(x.shape), False
    if x.ndim != 4:
        raise RuntimeError("triton multi-axis qk_norm_rope requires 3D or 4D q/k tensors")

    # BHSD becomes contiguous BSHD before batch and sequence are flattened, so
    # each token/head vector matches the rank-three Triton kernel contract.
    flat = (
        x.permute(0, 2, 1, 3).contiguous().reshape(x.shape[0] * x.shape[2], x.shape[1], x.shape[3])
    )
    return flat, tuple(x.shape), True


def _unflatten_heads(x: torch.Tensor, shape: tuple[int, ...], was_flattened: bool) -> torch.Tensor:
    """Restore a flattened token/head tensor to its recorded BHSD shape."""

    if not was_flattened:
        return x
    batch, heads, seq_len, dim = shape
    return x.reshape(batch, seq_len, heads, dim).permute(0, 2, 1, 3).contiguous()


def _axis_group(
    x: torch.Tensor,
    axis_dims: tuple[int, ...],
    start_axis: int,
    end_axis: int,
) -> torch.Tensor:
    """Slice the final dimension spanning axes ``[start_axis, end_axis)``."""

    start = sum(int(dim) for dim in axis_dims[:start_axis])
    width = sum(int(dim) for dim in axis_dims[start_axis:end_axis])
    return x[..., start : start + width]


def _flatten_axis_group(
    x: torch.Tensor,
    axis_dims: tuple[int, ...],
    start_axis: int,
    end_axis: int,
) -> tuple[torch.Tensor, tuple[int, ...], bool]:
    """Slice an axis group and flatten its optional batch/token dimensions."""

    return _flatten_heads(_axis_group(x, axis_dims, start_axis, end_axis))


def _rope_table_shape_matches(
    cos: torch.Tensor, sin: torch.Tensor, tokens: int, axis_dim: int
) -> bool:
    """Check paired half-width rotary tables for one declared axis."""

    return (
        cos.ndim == 2
        and sin.shape == cos.shape
        and int(cos.shape[-1]) * 2 == int(axis_dim)
        and _can_repeat_rope(cos, int(tokens))
    )


def _qk_common_eligible(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Check shared device, dtype, gradient, and nonempty-head Triton conditions."""

    q_heads = int(q.shape[1]) if q.ndim in (3, 4) else 0
    k_heads = int(k.shape[1]) if k.ndim in (3, 4) else 0
    return not (
        triton is None
        or torch.is_grad_enabled()
        or not triton_available(q.device)
        or q.device != k.device
        or q.dtype != k.dtype
        or q_heads <= 0
        or k_heads <= 0
    )


def _qk_norm_group_eligible(
    q: torch.Tensor,
    group_dim: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> bool:
    """Check normalization weights and width for one QK axis group."""

    group_dim = int(group_dim)
    return not (
        group_dim <= 0
        or group_dim > 1024
        or not q_weight.is_cuda
        or not k_weight.is_cuda
        or q_weight.device != q.device
        or k_weight.device != q.device
        or int(q_weight.numel()) != group_dim
        or int(k_weight.numel()) != group_dim
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
    )


def _qk_norm_rope_group_eligible(
    q: torch.Tensor,
    group_dim: int,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Extend group eligibility with even-width, device-local rotary factors."""

    if not _qk_norm_group_eligible(q, group_dim, q_weight, k_weight):
        return False
    group_dim = int(group_dim)
    return not (
        group_dim % 2 != 0
        or cos.device != q.device
        or sin.device != q.device
        or not cos.is_cuda
        or not sin.is_cuda
    )


def _packed_rope_axis_eligible(
    q: torch.Tensor,
    axis_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Check one packed rotary axis against Triton device and width constraints."""

    axis_dim = int(axis_dim)
    heads = int(q.shape[1]) if q.ndim in (3, 4) else 0
    return not (
        triton is None
        or torch.is_grad_enabled()
        or not triton_available(q.device)
        or axis_dim <= 0
        or axis_dim % 2 != 0
        or heads <= 0
        or not cos.is_cuda
        or not sin.is_cuda
        or cos.device != q.device
        or sin.device != q.device
    )


def _apply_rope_axis(
    q_normed: torch.Tensor,
    k_normed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one rotary axis across packed or BHSD Q/K tensor layouts."""

    # Rank-four inputs may receive rank-two tables for either every flattened
    # batch/token row or one repeatable sequence. Reshape both forms to
    # broadcast over the head axis without duplicating Q/K tensors.
    if q_normed.ndim == 4 and cos.ndim == 2:
        dim = int(q_normed.shape[-1])
        if dim % 2:
            return apply_rotary_pos_emb(q_normed, k_normed, cos, sin, None, unsqueeze_dim)

        batch = int(q_normed.shape[0])
        tokens = int(q_normed.shape[2])
        rows = int(cos.shape[0])
        if rows == batch * tokens:
            cos = cos.contiguous().view(batch, tokens, cos.shape[1])
            sin = sin.contiguous().view(batch, tokens, sin.shape[1])
            table_shape = (batch, 1, tokens)
        else:
            cos = _align_rope_table(cos, tokens)
            sin = _align_rope_table(sin, tokens)
            table_shape = (1, 1, tokens)

        # Full-width tables carry one phase per feature half after duplication;
        # shorter tables carry one phase per NeoX feature pair.
        if int(cos.shape[-1]) >= dim:
            cos_view = cos[..., :dim].view(*table_shape, dim)
            sin_view = sin[..., :dim].view(*table_shape, dim)
            half = dim // 2

            def rotate_full(x: torch.Tensor) -> torch.Tensor:
                """Rotate contiguous halves using duplicated full-width factors."""

                out = torch.empty_like(x)
                out[..., :half] = (
                    x[..., :half] * cos_view[..., :half] - x[..., half:dim] * sin_view[..., :half]
                )
                out[..., half:dim] = (
                    x[..., half:dim] * cos_view[..., half:dim]
                    + x[..., :half] * sin_view[..., half:dim]
                )
                if dim % 2:
                    out[..., -1:] = x[..., -1:]
                return out

            return rotate_full(q_normed), rotate_full(k_normed)

        cos_view = cos.view(*table_shape, cos.shape[-1])
        sin_view = sin.view(*table_shape, sin.shape[-1])
        half = dim // 2

        def rotate_packed(x: torch.Tensor) -> torch.Tensor:
            """Rotate contiguous halves using one factor per feature pair."""

            out = torch.empty_like(x)
            out[..., :half] = x[..., :half] * cos_view - x[..., half:] * sin_view
            out[..., half:] = x[..., half:] * cos_view + x[..., :half] * sin_view
            return out

        return rotate_packed(q_normed), rotate_packed(k_normed)

    # Other layouts use the public full-width or packed implementation based
    # on the table's feature coverage.
    if int(cos.shape[-1]) >= int(q_normed.shape[-1]):
        return apply_rotary_pos_emb(q_normed, k_normed, cos, sin, None, unsqueeze_dim)
    return apply_rotary_emb(q_normed, cos, sin), apply_rotary_emb(k_normed, cos, sin)


class TritonQKNormRope(Operator):
    """Triton provider for eligible QK RMSNorm-plus-RoPE layouts."""

    def __init__(self) -> None:
        """Register the Triton fused QK provider identity."""

        super().__init__("triton", "qk_norm_rope")

    def can_run(self, req: QKNormRopeRequest) -> bool:
        """Check single- or multi-axis geometry against the fused kernel family."""

        if isinstance(req, MultiAxisQKNormRopeReq):
            return self._can_run_multi_axis(req)

        # The in-place partial-rotation kernel has a distinct shape contract;
        # both paths require factor tables that are already position-indexed.
        if req.in_place:
            return (
                req.position_ids is None
                and req.unsqueeze_dim == 1
                and can_run_triton_qk_rms_norm_rope_inplace(
                    req.q,
                    req.k,
                    req.q_weight,
                    req.k_weight,
                    req.cos,
                    req.sin,
                )
            )
        if req.position_ids is not None or req.unsqueeze_dim != 1:
            return False

        return bool(
            can_run_triton_qk_rms_norm_rope(
                req.q,
                req.k,
                req.q_weight,
                req.k_weight,
                req.cos,
                req.sin,
                req.eps,
                req.eps,
            )
        )

    def run(self, req: QKNormRopeRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the eligible fused strategy and return normalized, rotated Q/K."""

        if isinstance(req, MultiAxisQKNormRopeReq):
            return self._run_multi_axis(req)

        if req.in_place:
            triton_qk_rms_norm_rope_inplace(
                req.q,
                req.k,
                req.q_weight,
                req.k_weight,
                req.cos,
                req.sin,
                req.eps,
            )
            return req.q, req.k

        out = try_triton_qk_rms_norm_rope(
            req.q, req.k, req.q_weight, req.k_weight, req.cos, req.sin, req.eps, req.eps
        )
        if out is None:
            raise RuntimeError("triton qk_norm_rope became ineligible")
        return out

    def _try_identity_tail(
        self, req: MultiAxisQKNormRopeReq, plan: QKNormRopePlan
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Try one launch for a rotated head group and an identity tail group."""

        if plan.strategy is not FusedStrategy.IDENTITY_TAIL:
            return None
        if req.q.ndim != 3 or req.k.ndim != 3:
            return None

        # Only the rotated head group needs factor tables; align them to the
        # packed token count before invoking the split head/tail kernel.
        tokens = int(req.q.shape[0])
        cos0, sin0 = plan.cos_tables[0], plan.sin_tables[0]
        if not (_can_repeat_rope(cos0, tokens) and _can_repeat_rope(sin0, tokens)):
            return None
        return try_triton_qk_split_rms_norm_rope(
            req.q,
            req.k,
            plan.q_weights[0],
            plan.q_weights[1],
            plan.k_weights[0],
            plan.k_weights[1],
            _align_rope_table(cos0, tokens),
            _align_rope_table(sin0, tokens),
            req.eps,
            req.eps,
            rope_dim=int(plan.axis_dims[0]),
        )

    def _try_rotated_tail(
        self, req: MultiAxisQKNormRopeReq, plan: QKNormRopePlan
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Try one launch for a head axis and two rotated shared-tail axes."""

        if plan.strategy is not FusedStrategy.ROTATED_SHARED_TAIL:
            return None
        if req.q.ndim != 3 or req.k.ndim != 3:
            return None
        tokens = int(req.q.shape[0])
        cos_tables: list[torch.Tensor] = []
        sin_tables: list[torch.Tensor] = []

        # The specialized three-axis kernel consumes one contiguous table per
        # axis with exactly one row per packed token.
        for axis in range(3):
            cos_a, sin_a = plan.cos_tables[axis], plan.sin_tables[axis]
            if not (_can_repeat_rope(cos_a, tokens) and _can_repeat_rope(sin_a, tokens)):
                return None
            cos_tables.append(_align_rope_table(cos_a, tokens))
            sin_tables.append(_align_rope_table(sin_a, tokens))
        return try_triton_qk_multi_axis_rms_norm_rope(
            req.q,
            req.k,
            plan.q_weights[0],
            plan.q_weights[1],
            plan.k_weights[0],
            plan.k_weights[1],
            (cos_tables[0], cos_tables[1], cos_tables[2]),
            (sin_tables[0], sin_tables[1], sin_tables[2]),
            req.eps,
            req.eps,
            axis_dims=(int(plan.axis_dims[0]), int(plan.axis_dims[1]), int(plan.axis_dims[2])),
        )

    def _can_run_multi_axis(self, req: MultiAxisQKNormRopeReq) -> bool:
        """Validate every normalization group and rotary axis for Triton execution."""

        if req.position_ids is not None or req.unsqueeze_dim != 1:
            return False

        try:
            plan = QKNormRopePlan.from_request(req)
        except RuntimeError:
            return False
        if req.q.ndim not in (3, 4) or req.k.ndim not in (3, 4):
            return False
        if not _same_token_layout(req.q, req.k):
            return False
        if not (req.q.is_cuda and req.k.is_cuda):
            return False
        tokens = _flattened_token_count(req.q)
        if tokens <= 0 or not _qk_common_eligible(req.q, req.k):
            return False

        # A single-axis group can fuse normalization and rotation. Shared
        # groups normalize together, then rotate each constituent axis through
        # the packed kernel, so both group and per-axis contracts must hold.
        for group in plan.groups:
            if group.single_axis:
                cos = plan.cos_tables[group.start]
                sin = plan.sin_tables[group.start]
                if not _can_repeat_rope(cos, tokens):
                    return False
                if not _rope_table_shape_matches(cos, sin, tokens, group.dim):
                    return False
                if not _qk_norm_rope_group_eligible(
                    req.q,
                    group.dim,
                    plan.q_weights[group.start],
                    plan.k_weights[group.start],
                    cos,
                    sin,
                ):
                    return False
            elif not _qk_norm_group_eligible(
                req.q,
                group.dim,
                plan.q_weights[group.start],
                plan.k_weights[group.start],
            ):
                return False
            for local_axis in range(group.start, group.end):
                cos = plan.cos_tables[local_axis]
                sin = plan.sin_tables[local_axis]
                axis_dim = int(plan.axis_dims[local_axis])
                if not _can_repeat_rope(cos, tokens):
                    return False
                if not _rope_table_shape_matches(cos, sin, tokens, axis_dim):
                    return False
                if not _packed_rope_axis_eligible(req.q, axis_dim, cos, sin):
                    return False
        return True

    def _run_multi_axis(self, req: MultiAxisQKNormRopeReq) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the planned fused strategy or compose eligible group kernels."""

        plan = QKNormRopePlan.from_request(req)

        # Prefer one-launch head/tail strategies when the plan identifies their
        # exact axis grouping and rotation pattern.
        fused = self._try_identity_tail(req, plan)
        if fused is not None:
            return fused
        fused = self._try_rotated_tail(req, plan)
        if fused is not None:
            return fused

        out_q = []
        out_k = []

        # The general path normalizes each shared group once. Independent axes
        # use the combined norm-plus-RoPE kernel; multi-axis groups split their
        # normalized output for per-axis packed rotation.
        for group in plan.groups:
            q_group, q_shape, q_was_flattened = _flatten_axis_group(
                req.q, plan.axis_dims, group.start, group.end
            )
            k_group, k_shape, k_was_flattened = _flatten_axis_group(
                req.k, plan.axis_dims, group.start, group.end
            )
            if group.single_axis:
                cos_flat = _align_rope_table(plan.cos_tables[group.start], int(q_group.shape[0]))
                sin_flat = _align_rope_table(plan.sin_tables[group.start], int(q_group.shape[0]))
                out = try_triton_qk_rms_norm_rope(
                    q_group,
                    k_group,
                    plan.q_weights[group.start],
                    plan.k_weights[group.start],
                    cos_flat,
                    sin_flat,
                    req.eps,
                    req.eps,
                )
                if out is None:
                    raise RuntimeError("triton multi-axis qk_norm_rope became ineligible")
                q_rot, k_rot = out
                out_q.append(_unflatten_heads(q_rot, q_shape, q_was_flattened))
                out_k.append(_unflatten_heads(k_rot, k_shape, k_was_flattened))
                continue

            out = run_qk_rms_norm(
                q_group,
                k_group,
                plan.q_weights[group.start],
                plan.k_weights[group.start],
                req.eps,
                req.eps,
            )
            if out is None:
                raise RuntimeError("triton multi-axis qk_norm became ineligible")
            q_normed, k_normed = out
            q_normed_parts = q_normed.split(plan.axis_dims[group.start : group.end], dim=-1)
            k_normed_parts = k_normed.split(plan.axis_dims[group.start : group.end], dim=-1)

            for local, (q_normed_part, k_normed_part) in enumerate(
                zip(q_normed_parts, k_normed_parts, strict=True)
            ):
                axis = group.start + local
                q_normed_part = q_normed_part.contiguous()
                k_normed_part = k_normed_part.contiguous()
                cos_flat = _align_rope_table(plan.cos_tables[axis], int(q_normed_part.shape[0]))
                sin_flat = _align_rope_table(plan.sin_tables[axis], int(k_normed_part.shape[0]))
                q_rot = _triton_packed.run(q_normed_part, cos_flat, sin_flat)
                k_rot = _triton_packed.run(k_normed_part, cos_flat, sin_flat)
                axis_dim = int(plan.axis_dims[axis])
                out_q.append(_unflatten_heads(q_rot, (*q_shape[:-1], axis_dim), q_was_flattened))
                out_k.append(_unflatten_heads(k_rot, (*k_shape[:-1], axis_dim), k_was_flattened))

        return torch.cat(out_q, dim=-1), torch.cat(out_k, dim=-1)


class EagerQKNormRope(Operator):
    """Portable tensor provider for QK RMSNorm-plus-RoPE."""

    def __init__(self) -> None:
        """Register the eager fused-QK provider."""

        super().__init__("eager", "qk_norm_rope")

    def can_run(self, req: QKNormRopeRequest) -> bool:
        """Accept all request layouts supported by the tensor implementation."""

        del req
        return True

    def run(self, req: QKNormRopeRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize Q/K and apply the request's rotary layout."""

        if isinstance(req, MultiAxisQKNormRopeReq):
            return self._run_multi_axis(req)

        q = eager_rms_norm(req.q.float(), req.q_weight.float(), req.eps)
        k = eager_rms_norm(req.k.float(), req.k_weight.float(), req.eps)
        if req.in_place:
            # In-place requests carry full-width duplicated factors for an even
            # rotary prefix; the remainder of each normalized head is retained.
            rotary_dim = int(req.cos.shape[-1])
            if (
                req.q.ndim != 3
                or req.k.ndim != 3
                or req.cos.ndim != 2
                or req.sin.shape != req.cos.shape
                or req.cos.shape[0] != req.q.shape[0]
                or rotary_dim <= 0
                or rotary_dim % 2
                or rotary_dim > req.q.shape[-1]
            ):
                raise ValueError("in-place partial qk_norm_rope geometry is invalid")

            def partial_rope(value: torch.Tensor) -> torch.Tensor:
                """Rotate the declared prefix of one normalized Q/K tensor."""

                head = value[..., :rotary_dim]
                first, second = head.chunk(2, dim=-1)
                rotated = torch.cat((-second, first), dim=-1)
                table_shape = (req.cos.shape[0], 1, rotary_dim)
                output = value.clone()
                output[..., :rotary_dim] = head * req.cos.view(
                    table_shape
                ) + rotated * req.sin.view(table_shape)
                return output

            req.q.copy_(partial_rope(q))
            req.k.copy_(partial_rope(k))
            return req.q, req.k

        q, k = _apply_rope_axis(q, k, req.cos, req.sin, unsqueeze_dim=req.unsqueeze_dim)
        return q.to(req.q.dtype), k.to(req.k.dtype)

    def _run_multi_axis(self, req: MultiAxisQKNormRopeReq) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize shared groups, rotate each axis, and restore feature order."""

        plan = QKNormRopePlan.from_request(req)
        q_parts = req.q.split(plan.axis_dims, dim=-1)
        k_parts = req.k.split(plan.axis_dims, dim=-1)
        out_q = []
        out_k = []

        # Shared groups concatenate before RMSNorm so their variance spans the
        # complete weight width, then split for independently parameterized
        # rotary tables.
        for group in plan.groups:
            q_group = torch.cat(q_parts[group.start : group.end], dim=-1)
            k_group = torch.cat(k_parts[group.start : group.end], dim=-1)
            q_normed_group = eager_rms_norm(
                q_group.float(), plan.q_weights[group.start].float(), req.eps
            )
            k_normed_group = eager_rms_norm(
                k_group.float(), plan.k_weights[group.start].float(), req.eps
            )
            q_normed_parts = q_normed_group.split(plan.axis_dims[group.start : group.end], dim=-1)
            k_normed_parts = k_normed_group.split(plan.axis_dims[group.start : group.end], dim=-1)
            for local, (q_normed, k_normed) in enumerate(
                zip(q_normed_parts, k_normed_parts, strict=True)
            ):
                axis = group.start + local
                q_rot, k_rot = _apply_rope_axis(
                    q_normed,
                    k_normed,
                    plan.cos_tables[axis],
                    plan.sin_tables[axis],
                    unsqueeze_dim=req.unsqueeze_dim,
                )
                out_q.append(q_rot)
                out_k.append(k_rot)

        return torch.cat(out_q, dim=-1).to(req.q.dtype), torch.cat(out_k, dim=-1).to(req.k.dtype)


@lru_cache(maxsize=1)
def rope_dispatcher() -> Dispatcher[PackedRopeReq, torch.Tensor]:
    """Return the process-wide packed-RoPE provider dispatcher."""

    return Dispatcher(
        "rope",
        [TritonPackedRope(), EagerPackedRope()],
        env_override="UNISERVE_ROPE_PROVIDER",
    )


@lru_cache(maxsize=1)
def qk_norm_rope_dispatcher() -> Dispatcher[QKNormRopeRequest, tuple[torch.Tensor, torch.Tensor]]:
    """Return the process-wide fused QK normalization/RoPE dispatcher."""

    return Dispatcher(
        "qk_norm_rope",
        [TritonQKNormRope(), EagerQKNormRope()],
        env_override="UNISERVE_QK_NORM_ROPE_PROVIDER",
    )


def qk_rms_norm_partial_rope_(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    query_bias: torch.Tensor | None = None,
    key_bias: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    value_bias: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q/K heads and rotate their leading split-half coordinates in place.

    Bias addition, normalization, and rotation accumulate in FP32 before the
    output store. When supplied, the value bias is also applied in place so the
    three attention projections share one launch.
    """

    if (query_bias is None) != (key_bias is None):
        raise ValueError("partial Q/K normalization requires both biases or neither")
    if (value is None) != (value_bias is None):
        raise ValueError("partial Q/K normalization requires both value and value bias or neither")

    if query.shape != key.shape or query.ndim < 3 or query.numel() == 0:
        raise ValueError("partial Q/K normalization requires equal nonempty head tensors")
    head_dim, heads = int(query.shape[-1]), int(query.shape[-2])
    if cosine.ndim < 1:
        raise ValueError("rotary tables must contain an explicit coordinate dimension")
    rotary_dim = int(cosine.shape[-1])
    rows = query.numel() // (heads * head_dim)
    if (
        rotary_dim < 2
        or rotary_dim % 2
        or rotary_dim > head_dim
        or cosine.shape != sine.shape
        or cosine.numel() != rows * rotary_dim
    ):
        raise ValueError("rotary tables must contain one even-width prefix per token")
    if any(tensor.device != query.device for tensor in (key, cosine, sine)):
        raise ValueError("partial Q/K normalization operands must share one device")
    if query.dtype != key.dtype or not query.is_floating_point():
        raise ValueError("partial Q/K normalization requires matching floating dtypes")
    for bias in (query_bias, key_bias, value_bias):
        if bias is not None and (bias.numel() != heads * head_dim or bias.device != query.device):
            raise ValueError("projection bias must match the complete head width and device")
    if value is not None and (value.shape != query.shape or value.device != query.device):
        raise ValueError("value projection must match Q/K geometry and device")
    operands = (query, key, cosine, sine, query_bias, key_bias, value, value_bias)
    if not (
        query.is_cuda
        and triton_available(query.device)
        and head_dim <= 256
        and all(tensor is None or tensor.is_contiguous() for tensor in operands)
    ):
        q, k = query.float(), key.float()
        if query_bias is not None and key_bias is not None:
            q = q + query_bias.reshape(heads, head_dim).float()
            k = k + key_bias.reshape(heads, head_dim).float()
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
        k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + eps)
        table_shape = (*query.shape[:-2], 1, rotary_dim)
        cos = cosine.reshape(table_shape).float()
        sin = sine.reshape(table_shape).float()
        for normalized, target in ((q, query), (k, key)):
            prefix = normalized[..., :rotary_dim].float()
            first, second = prefix.chunk(2, dim=-1)
            rotated = prefix * cos + torch.cat((-second, first), dim=-1) * sin
            normalized[..., :rotary_dim].copy_(rotated)
            target.copy_(normalized)
        if value is not None and value_bias is not None:
            value.copy_(
                (value.float() + value_bias.reshape(heads, head_dim).float()).to(value.dtype)
            )
        return query, key

    rows = query.numel() // (int(query.shape[-2]) * head_dim)
    heads = int(query.shape[-2])
    from .rope_kernels import _qk_rms_norm_partial_rope_kernel

    row_block = 8
    _qk_rms_norm_partial_rope_kernel[(triton.cdiv(rows, row_block), heads)](
        query,
        key,
        value,
        query_bias,
        key_bias,
        value_bias,
        cosine,
        sine,
        rows,
        heads,
        int(query.stride(-3)),
        int(query.stride(-2)),
        rotary_dim,
        HAS_BIAS=query_bias is not None,
        HAS_VALUE_BIAS=value_bias is not None,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        EPS=eps,
        HEAD_BLOCK=triton.next_power_of_2(head_dim),
        ROW_BLOCK=row_block,
        num_warps=4,
    )
    return query, key
