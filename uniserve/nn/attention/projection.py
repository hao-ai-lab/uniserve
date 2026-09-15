"""Q/K/V network composition with explicit normalization and rotary axes."""

import torch
from torch import nn

from uniserve.nn import functional
from uniserve.nn.linear import QKVParallelLinear
from uniserve.nn.norm import RMSNorm


class QKVProjection(nn.Module):
    """Project named branches and reshape their TP-local head dimensions."""

    def __init__(self, projection: QKVParallelLinear):
        super().__init__()
        self.projection = projection

    def forward(self, hidden, cos, sin):
        values = self.projection(hidden)
        query, key, value = (
            values[name].reshape(
                -1, values[name].shape[-1] // self.projection.head_dim, self.projection.head_dim
            )
            for name in ("q", "k", "v")
        )
        query, key = self.normalize(query, key, cos, sin)
        return query, key, value.to(query.dtype)

    def normalize(self, q, k, cos, sin):
        """Leave heads unchanged when the network has no positional transform."""

        if cos or sin:
            raise ValueError("unrotated QKV projection does not consume rotary factors")
        return q, k


class RotaryQKVProjection(QKVProjection):
    def __init__(self, projection, query_norm: nn.Module, key_norm: nn.Module):
        super().__init__(projection)
        self.query_norm, self.key_norm = query_norm, key_norm

    def normalize(self, q, k, cos, sin):
        if len(cos) != 1 or len(sin) != 1:
            raise ValueError("rotary QKV projection requires one factor pair")
        if (
            isinstance(self.query_norm, RMSNorm)
            and isinstance(self.key_norm, RMSNorm)
            and self.query_norm.eps == self.key_norm.eps
            and cos[0].shape[-1] * 2 == q.shape[-1]
        ):
            return functional.qk_norm_rope(
                q,
                k,
                self.query_norm.weight,
                self.key_norm.weight,
                cos,
                sin,
                eps=self.query_norm.eps,
                axis_dims=(q.shape[-1],),
            )
        return (
            functional.apply_rotary(
                self.query_norm(q.float()), cos[0], sin[0], rotation="split"
            ).to(q.dtype),
            functional.apply_rotary(self.key_norm(k.float()), cos[0], sin[0], rotation="split").to(
                k.dtype
            ),
        )


class AxialQKVProjection(QKVProjection):
    """Rotate head axes with full-head or ordered RMS normalization domains.

    An RMSNorm normalizes the complete head. A ModuleList of RMSNorm modules
    partitions the head in list order; each domain must cover complete rotary
    axes. Adjacent axes in one domain share the same checkpoint scale tensor.
    """

    def __init__(self, projection, query_norm, key_norm, *, axis_dims, rotations):
        super().__init__(projection)
        if (
            not isinstance(axis_dims, tuple)
            or sum(axis_dims) != projection.head_dim
            or any(type(width) is not int or width < 2 or width % 2 for width in axis_dims)
            or not isinstance(rotations, tuple)
            or len(rotations) != len(axis_dims)
            or any(rotation not in {"interleaved", "split"} for rotation in rotations)
        ):
            raise ValueError("rotary axes must partition the Q/K head width")
        self.query_norm, self.key_norm = query_norm, key_norm
        self.axis_dims, self.rotations = axis_dims, rotations
        if isinstance(query_norm, nn.ModuleList) or isinstance(key_norm, nn.ModuleList):
            query = self._axis_norms(query_norm)
            key = self._axis_norms(key_norm)
            if any(
                q.weight.shape != k.weight.shape or q.eps != k.eps
                for q, k in zip(query, key, strict=True)
            ):
                raise ValueError("query and key normalization domains must align")

    def _axis_norms(self, norms):
        if not isinstance(norms, nn.ModuleList) or not all(
            isinstance(norm, RMSNorm) for norm in norms
        ):
            raise ValueError("partitioned Q/K normalization requires ordered RMSNorm modules")
        result = []
        axis = 0
        for norm in norms:
            consumed = 0
            while axis < len(self.axis_dims) and consumed < norm.weight.numel():
                consumed += self.axis_dims[axis]
                result.append(norm)
                axis += 1
            if consumed != norm.weight.numel():
                raise ValueError("normalization domains must cover complete rotary axes")
        if axis != len(self.axis_dims):
            raise ValueError("normalization domains must cover the whole head")
        return tuple(result)

    def normalize(self, q, k, cos, sin):
        if len(cos) != len(self.axis_dims) or len(sin) != len(self.axis_dims):
            raise ValueError("each rotary axis requires one factor pair")
        partitioned = isinstance(self.query_norm, nn.ModuleList)
        if partitioned and all(rotation == "split" for rotation in self.rotations):
            from uniserve import ops

            query, key = self._axis_norms(self.query_norm), self._axis_norms(self.key_norm)
            if len({norm.eps for norm in (*query, *key)}) == 1:
                # Preserve shared normalization identities when several rotary
                # axes consume one domain; the fused kernel reduces it once.
                return ops.qk_norm_rope(
                    q,
                    k,
                    tuple(norm.weight for norm in query),
                    tuple(norm.weight for norm in key),
                    cos,
                    sin,
                    query[0].eps,
                    axis_dims=self.axis_dims,
                )
        if (
            isinstance(self.query_norm, RMSNorm)
            and isinstance(self.key_norm, RMSNorm)
            and self.query_norm.eps == self.key_norm.eps
            and all(rotation == "split" for rotation in self.rotations)
        ):
            return functional.qk_norm_rope(
                q,
                k,
                self.query_norm.weight,
                self.key_norm.weight,
                cos,
                sin,
                eps=self.query_norm.eps,
                axis_dims=self.axis_dims,
            )
        outputs = []
        for tensor, norm in ((q, self.query_norm), (k, self.key_norm)):
            if partitioned:
                widths = tuple(module.weight.numel() for module in norm)
                normalized = torch.cat(
                    tuple(
                        module(part)
                        for module, part in zip(
                            norm, tensor.float().split(widths, dim=-1), strict=True
                        )
                    ),
                    dim=-1,
                )
            else:
                normalized = norm(tensor.float())
            outputs.append(
                torch.cat(
                    tuple(
                        functional.apply_rotary(part, cosine, sine, rotation=rotation)
                        for part, cosine, sine, rotation in zip(
                            normalized.split(self.axis_dims, dim=-1),
                            cos,
                            sin,
                            self.rotations,
                            strict=True,
                        )
                    ),
                    dim=-1,
                ).to(tensor.dtype)
            )
        return tuple(outputs)
