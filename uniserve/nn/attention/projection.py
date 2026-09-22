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
        """Project to ``[tokens, heads, head_dim]`` and normalize/rotate Q
        and K.
        """  # noqa: D205
        values = self.projection(hidden)
        query, key, value = (
            values[name].reshape(
                -1,
                values[name].shape[-1] // self.projection.head_dim,
                self.projection.head_dim,
            )
            for name in ("q", "k", "v")
        )
        query, key = self.normalize(query, key, cos, sin)
        return query, key, value.to(query.dtype)

    def normalize(self, q, k, cos, sin):
        """Leave heads unchanged when the network has no positional
        transform.
        """  # noqa: D205
        if cos or sin:
            raise ValueError(
                "unrotated QKV projection does not consume rotary factors"
            )
        return q, k


class RotaryQKVProjection(QKVProjection):
    """Normalize Q/K heads and rotate them with one split-half factor pair."""

    def __init__(self, projection, query_norm: nn.Module, key_norm: nn.Module):
        super().__init__(projection)
        self.query_norm, self.key_norm = query_norm, key_norm

    def normalize(self, q, k, cos, sin):
        if len(cos) != 1 or len(sin) != 1:
            raise ValueError("rotary QKV projection requires one factor pair")

        # Full-head RMS norms with one epsilon are one normalization domain;
        # the fused call reduces and rotates each head once.
        if (
            isinstance(self.query_norm, RMSNorm)
            and isinstance(self.key_norm, RMSNorm)
            and self.query_norm.eps == self.key_norm.eps
        ):
            return functional.qk_norm_rope(
                q,
                k,
                (self.query_norm.weight,),
                (self.key_norm.weight,),
                cos,
                sin,
                eps=self.query_norm.eps,
                axis_dims=(q.shape[-1],),
            )

        return (
            functional.apply_rotary(
                self.query_norm(q.float()), cos[0], sin[0], rotation="split"
            ).to(q.dtype),
            functional.apply_rotary(
                self.key_norm(k.float()), cos[0], sin[0], rotation="split"
            ).to(k.dtype),
        )


class AxialQKVProjection(QKVProjection):
    """Rotate head axes with full-head or ordered RMS normalization domains.

    An RMSNorm normalizes the complete head. A ModuleList of RMSNorm modules
    partitions the head into ordered normalization domains; each domain covers
    complete rotary axes and every axis inside it shares its RMS denominator.
    """

    def __init__(
        self, projection, query_norm, key_norm, *, axis_dims, rotations
    ):
        super().__init__(projection)
        if (
            not isinstance(axis_dims, tuple)
            or sum(axis_dims) != projection.head_dim
            or any(
                type(width) is not int or width < 2 or width % 2
                for width in axis_dims
            )
            or not isinstance(rotations, tuple)
            or len(rotations) != len(axis_dims)
            or any(
                rotation not in {"interleaved", "split"}
                for rotation in rotations
            )
        ):
            raise ValueError("rotary axes must partition the Q/K head width")
        self.query_norm, self.key_norm = query_norm, key_norm
        self.axis_dims, self.rotations = axis_dims, rotations

        if isinstance(query_norm, nn.ModuleList) or isinstance(
            key_norm, nn.ModuleList
        ):
            for norms in (query_norm, key_norm):
                if not isinstance(norms, nn.ModuleList) or not all(
                    isinstance(norm, RMSNorm) for norm in norms
                ):
                    raise ValueError(
                        "partitioned Q/K normalization requires ordered "
                        "RMSNorm modules"
                    )
            if len(query_norm) != len(key_norm) or any(
                q.weight.shape != k.weight.shape or q.eps != k.eps
                for q, k in zip(query_norm, key_norm, strict=True)
            ):
                raise ValueError(
                    "query and key normalization domains must align"
                )
            # Every domain must end on an axis boundary.
            boundaries = {
                sum(axis_dims[:index]) for index in range(len(axis_dims) + 1)
            }
            ends = [0]
            for norm in query_norm:
                ends.append(ends[-1] + norm.weight.numel())
            if not set(ends) <= boundaries or ends[-1] != projection.head_dim:
                raise ValueError(
                    "normalization domains must cover complete rotary axes"
                )

    def _domains(self):
        """Return Q and K RMSNorm domains, or ``None`` for other norms."""
        norms: list[tuple[nn.Module, ...]] = []
        for norm in (self.query_norm, self.key_norm):
            if isinstance(norm, RMSNorm):
                norms.append((norm,))
            elif isinstance(norm, nn.ModuleList):
                norms.append(tuple(norm))
            else:
                return None
        return tuple(norms)

    def normalize(self, q, k, cos, sin):
        if len(cos) != len(self.axis_dims) or len(sin) != len(self.axis_dims):
            raise ValueError("each rotary axis requires one factor pair")

        # RMS domains with one epsilon and split rotation form one fused call;
        # the call reduces each domain once across the axes it covers.
        domains = self._domains()
        if (
            domains is not None
            and len({norm.eps for norm in (*domains[0], *domains[1])}) == 1
            and all(rotation == "split" for rotation in self.rotations)
        ):
            return functional.qk_norm_rope(
                q,
                k,
                tuple(norm.weight for norm in domains[0]),
                tuple(norm.weight for norm in domains[1]),
                cos,
                sin,
                eps=domains[0][0].eps,
                axis_dims=self.axis_dims,
            )

        # General path: normalize each norm domain, then rotate each axis.
        outputs = []
        for tensor, norm in ((q, self.query_norm), (k, self.key_norm)):
            if isinstance(norm, nn.ModuleList):
                widths = tuple(module.weight.numel() for module in norm)
                normalized = torch.cat(
                    tuple(
                        module(part)
                        for module, part in zip(
                            norm,
                            tensor.float().split(widths, dim=-1),
                            strict=True,
                        )
                    ),
                    dim=-1,
                )
            else:
                normalized = norm(tensor.float())
            outputs.append(
                torch.cat(
                    tuple(
                        functional.apply_rotary(
                            part, cosine, sine, rotation=rotation
                        )
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
