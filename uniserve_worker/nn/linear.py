"""Linear layers with a shared weight-loading and quantization seam."""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .mesh import DeviceMesh, divide, get_current_mesh
from .placement import (
    Partial,
    Replicate,
    Sharding,
    ShardPlan,
    ShardSlot,
    WeightMode,
    place_partitioned_tensor,
    reshard,
    set_shard_plan,
    shard_spec,
)
from .quant import (
    QuantizationConfig,
    QuantizeMethodBase,
    UnquantizedLinearMethod,
    get_current_quantization_config,
    warn_if_no_quant_context,
)
from .quant.load_state import has_weight_loader, set_weight_loader

__all__ = [
    'default_weight_loader',
    'LinearBase',
    'ColumnParallelLinear',
    'RowParallelLinear',
    'MergedColumnParallelLinear',
    'QKVParallelLinear',
]

# Reusable placement transitions for the tensor-parallel axis. A row-parallel
# GEMM produces a Partial sum reduced to Replicate (all-reduce); a column-parallel
# output is Sharded on its last dim and gathered to Replicate (all-gather). On a
# trivial tp axis ``reshard`` is a no-op, so tp=1 is byte-identical.
_TP_PARTIAL = Sharding((Partial("tp"),))
_TP_REPLICATE = Sharding((Replicate("tp"),))


def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    *,
    shard_id: int | str | None = None,
) -> None:
    """Copy a loaded tensor into ``param`` through its typed :class:`ShardPlan`.

    ``shard_id`` selects a named shard for merged/QKV parameters; the sharding
    layout (axis/rank/size, per-shard slices) is resolved entirely from the
    plan attached at construction, so placement semantics live in one helper.
    """
    place_partitioned_tensor(param, param.data, loaded_weight, shard_id=shard_id)


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        quant_method: QuantizeMethodBase | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mesh: DeviceMesh | None = None,
    ) -> None:
        super().__init__()
        self.mesh = mesh or get_current_mesh()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.prefix = str(prefix)
        self.has_bias = bool(bias)
        if quant_method is not None and quant_config is not None:
            raise ValueError("pass either quant_method or quant_config, not both")
        if quant_method is not None:
            config = None
        elif quant_config is not None:
            config = quant_config
        else:
            # No explicit config: fall back to the ambient quantization context
            # installed by the loader / model entry. When nothing is active the
            # layer is built unquantized; ``warn_if_no_quant_context`` emits a
            # once-per-process warning so that silent-unquantized construction is
            # observable rather than invisible.
            config = get_current_quantization_config()
            if config is None:
                warn_if_no_quant_context()
        self.quant_method = quant_method or (
            config.get_quant_method(self.prefix) if config is not None else UnquantizedLinearMethod()
        )
        self.quant_method.create_weights(
            self,
            input_size=self.input_size,
            output_size=self.output_size,
            bias=self.has_bias,
        )
        self._attach_default_weight_loaders()
        # Weights are created as ``torch.empty`` and are always populated by a
        # checkpoint loader (``default_weight_loader``) or the dummy loader
        # before use, so running a kaiming/uniform init at construction is pure
        # wasted compute that is immediately overwritten. Skipping it also keeps
        # never-loaded weights as raw uninitialized memory, so a missing-weight
        # bug surfaces as garbage/NaN rather than masquerading as a plausible
        # random init.

    def _attach_default_weight_loaders(self) -> None:
        weight = getattr(self, "weight", None)
        if isinstance(weight, nn.Parameter) and not has_weight_loader(weight):
            set_weight_loader(weight, default_weight_loader)
        bias = getattr(self, "bias", None)
        if isinstance(bias, nn.Parameter) and not has_weight_loader(bias):
            set_weight_loader(bias, default_weight_loader)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x)


def _attach_shard_plan(module: LinearBase, plan_for: Callable[[nn.Parameter], ShardPlan]) -> None:
    """Attach a per-parameter :class:`ShardPlan` to weight/weight_scale/bias."""
    for name in ("weight", "weight_scale", "bias"):
        param = getattr(module, name, None)
        if isinstance(param, nn.Parameter):
            set_shard_plan(param, plan_for(param))


class ColumnParallelLinear(LinearBase):
    """Output-dimension sharded linear.

    Real tensor-parallel collectives remain outside this class. At tp=1 this is
    byte-identical to ``LinearBase``; at tp>1 it provides the load-time shard
    layout needed by quantized/checkpoint loaders.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        quant_method: QuantizeMethodBase | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mesh: DeviceMesh | None = None,
    ) -> None:
        mesh = mesh or get_current_mesh()
        self.global_input_size = int(input_size)
        self.global_output_size = int(output_size)
        local_output = divide(self.global_output_size, mesh.tp_size)
        super().__init__(
            input_size,
            local_output,
            bias=bias,
            quant_method=quant_method,
            quant_config=quant_config,
            prefix=prefix,
            mesh=mesh,
        )
        spec = shard_spec(0, mesh)
        _attach_shard_plan(self, lambda _param: ShardPlan(spec=spec))


class RowParallelLinear(LinearBase):
    """Input-dimension sharded linear with a runtime TP all-reduce reshard."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        quant_method: QuantizeMethodBase | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mesh: DeviceMesh | None = None,
    ) -> None:
        mesh = mesh or get_current_mesh()
        self.global_input_size = int(input_size)
        self.global_output_size = int(output_size)
        local_input = divide(self.global_input_size, mesh.tp_size)
        super().__init__(
            local_input,
            output_size,
            bias=bias,
            quant_method=quant_method,
            quant_config=quant_config,
            prefix=prefix,
            mesh=mesh,
        )
        # The weight shards on the input axis; the per-channel weight_scale is
        # replicated across ranks (its rows index the unsharded output axis).
        set_shard_plan(self.weight, ShardPlan(spec=shard_spec(1, mesh)))
        weight_scale = getattr(self, "weight_scale", None)
        if isinstance(weight_scale, nn.Parameter):
            set_shard_plan(weight_scale, ShardPlan(spec=shard_spec(0, mesh, replicated=mesh.tp_size > 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return reshard(out, _TP_PARTIAL, _TP_REPLICATE, self.mesh, owner="RowParallelLinear")


class MergedColumnParallelLinear(LinearBase):
    """Column linear whose output axis packs several named shards."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int] | tuple[int, ...],
        *,
        bias: bool = True,
        quant_method: QuantizeMethodBase | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mesh: DeviceMesh | None = None,
        local_output_sizes: list[int] | tuple[int, ...] | None = None,
        weight_mode: WeightMode = WeightMode.VANILLA,
    ):
        mesh = mesh or get_current_mesh()
        self.global_output_sizes = tuple(int(s) for s in output_sizes)
        self.output_sizes = (
            tuple(int(s) for s in local_output_sizes)
            if local_output_sizes is not None
            else tuple(divide(size, mesh.tp_size) for size in self.global_output_sizes)
        )
        if len(self.output_sizes) != len(self.global_output_sizes):
            raise ValueError("local_output_sizes must match output_sizes")
        super().__init__(
            input_size,
            sum(self.output_sizes),
            bias=bias,
            quant_method=quant_method,
            quant_config=quant_config,
            prefix=prefix,
            mesh=mesh,
        )
        slots: dict[int | str, ShardSlot] = {}
        cursor = 0
        for idx, (size, global_size) in enumerate(zip(self.output_sizes, self.global_output_sizes)):
            replicated = size == global_size and mesh.tp_size > 1
            slots[idx] = ShardSlot(offset=cursor, size=size, spec=shard_spec(0, mesh, replicated=replicated))
            cursor += size
        _attach_shard_plan(
            self,
            lambda _param: ShardPlan(
                spec=shard_spec(0, mesh),
                mode=weight_mode,
                shard_axis=0,
                slots=dict(slots),
            ),
        )


class QKVParallelLinear(MergedColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        bias: bool = True,
        quant_method: QuantizeMethodBase | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mesh: DeviceMesh | None = None,
    ) -> None:
        mesh = mesh or get_current_mesh()
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        q_size = total_num_heads * head_size
        kv_size = total_num_kv_heads * head_size
        q_size_local = divide(q_size, mesh.tp_size)
        if total_num_kv_heads >= mesh.tp_size:
            kv_size_local = divide(kv_size, mesh.tp_size)
        else:
            kv_size_local = kv_size
        # The q/k/v -> 0/1/2 shard-id mapping is carried by WeightMode; the
        # k/v "replicated" decision falls out of the size==global_size test in
        # the base merged plan (a kv group too small to split stays whole).
        super().__init__(
            hidden_size,
            (q_size, kv_size, kv_size),
            bias=bias,
            quant_method=quant_method,
            quant_config=quant_config,
            prefix=prefix,
            mesh=mesh,
            local_output_sizes=(q_size_local, kv_size_local, kv_size_local),
            weight_mode=WeightMode.FUSED_QKV_LINEAR,
        )
