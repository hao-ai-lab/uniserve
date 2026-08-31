"""Linear layers with a shared weight-loading and quantization seam."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from ..execution.forward_batch import MeshView
from .layer import LayerConfig
from .mesh import DeviceMesh, TensorParallel, divide
from .placement import (
    ShardPlan,
    ShardSlot,
    WeightMode,
    set_shard_plan,
    shard_for,
)
from .quant.base import QuantizeMethodBase

__all__ = [
    "LinearBase",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "InterleavedMergedColumnParallelLinear",
    "QKVParallelLinear",
]


class LinearBase(nn.Module):
    weight: nn.Parameter
    bias: nn.Parameter | None
    weight_scale: torch.Tensor | None

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        layer_config: LayerConfig,
        quant_method: QuantizeMethodBase | None = None,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.prefix = str(prefix)
        self.has_bias = bool(bias)
        self.quant_method = (
            layer_config.quant_method(self.prefix) if quant_method is None else quant_method
        )
        self.quant_method.create_weights(
            self,
            input_size=self.input_size,
            output_size=self.output_size,
            bias=self.has_bias,
        )
        # Weights are created as ``torch.empty`` and are populated by the system loader before use. Skipping an initialization that would immediately be overwritten also makes a missing checkpoint tensor fail validation instead of appearing as a plausible random weight.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x)

    def forward_sequence_parallel(
        self,
        x: torch.Tensor,
        mesh: DeviceMesh,
        workspace: torch.Tensor,
        *,
        group: str,
    ) -> torch.Tensor:
        execute = getattr(self.quant_method, "apply_sequence_parallel", None)
        if not callable(execute):
            raise RuntimeError("linear quantization method has no sequence-parallel execution")
        return execute(self, x, mesh, workspace, group=group)


def _attach_shard_plan(module: LinearBase, plan_for: Callable[[nn.Parameter], ShardPlan]) -> None:
    """Attach a per-parameter :class:`ShardPlan` to weight/weight_scale/bias."""
    from ..loader.weight_loaders import (
        default_weight_loader,
        packed_weight_loader,
        sharded_weight_loader,
    )

    for name in ("weight", "weight_scale", "bias"):
        param = getattr(module, name, None)
        if isinstance(param, nn.Parameter):
            plan = plan_for(param)
            set_shard_plan(param, plan)
            current = getattr(param, "weight_loader", None)
            if current is default_weight_loader:
                setattr(
                    param,
                    "weight_loader",
                    packed_weight_loader if plan.slots else sharded_weight_loader,
                )


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
        layer_config: LayerConfig,
        quant_method: QuantizeMethodBase | None = None,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        parallel = layer_config.parallel
        self.global_input_size = int(input_size)
        self.global_output_size = int(output_size)
        local_output = divide(self.global_output_size, parallel.size)
        super().__init__(
            input_size,
            local_output,
            layer_config=layer_config,
            quant_method=quant_method,
            bias=bias,
            prefix=prefix,
        )
        partition = shard_for(0, parallel)
        _attach_shard_plan(self, lambda _param: ShardPlan(shard=partition))


class RowParallelLinear(LinearBase):
    """Input-dimension sharded linear with a runtime TP all-reduce reshard."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        layer_config: LayerConfig,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        parallel = layer_config.parallel
        self.global_input_size = int(input_size)
        self.global_output_size = int(output_size)
        local_input = divide(self.global_input_size, parallel.size)
        super().__init__(
            local_input,
            output_size,
            layer_config=layer_config,
            bias=bias,
            prefix=prefix,
        )
        # The weight shards on the input axis; the per-channel weight_scale is
        # replicated across ranks (its rows index the unsharded output axis).
        set_shard_plan(self.weight, ShardPlan(shard=shard_for(1, parallel)))
        from ..loader.weight_loaders import default_weight_loader, sharded_weight_loader

        if getattr(self.weight, "weight_loader", None) is default_weight_loader:
            setattr(self.weight, "weight_loader", sharded_weight_loader)
        weight_scale = getattr(self, "weight_scale", None)
        if isinstance(weight_scale, nn.Parameter):
            set_shard_plan(
                weight_scale,
                ShardPlan(shard=shard_for(0, parallel, replicated=parallel.size > 1)),
            )

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, mesh: MeshView, *, reduce: bool = True
    ) -> torch.Tensor:
        out = super().forward(x)
        if not reduce:
            return out
        return self.reduce_output(out, mesh)

    def reduce_output(self, out: torch.Tensor, mesh: MeshView) -> torch.Tensor:
        return mesh.all_reduce(out, "tp")


class MergedColumnParallelLinear(LinearBase):
    """Column linear whose output axis packs several named shards."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int] | tuple[int, ...],
        *,
        layer_config: LayerConfig,
        bias: bool = True,
        prefix: str = "",
        local_output_sizes: list[int] | tuple[int, ...] | None = None,
        weight_mode: WeightMode = WeightMode.VANILLA,
    ):
        parallel = layer_config.parallel
        self.global_output_sizes = tuple(int(s) for s in output_sizes)
        self.output_sizes = (
            tuple(int(s) for s in local_output_sizes)
            if local_output_sizes is not None
            else tuple(divide(size, parallel.size) for size in self.global_output_sizes)
        )
        if len(self.output_sizes) != len(self.global_output_sizes):
            raise ValueError("local_output_sizes must match output_sizes")
        super().__init__(
            input_size,
            sum(self.output_sizes),
            layer_config=layer_config,
            bias=bias,
            prefix=prefix,
        )
        slots: dict[int | str, ShardSlot] = {}
        cursor = 0
        for idx, (size, global_size) in enumerate(zip(self.output_sizes, self.global_output_sizes)):
            replicated = size == global_size and parallel.size > 1
            slots[idx] = ShardSlot(
                offset=cursor,
                size=size,
                shard=shard_for(0, parallel, replicated=replicated),
            )
            cursor += size
        _attach_shard_plan(
            self,
            lambda _param: ShardPlan(
                shard=shard_for(0, parallel),
                mode=weight_mode,
                shard_axis=0,
                slots=dict(slots),
            ),
        )


class InterleavedMergedColumnParallelLinear(LinearBase):
    """Column-parallel branches interleaved by fixed-width output groups."""

    def __init__(
        self,
        input_size: int,
        branch_output_size: int,
        branches: int,
        group_width: int,
        *,
        layer_config: LayerConfig,
        quant_method: QuantizeMethodBase | None = None,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        parallel = layer_config.parallel
        branch_output_size = int(branch_output_size)
        branches = int(branches)
        group_width = int(group_width)
        if branches <= 0 or group_width <= 0:
            raise ValueError("interleaved merged linear dimensions must be positive")
        if branch_output_size % group_width:
            raise ValueError("branch output size must divide into fixed-width groups")
        local_branch = divide(branch_output_size, parallel.size)
        if local_branch % group_width:
            raise ValueError("local branch output must divide into fixed-width groups")
        self.global_branch_output_size = branch_output_size
        self.local_branch_output_size = local_branch
        self.branches = branches
        self.group_width = group_width
        super().__init__(
            input_size,
            local_branch * branches,
            layer_config=layer_config,
            quant_method=quant_method,
            bias=bias,
            prefix=prefix,
        )
        from functools import partial
        from ..loader.weight_loaders import (
            attach_weight_loader,
            interleaved_packed_weight_loader,
        )

        attach_weight_loader(
            self.weight,
            partial(
                interleaved_packed_weight_loader,
                rank=parallel.rank,
                size=parallel.size,
                branches=branches,
                group_width=group_width,
            ),
        )


def local_attention_head_count(total_heads: int, *, parallel: TensorParallel) -> int:
    """This tensor-parallel rank's query-head count (query heads always shard)."""
    return divide(int(total_heads), parallel.size)


def local_kv_head_count(total_kv_heads: int, *, parallel: TensorParallel) -> int:
    """This tensor-parallel rank's KV-head count.

    The single owner of the attention KV sharding rule: a KV group divides
    across the tp axis when it is large enough and stays whole (replicated)
    when it is not. :class:`QKVParallelLinear` bakes the same decision into its
    local shard sizes, and KV-pool/caps geometry must use this helper so pool
    layouts can never drift from what sharded attention actually writes.
    """
    total = int(total_kv_heads)
    tp_size = int(parallel.size)
    if tp_size <= 1 or total < tp_size:
        return total
    return divide(total, tp_size)


class QKVParallelLinear(MergedColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        layer_config: LayerConfig,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        parallel = layer_config.parallel
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        q_size = total_num_heads * head_size
        kv_size = total_num_kv_heads * head_size
        q_size_local = local_attention_head_count(total_num_heads, parallel=parallel) * head_size
        kv_size_local = local_kv_head_count(total_num_kv_heads, parallel=parallel) * head_size
        # The q/k/v -> 0/1/2 shard-id mapping is carried by WeightMode; the
        # k/v "replicated" decision falls out of the size==global_size test in
        # the base merged plan (a kv group too small to split stays whole).
        super().__init__(
            hidden_size,
            (q_size, kv_size, kv_size),
            layer_config=layer_config,
            bias=bias,
            prefix=prefix,
            local_output_sizes=(q_size_local, kv_size_local, kv_size_local),
            weight_mode=WeightMode.FUSED_QKV_LINEAR,
        )
