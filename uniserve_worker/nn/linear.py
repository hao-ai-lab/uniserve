"""Linear layers with a shared weight-loading and quantization seam."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .layer import LayerConfig
from .mesh import GroupCoordinator, TensorParallel, divide
from .placement import (
    ShardPlan,
    ShardSlot,
    WeightMode,
    set_shard_plan,
    shard_for,
)
from .quant.base import (
    PreparedLinearInput,
    QuantizeMethodBase,
)

__all__ = [
    "LinearBase",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "InterleavedMergedColumnParallelLinear",
    "QKVParallelLinear",
]


class LinearBase(nn.Module):
    """Defines the quantization-aware weight lifecycle and execution interface shared by linear layers."""

    weight: nn.Parameter
    bias: nn.Parameter | None
    weight_scale: torch.Tensor | None
    logical_weight_absmax: torch.Tensor | None

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        layer_config: LayerConfig,
        quant_method: QuantizeMethodBase | None = None,
        bias: bool = True,
        prefix: str = "",
        sequence_group: GroupCoordinator | None = None,
        input_scale_group: GroupCoordinator | None = None,
    ) -> None:
        """Create loadable weight storage through the layer's selected quantization method."""

        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.sequence_group = sequence_group
        self.input_scale_group = input_scale_group
        self.logical_input_row_partitions = 1
        self.weight_group = GroupCoordinator()
        self.weight_shard_axis: int | None = None
        self.weight_output_offset = 0
        self.weight_global_output = self.output_size
        self.weight_scale_partition_size: int | None = None
        self.weight_output_partitions: tuple[int, ...] = (self.output_size,)
        self.register_buffer("logical_weight_absmax", None, persistent=False)
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
        # Weights remain uninitialized until the loader assigns checkpoint data.
        # A missing assignment therefore fails validation instead of resembling a valid random weight.

    @property
    def execution_bias(self) -> torch.Tensor | None:
        """Bias applied by the numerical GEMM; row reductions apply it afterward."""

        return self.bias

    def forward(self, x: torch.Tensor, *, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Apply the layer's selected weight and activation precision method."""

        partitions = (
            self.logical_input_row_partitions
            if self.quant_method.input_scale_domain == "tensor"
            else 1
        )
        if partitions == 1:
            return self._project(x, output_dtype=output_dtype)
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] % partitions:
            raise ValueError("activation rows must divide their logical quantization partitions")
        outputs = [
            self._project(part, output_dtype=output_dtype) for part in flat.chunk(partitions, dim=0)
        ]
        return torch.cat(outputs, dim=0).reshape(*x.shape[:-1], self.output_size)

    def _project(self, x: torch.Tensor, *, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Compose scale-domain communication with numerical input preparation."""

        groups = []
        domain = self.quant_method.input_scale_domain
        if domain != "block" and self.weight_shard_axis == 1 and self.weight_group.world_size > 1:
            groups.append(self.weight_group)
        if (
            domain == "tensor"
            and self.input_scale_group is not None
            and self.input_scale_group.world_size > 1
        ):
            groups.append(self.input_scale_group)
        output_dtype = x.dtype if output_dtype is None else output_dtype
        if not groups and output_dtype == x.dtype:
            return self.quant_method.apply(self, x)
        flat = x.reshape(-1, x.shape[-1])
        scale = self.quant_method.input_scale(flat)
        if scale is not None:
            for group in groups:
                group.all_reduce_max(scale)
        prepared = self.quant_method.prepare_input(flat, scale)
        return self.quant_method.apply_prepared(self, prepared, output_dtype=output_dtype).reshape(
            *x.shape[:-1], self.output_size
        )

    @torch.no_grad()
    def finalize_weights(self) -> None:
        """Resolve logical scale domains before replacing checkpoint storage.

        Output partitions describe numerical domains independent of GPU count.
        Input shards contribute maxima to the same output rows. Communication
        stays at the shared parallel layer; numerical methods only pack values.
        """

        domain = self.quant_method.weight_scale_domain
        if self.weight.dtype in (torch.float32, torch.float16, torch.bfloat16):
            if domain == "row":
                maximum = self.weight.abs().amax(dim=1, keepdim=True).float()
                if self.weight_shard_axis == 1:
                    self.weight_group.all_reduce_max(maximum)
                self.logical_weight_absmax = maximum
            elif domain == "tensor":
                width = self.weight_scale_partition_size or self.weight_global_output
                count = (self.weight_global_output + width - 1) // width
                maximum = torch.zeros(count, device=self.weight.device, dtype=torch.float32)
                start = self.weight_output_offset
                end = start + self.output_size
                partitions = []
                domains = []
                while start < end:
                    domain_id = start // width
                    stop = min(end, (domain_id + 1) * width)
                    begin = start - self.weight_output_offset
                    maximum[domain_id] = (
                        self.weight[begin : begin + stop - start].abs().amax().float()
                    )
                    partitions.append(stop - start)
                    domains.append(domain_id)
                    start = stop
                self.weight_group.all_reduce_max(maximum)
                self.weight_output_partitions = tuple(partitions)
                self.logical_weight_absmax = maximum[domains].reshape(-1, 1)
        self.quant_method.process_weights_after_loading(self)

    def forward_prequantized(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Apply a quantized linear to caller-supplied FP8 activation storage and scale."""

        return self.quant_method.apply_prequantized(
            self,
            x,
            scale,
            output_dtype=output_dtype,
        )

    def forward_sequence_parallel(
        self,
        x: torch.Tensor,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        """Gather prepared row shards and execute their shared numerical GEMM.

        Values and block scales use caller-owned byte storage. Their common
        numerical scale is resolved before quantization, independent of physical
        row ownership. Format preparation never initiates communication.
        """

        group = self.sequence_group
        if group is None:
            raise RuntimeError("sequence projection requires a construction-time sequence_group")
        if group.world_size == 1:
            return self.forward(x)
        scale = self.quant_method.input_scale(x)
        if scale is not None and self.quant_method.input_scale_domain == "tensor":
            group.all_reduce_max(scale)
        prepared = self.quant_method.prepare_input(x, scale)
        cursor = 0

        def gather(value: torch.Tensor) -> torch.Tensor:
            nonlocal cursor
            elements = value.numel() * group.world_size
            byte_count = elements * value.element_size()
            if cursor + byte_count > workspace.numel() * workspace.element_size():
                raise ValueError("sequence projection workspace cannot hold prepared input")
            target = workspace.view(torch.uint8)[cursor : cursor + byte_count].view(value.dtype)
            target = target.view(value.shape[0] * group.world_size, *value.shape[1:])
            group.all_gather_into_tensor(target, value)
            cursor += byte_count
            return target

        values = gather(prepared.values)
        scales = gather(prepared.block_scales) if prepared.block_scales is not None else None
        global_scale = prepared.global_scale
        if global_scale is not None and self.quant_method.input_scale_domain == "row":
            global_scale = gather(global_scale)
        return self.quant_method.apply_prepared(
            self,
            PreparedLinearInput(values, scales, global_scale),
            output_dtype=x.dtype,
        )


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
        """Shard the output dimension across tensor-parallel ranks at construction time."""

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
        self.weight_group = layer_config.tensor_group()
        self.weight_shard_axis = 0
        self.weight_output_offset = parallel.rank * local_output
        self.weight_global_output = self.global_output_size
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
        quant_method: QuantizeMethodBase | None = None,
    ) -> None:
        """Shard the input dimension and configure tensor-parallel output reduction."""

        parallel = layer_config.parallel
        self.global_input_size = int(input_size)
        self.global_output_size = int(output_size)
        local_input = divide(self.global_input_size, parallel.size)
        super().__init__(
            local_input,
            output_size,
            layer_config=layer_config,
            quant_method=quant_method,
            bias=bias,
            prefix=prefix,
        )
        self.tp_group = layer_config.tensor_group()
        self.weight_group = self.tp_group
        self.weight_shard_axis = 1
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        reduce: bool = True,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Project an input shard and optionally sum partial outputs across tensor ranks."""

        out = super().forward(x, output_dtype=output_dtype)
        if not reduce:
            return out
        return self.reduce_output(out)

    @property
    def execution_bias(self) -> torch.Tensor | None:
        return None

    def reduce_output(self, out: torch.Tensor) -> torch.Tensor:
        """Sum input-shard partial outputs across the tensor-parallel axis."""

        out = self.tp_group.all_reduce(out)
        return out if self.bias is None else out + self.bias


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
        quant_method: QuantizeMethodBase | None = None,
    ):
        """Pack multiple output branches while retaining their global and local extents."""

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
            quant_method=quant_method,
            bias=bias,
            prefix=prefix,
        )
        slots: dict[int | str, ShardSlot] = {}
        self.weight_group = layer_config.tensor_group()
        self.weight_shard_axis = 0
        self.weight_global_output = sum(self.global_output_sizes)
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
        sequence_group: GroupCoordinator | None = None,
        weight_scale_partition_size: int | None = None,
        input_scale_group: GroupCoordinator | None = None,
    ) -> None:
        """Partition fixed-width groups from every branch across tensor-parallel ranks."""

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
            sequence_group=sequence_group,
            input_scale_group=input_scale_group,
        )
        from functools import partial

        self.weight_group = layer_config.tensor_group()
        self.weight_shard_axis = 0
        self.weight_output_offset = parallel.rank * local_branch * branches
        self.weight_global_output = branch_output_size * branches
        self.weight_scale_partition_size = weight_scale_partition_size

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
    """Return this tensor-parallel rank's query-head count."""
    return divide(int(total_heads), parallel.size)


def local_kv_head_count(total_kv_heads: int, *, parallel: TensorParallel) -> int:
    """Return this tensor-parallel rank's KV-head count.

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
    """Projects sharded queries, keys, and values with head-aware tensor-parallel partitioning."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        *,
        layer_config: LayerConfig,
        quant_method: QuantizeMethodBase | None = None,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        """Derive rank-local query and KV head counts before packing their projections."""

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
            quant_method=quant_method,
            bias=bias,
            prefix=prefix,
            local_output_sizes=(q_size_local, kv_size_local, kv_size_local),
            weight_mode=WeightMode.FUSED_QKV_LINEAR,
        )
