"""Linear layers with a shared weight-loading and quantization seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import torch
import torch.nn as nn

from .layer import LayerConfig
from .mesh import Communicator, RowGather, divide
from .parallel_attention import HeadRowExchange
from .quant.base import (
    LinearMethod,
    PreparedLinearInput,
)
from .shard import (
    Shard,
    ShardPlan,
    ShardSlot,
    WeightMode,
    set_shard_plan,
    shard_for,
)

__all__ = [
    "LinearBase",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "InterleavedMergedColumnParallelLinear",
    "QKVParallelLinear",
]


class GatheredLinear:
    """Project streamed row shards through their owning layer's numerical method.

    Row and block quantization domains are independent across publications.
    Tensor-wide domains require the complete input and are handled by
    ``forward_sequence_parallel``. Transport owns only readiness and scratch;
    this object owns output storage and optional downstream row consumption.
    """

    def __init__(
        self,
        layer: "LinearBase",
        rows: int,
        workspace: torch.Tensor,
        row_consumer: Callable[[slice, torch.Tensor], None] | None,
    ) -> None:
        self.layer = layer
        self.rows = rows
        self.workspace = workspace
        self.row_consumer = row_consumer
        self.transport: RowGather | None = None
        self.output: torch.Tensor | None = None

    @staticmethod
    def _project(
        layer: "LinearBase",
        output: torch.Tensor,
        row_consumer: Callable[[slice, torch.Tensor], None] | None,
        interval: slice,
        values: torch.Tensor,
    ) -> None:
        target = output[interval]
        layer.project_into(values, target)
        if row_consumer is not None:
            row_consumer(interval, target)

    def append(self, start: int, values: torch.Tensor) -> None:
        """Publish the next local hidden interval on the current stream."""

        if self.transport is None:
            group = self.layer.sequence_group
            assert group is not None
            self.output = values.new_empty((self.rows * group.world_size, self.layer.output_size))
            # Borrow the numerical operands directly. A bound self callback
            # would form a transport/projection cycle and retain symmetric
            # scratch beyond its communicator's explicit retirement.
            self.transport = RowGather(
                group,
                self.rows,
                self.layer.input_size,
                values.dtype,
                self.workspace,
                partial(self._project, self.layer, self.output, self.row_consumer),
            )
        self.transport.append(start, values)

    def finish(self) -> torch.Tensor:
        """Finish consuming transfers and return independently owned logical rows."""

        if self.transport is None or self.output is None:
            raise ValueError("streamed projection has no published rows")
        self.transport.finish()
        output, self.output = self.output, None
        self.row_consumer = None
        return output


class LinearBase(nn.Module):
    """Defines the quantization-aware weight lifecycle and execution interface shared by linear layers."""

    weight: nn.Parameter
    bias: nn.Parameter | None
    weight_scale: torch.Tensor | None
    weight_scale_2: torch.Tensor | None
    logical_weight_absmax: torch.Tensor | None

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        layer_config: LayerConfig,
        quant_method: LinearMethod | None = None,
        bias: bool = True,
        prefix: str = "",
        sequence_group: Communicator | None = None,
        input_scale_group: Communicator | None = None,
        weight_group: Communicator | None = None,
        weight_shard_axis: int | None = None,
        weight_output_offset: int = 0,
        weight_global_output: int | None = None,
    ) -> None:
        """Create loadable weight storage through the layer's selected quantization method."""

        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("linear dimensions must be positive")
        self.sequence_group = sequence_group
        self.input_scale_group = (
            layer_config.sequence if input_scale_group is None else input_scale_group
        )
        self.weight_group = Communicator() if weight_group is None else weight_group
        self.weight_shard_axis = weight_shard_axis
        self.weight_output_offset = weight_output_offset
        self.weight_global_output = (
            self.output_size if weight_global_output is None else weight_global_output
        )
        self.weight_output_partitions: tuple[int, ...] = (self.output_size,)
        self.register_buffer("logical_weight_absmax", None, persistent=False)
        self.prefix = layer_config.qualify(str(prefix))
        self.has_bias = bool(bias)
        self.quant_method = (
            layer_config.quant_method(str(prefix)) if quant_method is None else quant_method
        )
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
        self.input_scale_groups = tuple(groups)
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

        if x.ndim < 1 or x.shape[-1] != self.input_size:
            raise ValueError("linear input must retain its declared feature width")
        return self._project(x, output_dtype=output_dtype)

    def _project(
        self,
        x: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compose scale-domain communication with numerical input preparation."""

        groups = self.input_scale_groups
        output_dtype = x.dtype if output_dtype is None else output_dtype
        if x.numel() == 0:
            # Empty routes still enter tensor-wide scale collectives. No GEMM
            # or format packing is needed for their empty numerical result.
            if self.quant_method.input_scale_domain == "tensor":
                scale = self.quant_method.input_scale(
                    x.reshape(-1, x.shape[-1]), absmax=x.new_zeros((), dtype=torch.float32)
                )
                if scale is not None:
                    for group in groups:
                        group.all_reduce_max(scale)
            return x.new_empty((*x.shape[:-1], self.output_size), dtype=output_dtype)
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

    def prepare_input(
        self, value: torch.Tensor, *, absmax: torch.Tensor | None = None
    ) -> PreparedLinearInput:
        """Prepare reusable activations in this GEMM's preferred physical layout.

        A supplied magnitude covers every local row and channel. Shared scale
        domains still combine their ranks before format packing. Row-addressable
        exchange calls the format method with its explicit linear layout.
        """

        flat = value.reshape(-1, value.shape[-1])
        scale = self.quant_method.input_scale(flat, absmax=absmax)
        if scale is not None:
            for group in self.input_scale_groups:
                group.all_reduce_max(scale)
        prepared = self.quant_method.prepare_input(
            flat, scale, block_scale_layout=self.quant_method.preferred_block_scale_layout
        )
        return replace(
            prepared,
            values=prepared.values.reshape(*value.shape[:-1], prepared.values.shape[-1]),
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
                maximum = self.weight.abs().amax().float().reshape(1, 1)
                self.weight_group.all_reduce_max(maximum)
                self.logical_weight_absmax = maximum
        self.quant_method.process_weights_after_loading(self)

    def forward_prepared(
        self,
        prepared: PreparedLinearInput,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
        include_bias: bool = True,
    ) -> torch.Tensor:
        """Consume a fused producer's typed values and scales without requantizing."""

        return self.quant_method.apply_prepared(
            self,
            prepared,
            output_dtype=output_dtype,
            include_bias=include_bias,
        )

    def stream_sequence_parallel(
        self,
        rows: int,
        workspace: torch.Tensor,
        *,
        row_consumer: Callable[[slice, torch.Tensor], None] | None = None,
    ) -> GatheredLinear:
        """Accept ordered row production before the complete input is ready.

        Tensor-wide quantization scales require their complete input domain and
        use ``forward_sequence_parallel``. The returned projection owns stream
        dependencies; the caller supplies registered scratch through completion.
        An optional consumer receives each projected logical row interval on
        the current stream, before later peer intervals have completed. It may
        transform those output rows in place and publish independent outputs.
        """

        if self.sequence_group is None or self.quant_method.input_scale_domain == "tensor":
            raise ValueError("streamed projection requires a sequence-bound row-local scale domain")
        return GatheredLinear(self, rows, workspace, row_consumer)

    def project_into(self, values: torch.Tensor, output: torch.Tensor) -> None:
        """Write inference rows while preserving scale communication and accumulation."""

        if output.shape != (*values.shape[:-1], self.output_size) or output.dtype != values.dtype:
            raise ValueError("projection output must match the input rows, dtype and output width")
        if self.input_scale_groups:
            output.copy_(self(values))
        else:
            self.quant_method.apply_into(self, values, output)

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
        if self.quant_method.input_scale_domain != "tensor":
            flat = x.reshape(-1, x.shape[-1])
            projected = x.new_empty((flat.shape[0] * group.world_size, self.output_size))
            for interval, values in group.gather_row_chunks(flat, workspace):
                self.project_into(values, projected[interval])
            return projected.view(x.shape[0] * group.world_size, *x.shape[1:-1], self.output_size)
        scale = self.quant_method.input_scale(x)
        if scale is not None and self.quant_method.input_scale_domain == "tensor":
            group.all_reduce_max(scale)
        prepared = self.quant_method.prepare_input(x, scale, block_scale_layout="linear")
        if prepared.block_scales is not None and prepared.block_scale_layout != "linear":
            raise ValueError("sequence row exchange requires row-addressable block scales")
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
        row_scales = gather(prepared.row_scales) if prepared.row_scales is not None else None
        return self.quant_method.apply_prepared(
            self,
            PreparedLinearInput(
                values, scales, prepared.tensor_scale, row_scales, prepared.block_scale_layout
            ),
            output_dtype=x.dtype,
        )


def project_with_deferred_bias(
    linear: nn.Module,
    value: torch.Tensor,
    *,
    absmax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Project values and expose quantized bias for a fused numerical consumer.

    Dense providers apply bias in their GEMM epilogue. Quantized projections
    return the unapplied channel bias after any required row-shard reduction.
    """

    if isinstance(linear, LinearBase) and linear.quant_method.is_quantized:
        prepared = linear.prepare_input(value, absmax=absmax)
        output = linear.forward_prepared(prepared, include_bias=False)
        if isinstance(linear, RowParallelLinear):
            output = linear.weight_group.all_reduce(output)
        return output, linear.bias
    return linear(value), None


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

    Real tensor-parallel collectives remain outside this class. At tp=1 this
    uses the complete weight matrix; at tp>1 it provides the load-time shard
    layout needed by quantized/checkpoint loaders.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        layer_config: LayerConfig,
        quant_method: LinearMethod | None = None,
        bias: bool = True,
        prefix: str = "",
    ) -> None:
        """Shard the output dimension across tensor-parallel ranks at construction time."""

        parallel = layer_config.communicator
        self.global_input_size = int(input_size)
        self.global_output_size = int(output_size)
        local_output = divide(self.global_output_size, parallel.world_size)
        super().__init__(
            input_size,
            local_output,
            layer_config=layer_config,
            quant_method=quant_method,
            bias=bias,
            prefix=prefix,
            weight_group=layer_config.communicator,
            weight_shard_axis=0,
            weight_output_offset=parallel.rank_in_group * local_output,
            weight_global_output=self.global_output_size,
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
        quant_method: LinearMethod | None = None,
    ) -> None:
        """Shard the input dimension and configure tensor-parallel output reduction."""

        parallel = layer_config.communicator
        self.global_input_size = int(input_size)
        self.global_output_size = int(output_size)
        local_input = divide(self.global_input_size, parallel.world_size)
        super().__init__(
            local_input,
            output_size,
            layer_config=layer_config,
            quant_method=quant_method,
            bias=bias,
            prefix=prefix,
            weight_group=layer_config.communicator,
            weight_shard_axis=1,
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
                ShardPlan(shard=shard_for(0, parallel, replicated=parallel.world_size > 1)),
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

        out = self.weight_group.all_reduce(out)
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
        quant_method: LinearMethod | None = None,
    ):
        """Pack multiple output branches while retaining their global and local extents."""

        parallel = layer_config.communicator
        self.global_output_sizes = tuple(int(s) for s in output_sizes)
        self.output_sizes = (
            tuple(int(s) for s in local_output_sizes)
            if local_output_sizes is not None
            else tuple(divide(size, parallel.world_size) for size in self.global_output_sizes)
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
            weight_group=layer_config.communicator,
            weight_shard_axis=0,
            weight_global_output=sum(self.global_output_sizes),
        )
        slots: dict[int | str, ShardSlot] = {}
        cursor = 0
        for idx, (size, global_size) in enumerate(zip(self.output_sizes, self.global_output_sizes)):
            partitions = divide(global_size, size)
            replicas = divide(parallel.world_size, partitions)
            slots[idx] = ShardSlot(
                offset=cursor,
                size=size,
                shard=Shard(
                    axis=0,
                    rank=parallel.rank_in_group // replicas,
                    size=partitions,
                    replicated=partitions == 1,
                ),
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

    def forward_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return named packed branches while respecting their weight representation.

        A single dense row uses the independent projection GEMVs. Quantized
        weights retain their common preparation, scale and GEMM contract before
        the logical output is split; packed bytes are never treated as dense weights.
        """

        if not self.quant_method.is_quantized and x.numel() == self.input_size:
            weights = self.weight.split(self.output_sizes, dim=0)
            biases = (
                (None,) * len(weights)
                if self.bias is None
                else self.bias.split(self.output_sizes, dim=0)
            )
            return tuple(
                torch.nn.functional.linear(x, weight, bias)
                for weight, bias in zip(weights, biases, strict=True)
            )
        return self(x).split(self.output_sizes, dim=-1)


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
        quant_method: LinearMethod | None = None,
        bias: bool = True,
        prefix: str = "",
        sequence_group: Communicator | None = None,
        input_scale_group: Communicator | None = None,
    ) -> None:
        """Partition fixed-width groups from every branch across tensor-parallel ranks."""

        parallel = layer_config.communicator
        branch_output_size = int(branch_output_size)
        branches = int(branches)
        group_width = int(group_width)
        if branches <= 0 or group_width <= 0:
            raise ValueError("interleaved merged linear dimensions must be positive")
        if branch_output_size % group_width:
            raise ValueError("branch output size must divide into fixed-width groups")
        local_branch = divide(branch_output_size, parallel.world_size)
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
            weight_group=layer_config.communicator,
            weight_shard_axis=0,
            weight_output_offset=parallel.rank_in_group * local_branch * branches,
            weight_global_output=branch_output_size * branches,
            sequence_group=sequence_group,
            input_scale_group=input_scale_group,
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
                rank=parallel.rank_in_group,
                size=parallel.world_size,
                branches=branches,
                group_width=group_width,
            ),
        )


def local_attention_head_count(
    total_heads: int, *, parallel: Communicator, sequence: Communicator | None = None
) -> int:
    """Return this tensor-parallel rank's query-head count."""
    return divide(
        int(total_heads), parallel.world_size * (1 if sequence is None else sequence.world_size)
    )


def local_kv_head_count(
    total_kv_heads: int, *, parallel: Communicator, sequence: Communicator | None = None
) -> int:
    """Return local KV heads after tensor sharding and optional row exchange.

    A head is replicated across adjacent query owners when membership exceeds
    the KV head count. It remains one logical head in cache publications.
    """

    count = HeadRowExchange(parallel).head_region(int(total_kv_heads))[0]
    return count if sequence is None else HeadRowExchange(sequence).head_region(count)[0]


def local_kv_head_offset(
    total_kv_heads: int, *, parallel: Communicator, sequence: Communicator | None = None
) -> int:
    """Return the global coordinate of the rank's actual KV head interval."""

    count, offset = HeadRowExchange(parallel).head_region(int(total_kv_heads))
    return offset if sequence is None else offset + HeadRowExchange(sequence).head_region(count)[1]


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
        quant_method: LinearMethod | None = None,
        bias: bool = True,
        prefix: str = "",
        packed_names: tuple[str, str, str] = ("q_proj", "k_proj", "v_proj"),
    ) -> None:
        """Derive rank-local query and KV head counts before packing their projections."""

        parallel = layer_config.communicator
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        q_size = total_num_heads * head_size
        kv_size = total_num_kv_heads * head_size
        q_size_local = local_attention_head_count(total_num_heads, parallel=parallel) * head_size
        kv_size_local = local_kv_head_count(total_num_kv_heads, parallel=parallel) * head_size
        # The merged loader records each branch's distinct partition count;
        # adjacent query owners may therefore load the same single K/V head.
        super().__init__(
            hidden_size,
            (q_size, kv_size, kv_size),
            layer_config=layer_config,
            quant_method=(
                layer_config.quant_method(prefix, packed_names=packed_names)
                if quant_method is None
                else quant_method
            ),
            bias=bias,
            prefix=prefix,
            local_output_sizes=(q_size_local, kv_size_local, kv_size_local),
            weight_mode=WeightMode.FUSED_QKV_LINEAR,
        )
