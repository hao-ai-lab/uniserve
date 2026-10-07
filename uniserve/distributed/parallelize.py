"""Bind mathematical layer partitions without owning communication resources."""

from __future__ import annotations

from types import MappingProxyType

import torch
from torch import nn
from torch.distributed.tensor import Replicate, Shard

from uniserve.nn.attention._parallel import ParallelAttention
from uniserve.nn.attention.config import AttentionParallelConfig
from uniserve.nn.attention.layer import Attention
from uniserve.nn.attention.vsa import Attention as VsaAttention
from uniserve.nn.attention.vsa import RegionAttention
from uniserve.nn.linear import (
    ColumnParallelLinear,
    Linear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelHead,
    _padded_vocabulary,
    _vocabulary,
)
from uniserve.nn.moe import FusedMoE
from uniserve.quantization import QuantizedTensor

from .distribution import Distribution
from .mesh import Communicator, DeviceMesh
from .tokens import HeadExchange, kv_head_partition


def communication_axes(
    module: nn.Module,
    mesh: DeviceMesh,
    *,
    attention: AttentionParallelConfig = AttentionParallelConfig(),
) -> tuple[tuple[str, ...], ...]:
    """Declare the fibers used by numerical partitioning and quantization.

    This query reads an unpartitioned model, including a meta model. Runtime
    owners create these groups on every rank before loading local parameters.
    It does not create resources or depend on a rank's resident pipeline stage.
    """
    return tuple(
        sorted(
            {
                axes
                for requirements in _communication_axes(
                    module, mesh, attention
                ).values()
                for axes in requirements
            }
        )
    )


def _communication_axes(module, mesh, attention):
    from uniserve.model import Denoiser, Encoder, TransformerDecoder

    requirements = {}

    def visit(child, config, projected=False):
        required = set()

        def add(*axes):
            selected = tuple(axis for axis in mesh.axes if axis in axes)
            if mesh.size(selected) > 1:
                required.add(selected)

        if child is not module and isinstance(child, Encoder):
            config = AttentionParallelConfig()
        head = None if config.heads is None else config.heads.axis
        context = None if config.context is None else config.context.gather_axis
        tokens = tuple(axis for axis in mesh.axes if axis in (head, context))
        if isinstance(child, (Denoiser, TransformerDecoder)):
            add("pp")
            add(*tokens)
        if isinstance(child, (Attention, VsaAttention, RegionAttention)):
            add(head)
            add(context)
        if isinstance(child, FusedMoE):
            # Tensor-parallel shards of I sum once after combining experts.
            add("tp")
        if isinstance(child, (Linear, VocabParallelEmbedding)):
            # Quantization reduces statistics across each sharded dimension.
            for axis in tokens:
                add(axis)
            if isinstance(
                child,
                (
                    ColumnParallelLinear,
                    RowParallelLinear,
                    VocabParallelEmbedding,
                ),
            ):
                add("tp")
            if isinstance(child, ColumnParallelLinear):
                add(head) if projected else add(*tokens)
        branches = (
            isinstance(child, MergedColumnParallelLinear)
            and child.branch_width is not None
        )
        if branches:
            add("tp", head)
        for nested in child.children():
            visit(nested, config, projected or branches)
        requirements[child] = tuple(sorted(required))

    visit(module, attention)
    return requirements


def communicators(module: nn.Module) -> tuple:
    """Read declared borrowed groups, deduplicated by physical instance.

    Numerical bindings and custom modules expose ``communication_groups``.
    No other module attributes participate in communication discovery.
    """
    groups = {}
    for child in module.modules():
        for group in getattr(child, "communication_groups", ()):
            if group.size > 1:
                groups[group._require()] = group
    return tuple(groups.values())


def _partition_experts(module, mesh, group, attention) -> None:
    """Keep this rank's interval of every expert's intermediate channels.

    Each rank holds the same experts with ``I / tp`` of their up, gate and
    down channels; ``up_gate`` keeps the local up rows followed by the local
    gate rows, preserving the resident order.
    """
    bound = getattr(module, "_parallel_mesh", None)
    if bound is not None:
        if bound != mesh or module._attention_parallel != attention:
            raise ValueError(
                "a resident layer cannot change its mathematical partition"
            )
        return
    intermediate = module.intermediate_size
    if intermediate % group.size:
        raise ValueError(
            "expert intermediate channels must divide the tensor-parallel group"
        )
    if isinstance(module.up_gate.weight, QuantizedTensor):
        raise ValueError("parallel binding must precede weight quantization")
    width = intermediate // group.size
    local = slice(group.rank * width, (group.rank + 1) * width)
    if width != intermediate:
        up_gate, down = module.up_gate.weight, module.down.weight
        module.up_gate.weight = nn.Parameter(
            torch.cat(
                (
                    up_gate[:, local],
                    up_gate[
                        :,
                        intermediate + local.start : intermediate + local.stop,
                    ],
                ),
                dim=1,
            ),
            requires_grad=False,
        )
        module.down.weight = nn.Parameter(
            down[:, :, local].contiguous(), requires_grad=False
        )
        module.up_gate.out_features = 2 * width
        module.down.in_features = width
    module.intermediate_slice = local
    module.group = group
    module._parallel_mesh = mesh
    module._attention_parallel = attention


def partition_experts(model: nn.Module, group: Communicator) -> None:
    """Shard every ``FusedMoE`` of ``model`` over the expert group ``group``.

    Rank ``r`` of ``P`` keeps the contiguous global experts
    ``[r * E / P, (r + 1) * E / P)`` of every layer, the placement the
    FlashInfer and TensorRT-LLM all-to-all kernels route tokens by. Each
    stacked weight keeps those experts only; routing still names global
    experts. Binding happens on the meta skeleton, before weights load, so a
    rank reads only its own experts' tensors.

    Raises:
        ValueError: A layer's expert count does not divide the group, the
            layer is already tensor-parallel or expert-parallel, or its
            weights are already quantized.
    """
    if group.size == 1:
        return
    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        if module.expert_group.size > 1 or module.group.size > 1:
            raise ValueError(
                "an expert layer is partitioned over one group at a time"
            )
        if module.num_experts % group.size:
            raise ValueError(
                f"{module.num_experts} experts do not divide an expert group "
                f"of {group.size} ranks"
            )
        if isinstance(module.up_gate.weight, QuantizedTensor):
            raise ValueError("expert binding must precede weight quantization")

        count = module.num_experts // group.size
        local = slice(group.rank * count, (group.rank + 1) * count)
        for linear in (module.up_gate, module.down):
            linear.weight = nn.Parameter(
                linear.weight[local].contiguous(), requires_grad=False
            )
            linear.num_experts = count
        module.expert_slice = local
        module.expert_group = group


def _replicated_heads(mesh: DeviceMesh, heads: int) -> Distribution:
    """Factor TP into distinct KV heads and adjacent copies of each head."""
    index = mesh.axes.index("tp")
    replica_axis = "kv_replica"
    if replica_axis in mesh.axes:
        raise ValueError("KV replication requires an unambiguous replica axis")

    axes = (*mesh.axes[:index], "tp", replica_axis, *mesh.axes[index + 1 :])
    shape = (
        *mesh.shape[:index],
        heads,
        mesh.size("tp") // heads,
        *mesh.shape[index + 1 :],
    )
    result = DeviceMesh(
        ranks=mesh.ranks, shape=shape, axes=axes, rank=mesh.rank
    )
    # Existing fibers spanning TP also span its two factors. No process group
    # is created by a mathematical binding; statistical max can include the
    # replica factor because every replica represents identical logical values.
    groups = {
        tuple(
            part
            for axis in selected
            for part in (("tp", replica_axis) if axis == "tp" else (axis,))
        ): group
        for selected, group in mesh._groups.items()
    }
    object.__setattr__(result, "_groups", MappingProxyType(groups))
    object.__setattr__(result, "_device", mesh._device)
    return Distribution(
        result,
        tuple(Shard(0) if axis == "tp" else Replicate() for axis in axes),
    )


def parallelize_(
    module: nn.Module,
    mesh: DeviceMesh,
    *,
    attention: AttentionParallelConfig = AttentionParallelConfig(),
    exclude: frozenset[nn.Module] = frozenset(),
) -> None:
    """Partition resident numerical parameters on an already bound mesh.

    Call before loading or quantizing weights. Shared Parameters with the same
    partition remain identical objects. Rebinding to a different partition is
    rejected because discarded source values cannot be reconstructed locally.
    Attention axis names select token placements; TP owns matrix channels.
    ``exclude`` leaves the named subtrees, including shared aliases, on their
    own mathematical partition. In particular, remote experts do not inherit
    the calling attention module's tensor-parallel reduction.
    """
    if mesh.rank not in mesh.ranks:
        raise ValueError("a nonparticipating rank cannot bind resident layers")

    # Token axes named by the attention config must be declared by the mesh
    # and stay distinct from the channel (tp) and stage (pp) partitions.
    selected = set()
    if attention.heads is not None:
        selected.add(attention.heads.axis)
    if attention.context is not None:
        selected.add(attention.context.gather_axis)
    if not selected.issubset(mesh.axes) or selected.intersection({"tp", "pp"}):
        raise ValueError(
            "attention requires distinct token axes declared by the mesh"
        )

    # Deferred import: uniserve.model depends on the modules bound here.
    from uniserve.model import CausalLM, Encoder, TransformerDecoder

    # An encoder consumes complete samples through its own numerical call.
    # A surrounding denoiser's token partition does not partition those input
    # documents; each sequence coordinate receives the same refined features.
    modules = []
    seen = {id(nested) for child in exclude for nested in child.modules()}

    def visit(child):
        if id(child) in seen:
            return
        seen.add(id(child))
        if child is not module and isinstance(child, Encoder):
            parallelize_(child, mesh, exclude=exclude)
            return
        modules.append(child)
        for nested in child.children():
            visit(nested)

    visit(module)

    pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
    token_axes = tuple(axis for axis in mesh.axes if axis in selected)
    for child in modules:
        if isinstance(child.__dict__.get("mesh"), DeviceMesh):
            child.mesh = mesh
        if not isinstance(child, TransformerDecoder):
            continue

        bound = getattr(child, "_parallel_mesh", None)
        if bound is not None:
            if bound != mesh or child._attention_parallel != attention:
                raise ValueError(
                    "a decoder cannot change its mathematical partition"
                )
            continue

        # Keep only this pipeline stage's contiguous share of the layers.
        # Stage-boundary modules live on the stages that consume them.
        layers = tuple(child.layers)
        if len(layers) < pipeline.size:
            raise ValueError(
                "every pipeline stage requires at least one decoder layer"
            )
        start = len(layers) * pipeline.rank // pipeline.size
        stop = len(layers) * (pipeline.rank + 1) // pipeline.size
        child.layers = nn.ModuleDict(
            {name: child.layers[name] for name in layers[start:stop]}
        )
        if pipeline.rank != 0:
            child.embedding = None
        if pipeline.rank != pipeline.size - 1:
            child.norm = None

        child.mesh = mesh
        child._pipeline = pipeline
        child._tokens = mesh.get_group(token_axes)
        child._parallel_mesh = mesh
        child._attention_parallel = attention

    # Pipeline trimming detached layers held by other stages; only rebind
    # modules that remain reachable from the root.
    retained = {id(child) for child in module.modules()}
    modules = [child for child in modules if id(child) in retained]
    requirements = _communication_axes(module, mesh, attention)
    for child in modules:
        child.communication_groups = tuple(
            mesh.get_group(axes) for axes in requirements[child]
        )
    for child in modules:
        # Vocabulary heads project only on the last pipeline stage.
        if (
            isinstance(child, CausalLM)
            and child.backbone._pipeline.rank != pipeline.size - 1
        ):
            child.lm_head = None

    group = mesh.get_group("tp" if "tp" in mesh.axes else ())
    parameters = {}
    expanded = {}

    def expand_vocabulary(parameter, size, padded):
        if parameter is None or parameter.shape[0] == padded:
            return parameter
        key = (id(parameter), padded)
        if key not in expanded:
            value = torch.zeros(
                (padded, *parameter.shape[1:]),
                dtype=parameter.dtype,
                device=parameter.device,
            )
            value[:size].copy_(parameter[:size])
            expanded[key] = nn.Parameter(
                value, requires_grad=parameter.requires_grad
            )
        return expanded[key]

    def shard(parameter, region):
        if parameter is None:
            return None
        key = (id(parameter), tuple((part.start, part.stop) for part in region))
        if key not in parameters:
            if isinstance(parameter, QuantizedTensor):
                raise ValueError(
                    "parallel binding must precede weight quantization"
                )
            complete = all(
                part.start == 0 and part.stop == width
                for part, width in zip(region, parameter.shape, strict=True)
            )
            parameters[key] = (
                parameter
                if complete
                else nn.Parameter(
                    parameter[region].contiguous(),
                    requires_grad=parameter.requires_grad,
                )
            )
        return parameters[key]

    # Record per-module channel facts before partitioning any weights:
    # interleaved-branch projections and each QKV layer's KV head geometry.
    kv: dict[int, tuple[int, int, slice]] = {}
    projected = {}
    for child in modules:
        if (
            isinstance(child, MergedColumnParallelLinear)
            and child.branch_width is not None
        ):
            # Interleaved channel groups can project the consuming Ulysses
            # heads directly. Context coordinates retain separate query rows;
            # only the head axis gathers their input-token contributions.
            head_axis = (
                None if attention.heads is None else attention.heads.axis
            )
            axes = tuple(
                axis for axis in mesh.axes if axis in ("tp", head_axis)
            )
            projection_group = mesh.get_group(axes)
            for _, branch in child._branches():
                if branch.out_features % (
                    projection_group.size * child.branch_width
                ):
                    raise ValueError(
                        "projected heads must divide whole interleaved "
                        "channel groups"
                    )
                projected[id(branch)] = projection_group, axes, head_axis

        if isinstance(child, QKVParallelLinear):
            if child.num_heads % group.size:
                raise ValueError(
                    "query heads must divide the tensor-parallel group"
                )
            heads = kv_head_partition(child.num_kv_heads, group)
            for name in ("k", "v"):
                kv[id(child.projections[name])] = (
                    child.num_kv_heads,
                    child.head_dim,
                    heads,
                )

    for child in modules:
        if isinstance(child, RegionAttention):
            bound = getattr(child, "_parallel_mesh", None)
            if bound is not None:
                if bound != mesh or child._attention_parallel != attention:
                    raise ValueError(
                        "VSA cannot change its mathematical partition"
                    )
                continue
            # Region VSA attends the complete sequence for this rank's head
            # shard and returns rows to their Ulysses owners; it has no
            # context partition of its keys.
            if attention.context is not None:
                raise ValueError("region VSA has no context partition")
            child.exchange = HeadExchange(
                mesh.get_group(
                    () if attention.heads is None else attention.heads.axis
                )
            )
            child._parallel_mesh, child._attention_parallel = mesh, attention
            continue
        if isinstance(child, VsaAttention):
            bound = getattr(child, "_parallel_mesh", None)
            if bound is not None and (
                bound != mesh or child._attention_parallel != attention
            ):
                raise ValueError("VSA cannot change its mathematical partition")
            if bound is None:
                child.mesh = mesh
                child.parallel = ParallelAttention(
                    mesh=mesh, parallel=attention
                )
                child._parallel_mesh, child._attention_parallel = (
                    mesh,
                    attention,
                )
            continue
        if isinstance(child, Attention):
            bound = getattr(child, "_parallel_mesh", None)
            if bound is not None:
                if bound != mesh or child._attention_parallel != attention:
                    raise ValueError(
                        "attention cannot change its mathematical partition"
                    )
                continue

            if child.num_heads % group.size:
                raise ValueError(
                    "attention query heads must divide tensor parallelism"
                )
            kv_heads = kv_head_partition(child.num_kv_heads, group)
            heads, start = kv_heads.stop - kv_heads.start, kv_heads.start

            exchange = HeadExchange(
                mesh.get_group(
                    () if attention.heads is None else attention.heads.axis
                )
            )
            query_heads = child.num_heads // group.size
            if query_heads % exchange.group.size:
                raise ValueError(
                    "local query heads must divide Ulysses membership"
                )

            head_slice = exchange.head_slice(heads)
            child.local_heads = query_heads // exchange.group.size
            child.local_kv_heads = head_slice.stop - head_slice.start
            child.head_indices = tuple(
                range(start + head_slice.start, start + head_slice.stop)
            )
            child.exchange = exchange
            if attention.context is not None:
                child.context_parallel = ParallelAttention(
                    mesh=mesh, parallel=attention
                )
            child._parallel_mesh = mesh
            child._attention_parallel = attention
            continue
        if isinstance(child, FusedMoE):
            _partition_experts(child, mesh, group, attention)
            continue
        if not isinstance(child, (Linear, VocabParallelEmbedding)):
            continue
        bound = getattr(child, "_parallel_mesh", None)
        if bound is not None:
            if bound != mesh or child._attention_parallel != attention:
                raise ValueError(
                    "a resident layer cannot change its mathematical partition"
                )
            continue

        if isinstance(child, (VocabParallelEmbedding, VocabParallelHead)):
            if child.weight.shape[0] != child.vocab.padded_size:
                raise ValueError(
                    "parallelize_ requires an unpartitioned vocabulary source"
                )
            size = child.vocab.size
            padded = _padded_vocabulary(size, group.size)
            child.weight = expand_vocabulary(child.weight, size, padded)
            child.vocab = _vocabulary(size, group)
            if isinstance(child, VocabParallelEmbedding):
                child.weight = shard(
                    child.weight,
                    (child.vocab.local_slice, slice(0, child.embedding_dim)),
                )
                child.weight_distribution = Distribution(
                    mesh,
                    tuple(
                        Shard(0) if axis == "tp" else Replicate()
                        for axis in mesh.axes
                    ),
                )
                child.group = group
                child._parallel_mesh = mesh
                child._attention_parallel = attention
                continue
            child.bias = expand_vocabulary(child.bias, size, padded)
            child.out_features = padded
        if tuple(child.weight.shape) != (child.out_features, child.in_features):
            raise ValueError(
                "parallelize_ requires complete unpartitioned source parameters"
            )

        # Default partition: tokens shard along the selected token axes, and
        # the weight stays whole unless a column/row branch narrows it below.
        weight_dim = None
        input_places = [
            Shard(0) if axis in selected else Replicate() for axis in mesh.axes
        ]
        output_places = list(input_places)
        row = slice(0, child.out_features)
        column = slice(0, child.in_features)
        projection = projected.get(id(child))
        column_group = group if projection is None else projection[0]

        if isinstance(child, ColumnParallelLinear):
            weight_dim = 0
            kv_partition = kv.get(id(child))
            if kv_partition is not None and kv_partition[0] < group.size:
                # Replicated KV heads: this rank projects its one head.
                _, head_dim, local = kv_partition
                row = slice(local.start * head_dim, local.stop * head_dim)
            else:
                if child.out_features % column_group.size:
                    raise ValueError(
                        "column output channels must divide the "
                        "tensor-parallel group"
                    )
                width = child.out_features // column_group.size
                row = slice(
                    column_group.rank * width, (column_group.rank + 1) * width
                )
            if "tp" in mesh.axes:
                output_places[mesh.axes.index("tp")] = Shard(1)
            if projection is not None:
                if projection[2] is not None:
                    output_places[mesh.axes.index(projection[2])] = Shard(1)
                child.gather_axes = (
                    () if projection[2] is None else (projection[2],)
                )
        elif isinstance(child, RowParallelLinear):
            weight_dim = 1
            if child.in_features % group.size:
                raise ValueError(
                    "row input channels must divide the tensor-parallel group"
                )
            width = child.in_features // group.size
            column = slice(group.rank * width, (group.rank + 1) * width)
            if "tp" in mesh.axes:
                input_places[mesh.axes.index("tp")] = Shard(1)

        child.weight = shard(child.weight, (row, column))
        child.bias = shard(child.bias, (row,))
        child._weight_slice = (row, column)
        child.input_distribution = Distribution(mesh, tuple(input_places))
        child.output_distribution = Distribution(mesh, tuple(output_places))
        child.weight_distribution = Distribution(
            mesh,
            tuple(
                Shard(weight_dim)
                if weight_dim is not None
                and axis in (("tp",) if projection is None else projection[1])
                else Replicate()
                for axis in mesh.axes
            ),
        )

        if id(child) in kv and kv[id(child)][0] < group.size:
            child.weight_distribution = _replicated_heads(
                mesh, kv[id(child)][0]
            )
            output_mesh = child.weight_distribution.mesh
            child.output_distribution = Distribution(
                output_mesh,
                tuple(
                    Shard(1)
                    if axis == "tp"
                    else Shard(0)
                    if axis in selected
                    else Replicate()
                    for axis in output_mesh.axes
                ),
            )
        if isinstance(child, (ColumnParallelLinear, RowParallelLinear)):
            child.group = (
                column_group
                if isinstance(child, ColumnParallelLinear)
                else group
            )
        child._parallel_mesh = mesh
        child._attention_parallel = attention
