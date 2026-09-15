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
from uniserve.quantization import QuantizedTensor

from ._tokens import _HeadExchange
from .distribution import Distribution
from .mesh import DeviceMesh


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
) -> None:
    """Partition resident numerical parameters on an already bound mesh.

    Call before loading or quantizing weights. Shared Parameters with the same
    partition remain identical objects. Rebinding to a different partition is
    rejected because discarded source values cannot be reconstructed locally.
    Attention axis names select token placements; TP owns matrix channels.
    """
    if mesh.rank not in mesh.ranks:
        raise ValueError("a nonparticipating rank cannot bind resident layers")

    # Token axes named by the attention config must be declared by the mesh
    # and stay distinct from the channel (tp) and stage (pp) partitions.
    selected = set()
    if attention.heads is not None:
        selected.add(attention.heads.axis)
    if attention.context is not None:
        selected.update(
            axis
            for axis in (
                attention.context.gather_axis,
                attention.context.peer_axis,
            )
            if axis is not None
        )
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
    seen = set()

    def visit(child):
        if id(child) in seen:
            return
        seen.add(id(child))
        if child is not module and isinstance(child, Encoder):
            parallelize_(child, mesh)
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
    for child in modules:
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
    kv = {}
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
            for branch in child.projections.values():
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
            if child.num_kv_heads >= group.size:
                if child.num_kv_heads % group.size:
                    raise ValueError(
                        "KV heads must divide their tensor-parallel group"
                    )
            elif group.size % child.num_kv_heads:
                raise ValueError(
                    "KV heads must replicate evenly across tensor-parallel "
                    "ranks"
                )
            for name in ("k", "v"):
                kv[id(child.projections[name])] = (
                    child.num_kv_heads,
                    child.head_dim,
                )

    for child in modules:
        if isinstance(child, VsaAttention):
            bound = getattr(child, "_parallel_mesh", None)
            if bound is not None and (
                bound != mesh or child._attention_parallel != attention
            ):
                raise ValueError("VSA cannot change its mathematical partition")
            if bound is None:
                child.mesh = mesh
                child._parallel = ParallelAttention(
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
            if child.num_kv_heads >= group.size:
                if child.num_kv_heads % group.size:
                    raise ValueError(
                        "attention KV heads must divide tensor parallelism"
                    )
                heads = child.num_kv_heads // group.size
                start = group.rank * heads
            else:
                # Fewer KV heads than TP ranks: several ranks host adjacent
                # copies of the same head, indexed by this rank's copy slot.
                if group.size % child.num_kv_heads:
                    raise ValueError(
                        "attention KV replicas must divide tensor parallelism"
                    )
                heads = 1
                start = group.rank // (group.size // child.num_kv_heads)

            exchange = _HeadExchange(
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
            child._local_heads = query_heads // exchange.group.size
            child._local_kv_heads = head_slice.stop - head_slice.start
            child._head_indices = tuple(
                range(start + head_slice.start, start + head_slice.stop)
            )
            child._exchange = exchange
            if attention.context is not None:
                child._context = ParallelAttention(
                    mesh=mesh, parallel=attention
                )
            child._parallel_mesh = mesh
            child._attention_parallel = attention
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
            heads = kv.get(id(child))
            if heads is not None and heads[0] < group.size:
                replicas = group.size // heads[0]
                start = group.rank // replicas * heads[1]
                row = slice(start, start + heads[1])
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
                child._gather_axes = (
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
