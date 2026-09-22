"""Matrix projections with explicit numerical placements and named branches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from math import sqrt

import torch
from torch import nn
from torch.distributed.tensor import Replicate, Shard

from uniserve.distributed import Communicator, DeviceMesh, Distribution
from uniserve.quantization import QuantizedTensor, Quantizer

from . import _binding
from .functional._linear import apply_linear, apply_merged_linear
from .functional._tensors import as_matrix


def _distribution(
    group: Communicator | None = None, dim: int | None = None
) -> Distribution:
    """View a communicator as a single-axis tensor-parallel distribution."""
    group = Communicator() if group is None else group
    mesh = DeviceMesh(
        ranks=group.ranks,
        shape=(group.size,),
        axes=("tp",),
        rank=group.global_rank,
    )
    object.__setattr__(mesh, "_groups", {("tp",): group})
    return Distribution(mesh, (Replicate() if dim is None else Shard(dim),))


def _input(module: Linear, x: torch.Tensor) -> torch.Tensor:
    """Interpret leading axes as logical matrix rows before choosing
    statistics.
    """  # noqa: D205
    if module.input_quantizer is None or isinstance(x, QuantizedTensor):
        return x
    matrix = as_matrix(x)
    binding = _binding.matmul.get().get(id(module))
    if binding is not None:
        return binding.quantize(
            matrix, module.input_quantizer, module.input_distribution
        )
    return module.input_quantizer.quantize(
        matrix, distribution=module.input_distribution
    )


def _coalesce(projections, *, assigned=None):
    """Make compatible logical parameters views of one resident backing.

    Parameter identities and branch scale domains remain unchanged. A shared
    parameter cannot occupy two different fused orders; its first placement
    owns the backing and other compositions borrow the individual matrices.
    """
    assigned = set() if assigned is None else assigned
    for field in ("weight", "bias"):
        parameters = tuple(
            getattr(branch, field) for branch in projections.values()
        )
        if any(value is None or id(value) in assigned for value in parameters):
            continue
        if len({id(value) for value in parameters}) != len(parameters):
            continue

        encoded = tuple(
            isinstance(value, QuantizedTensor) for value in parameters
        )
        if any(encoded) and not all(encoded):
            continue
        if all(encoded):
            if any(value.quantizer.format != "fp8" for value in parameters):
                continue
            tensors = tuple(value.buffers()["values"] for value in parameters)
        else:
            tensors = parameters

        first = tensors[0]
        if any(
            value.dtype != first.dtype
            or value.device != first.device
            or value.shape[1:] != first.shape[1:]
            for value in tensors
        ):
            continue

        backing = torch.cat(tensors, dim=0)
        for parameter, tensor, view in zip(
            parameters,
            tensors,
            backing.split([value.shape[0] for value in tensors]),
            strict=True,
        ):
            # Encoded tensors expose ordinary storage fields. Updating that
            # field preserves every QuantizedTensor/Parameter alias as well.
            tensor.data = view
            assigned.add(id(parameter))


class Linear(nn.Module):
    """Compute a local projection over a logical weight[N, K] parameter.

    Construction follows ordinary module composition. Parallel binding changes
    local parameter storage before loading; calls borrow their active operator
    without retaining an execution context.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if min(in_features, out_features) < 1:
            raise ValueError("linear dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=False,
            )
            if bias
            else None
        )
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(
                self.bias, -1 / sqrt(in_features), 1 / sqrt(in_features)
            )
        self.input_quantizer: Quantizer | None = None
        self.input_distribution = _distribution()
        self.weight_distribution = _distribution()
        self.output_distribution = _distribution()
        self._weight_slice = (slice(0, out_features), slice(0, in_features))

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (*x.shape[:-1], self.weight.shape[0])
        dtype = x.dtype if output_dtype is None else output_dtype
        _check_output(out, shape, dtype, x.device)
        encoded = _input(self, x)

        if encoded is x:
            return apply_linear(
                x,
                self.weight,
                self.bias,
                output_dtype=dtype,
                out=out,
                operator=_binding.matmul.get().get(id(self)),
            )

        # Quantization flattens leading axes into GEMM rows; a contiguous
        # caller output can absorb the result directly at that row shape.
        target = (
            out.reshape(-1, shape[-1])
            if out is not None and out.is_contiguous()
            else None
        )
        result = apply_linear(
            encoded,
            self.weight,
            self.bias,
            output_dtype=dtype,
            out=target,
            operator=_binding.matmul.get().get(id(self)),
        ).reshape(shape)
        return result if out is None else out.copy_(result)


def _check_output(out, shape, dtype, device):
    if out is not None and (
        out.shape != shape or out.dtype != dtype or out.device != device
    ):
        raise ValueError("projection output must match shape, dtype and device")


class ColumnParallelLinear(Linear):
    """Project replicated channels into the local output-channel shard."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: Communicator | None = None,
        input_distribution: Distribution | None = None,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.group = Communicator() if group is None else group
        self.communication_groups = (self.group,)
        if out_features % self.group.size:
            raise ValueError(
                "column output channels must divide the tensor-parallel group"
            )

        width = out_features // self.group.size
        super().__init__(
            in_features, width, bias=bias, device=device, dtype=dtype
        )
        self.out_features = out_features
        start = self.group.rank * width
        self._weight_slice = (
            slice(start, start + width),
            slice(0, in_features),
        )
        self.input_distribution = input_distribution or _distribution(
            self.group
        )
        self.weight_distribution = _distribution(self.group, 0)
        self.output_distribution = _distribution(self.group, 1)

    def forward_chunks(
        self,
        x: torch.Tensor | Iterator[tuple[slice, torch.Tensor]],
        *,
        token_slice: slice,
        num_tokens: int,
        output_dtype: torch.dtype | None = None,
    ) -> Iterator[tuple[slice, torch.Tensor]]:
        """Project gathered token intervals as the input exchange yields
        them.
        """  # noqa: D205
        from ._chunks import projection_inputs

        for interval, values in projection_inputs(
            self, x, token_slice, num_tokens
        ):
            yield interval, self(values, output_dtype=output_dtype)


class RowParallelLinear(Linear):
    """Sum local contraction shards, then add the replicated bias once."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: Communicator | None = None,
        input_distribution: Distribution | None = None,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.group = Communicator() if group is None else group
        self.communication_groups = (self.group,)
        if in_features % self.group.size:
            raise ValueError(
                "row input channels must divide the tensor-parallel group"
            )

        width = in_features // self.group.size
        super().__init__(
            width, out_features, bias=bias, device=device, dtype=dtype
        )
        self.in_features = in_features
        start = self.group.rank * width
        self._weight_slice = (
            slice(0, out_features),
            slice(start, start + width),
        )
        self.input_distribution = input_distribution or _distribution(
            self.group, 1
        )
        self.weight_distribution = _distribution(self.group, 1)
        self.output_distribution = _distribution(self.group)

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.group.size == 1:
            # A singleton partition is the ordinary affine map: dense bias
            # belongs in GEMM before its final output rounding. Only multiple
            # input shards require bias after the sum of their local products.
            return super().forward(x, output_dtype=output_dtype, out=out)

        shape = (*x.shape[:-1], self.weight.shape[0])
        dtype = x.dtype if output_dtype is None else output_dtype
        _check_output(out, shape, dtype, x.device)
        encoded = _input(self, x)

        target = (
            out
            if encoded is x
            else (
                out.reshape(-1, shape[-1])
                if out is not None and out.is_contiguous()
                else None
            )
        )
        result = apply_linear(
            encoded,
            self.weight,
            output_dtype=dtype,
            out=target,
            operator=_binding.matmul.get().get(id(self)),
        ).reshape(shape)

        self.group.all_reduce(result)
        if self.bias is not None:
            result.add_(self.bias.to(dtype))
        return result if out is None else out.copy_(result)

    def forward_chunks(
        self,
        chunks: Iterator[tuple[slice, torch.Tensor]],
        *,
        output_dtype: torch.dtype | None = None,
    ) -> Iterator[tuple[slice, torch.Tensor]]:
        """Reduce each gathered interval, quantizing whole-tensor scales
        jointly.
        """  # noqa: D205
        from ._chunks import tensor_statistics

        if not tensor_statistics(self.input_quantizer):
            for interval, values in chunks:
                yield interval, self(values, output_dtype=output_dtype)
            return

        # Tensor scales span every supplied row, even when publication order
        # differs from logical token order. Wait for that numerical dependency.
        rows = list(chunks)
        if all(isinstance(values, QuantizedTensor) for _, values in rows):
            for interval, values in rows:
                yield interval, self(values, output_dtype=output_dtype)
            return

        maximum = torch.zeros(
            (), dtype=torch.float32, device=self.weight.device
        )
        for _, values in rows:
            if values.numel():
                maximum = torch.maximum(maximum, values.abs().amax().float())
        for axis in (
            *self.input_distribution.shard_axes(0),
            *self.input_distribution.shard_axes(1),
        ):
            self.input_distribution.mesh.get_group(axis).all_reduce(
                maximum, op="max"
            )

        for interval, values in rows:
            encoded = self.input_quantizer.quantize(
                as_matrix(values), amax=maximum
            )
            yield interval, self(encoded, output_dtype=output_dtype)


class MergedColumnParallelLinear(nn.Module):
    """Fuse named column projections while preserving each logical parameter."""

    def __init__(
        self,
        in_features: int,
        outputs: Mapping[str, int],
        *,
        branch_width: int | None = None,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if not outputs or any(not name or "." in name for name in outputs):
            raise ValueError(
                "merged projections require nonempty module branch names"
            )
        if branch_width is not None and (
            type(branch_width) is not int
            or branch_width < 1
            or any(width % branch_width for width in outputs.values())
        ):
            raise ValueError(
                "branch group width must positively divide every output branch"
            )
        self.projections = nn.ModuleDict(
            {
                name: ColumnParallelLinear(
                    in_features, width, bias=bias, device=device, dtype=dtype
                )
                for name, width in outputs.items()
            }
        )
        self.branch_width = branch_width
        _coalesce(self.projections)

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
        out: Mapping[str, torch.Tensor] | None = None,
    ) -> Mapping[str, torch.Tensor]:
        if out is not None and set(out) != set(self.projections):
            raise ValueError(
                "merged output names must match the projection branches"
            )

        branches = tuple(self.projections.values())
        dtype = x.dtype if output_dtype is None else output_dtype
        for name, branch in self.projections.items():
            _check_output(
                None if out is None else out[name],
                (*x.shape[:-1], branch.weight.shape[0]),
                dtype,
                x.device,
            )

        # Branches with divergent input quantization encode separately and
        # cannot share one fused GEMM over a common encoded input.
        if any(
            branch.input_quantizer != branches[0].input_quantizer
            for branch in branches[1:]
        ):
            return {
                name: branch(
                    x,
                    output_dtype=dtype,
                    out=None if out is None else out[name],
                )
                for name, branch in self.projections.items()
            }

        encoded = _input(branches[0], x)
        targets = None if encoded is not x or out is None else out
        values = apply_merged_linear(
            encoded,
            {name: branch.weight for name, branch in self.projections.items()},
            {name: branch.bias for name, branch in self.projections.items()},
            branch_width=self.branch_width,
            output_dtype=dtype,
            out=targets,
            operator=_binding.merged_matmul.get().get(id(self)),
        )

        result = {
            name: value.reshape(*x.shape[:-1], value.shape[-1])
            for name, value in values.items()
        }
        if out is not None:
            for name, value in result.items():
                out[name].copy_(value)
            return out
        return result

    def forward_chunks(
        self,
        x: torch.Tensor | Iterator[tuple[slice, torch.Tensor]],
        *,
        token_slice: slice,
        num_tokens: int,
        output_dtype: torch.dtype | None = None,
    ) -> Iterator[tuple[slice, Mapping[str, torch.Tensor]]]:
        """Project gathered token intervals, fusing branches that share
        quantization.
        """  # noqa: D205
        from ._chunks import (
            _assemble,
            _partition,
            materialize_input,
            projection_inputs,
        )

        branches = tuple(self.projections.values())
        if len({branch.input_quantizer for branch in branches}) != 1:
            if not isinstance(x, torch.Tensor):
                x = _assemble(
                    x,
                    token_slice,
                    branches[0].weight.shape[1],
                    branches[0].weight,
                )
            _, _, domain = _partition(branches[0], x, token_slice, num_tokens)
            with materialize_input(
                branches[0], x, token_slice, num_tokens
            ) as values:
                yield domain, self(values, output_dtype=output_dtype)
            return

        for interval, values in projection_inputs(
            branches[0], x, token_slice, num_tokens
        ):
            yield interval, self(values, output_dtype=output_dtype)


class QKVParallelLinear(MergedColumnParallelLinear):
    """Named Q/K/V projections with whole-head partitioning and GQA
    replication.
    """  # noqa: D205

    def __init__(
        self,
        in_features: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if (
            min(num_heads, num_kv_heads, head_dim) < 1
            or num_heads % num_kv_heads
        ):
            raise ValueError(
                "query heads must be a positive multiple of KV heads"
            )
        super().__init__(
            in_features,
            {
                "q": num_heads * head_dim,
                "k": num_kv_heads * head_dim,
                "v": num_kv_heads * head_dim,
            },
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def _padded_vocabulary(size: int, members: int) -> int:
    from math import lcm

    multiple = lcm(64, members)
    return (size + multiple - 1) // multiple * multiple


def _vocabulary(size: int, group: Communicator):
    from uniserve.model.logits import VocabShard

    padded = _padded_vocabulary(size, group.size)
    width = padded // group.size
    return VocabShard(
        size, slice(group.rank * width, (group.rank + 1) * width), padded, group
    )


class VocabParallelEmbedding(nn.Module):
    """Embed owned vocabulary IDs, masking padding before a numerical sum."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        group: Communicator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if min(num_embeddings, embedding_dim) < 1:
            raise ValueError("embedding dimensions must be positive")
        if padding_idx is not None and not 0 <= padding_idx < num_embeddings:
            raise ValueError("padding token must be within the vocabulary")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.group = Communicator() if group is None else group
        self.communication_groups = (self.group,)
        self.vocab = _vocabulary(num_embeddings, self.group)
        width = self.vocab.local_slice.stop - self.vocab.local_slice.start
        self.weight = nn.Parameter(
            torch.empty(width, embedding_dim, device=device, dtype=dtype),
            requires_grad=False,
        )
        nn.init.normal_(self.weight)
        start = self.vocab.local_slice.start
        self.weight[max(0, num_embeddings - start) :].zero_()
        if padding_idx is not None and start <= padding_idx < start + width:
            self.weight[padding_idx - start].zero_()
        self.weight_distribution = _distribution(self.group, 0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # IDs owned by other shards embed as zeros here; the all-reduce sums
        # every shard's partial rows into the complete embedding.
        region = self.vocab.local_slice
        outside = (input_ids < region.start) | (
            input_ids >= min(region.stop, self.vocab.size)
        )
        local_ids = (input_ids - region.start).masked_fill(outside, 0)
        result = torch.nn.functional.embedding(local_ids, self.weight)
        result.masked_fill_(outside.unsqueeze(-1), 0)
        return self.group.all_reduce(result)


class VocabParallelHead(ColumnParallelLinear):
    """Project local padded vocabulary columns; Logits.gather gathers
    explicitly.
    """  # noqa: D205

    def __init__(
        self,
        in_features: int,
        num_embeddings: int,
        *,
        group: Communicator | None = None,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        group = Communicator() if group is None else group
        if num_embeddings < 1:
            raise ValueError("vocabulary size must be positive")
        super().__init__(
            in_features,
            _padded_vocabulary(num_embeddings, group.size),
            group=group,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.vocab = _vocabulary(num_embeddings, self.group)
        valid = max(0, num_embeddings - self.vocab.local_slice.start)
        self.weight[valid:].zero_()
        if self.bias is not None:
            self.bias[valid:].zero_()
