"""Dense and quantized projections over bound or standalone matmul operators.

``apply_linear`` and ``apply_merged_linear`` also accept an already-bound
operator; the layers in :mod:`uniserve.nn.linear` supply theirs through the
active execution context.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from uniserve.quantization import QuantizedTensor

from ._tensors import as_matrix


def _operator(weight, x, output_dtype):
    """Prepare a standalone matmul operator with its own workspace."""
    from uniserve.runtime.backends.matmul import resolve

    provider = resolve("auto", weight)
    quantizer = x.quantizer if isinstance(x, QuantizedTensor) else None
    configs = provider.workspace_buffers(
        weight,
        input_dtype=x.dtype,
        input_quantizer=quantizer,
        max_rows=x.shape[0],
        output_dtype=output_dtype,
    )
    workspace = {
        name: torch.empty(config.shape, dtype=config.dtype, device=x.device)
        for name, config in configs.items()
    }
    return provider.prepare(
        weight,
        input_dtype=x.dtype,
        input_quantizer=quantizer,
        max_rows=x.shape[0],
        output_dtype=output_dtype,
        workspace=workspace,
    )


def apply_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    operator=None,
) -> torch.Tensor:
    """Compute local ``x @ weight.T + bias`` without implicit collectives.

    Quantized operands carry their numerical scales. Bound operators borrow
    caller workspace; a standalone call prepares its own execution resources.
    Output storage, when supplied, must exactly match the result representation.
    """
    if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            "linear operands must agree on their contraction dimension"
        )
    shape = (*x.shape[:-1], weight.shape[0])
    dtype = x.dtype if output_dtype is None else output_dtype
    if out is not None and (
        out.shape != shape or out.dtype != dtype or out.device != x.device
    ):
        raise ValueError("linear output must match shape, dtype and device")
    if weight.device != x.device or (
        bias is not None
        and (bias.shape != (weight.shape[0],) or bias.device != x.device)
    ):
        raise ValueError(
            "linear weights and bias must match output channels and device"
        )

    result = (
        torch.empty(shape, dtype=dtype, device=x.device) if out is None else out
    )
    matrix = as_matrix(x)
    if operator is None:
        from uniserve.nn import _binding

        operator = _binding.matmul.get().get(id(weight))
    if operator is None:
        operator = _operator(weight, matrix, dtype)

    destination = (
        result
        if result.is_contiguous()
        else torch.empty_like(result, memory_format=torch.contiguous_format)
    )
    operator(
        matrix, bias, out=destination.view(matrix.shape[0], weight.shape[0])
    )
    return result if destination is result else result.copy_(destination)


def apply_merged_linear(
    x: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    biases: Mapping[str, torch.Tensor | None],
    *,
    branch_width: int | None = None,
    output_dtype: torch.dtype | None = None,
    out: Mapping[str, torch.Tensor] | None = None,
    operator=None,
) -> Mapping[str, torch.Tensor]:
    """Project named branches while retaining each branch's scale domain."""
    from uniserve.runtime.backends.matmul import resolve

    if (
        not weights
        or set(weights) != set(biases)
        or (out is not None and set(out) != set(weights))
    ):
        raise ValueError(
            "merged projections require matching nonempty branch mappings"
        )
    if x.ndim < 1 or any(
        weight.ndim != 2
        or weight.shape[1] != x.shape[-1]
        or weight.device != x.device
        for weight in weights.values()
    ):
        raise ValueError(
            "merged operands must agree on contraction dimension and device"
        )
    if any(
        bias is not None
        and (bias.shape != (weights[name].shape[0],) or bias.device != x.device)
        for name, bias in biases.items()
    ):
        raise ValueError(
            "merged bias must match branch output channels and device"
        )
    dtype = x.dtype if output_dtype is None else output_dtype
    if out is not None and any(
        value.shape != (*x.shape[:-1], weights[name].shape[0])
        or value.dtype != dtype
        or value.device != x.device
        for name, value in out.items()
    ):
        raise ValueError(
            "merged outputs must match branch shape, dtype and device"
        )

    if out is None:
        block_scaled = all(
            isinstance(weight, QuantizedTensor)
            and weight.quantizer.format in {"nvfp4", "mxfp8"}
            for weight in weights.values()
        )
        if block_scaled:
            # Independent block-scaled branches have distinct tensor-scale
            # domains and therefore distinct GEMMs. Give each native kernel a
            # contiguous destination instead of copying from a temporary into
            # channel views of an output it cannot fuse.
            outputs = {
                name: torch.empty(
                    (*x.shape[:-1], weight.shape[0]),
                    dtype=dtype,
                    device=x.device,
                )
                for name, weight in weights.items()
            }
        else:
            # Fusable branches share caller-owned output storage so one GEMM
            # can write the complete channel matrix directly.
            widths = tuple(weight.shape[0] for weight in weights.values())
            packed = torch.empty(
                (*x.shape[:-1], sum(widths)), dtype=dtype, device=x.device
            )
            outputs = dict(
                zip(weights, packed.split(widths, dim=-1), strict=True)
            )
    else:
        outputs = out

    matrix = as_matrix(x)
    if operator is None:
        from uniserve.nn import _binding

        key = (
            tuple((name, id(weight)) for name, weight in weights.items()),
            branch_width,
        )
        operator = _binding.merged_matmul.get().get(key)
    if operator is None:
        provider = resolve("auto", next(iter(weights.values())))
        quantizer = x.quantizer if isinstance(x, QuantizedTensor) else None
        configs = provider.merged_workspace_buffers(
            weights,
            input_dtype=x.dtype,
            input_quantizer=quantizer,
            max_rows=matrix.shape[0],
            branch_width=branch_width,
            output_dtype=dtype,
        )
        workspace = {
            name: torch.empty(config.shape, dtype=config.dtype, device=x.device)
            for name, config in configs.items()
        }
        operator = provider.prepare_merged(
            weights,
            input_dtype=x.dtype,
            input_quantizer=quantizer,
            max_rows=matrix.shape[0],
            branch_width=branch_width,
            output_dtype=dtype,
            workspace=workspace,
        )

    destinations = {}
    for name, value in outputs.items():
        try:
            destinations[name] = value.view(
                matrix.shape[0], weights[name].shape[0]
            )
        except RuntimeError:
            # An arbitrary caller layout may not flatten its leading axes as
            # a view. Only that branch needs a temporary matrix and copy-back.
            destinations[name] = torch.empty(
                (matrix.shape[0], weights[name].shape[0]),
                dtype=dtype,
                device=x.device,
            )

    operator(matrix, biases, out=destinations)
    for name, value in destinations.items():
        if (
            value.untyped_storage().data_ptr()
            != outputs[name].untyped_storage().data_ptr()
        ):
            outputs[name].copy_(value.view(outputs[name].shape))
    return outputs


def linear(x, weight, bias=None, *, output_dtype=None, out=None):
    """Compute local x @ weight.T + bias with explicit encoded operand
    scales.
    """  # noqa: D205
    return apply_linear(x, weight, bias, output_dtype=output_dtype, out=out)


def merged_linear(
    x, weights, biases, *, branch_width=None, output_dtype=None, out=None
):
    """Project independent named branches without changing their scale
    domains.
    """  # noqa: D205
    return apply_merged_linear(
        x,
        weights,
        biases,
        branch_width=branch_width,
        output_dtype=output_dtype,
        out=out,
    )
