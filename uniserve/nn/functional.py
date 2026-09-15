"""Callable numerical primitives over local tensors and borrowed operators."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Literal

import torch

from uniserve.quantization import QuantizedTensor


def _result(value: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return value
    if (
        out.shape != value.shape
        or out.dtype != value.dtype
        or out.device != value.device
    ):
        raise ValueError(
            "output must match the numerical result's shape, dtype and device"
        )
    return out.copy_(value)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize the last axis using FP32 variance accumulation."""
    from uniserve import ops

    return _result(ops.rms_norm(x, weight, eps), out)


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized and residual sums, preserving caller inputs by
    default.
    """  # noqa: D205
    from uniserve import ops

    normalized, summed = ops.add_rms_norm(
        x, residual, weight, eps, in_place=False
    )
    if out is None:
        return normalized, summed
    return _result(normalized, out[0]), _result(summed, out[1])


def silu_and_mul(
    x: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply SiLU to the first channel half and multiply by the second."""
    from uniserve import ops

    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError("gating requires equal channel halves")

    if isinstance(out, QuantizedTensor):
        from uniserve.quantization import Quantizer

        if out.quantizer != Quantizer("fp8", axis=0):
            raise ValueError("fused SiLU output requires row-scaled FP8")
        if x.ndim != 2:
            gate, value = x.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate.float()) * value.float()
            # Axis zero of a higher-rank tensor retains that axis alone; it
            # must not silently become one separate scale per flattened row.
            encoded = out.quantizer.quantize(activated)
            return _result(encoded.to(dtype=x.dtype), out)

        values, scales = ops.silu_and_mul_fp8(x)
        encoded = out.quantizer.from_tensors(
            {"values": values, "scale": scales.reshape(*x.shape[:-1], 1)},
            shape=tuple(values.shape),
            dtype=x.dtype,
        )
        return _result(encoded, out)

    return _result(ops.silu_and_mul(x), out)


def gelu_and_mul(
    x: torch.Tensor,
    *,
    approximate: Literal["none", "tanh"],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the selected GELU formula to the first channel half, then
    multiply.
    """  # noqa: D205
    if x.ndim < 1 or x.shape[-1] % 2 or approximate not in {"none", "tanh"}:
        raise ValueError(
            "GELU gating requires equal channel halves and a supported "
            "approximation"
        )
    gate, value = x.chunk(2, dim=-1)
    return _result(
        torch.nn.functional.gelu(gate, approximate=approximate) * value, out
    )


def attention(q, k, v, batch, *, scale: float, out=None):
    """Apply local attention without implicit cache allocation or collectives.

    Paged read-only calls pass physical K/V tensors and write_indices=None.
    Persistent cache writes use Attention bound through ExecutionContext.
    """
    from uniserve.model.inputs import TextSize
    from uniserve.nn.attention.inputs import DenseInput
    from uniserve.runtime.backends.attention import resolve
    from uniserve.runtime.backends.attention._sequences import host_lengths
    from uniserve.runtime.tensor_buffers import TensorBuffers

    if q.ndim not in {3, 4}:
        raise ValueError("attention requires packed THD or dense BHTD tensors")

    size = (
        TextSize(
            q.shape[0] if q.ndim == 3 else q.shape[0] * q.shape[2],
            1 if q.ndim == 3 else q.shape[0],
        )
        if isinstance(batch, DenseInput)
        else TextSize(q.shape[0], batch.queries.batch_size)
    )
    kv_axis = 1 if k.ndim == 3 or isinstance(batch, DenseInput) else 2
    options = {
        "num_heads": q.shape[1],
        "num_kv_heads": k.shape[kv_axis],
        "head_dim": q.shape[-1],
        "dtype": q.dtype,
        "size": size,
        "cache": None,
    }

    provider = resolve("auto", device=q.device)
    requirements = provider.workspace_buffers(**options)
    with TensorBuffers.allocate(requirements, device=q.device) as buffers:
        operator = provider.prepare(
            **options, workspace=buffers.view(requirements)
        )
        try:
            if operator.requires_host_lengths(batch):
                batch = host_lengths(batch)
            operator.bind(batch)
            destination = torch.empty_like(q) if out is None else out
            return operator(q, k, v, batch, scale=scale, out=destination)
        finally:
            operator.close()


def apply_rotary(
    x, cos, sin, *, rotation: Literal["interleaved", "split"], out=None
):
    """Rotate the leading coordinates of token/head vectors with compact
    factors.

    Factors have shape [..., rotary_dim / 2], matching x's token dimensions;
    their omitted penultimate head axis broadcasts over every head. A trailing
    non-rotary feature interval is preserved. Arithmetic accumulates in FP32.
    """  # noqa: D205
    if (
        rotation not in {"interleaved", "split"}
        or x.ndim < 2
        or cos.shape != sin.shape
        or cos.ndim != x.ndim - 1
        or cos.shape[:-1] != x.shape[:-2]
        or cos.shape[-1] * 2 > x.shape[-1]
        or cos.device != x.device
        or sin.device != x.device
    ):
        raise ValueError(
            "rotary factors must match tokens and a prefix of the head width"
        )
    width = cos.shape[-1] * 2
    if width == x.shape[-1] and x.ndim == 3 and rotation == "split":
        from uniserve import ops

        return _result(ops.apply_rotary_emb(x, cos, sin), out)

    # Factors broadcast over the omitted head axis: [..., 1, width / 2].
    cosine, sine = cos.float().unsqueeze(-2), sin.float().unsqueeze(-2)
    prefix = x[..., :width].float()
    # Split pairs the two half-widths; interleaved pairs adjacent coordinates.
    first, second = (
        prefix.chunk(2, dim=-1)
        if rotation == "split"
        else (prefix[..., ::2], prefix[..., 1::2])
    )
    left, right = first * cosine - second * sine, second * cosine + first * sine
    rotated = (
        torch.cat((left, right), dim=-1)
        if rotation == "split"
        else torch.stack((left, right), dim=-1).flatten(-2)
    ).to(x.dtype)

    if width < x.shape[-1]:
        rotated = torch.cat((rotated, x[..., width:]), dim=-1)
    return _result(rotated, out)


def qk_norm_rope(
    q, k, q_weight, k_weight, cos, sin, *, eps: float, axis_dims, out=None
):
    """Normalize each full head in FP32, then rotate its independent axes.

    Each axis uses split-half rotation and its own compact factors. The shared
    head normalization spans all axes; separate mathematical normalization
    domains are expressed by separate calls on the corresponding tensor views.
    """
    from uniserve import ops

    if (
        not axis_dims
        or len(axis_dims) != len(cos)
        or len(cos) != len(sin)
        or any(
            type(width) is not int or width < 2 or width % 2
            for width in axis_dims
        )
        or sum(axis_dims) != q.shape[-1]
        or k.shape[-1] != q.shape[-1]
        or q_weight.shape != (q.shape[-1],)
        or k_weight.shape != q_weight.shape
    ):
        raise ValueError(
            "Q/K rotary axes must partition the normalized head width"
        )

    if len(axis_dims) == 1 and cos[0].shape[-1] * 2 < axis_dims[0]:
        # Normalize the complete head before rotating its leading coordinates.
        # Compact factors are consumed directly by the native kernel, including
        # strided projections whose caller supplies the same Q/K views as out.
        query, key = (
            (q.clone(), k.clone())
            if out is None
            else (_result(q, out[0]), _result(k, out[1]))
        )
        from uniserve.ops.rope_kernels import (
            can_run_triton_qk_rms_norm_rope_inplace,
            triton_qk_rms_norm_rope_inplace,
        )

        if can_run_triton_qk_rms_norm_rope_inplace(
            query, key, q_weight, k_weight, cos[0], sin[0], compact=True
        ):
            triton_qk_rms_norm_rope_inplace(
                query,
                key,
                q_weight,
                k_weight,
                cos[0],
                sin[0],
                eps,
                compact=True,
            )
        else:
            for source, weight, target in (
                (q, q_weight, query),
                (k, k_weight, key),
            ):
                value = source.float()
                value = value * torch.rsqrt(
                    value.square().mean(-1, keepdim=True) + eps
                )
                value = value * weight.float()
                target.copy_(
                    apply_rotary(value, cos[0], sin[0], rotation="split").to(
                        source.dtype
                    )
                )
        return query, key

    if len(axis_dims) == 1:
        query, key = ops.qk_norm_rope(
            q, k, q_weight, k_weight, cos[0], sin[0], eps
        )
    else:
        query, key = ops.qk_norm_rope(
            q,
            k,
            (q_weight,) * len(axis_dims),
            (k_weight,) * len(axis_dims),
            cos,
            sin,
            eps,
            axis_dims=axis_dims,
        )
    if out is None:
        return query, key
    return _result(query, out[0]), _result(key, out[1])


def _matrix(x: torch.Tensor) -> torch.Tensor:
    """Flatten leading axes into logical GEMM rows, preserving encoded
    layouts.
    """  # noqa: D205
    if x.ndim == 2:
        return x
    shape = (prod(x.shape[:-1]), x.shape[-1])
    if not isinstance(x, QuantizedTensor):
        return x.reshape(shape)

    fields = dict(x.buffers())
    # nvfp4 packs two values per byte, so its physical row width halves.
    fields["values"] = fields["values"].reshape(
        shape if x.quantizer.format != "nvfp4" else (shape[0], shape[1] // 2)
    )
    if x.quantizer.axis == 0:
        fields["scale"] = (
            fields["scale"].expand(*x.shape[:-1], 1).reshape(shape[0], 1)
        )
    return x.quantizer.from_tensors(
        fields, shape=shape, dtype=x.dtype, scale_layout=x.scale_layout
    )


def _operator(weight, x, output_dtype):
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


def _linear(
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
    matrix = _matrix(x)
    if operator is None:
        from . import _binding

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


def _merged_linear(
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
        # Logical branches share caller-owned output storage. A fused GEMM can
        # write this matrix directly, and its consumers can borrow adjacent
        # channels without concatenating or depending on execution scratch.
        widths = tuple(weight.shape[0] for weight in weights.values())
        packed = torch.empty(
            (*x.shape[:-1], sum(widths)), dtype=dtype, device=x.device
        )
        outputs = dict(zip(weights, packed.split(widths, dim=-1), strict=True))
    else:
        outputs = out

    matrix = _matrix(x)
    if operator is None:
        from . import _binding

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
    return _linear(x, weight, bias, output_dtype=output_dtype, out=out)


def merged_linear(
    x, weights, biases, *, branch_width=None, output_dtype=None, out=None
):
    """Project independent named branches without changing their scale
    domains.
    """  # noqa: D205
    return _merged_linear(
        x,
        weights,
        biases,
        branch_width=branch_width,
        output_dtype=output_dtype,
        out=out,
    )


def patchify(images: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    """Pack CHW or NCHW images in spatial-patch, pixel, then channel order."""
    if (
        type(patch_size) is not int
        or patch_size < 1
        or images.ndim not in {3, 4}
    ):
        raise ValueError(
            "patching requires CHW/NCHW images and a positive patch size"
        )
    channels, height, width = images.shape[-3:]
    if height % patch_size or width % patch_size:
        raise ValueError("image dimensions must be divisible by the patch size")

    batch = images.shape[0] if images.ndim == 4 else 1
    rows, columns = height // patch_size, width // patch_size
    # [batch, rows * columns, patch_size**2 * channels]
    result = (
        images.reshape(batch, channels, rows, patch_size, columns, patch_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, rows * columns, patch_size**2 * channels)
    )
    return result if images.ndim == 4 else result[0]


def unpatchify(
    patches: torch.Tensor, size, *, patch_size: int, channels: int
) -> torch.Tensor:
    """Restore canonical patch rows into a CHW or NCHW image of the explicit
    size.
    """  # noqa: D205
    if (
        type(patch_size) is not int
        or patch_size < 1
        or type(channels) is not int
        or channels < 1
        or patches.ndim not in {2, 3}
    ):
        raise ValueError(
            "unpatching requires token rows and positive patch/channel "
            "dimensions"
        )
    if size.height % patch_size or size.width % patch_size:
        raise ValueError("image dimensions must be divisible by the patch size")
    rows, columns = size.height // patch_size, size.width // patch_size
    if patches.shape[-2:] != (rows * columns, patch_size**2 * channels):
        raise ValueError(
            "patch rows do not match the requested image dimensions"
        )

    batch = patches.shape[0] if patches.ndim == 3 else 1
    # Inverse of patchify: [batch, channels, height, width]
    result = (
        patches.reshape(batch, rows, columns, patch_size, patch_size, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, size.height, size.width)
    )
    return result if patches.ndim == 3 else result[0]
