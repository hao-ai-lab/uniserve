"""Matmul operators prepared for one projection call site."""

from __future__ import annotations

from collections.abc import Mapping

from uniserve.nn.linear import MergedColumnParallelLinear
from uniserve.quantization import QuantizedTensor

from ..backends import matmul as matmul_backend
from . import capturing


class MatmulBinding:
    """Specialize one numerical call site.

    Specialize one numerical call site while retaining all borrowed backing.
    """

    def __init__(self, module, backend, max_rows, allocate):
        self.module, self.backend, self.max_rows = module, backend, max_rows
        self.allocate = allocate
        self.operators = {}

    def _prepare(self, dtype, output_dtype, quantizer, rows):
        if self.max_rows is not None and rows > self.max_rows:
            raise ValueError("matmul exceeds the prepared token capacity")
        rows = max(rows, self.max_rows or 0)
        key = (dtype, output_dtype, quantizer)
        previous = self.operators.get(key)
        if previous is not None and previous[0] >= rows:
            return previous[1]

        if capturing(self._weight().device):
            raise RuntimeError(
                "matmul shape and representation must be prepared before "
                "capture"
            )

        options = {
            "input_dtype": dtype,
            "input_quantizer": quantizer,
            "max_rows": rows,
            "output_dtype": output_dtype,
        }
        provider = matmul_backend.resolve(self.backend, self._weight())
        if isinstance(self.module, MergedColumnParallelLinear):
            weights = {
                name: child.weight
                for name, child in self.module.projections.items()
            }
            options["branch_width"] = self.module.branch_width
            requirements = provider.merged_workspace_buffers(weights, **options)
            workspace = self.allocate(requirements, self._weight().device)
            operator = provider.prepare_merged(
                weights, **options, workspace=workspace
            )
        else:
            weight = self.module.weight
            requirements = provider.workspace_buffers(weight, **options)
            workspace = self.allocate(requirements, weight.device)
            operator = provider.prepare(weight, **options, workspace=workspace)

        self.operators[key] = (rows, operator)
        return operator

    def _weight(self):
        if isinstance(self.module, MergedColumnParallelLinear):
            return next(iter(self.module.projections.values())).weight
        return self.module.weight

    def quantize(self, x, quantizer, distribution):
        """Encode a complete logical domain.

        Encode a complete logical domain in this context's activation
        storage.
        """
        operator = self._prepare(x.dtype, x.dtype, quantizer, x.shape[0])
        target = operator._input_storage(x)
        return quantizer.quantize(x, distribution=distribution, out=target)

    def __call__(self, x, bias, *, out):
        destination = (
            next(iter(out.values())) if isinstance(out, Mapping) else out
        )
        quantizer = x.quantizer if isinstance(x, QuantizedTensor) else None
        return self._prepare(x.dtype, destination.dtype, quantizer, x.shape[0])(
            x, bias, out=out
        )
