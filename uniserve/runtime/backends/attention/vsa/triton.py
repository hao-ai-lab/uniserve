"""Triton block-64 attention with live device selection and validity."""

from . import Backend as BaseBackend
from . import Operator as BaseOperator
from . import _triton


def available(device):
    return _triton.available(device)


class _Operator(BaseOperator):
    def __call__(self, q, k, v, batch, *, scale, out):
        self._validate(q, k, v, batch, out)
        _triton.block_sparse_attention(
            q, k, v, out, batch.block_indices, batch.block_counts, batch.valid_sizes, scale=scale
        )
        return out

    def rows(self, q, k, v, batch, *, gate, compressed, out, owners, chunk_tokens, packed, scale):
        from functools import partial

        from ._rows import _Rows

        self.bind(batch)
        key = ("rows", scale)
        if not hasattr(self, "_row_operators"):
            self._row_operators = {}
        if key not in self._row_operators:
            self._row_operators[key] = _Rows(partial(_triton.block_sparse_attention, scale=scale))
        return self._row_operators[key].prepare(
            q,
            k,
            v,
            mask_block_indices=batch.block_indices,
            mask_block_count=batch.block_counts,
            valid_sizes=batch.valid_sizes,
            pattern=batch.pattern,
            gate=gate,
            compressed=compressed,
            attention_output=out,
            owners=owners,
            chunk_rows=chunk_tokens,
            packed=packed,
        )

    def close(self):
        if hasattr(self, "_row_operators"):
            self._row_operators.clear()
        super().close()


class Backend(BaseBackend):
    operator_class = _Operator
