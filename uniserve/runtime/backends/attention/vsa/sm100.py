"""SM100 block attention with the established shape-dependent kernel choice."""

from . import Backend as BaseBackend
from . import Operator as BaseOperator
from . import _cute

try:
    from uniserve_kernel import sparse_attention as _kernel
except ImportError:
    _kernel = None


def available(device):
    return (
        _kernel is not None
        and _kernel.available(device)
        and _cute.available(device)
    )


class _Operator(BaseOperator):
    def __call__(self, q, k, v, batch, *, scale, out):
        self._validate(q, k, v, batch, out)

        # Non-default scales and shapes near the provider boundary take the
        # CuTe path; the custom kernel serves only the default-scale shapes it
        # was measured for.
        if scale != q.shape[-1] ** -0.5 or _cute.should_use(
            rows=k.shape[0], prefix_tiles=self.pattern.dense_prefix_tiles
        ):
            _cute.block_sparse_attention(
                q,
                k,
                v,
                out,
                batch.block_indices,
                batch.block_counts,
                batch.valid_sizes,
                scale=scale,
            )
        else:
            from uniserve.ops.video_sparse import pack_qkv

            if q.shape == k.shape:
                query, key, value = pack_qkv(q, k, v).unbind(0)
            else:
                query, key, value = (
                    q.transpose(0, 1).contiguous(),
                    k.transpose(0, 1),
                    v.transpose(0, 1),
                )

            attended = _kernel.block_sparse_attention(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                batch.block_indices,
                batch.block_counts,
                batch.valid_sizes,
            )
            out.copy_(attended[0].transpose(0, 1))
        return out

    def rows(
        self,
        q,
        k,
        v,
        batch,
        *,
        gate,
        compressed,
        out,
        owners,
        chunk_tokens,
        packed,
        scale,
    ):
        from functools import partial

        from ._rows import _Rows

        self.bind(batch)

        # Reuse one row operator per softmax scale; its query-map plans are
        # cached inside _Rows.
        key = ("rows", scale)
        if not hasattr(self, "_row_operators"):
            self._row_operators = {}
        if key not in self._row_operators:
            self._row_operators[key] = _Rows(
                partial(_cute.block_sparse_attention, scale=scale)
            )
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
