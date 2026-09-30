"""FlashInfer head-flattened BSR with live block IDs and per-key validity."""

import torch
from uniserve_kernels.attention.vsa_rows import pack_sparse_input_rows

from uniserve.tensors import BufferConfig

from . import Backend as BaseBackend
from . import Operator as BaseOperator
from . import _flashinfer


def available(device):
    return _flashinfer.available(device)


class _Operator(BaseOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scratch = self.workspace["scratch"]
        self._state = _flashinfer.SparseExecutionState(
            workspaces={scratch.device: scratch}, transient=self.transient
        )

    def __call__(self, q, k, v, batch, *, scale, out):
        self._validate(q, k, v, batch, out)

        # Equal Q/K extents share one packed row domain; unequal extents need
        # explicit head-first layouts with invalid keys zeroed by hand.
        if q.shape == k.shape:
            query, key, value = pack_sparse_input_rows(
                q, k, v, batch.valid_sizes
            ).unbind(0)
        else:
            # BSR masks exclude invalid keys. Zero their payload as well so a
            # masked padding NaN cannot enter the matrix multiplication.
            index = torch.arange(k.shape[0], device=k.device)
            invalid = (index % 64 >= batch.valid_sizes[index // 64]).view(
                -1, 1, 1
            )
            query = q.transpose(0, 1).contiguous()
            key = k.masked_fill(invalid, 0).transpose(0, 1).contiguous()
            value = v.masked_fill(invalid, 0).transpose(0, 1).contiguous()

        prefix_rows = (
            self.pattern.dense_prefix_tiles * 64
            if torch.cuda.get_device_capability(q.device)[0] == 9
            else 0
        )
        if prefix_rows:
            # Context partitions can have fewer Q rows than their complete
            # K/V domain. Their local prefix keeps the same globally visible
            # keys as interval production and uses the same SM90 prefill.
            key_rows = self.pattern.dense_key_tiles * 64
            out[:prefix_rows].copy_(
                _flashinfer.dense_prefix(
                    query[:, :prefix_rows],
                    key[:, :key_rows],
                    value[:, :key_rows],
                    batch.valid_sizes,
                    scale=scale,
                )
            )
        sparse_rows = q.shape[0] - prefix_rows
        if not sparse_rows:
            return out

        plan = _flashinfer._plan_for(
            self._state,
            q,
            k,
            pattern=self.pattern,
            index_width=batch.block_indices.shape[-1],
            scale=scale,
            row_start=prefix_rows,
            row_count=sparse_rows,
        )
        result = _flashinfer.run_sparse(
            plan,
            query[:, prefix_rows:].contiguous().view(-1, 1, self.head_dim),
            key.reshape(-1, 1, self.head_dim),
            value.reshape(-1, 1, self.head_dim),
            block_indices=batch.block_indices,
            block_counts=batch.block_counts,
            valid_sizes=batch.valid_sizes,
        )
        out[prefix_rows:].copy_(
            result.view(self.num_heads, sparse_rows, self.head_dim).transpose(
                0, 1
            )
        )
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
        self.bind(batch)

        # Callers hand over row-major prepared rows; the head-flattened BSR
        # path needs them head-first. Without prepared rows, prepare_rows
        # packs the inputs itself in the layout this device consumes.
        if packed is not None and not _flashinfer.uses_row_major_inputs(
            q.device
        ):
            rows, heads, width = q.shape
            converted = torch.empty(
                (3, heads, rows, width), dtype=q.dtype, device=q.device
            )
            converted[1:].copy_(packed[1:].transpose(1, 2))
            # Q is interval-major: transpose heads within each interval, not
            # across the complete row domain. The final interval can be short.
            owner_rows = rows // owners
            for start in range(0, owner_rows, chunk_tokens):
                count = owners * min(chunk_tokens, owner_rows - start)
                offset = start * owners
                destination = (
                    converted[0]
                    .view(-1)
                    .narrow(0, offset * heads * width, count * heads * width)
                )
                destination.view(heads, count, width).copy_(
                    packed[0, offset : offset + count].transpose(0, 1)
                )
            packed = converted
        return _flashinfer.prepare_rows(
            self._state,
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
            scale=scale,
        )

    def close(self):
        self._state.plans.clear()
        self._state.native_plans.clear()
        self._state.workspaces.clear()
        super().close()


class Backend(BaseBackend):
    operator_class = _Operator

    def workspace_buffers(self, pattern, *, num_heads, head_dim, dtype):
        return {
            "scratch": BufferConfig(
                (_flashinfer._FLOAT_WORKSPACE_BYTES,), torch.uint8
            )
        }
