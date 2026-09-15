"""FlashInfer head-flattened BSR with live block IDs and per-key validity."""

import torch

from uniserve.ops.video_sparse_rows import pack_sparse_input_rows
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
        self._state = _flashinfer.SparseExecutionState(workspaces={scratch.device: scratch})

    def __call__(self, q, k, v, batch, *, scale, out):
        self._validate(q, k, v, batch, out)

        plan = _flashinfer._plan_for(
            self._state,
            q,
            k,
            pattern=self.pattern,
            index_width=batch.block_indices.shape[-1],
            scale=scale,
        )
        _flashinfer._fill_flattened_bsr(plan, batch.block_indices, batch.valid_sizes)

        # Equal Q/K extents share one packed row domain; unequal extents need
        # explicit head-first layouts with invalid keys zeroed by hand.
        if q.shape == k.shape:
            query, key, value = pack_sparse_input_rows(q, k, v, batch.valid_sizes).unbind(0)
        else:
            # BSR masks exclude invalid keys. Zero their payload as well so a
            # masked padding NaN cannot enter the matrix multiplication.
            index = torch.arange(k.shape[0], device=k.device)
            invalid = (index % 64 >= batch.valid_sizes[index // 64]).view(-1, 1, 1)
            query = q.transpose(0, 1).contiguous()
            key = k.masked_fill(invalid, 0).transpose(0, 1).contiguous()
            value = v.masked_fill(invalid, 0).transpose(0, 1).contiguous()

        result = plan.wrapper.run(
            query.reshape(-1, 1, self.head_dim),
            key.reshape(-1, 1, self.head_dim),
            value.reshape(-1, 1, self.head_dim),
        )
        out.copy_(result.view(self.num_heads, q.shape[0], self.head_dim).transpose(0, 1))
        return out

    def rows(self, q, k, v, batch, *, gate, compressed, out, owners, chunk_tokens, packed, scale):
        self.bind(batch)

        if not _flashinfer.uses_row_major_inputs(q.device):
            packed = packed.transpose(1, 2).contiguous()
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
        return {"scratch": BufferConfig((_flashinfer._FLOAT_WORKSPACE_BYTES,), torch.uint8)}
