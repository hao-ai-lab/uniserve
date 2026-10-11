"""FlashInfer head-flattened BSR with live block IDs and per-key validity."""

import torch

from uniserve.tensors import BufferConfig

from . import Backend as BaseBackend
from . import Operator as BaseOperator
from . import _flashinfer

# Tile sizes this provider's kernels address.
# FIXME: no 128-row build exists here, so 128-row tiles (the ref2va FastH3
# students) run only on SM100 (``sm100.TILES``).
TILES = (64,)


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

        if _flashinfer.uses_row_major_inputs(q.device):
            # The SM120 block-64 kernel reads row-major [rows, heads, dim].
            from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
                bsa_attn_sm120_blk64_fwd,
            )

            index_width = batch.q2k_index.shape[-1]
            bsa_attn_sm120_blk64_fwd(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                batch.q2k_index.unsqueeze(0),
                index_width,
                block_sizes=batch.variable_block_sizes,
                q2k_block_nums=batch.q2k_num.unsqueeze(0),
                softmax_scale=scale,
                out=out.unsqueeze(0),
            )
            return out

        # The head-flattened BSR reads head-first rows; its mask excludes the
        # keys past each tile's valid size, whose payload is finite.
        query = q.transpose(0, 1).contiguous()
        key = k.transpose(0, 1).contiguous()
        value = v.transpose(0, 1).contiguous()

        prefix_rows = (
            self.pattern.dense_prefix_tiles * 64
            if torch.cuda.get_device_capability(q.device)[0] == 9
            else 0
        )
        if prefix_rows:
            # Context partitions can have fewer Q rows than their complete
            # K/V domain; the dense prefix attends every valid key.
            key_rows = self.pattern.dense_key_tiles * 64
            out[:prefix_rows].copy_(
                _flashinfer.dense_prefix(
                    query[:, :prefix_rows],
                    key[:, :key_rows],
                    value[:, :key_rows],
                    batch.variable_block_sizes,
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
            index_width=batch.q2k_index.shape[-1],
            scale=scale,
            row_start=prefix_rows,
            row_count=sparse_rows,
        )
        result = _flashinfer.run_sparse(
            plan,
            query[:, prefix_rows:].contiguous().view(-1, 1, self.head_dim),
            key.reshape(-1, 1, self.head_dim),
            value.reshape(-1, 1, self.head_dim),
            q2k_index=batch.q2k_index,
            q2k_num=batch.q2k_num,
            variable_block_sizes=batch.variable_block_sizes,
        )
        out[prefix_rows:].copy_(
            result.view(self.num_heads, sparse_rows, self.head_dim).transpose(
                0, 1
            )
        )
        return out

    def close(self):
        self._state.plans.clear()
        self._state.workspaces.clear()
        super().close()


class Backend(BaseBackend):
    name = "flashinfer"
    operator_class = _Operator

    def workspace_buffers(self, pattern, *, num_heads, head_dim, dtype):
        return {
            "scratch": BufferConfig(
                (_flashinfer._FLOAT_WORKSPACE_BYTES,), torch.uint8
            )
        }
