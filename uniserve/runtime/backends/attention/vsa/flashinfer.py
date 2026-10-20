"""FlashInfer head-flattened BSR with live block IDs and per-key validity."""

import torch

from uniserve.tensors import BufferConfig

from . import Backend as BaseBackend
from . import Operator as BaseOperator
from . import _flashinfer

# Tile sizes this provider's kernels address.
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

        # The head-flattened BSR reads head-first rows; its mask excludes the
        # keys past each tile's valid size, whose payload is finite.
        query = q.transpose(0, 1).contiguous()
        key = k.transpose(0, 1).contiguous()
        value = v.transpose(0, 1).contiguous()

        plan = _flashinfer._plan_for(
            self._state,
            q,
            k,
            pattern=self.pattern,
            index_width=batch.q2k_index.shape[-1],
            scale=scale,
        )
        result = _flashinfer.run_sparse(
            plan,
            query.contiguous().view(-1, 1, self.head_dim),
            key.reshape(-1, 1, self.head_dim),
            value.reshape(-1, 1, self.head_dim),
            q2k_index=batch.q2k_index,
            q2k_num=batch.q2k_num,
            variable_block_sizes=batch.variable_block_sizes,
        )
        out.copy_(
            result.view(self.num_heads, q.shape[0], self.head_dim).transpose(
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
