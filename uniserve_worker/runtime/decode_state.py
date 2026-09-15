"""Fixed device continuation tensors indexed by scheduler request slot."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.runtime.triton import triton_available
from uniserve.tensors import BufferConfig

from ..ops import decode_state as kernels


class DecodeState:
    """Own graph-safe token continuation state for every request slot.

    Slot ``0`` is the permanent inactive sentinel. Public mutators accept only
    real slots, so padding cannot alter sentinel state.
    """

    def __init__(
        self,
        *,
        request_pool_size: int,
        vocab_size: int,
        continuation_width: int,
        device: torch.device | str,
        logits_dtype: torch.dtype = torch.float32,
        valid_cache_lengths: torch.Tensor | None = None,
    ) -> None:
        """Allocate request-indexed continuation tensors and prewarm update kernels."""

        if request_pool_size < 1 or vocab_size < 1 or continuation_width < 1:
            raise ValueError("runtime-state dimensions must be positive")
        self.request_pool_size = int(request_pool_size)
        self.vocab_size = int(vocab_size)
        self.continuation_width = int(continuation_width)
        self.device = torch.device(device)

        if not logits_dtype.is_floating_point:
            raise ValueError("runtime prompt-logit dtype must be floating point")
        self.logits_dtype = logits_dtype

        # Row zero is the immutable padding sentinel. Real request slots use the
        # same one-based indexes assigned by the request pool.
        rows = self.request_pool_size + 1
        if valid_cache_lengths is None:
            self.valid_cache_lengths = torch.zeros(rows, dtype=torch.int32, device=self.device)
        else:
            if (
                valid_cache_lengths.shape != (rows,)
                or valid_cache_lengths.dtype != torch.int32
                or valid_cache_lengths.device != self.device
            ):
                raise ValueError("runtime cache-length storage is incompatible")
            self.valid_cache_lengths = valid_cache_lengths

        buffer_configs = self.buffers(
            request_pool_size=self.request_pool_size,
            vocab_size=self.vocab_size,
            continuation_width=self.continuation_width,
            logits_dtype=self.logits_dtype,
        )
        tensors = TensorBuffers.allocate(buffer_configs, device=self.device).view(buffer_configs)
        for name, value in {
            "logical_lengths": 0,
            "sampling_positions": 0,
            "future_input_tokens": 1,
            "penalty_counts": 0,
            "predicates": 0,
            "_ones_int32": 1,
            "_ones_int64": 1,
        }.items():
            tensors[name].fill_(value)

        self.logical_lengths = tensors["logical_lengths"]
        self.sampling_positions = tensors["sampling_positions"]
        self.future_input_tokens = tensors["future_input_tokens"]
        self.penalty_counts = tensors["penalty_counts"]
        self.prompt_logits = tensors["prompt_logits"]
        self.predicates = tensors["predicates"]
        self._ones_int32 = tensors["_ones_int32"]
        self._ones_int64 = tensors["_ones_int64"]

        # Prepare the same capacity-bounded kernel used by live publications.
        # A zero count leaves all request rows unchanged.
        if (
            kernels.triton is not None
            and self.device.type == "cuda"
            and triton_available(self.device)
        ):
            self._reset_device_row(0, 0, 0, 0)
            kernels._publish_decode_kernel[(1,)](
                self._ones_int64,
                self.future_input_tokens[:, 0],
                self.predicates,
                self.future_input_tokens,
                self.predicates,
                self.logical_lengths,
                self.sampling_positions,
                self.valid_cache_lengths,
                count=0,
                continuation_width=self.continuation_width,
                block_size=kernels.triton.next_power_of_2(self.request_pool_size),
            )

    @staticmethod
    def buffers(
        *,
        request_pool_size: int,
        vocab_size: int,
        continuation_width: int,
        logits_dtype: torch.dtype,
    ) -> dict[str, BufferConfig]:
        """Describe continuation storage; verified lengths belong to the page-table owner."""

        if min(request_pool_size, vocab_size, continuation_width) < 1:
            raise ValueError("runtime-state dimensions must be positive")
        if not logits_dtype.is_floating_point:
            raise ValueError("runtime prompt-logit dtype must be floating point")
        rows = request_pool_size + 1
        return {
            "logical_lengths": BufferConfig((rows,), torch.int32),
            "sampling_positions": BufferConfig((rows,), torch.int64),
            "future_input_tokens": BufferConfig((rows, continuation_width), torch.int64),
            "penalty_counts": BufferConfig((rows, vocab_size), torch.int32),
            "prompt_logits": BufferConfig((rows, vocab_size), logits_dtype),
            "predicates": BufferConfig((rows,), torch.bool),
            "_ones_int32": BufferConfig((request_pool_size,), torch.int32),
            "_ones_int64": BufferConfig((request_pool_size,), torch.int64),
        }

    def reset(
        self,
        request_pool_indices: torch.Tensor | Sequence[int],
        *,
        valid_cache_lengths: torch.Tensor | Sequence[int] | None = None,
        logical_lengths: torch.Tensor | Sequence[int] | None = None,
        sampling_positions: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Initialize selected request rows with validated cache length and sampling state."""

        # Host fast path: fully CPU-resident columns reset their rows directly,
        # without building device index tensors.
        if not isinstance(request_pool_indices, torch.Tensor) and self.device.type == "cuda":
            host = tuple(int(value) for value in request_pool_indices)
            self._validate_host_indices(host)
            valid = self._host_reset_column(valid_cache_lengths, len(host))
            logical = self._host_reset_column(logical_lengths, len(host))
            sampling = self._host_reset_column(sampling_positions, len(host))
            if valid is not None and logical is not None and sampling is not None:
                for position, row in enumerate(host):
                    self._reset_device_row(
                        row,
                        valid[position],
                        logical[position],
                        sampling[position],
                    )
                return

        indices = self._indices(request_pool_indices)
        self.future_input_tokens.index_fill_(0, indices, 1)
        self.penalty_counts.index_fill_(0, indices, 0)
        self.predicates.index_fill_(0, indices, False)
        self._copy_or_zero(self.valid_cache_lengths, indices, valid_cache_lengths)
        self._copy_or_zero(self.logical_lengths, indices, logical_lengths)
        self._copy_or_zero(self.sampling_positions, indices, sampling_positions)

    def set_cache_length(self, slot: int, length: int | torch.Tensor) -> None:
        """Update the borrowed verified-KV column in execution submission order."""

        self._validate_host_indices((slot,))
        self._copy_scalar(self.valid_cache_lengths[slot : slot + 1], length)

    def set_prompt_logits(self, slot: int, logits: torch.Tensor) -> None:
        """Publish prompt logits into stable request storage before successors use it."""

        self._validate_host_indices((slot,))
        self.prompt_logits[slot].copy_(logits.to(dtype=self.prompt_logits.dtype))

    def apply_tokens(
        self,
        slots: Sequence[int],
        *,
        tokens: torch.Tensor,
        predicates: torch.Tensor,
        valid: torch.Tensor,
        active: torch.Tensor,
        penalty_bases: Sequence[torch.Tensor | None],
        device_slots: torch.Tensor | None = None,
        logical_position: int | torch.Tensor | None = None,
        sampling_position: int | torch.Tensor | None = None,
    ) -> None:
        """Commit selected tokens, continuation coordinates, and occurrence counts.

        Batched decode advances the existing device coordinates by one. Prefill
        and verification supply their actual coordinates for a single slot.
        The token's high continuation bit is never stored as a token ID.
        """

        indices = tuple(int(slot) for slot in slots)
        self._validate_host_indices(indices)
        if len(indices) != len(penalty_bases):
            raise ValueError("decode penalty rows do not align with request slots")

        if device_slots is not None:
            if logical_position is not None or sampling_position is not None:
                raise ValueError("batched decode does not replace explicit coordinates")
            self._advance_tokens(
                indices, device_indices=device_slots, tokens=tokens, predicates=predicates
            )
            penalty_tokens = tokens.reshape(-1)
        else:
            if len(indices) != 1 or logical_position is None or sampling_position is None:
                raise ValueError("explicit token publication requires one complete request row")
            slot = indices[0]
            future_token = self.future_input_tokens[slot, :1]
            future_token.copy_(tokens.reshape(-1)[:1])
            future_token.bitwise_and_((1 << 31) - 1)
            self.predicates[slot : slot + 1].copy_(predicates.reshape(-1)[:1].to(torch.bool))
            self._copy_scalar(self.logical_lengths[slot : slot + 1], logical_position)
            self._copy_scalar(self.sampling_positions[slot : slot + 1], sampling_position)
            penalty_tokens = future_token
        # Occurrence counts grow only for tokens that are valid and active.
        for index, counts in enumerate(penalty_bases):
            if counts is None:
                continue
            weight = (
                valid.reshape(-1)[index : index + 1] & active.reshape(-1)[index : index + 1]
            ).to(counts.dtype)
            counts.scatter_add_(0, penalty_tokens[index : index + 1].to(torch.int64), weight)

    @staticmethod
    def _copy_scalar(target: torch.Tensor, value: int | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            target.copy_(value.reshape(-1)[:1].to(device=target.device, dtype=target.dtype))
        else:
            target.fill_(int(value))

    def _advance_tokens(
        self,
        request_pool_indices: Sequence[int],
        *,
        device_indices: torch.Tensor,
        tokens: torch.Tensor,
        predicates: torch.Tensor,
    ) -> None:
        """Commit device-selected decode transitions into request-indexed continuation tensors."""

        host = tuple(int(value) for value in request_pool_indices)
        self._validate_host_indices(host)
        count = len(host)
        indices = device_indices.reshape(-1)
        if (
            count == 0
            or int(indices.numel()) != count
            or indices.device != self.device
            or indices.dtype != torch.int64
        ):
            raise ValueError("decode runtime-state indices are not aligned")
        values = (
            tokens.reshape(-1),
            predicates.reshape(-1),
        )
        if any(int(value.numel()) != count for value in values):
            raise ValueError("decode runtime-state values are not aligned")
        ones_i32 = self._ones_int32[:count]
        ones_i64 = self._ones_int64[:count]
        if (
            kernels.triton is not None
            and self.device.type == "cuda"
            and triton_available(self.device)
        ):
            # Unique validated slots bound count by the owned pool capacity.
            # Mask live rows without specializing on each arriving batch size.
            block_size = kernels.triton.next_power_of_2(self.request_pool_size)
            kernels._publish_decode_kernel[(1,)](
                indices,
                values[0],
                values[1],
                self.future_input_tokens,
                self.predicates,
                self.logical_lengths,
                self.sampling_positions,
                self.valid_cache_lengths,
                count=count,
                continuation_width=self.continuation_width,
                block_size=block_size,
            )
            return

        # Fallback without Triton: the same scatter updates via index ops.
        self.future_input_tokens[:, 0].index_copy_(
            0,
            indices,
            values[0].to(dtype=torch.int64).bitwise_and((1 << 31) - 1),
        )
        self.predicates.index_copy_(
            0,
            indices,
            values[1].to(dtype=torch.bool),
        )
        self.logical_lengths.index_add_(0, indices, ones_i32)
        self.sampling_positions.index_add_(0, indices, ones_i64)
        self.valid_cache_lengths.index_add_(0, indices, ones_i32)

    def _indices(self, values: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Normalize host or device row indices onto the runtime-state device."""

        if isinstance(values, torch.Tensor):
            source = values.reshape(-1)
            count = int(source.numel())
            if count == 0:
                return source.to(device=self.device, dtype=torch.long)
            if source.device.type == "cpu":
                host = tuple(int(value) for value in source.tolist())
                self._validate_host_indices(host)
            indices = source.to(device=self.device, dtype=torch.long)
            if indices.device.type == "cuda":
                torch._assert_async(
                    torch.all((indices >= 1) & (indices <= self.request_pool_size)),
                    "request-pool index is outside runtime-state capacity",
                )
                if count > 1:
                    ordered = torch.sort(indices).values
                    torch._assert_async(
                        torch.all(ordered[1:] != ordered[:-1]),
                        "runtime-state mutation repeats a request-pool index",
                    )
            return indices
        host = tuple(int(value) for value in values)
        self._validate_host_indices(host)
        return torch.tensor(host, dtype=torch.long, device=self.device)

    def _validate_host_indices(self, values: tuple[int, ...]) -> None:
        """Validate host reset indices are unique and within runtime row bounds."""

        if any(value < 1 or value > self.request_pool_size for value in values):
            raise ValueError("request-pool index is outside runtime-state capacity")
        if len(set(values)) != len(values):
            raise ValueError("runtime-state mutation repeats a request-pool index")

    @staticmethod
    def _host_reset_column(
        values: torch.Tensor | Sequence[int] | None,
        count: int,
    ) -> tuple[int, ...] | None:
        """Normalize an optional host reset column to the requested row count."""

        if values is None:
            return (0,) * count
        if isinstance(values, torch.Tensor):
            flat = values.reshape(-1)
            if int(flat.numel()) != count:
                raise ValueError("runtime-state reset columns are not aligned")
            if flat.device.type != "cpu":
                return None
            return tuple(int(value) for value in flat.tolist())
        result = tuple(int(value) for value in values)
        if len(result) != count:
            raise ValueError("runtime-state reset columns are not aligned")
        return result

    def _reset_device_row(
        self,
        row: int,
        valid_cache_length: int,
        logical_length: int,
        sampling_position: int,
    ) -> None:
        """Reset one device row through the fused kernel or tensor fallback path."""

        if kernels.triton is not None and triton_available(self.device):
            block_size = 256
            span = max(self.continuation_width, self.vocab_size)
            kernels._reset_row_kernel[(kernels.triton.cdiv(span, block_size),)](
                self.future_input_tokens,
                self.penalty_counts,
                self.predicates,
                self.logical_lengths,
                self.sampling_positions,
                self.valid_cache_lengths,
                row,
                valid_cache_length,
                logical_length,
                sampling_position,
                continuation_width=self.continuation_width,
                vocab_size=self.vocab_size,
                block_size=block_size,
            )
            return

        self.future_input_tokens[row].fill_(1)
        self.penalty_counts[row].zero_()
        self.predicates[row].fill_(False)
        self.logical_lengths[row].fill_(logical_length)
        self.sampling_positions[row].fill_(sampling_position)
        self.valid_cache_lengths[row].fill_(valid_cache_length)

    def _copy_or_zero(
        self,
        target: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor | Sequence[int] | None,
    ) -> None:
        """Scatter supplied values into indexed rows or clear those rows when absent."""

        if values is None:
            target.index_fill_(0, indices, 0)
            return
        source = torch.as_tensor(values, dtype=target.dtype, device=self.device).reshape(-1)
        if int(source.numel()) != int(indices.numel()):
            raise ValueError("runtime-state reset columns are not aligned")
        target[indices] = source


__all__ = ["DecodeState"]
