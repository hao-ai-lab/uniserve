"""Fixed device continuation tensors indexed by scheduler request slot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from uniserve_worker.backends.triton import triton_available

from ..execution.bounded_storage import BoundedTensorStorage, TensorSchema

try:  # pragma: no cover - availability depends on the serving environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _reset_row_kernel(
        future_tokens_ptr,
        penalty_counts_ptr,
        predicates_ptr,
        selected_points_ptr,
        logical_lengths_ptr,
        sampling_positions_ptr,
        cache_lengths_ptr,
        row,
        valid_cache_length,
        logical_length,
        sampling_position,
        continuation_width: tl.constexpr,
        vocab_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Reset one device runtime row while preserving its declared logical coordinates."""

        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        tl.store(
            future_tokens_ptr + row * continuation_width + offsets,
            1,
            mask=offsets < continuation_width,
        )
        tl.store(
            penalty_counts_ptr + row * vocab_size + offsets,
            0,
            mask=offsets < vocab_size,
        )
        scalar = offsets == 0
        tl.store(predicates_ptr + row + offsets, 0, mask=scalar)
        tl.store(selected_points_ptr + row + offsets, 0, mask=scalar)
        tl.store(logical_lengths_ptr + row + offsets, logical_length, mask=scalar)
        tl.store(sampling_positions_ptr + row + offsets, sampling_position, mask=scalar)
        tl.store(cache_lengths_ptr + row + offsets, valid_cache_length, mask=scalar)

    @triton.jit
    def _publish_decode_kernel(
        indices_ptr,
        tokens_ptr,
        predicates_in_ptr,
        selected_in_ptr,
        future_tokens_ptr,
        predicates_out_ptr,
        selected_out_ptr,
        logical_lengths_ptr,
        sampling_positions_ptr,
        cache_lengths_ptr,
        count,
        continuation_width: tl.constexpr,
        has_selected: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Publish batched decode tokens and advance device-resident runtime coordinates."""

        offsets = tl.arange(0, block_size)
        mask = offsets < count
        indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
        tokens = tl.load(tokens_ptr + offsets, mask=mask, other=0).to(tl.int64)
        predicates = tl.load(predicates_in_ptr + offsets, mask=mask, other=0)
        selected = (
            tl.load(selected_in_ptr + offsets, mask=mask, other=1)
            if has_selected
            else tl.full((block_size,), 1, tl.int32)
        )
        tl.store(
            future_tokens_ptr + indices * continuation_width,
            tokens & ((1 << 31) - 1),
            mask=mask,
        )
        tl.store(predicates_out_ptr + indices, predicates, mask=mask)
        tl.store(selected_out_ptr + indices, selected, mask=mask)
        logical = tl.load(logical_lengths_ptr + indices, mask=mask, other=0)
        sampling = tl.load(sampling_positions_ptr + indices, mask=mask, other=0)
        cache = tl.load(cache_lengths_ptr + indices, mask=mask, other=0)
        tl.store(logical_lengths_ptr + indices, logical + 1, mask=mask)
        tl.store(sampling_positions_ptr + indices, sampling + 1, mask=mask)
        tl.store(cache_lengths_ptr + indices, cache + 1, mask=mask)


@dataclass(frozen=True, slots=True)
class RuntimeStateSnapshot:
    """Captures continuation tensors and cache length for rollback-safe request execution."""

    valid_cache_length: int
    logical_length: int
    sampling_position: int
    future_input_tokens: torch.Tensor
    penalty_counts: torch.Tensor
    predicate: bool
    selected_point: int
    prompt_logits: torch.Tensor | None


class RuntimeStates:
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
            raise ValueError("runtime-state geometry must be positive")
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
        tensors = BoundedTensorStorage.allocate(
            self.tensor_schema(
                request_pool_size=self.request_pool_size,
                vocab_size=self.vocab_size,
                continuation_width=self.continuation_width,
                logits_dtype=self.logits_dtype,
            ),
            self.device,
        ).capacity
        self.logical_lengths = tensors["logical_lengths"]
        self.sampling_positions = tensors["sampling_positions"]
        self.future_input_tokens = tensors["future_input_tokens"]
        self.penalty_counts = tensors["penalty_counts"]
        self.prompt_logits = tensors["prompt_logits"]
        self.predicates = tensors["predicates"]
        self.selected_points = tensors["selected_points"]
        self._ones_int32 = tensors["_ones_int32"]
        self._ones_int64 = tensors["_ones_int64"]

        # Zero-count launches compile representative block sizes without
        # modifying any request row.
        if triton is not None and self.device.type == "cuda" and triton_available(self.device):
            self._reset_device_row(0, 0, 0, 0)
            for block_size in (1, 2, 4, 8, 16, 32, 64, 128):
                if block_size > self.request_pool_size:
                    break
                _publish_decode_kernel[(1,)](
                    self._ones_int64,
                    self.future_input_tokens[:, 0],
                    self.predicates,
                    self._ones_int32,
                    self.future_input_tokens,
                    self.predicates,
                    self.selected_points,
                    self.logical_lengths,
                    self.sampling_positions,
                    self.valid_cache_lengths,
                    count=0,
                    continuation_width=self.continuation_width,
                    has_selected=False,
                    block_size=block_size,
                )

    @staticmethod
    def tensor_schema(
        *,
        request_pool_size: int,
        vocab_size: int,
        continuation_width: int,
        logits_dtype: torch.dtype,
    ) -> dict[str, TensorSchema]:
        """Describe continuation storage; verified lengths belong to the page-table owner."""

        if min(request_pool_size, vocab_size, continuation_width) < 1:
            raise ValueError("runtime-state geometry must be positive")
        if not logits_dtype.is_floating_point:
            raise ValueError("runtime prompt-logit dtype must be floating point")
        rows = request_pool_size + 1
        return {
            "logical_lengths": TensorSchema((rows,), torch.int32, fill=0),
            "sampling_positions": TensorSchema((rows,), torch.int64, fill=0),
            "future_input_tokens": TensorSchema((rows, continuation_width), torch.int64, fill=1),
            "penalty_counts": TensorSchema((rows, vocab_size), torch.int32, fill=0),
            "prompt_logits": TensorSchema((rows, vocab_size), logits_dtype),
            "predicates": TensorSchema((rows,), torch.bool, fill=0),
            "selected_points": TensorSchema((rows,), torch.int32, fill=0),
            "_ones_int32": TensorSchema((request_pool_size,), torch.int32, fill=1),
            "_ones_int64": TensorSchema((request_pool_size,), torch.int64, fill=1),
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
        self.selected_points.index_fill_(0, indices, 0)
        self._copy_or_zero(self.valid_cache_lengths, indices, valid_cache_lengths)
        self._copy_or_zero(self.logical_lengths, indices, logical_lengths)
        self._copy_or_zero(self.sampling_positions, indices, sampling_positions)

    def release(self, request_pool_indices: torch.Tensor | Sequence[int]) -> None:
        """Mark selected continuation-state rows inactive."""

        self.reset(request_pool_indices)

    def snapshot_rows(
        self,
        request_pool_indices: Sequence[int],
        *,
        prompt_logits_ready: Sequence[bool],
    ) -> tuple[RuntimeStateSnapshot, ...]:
        """Copy selected continuation rows and prompt-logit readiness into rollback snapshots."""

        rows = tuple(int(value) for value in request_pool_indices)
        flags = tuple(bool(value) for value in prompt_logits_ready)
        self._validate_host_indices(rows)
        if len(rows) != len(flags):
            raise ValueError("runtime-state snapshot columns are not aligned")
        return tuple(
            RuntimeStateSnapshot(
                valid_cache_length=int(self.valid_cache_lengths[row]),
                logical_length=int(self.logical_lengths[row]),
                sampling_position=int(self.sampling_positions[row]),
                future_input_tokens=self.future_input_tokens[row].detach().cpu().contiguous(),
                penalty_counts=self.penalty_counts[row].detach().cpu().contiguous(),
                predicate=bool(self.predicates[row]),
                selected_point=int(self.selected_points[row]),
                prompt_logits=(
                    self.prompt_logits[row].detach().cpu().contiguous() if ready else None
                ),
            )
            for row, ready in zip(rows, flags, strict=True)
        )

    def restore_rows(
        self,
        rows: Sequence[tuple[int, RuntimeStateSnapshot]],
    ) -> None:
        """Restore continuation tensors and metadata from rollback snapshots."""

        for raw_index, snapshot in rows:
            index = int(raw_index)
            self._validate_host_indices((index,))
            if tuple(snapshot.future_input_tokens.shape) != (self.continuation_width,):
                raise ValueError("runtime-state continuation snapshot has an invalid shape")
            if tuple(snapshot.penalty_counts.shape) != (self.vocab_size,):
                raise ValueError("runtime-state penalty snapshot has an invalid shape")
            if snapshot.prompt_logits is not None and tuple(snapshot.prompt_logits.shape) != (
                self.vocab_size,
            ):
                raise ValueError("runtime-state prompt-logit snapshot has an invalid shape")
            self.future_input_tokens[index].copy_(
                snapshot.future_input_tokens.to(self.device, dtype=torch.int64)
            )
            self.penalty_counts[index].copy_(
                snapshot.penalty_counts.to(self.device, dtype=torch.int32)
            )
            self.predicates[index] = snapshot.predicate
            self.selected_points[index] = snapshot.selected_point
            self.valid_cache_lengths[index] = snapshot.valid_cache_length
            self.logical_lengths[index] = snapshot.logical_length
            self.sampling_positions[index] = snapshot.sampling_position
            if snapshot.prompt_logits is not None:
                self.prompt_logits[index].copy_(
                    snapshot.prompt_logits.to(self.device, dtype=self.logits_dtype)
                )

    def publish_decode(
        self,
        request_pool_indices: Sequence[int],
        *,
        device_indices: torch.Tensor,
        tokens: torch.Tensor,
        predicates: torch.Tensor,
        selected_points: torch.Tensor | None,
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
        selected = ones_i32 if selected_points is None else selected_points.reshape(-1)
        if int(selected.numel()) != count:
            raise ValueError("decode selected points are not row-aligned")
        if triton is not None and self.device.type == "cuda" and triton_available(self.device):
            block_size = triton.next_power_of_2(count)
            selected_input = ones_i32 if selected_points is None else selected
            _publish_decode_kernel[(1,)](
                indices,
                values[0],
                values[1],
                selected_input,
                self.future_input_tokens,
                self.predicates,
                self.selected_points,
                self.logical_lengths,
                self.sampling_positions,
                self.valid_cache_lengths,
                count=count,
                continuation_width=self.continuation_width,
                has_selected=selected_points is not None,
                block_size=block_size,
            )
            return
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
        self.selected_points.index_copy_(
            0,
            indices,
            selected.to(dtype=torch.int32),
        )
        self.logical_lengths.index_add_(0, indices, ones_i32)
        self.sampling_positions.index_add_(0, indices, ones_i64)
        self.valid_cache_lengths.index_add_(0, indices, ones_i32)

    def penalty_rows(self, request_pool_indices: torch.Tensor) -> torch.Tensor:
        """Gather vocabulary-wide occurrence counts for selected request slots."""

        return self.penalty_counts.index_select(0, self._indices(request_pool_indices))

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

        if triton is not None and triton_available(self.device):
            block_size = 256
            span = max(self.continuation_width, self.vocab_size)
            _reset_row_kernel[(triton.cdiv(span, block_size),)](
                self.future_input_tokens,
                self.penalty_counts,
                self.predicates,
                self.selected_points,
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
        self.selected_points[row].zero_()
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


__all__ = ["RuntimeStateSnapshot", "RuntimeStates"]
