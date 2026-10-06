"""Fixed device continuation tensors indexed by scheduler request slot.

``DecodeState`` keeps, per request-pool slot, the device-resident
continuation state of token calls: the next input token and its
continuation predicate, the logical position, the sampling counter that
mirrors the request's ``rng_counter``, committed penalty counts, and, when
prompt scoring is requested, the last prompt logits of a prefill chunk.
The native executor retains each call's pending numerical values and applies
them when the batch commits. The
executor resets a slot's rows when a request is admitted to it and when the
request's storage is released. Input preparation (``TokenBuffers``) reads next
tokens and logical lengths by slot, and token sampling reads the committed
penalty counts.

The verified KV length column is borrowed: the worker passes
``BlockTables.verified_lengths`` as ``valid_cache_lengths``, so batched
decode advances the same tensor the block tables expose.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from uniserve_kernels.triton import launchable

from uniserve.runtime.device import async_tensor_h2d
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig
from uniserve_worker.storage import _decode_state as kernels


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
        """Allocate request-indexed continuation tensors.

        On a CUDA device where Triton kernels are launchable, both update
        kernels are launched once here, on the sentinel row and with an empty
        batch, so later launches reuse their compiled form.

        Args:
            request_pool_size: Number of real request slots; tensors get one
                extra leading row for the slot ``0`` sentinel.
            vocab_size: Width of the penalty-count and prompt-logit rows.
            continuation_width: Token slots per row in
                ``future_input_tokens``. Token export writes only column
                ``0``; a reset fills the whole row.
            device: Device of every tensor.
            logits_dtype: Floating dtype of ``prompt_logits``.
            valid_cache_lengths: Optional borrowed ``int32`` column of shape
                ``[request_pool_size + 1]`` on ``device``; a zeroed column is
                allocated when absent.

        Raises:
            ValueError: When a dimension is not positive, ``logits_dtype`` is
                not floating point, or ``valid_cache_lengths`` has another
                shape, dtype or device.
        """
        if request_pool_size < 1 or vocab_size < 1 or continuation_width < 1:
            raise ValueError("runtime-state dimensions must be positive")
        self.request_pool_size = int(request_pool_size)
        self.vocab_size = int(vocab_size)
        self.continuation_width = int(continuation_width)
        self.device = torch.device(device)

        if not logits_dtype.is_floating_point:
            raise ValueError(
                "runtime prompt-logit dtype must be floating point"
            )
        self.logits_dtype = logits_dtype

        # Row zero is the immutable padding sentinel. Real request slots use the
        # same one-based indexes assigned by the request pool.
        rows = self.request_pool_size + 1
        if valid_cache_lengths is None:
            self.valid_cache_lengths = torch.zeros(
                rows, dtype=torch.int32, device=self.device
            )
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
        tensors = TensorBuffers.allocate(
            buffer_configs, device=self.device
        ).view(buffer_configs)

        # Initial values match what a row reset with omitted columns writes:
        # continuation tokens are 1, and counts, predicates and coordinates
        # are 0. ``prompt_logits`` stays uninitialized until prompt scoring
        # of a prefill publishes into its row.
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

        # Whether row resets run as one fused Triton launch; decided once, so
        # a reset does not re-query device availability.
        self._fused_reset = (
            kernels.triton is not None
            and self.device.type == "cuda"
            and launchable(self.device)
        )

        # Prepare the same capacity-bounded kernel used by live exports.
        # A zero count leaves all request rows unchanged, and the row reset
        # rewrites the sentinel with its initial values.
        if self._fused_reset:
            self._reset_rows(
                async_tensor_h2d(
                    (0, 0, 0, 0), dtype=torch.int64, device=self.device
                ),
                1,
            )
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
                block_size=kernels.triton.next_power_of_2(
                    self.request_pool_size
                ),
            )

    @staticmethod
    def buffers(
        *,
        request_pool_size: int,
        vocab_size: int,
        continuation_width: int,
        logits_dtype: torch.dtype,
    ) -> dict[str, BufferConfig]:
        """Describe the tensors this state allocates, keyed by attribute name.

        ``valid_cache_lengths`` is not included: ``__init__`` either borrows
        it from its owner or allocates it separately. Startup sizing
        (``bootstrap.report``) calls this without an instance and counts the
        verified-length column under ``BlockTables.buffers``.

        Raises:
            ValueError: When a dimension is not positive or ``logits_dtype``
                is not floating point.
        """
        if min(request_pool_size, vocab_size, continuation_width) < 1:
            raise ValueError("runtime-state dimensions must be positive")
        if not logits_dtype.is_floating_point:
            raise ValueError(
                "runtime prompt-logit dtype must be floating point"
            )
        rows = request_pool_size + 1
        # Request-indexed tensors have ``rows`` leading rows; the unit
        # increments used by the non-Triton decode advance are sized to the
        # largest batch, one entry per real slot.
        return {
            "logical_lengths": BufferConfig((rows,), torch.int32),
            "sampling_positions": BufferConfig((rows,), torch.int64),
            "future_input_tokens": BufferConfig(
                (rows, continuation_width), torch.int64
            ),
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
        """Reinitialize selected request rows.

        Continuation tokens become 1, penalty counts and predicates are
        cleared, and the cache length, logical length and sampling position
        take the supplied per-row values, or 0 when a column is omitted.

        Raises:
            ValueError: When an index is outside ``1..request_pool_size`` or
                repeated, or a supplied column does not have one value per
                index. Device-tensor indices are not checked on the host;
                on a CUDA state ``torch._assert_async`` checks them and fails
                asynchronously on the device.
        """
        # Host fast path: with host indices and host columns on a device
        # where Triton launches, the rows and their columns copy in one
        # non-blocking copy and every row resets in one fused launch. A
        # device column makes ``_host_reset_column`` return None and falls
        # through to the indexed path.
        if (
            not isinstance(request_pool_indices, torch.Tensor)
            and self._fused_reset
        ):
            host = tuple(int(value) for value in request_pool_indices)
            self._validate_host_indices(host)
            if not host:
                return
            valid = self._host_reset_column(valid_cache_lengths, len(host))
            logical = self._host_reset_column(logical_lengths, len(host))
            sampling = self._host_reset_column(sampling_positions, len(host))
            if (
                valid is not None
                and logical is not None
                and sampling is not None
            ):
                columns = async_tensor_h2d(
                    host + valid + logical + sampling,
                    dtype=torch.int64,
                    device=self.device,
                )
                self._reset_rows(columns, len(host))
                return

        indices = self._indices(request_pool_indices)
        self.future_input_tokens.index_fill_(0, indices, 1)
        self.penalty_counts.index_fill_(0, indices, 0)
        self.predicates.index_fill_(0, indices, False)
        self._copy_or_zero(
            self.valid_cache_lengths, indices, valid_cache_lengths
        )
        self._copy_or_zero(self.logical_lengths, indices, logical_lengths)
        self._copy_or_zero(self.sampling_positions, indices, sampling_positions)

    def set_cache_length(self, slot: int, length: int | torch.Tensor) -> None:
        """Set one slot's verified KV length.

        The write is enqueued on the current stream, so it is ordered with
        the execution work submitted around it. A tensor ``length``
        contributes its first element through a tensor copy, so a device
        value is never synchronized to the host.
        """
        self._validate_host_indices((slot,))
        self._copy_scalar(self.valid_cache_lengths[slot : slot + 1], length)

    def set_prompt_logits(self, slot: int, logits: torch.Tensor) -> None:
        """Copy one slot's last prompt logits into its ``prompt_logits`` row.

        Prompt scoring of the request's next prefill chunk
        reads this row to score that chunk's first token.
        """
        self._validate_host_indices((slot,))
        self.prompt_logits[slot].copy_(
            logits.to(dtype=self.prompt_logits.dtype)
        )

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
        """Commit selected tokens, continuation coordinates and penalty counts.

        With ``device_slots``, a batched decode advances the logical length,
        sampling position and verified cache length of every slot by one.
        Without it, prefill and verification supply their actual logical and
        sampling coordinates for exactly one slot; their cache length is set
        separately through ``set_cache_length``. The continuation bit
        (``TOKEN_CONTINUATION_BIT``) is never stored as part of a token ID.

        Args:
            slots: Host request slots, one per sampled row.
            tokens: Selected token per row.
            predicates: Continuation predicate per row.
            valid: Per-row validity; with ``active``, gates penalty counting.
            active: Per-row activity.
            penalty_bases: Per row, the ``penalty_counts`` row to accumulate
                into, or None to skip counting for that row (for example,
                a request without penalties).
            device_slots: ``slots`` as an ``int64`` tensor on this state's
                device; selects the batched decode path.
            logical_position: Explicit logical length for the single slot.
            sampling_position: Explicit sampling position for the single
                slot.

        Raises:
            ValueError: When slots are out of range or repeated,
                ``penalty_bases`` (or, with ``device_slots``, ``tokens`` and
                ``predicates``) do not align with ``slots``, ``device_slots``
                is not an aligned ``int64`` tensor on this device, or the
                arguments form neither one non-empty batched decode nor one
                complete explicit row.
        """
        indices = tuple(int(slot) for slot in slots)
        self._validate_host_indices(indices)
        if len(indices) != len(penalty_bases):
            raise ValueError(
                "decode penalty rows do not align with request slots"
            )

        if device_slots is not None:
            if logical_position is not None or sampling_position is not None:
                raise ValueError(
                    "batched decode does not replace explicit coordinates"
                )
            self._advance_tokens(
                indices,
                device_indices=device_slots,
                tokens=tokens,
                predicates=predicates,
            )
            penalty_tokens = tokens.reshape(-1)
        else:
            if (
                len(indices) != 1
                or logical_position is None
                or sampling_position is None
            ):
                raise ValueError(
                    "explicit token export requires one complete request row"
                )
            slot = indices[0]
            future_token = self.future_input_tokens[slot, :1]
            future_token.copy_(tokens.reshape(-1)[:1])
            future_token.bitwise_and_((1 << 31) - 1)
            self.predicates[slot : slot + 1].copy_(
                predicates.reshape(-1)[:1].to(torch.bool)
            )
            self._copy_scalar(
                self.logical_lengths[slot : slot + 1], logical_position
            )
            self._copy_scalar(
                self.sampling_positions[slot : slot + 1], sampling_position
            )
            penalty_tokens = future_token

        # Occurrence counts grow only for tokens that are valid and active.
        for index, counts in enumerate(penalty_bases):
            if counts is None:
                continue
            weight = (
                valid.reshape(-1)[index : index + 1]
                & active.reshape(-1)[index : index + 1]
            ).to(counts.dtype)
            counts.scatter_add_(
                0, penalty_tokens[index : index + 1].to(torch.int64), weight
            )

    @staticmethod
    def _copy_scalar(target: torch.Tensor, value: int | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            target.copy_(
                value.reshape(-1)[:1].to(
                    device=target.device, dtype=target.dtype
                )
            )
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
        """Commit device-selected decode transitions for several slots.

        Each slot's first continuation token and predicate are replaced and
        its logical length, sampling position and verified cache length grow
        by one.
        """
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
            and launchable(self.device)
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

        # Fallback without launchable Triton: the same scatter updates via
        # index ops, using the preallocated unit increments.
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
        """Normalize host or device row indices onto the runtime-state device.

        Host sequences and CPU tensors are validated synchronously. On a
        CUDA state device, tensor indices are also checked with
        ``torch._assert_async``, which reports a violation as a device
        assertion instead of raising here.
        """
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
                    torch.all(
                        (indices >= 1) & (indices <= self.request_pool_size)
                    ),
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
        return async_tensor_h2d(host, dtype=torch.long, device=self.device)

    def _validate_host_indices(self, values: tuple[int, ...]) -> None:
        """Require unique host indices within ``1..request_pool_size``."""
        if any(value < 1 or value > self.request_pool_size for value in values):
            raise ValueError(
                "request-pool index is outside runtime-state capacity"
            )
        if len(set(values)) != len(values):
            raise ValueError(
                "runtime-state mutation repeats a request-pool index"
            )

    @staticmethod
    def _host_reset_column(
        values: torch.Tensor | Sequence[int] | None,
        count: int,
    ) -> tuple[int, ...] | None:
        """Normalize an optional host reset column to ``count`` integers.

        Returns zeros for an omitted column and None for a device tensor,
        which the caller handles on the indexed path.

        Raises:
            ValueError: When the column does not hold ``count`` values.
        """
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

    def _reset_rows(self, columns: torch.Tensor, count: int) -> None:
        """Reset ``count`` rows from a device ``[4, count]`` int64 table.

        The table holds the rows, then their verified cache lengths, logical
        lengths and sampling positions (`_decode_state._reset_rows_kernel`).
        """
        # One program per row and block of the wider of the two row-wide
        # spans.
        block_size = 256
        span = max(self.continuation_width, self.vocab_size)
        kernels._reset_rows_kernel[
            (count, kernels.triton.cdiv(span, block_size))
        ](
            columns,
            self.future_input_tokens,
            self.penalty_counts,
            self.predicates,
            self.logical_lengths,
            self.sampling_positions,
            self.valid_cache_lengths,
            count,
            continuation_width=self.continuation_width,
            vocab_size=self.vocab_size,
            block_size=block_size,
        )

    def _copy_or_zero(
        self,
        target: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor | Sequence[int] | None,
    ) -> None:
        """Scatter supplied values into indexed rows, or zero them if absent.

        Raises:
            ValueError: When ``values`` does not hold one value per index.
        """
        if values is None:
            target.index_fill_(0, indices, 0)
            return
        source = torch.as_tensor(
            values, dtype=target.dtype, device=self.device
        ).reshape(-1)
        if int(source.numel()) != int(indices.numel()):
            raise ValueError("runtime-state reset columns are not aligned")
        target[indices] = source


__all__ = ["DecodeState"]
