"""System-side text backend gating — which forward shape a batch can run on.

Relocated from the model (``Qwen3.can_run_text_batch`` and the
``_*_backend_available`` probes): deciding whether a text op group can run as one
**batched paged/varlen** forward, or must fall back to **per-op dense**
(single-request) execution, is a *system* capability question — it depends on the
installed attention backends, the page-block multiple, and whether the system KV
pool's storage is paged-attention-friendly — not on the model. The model only
declares its ``head_dim``; the system owns the rest.

Both the batched and the per-op path call the same thin ``model.forward``; this
gate only picks which one the driver builds.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...contracts.forward_mode import ForwardMode
from ...foundation.runtime_config import get_worker_config

if TYPE_CHECKING:
    from ...contracts.batches import TextBatch

__all__ = ["TextBackendGate"]

_TEXT_MODES = frozenset({ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.VERIFY_DRAFT})


class TextBackendGate:
    """Decide batched-paged vs per-op execution for a text group.

    Parameterized on the model geometry (``head_dim``), the system pool sizing
    (``block_size``) and storage flag (``paged_storage_ok``), and the device —
    primitives only, so it stays in the backends layer with no pool dependency.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        block_size: int,
        device_type: str,
        paged_storage_ok: bool = True,
    ) -> None:
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.device_type = str(device_type)
        self.paged_storage_ok = bool(paged_storage_ok)

    def batched_capable(self, text: "TextBatch", *, attention_preference: str | None) -> bool:
        """Whether ``text`` can run as one batched paged/varlen forward."""

        if text.mode not in _TEXT_MODES and text.mode != ForwardMode.MIXED:
            return False
        lengths = [len(tokens) for tokens in text.token_ids]
        if not lengths or any(length <= 0 for length in lengths):
            return False
        if self.device_type != "cuda":
            return False
        if text.mode == ForwardMode.MIXED:
            # Heterogeneous extend+decode fuses into one varlen-paged forward
            # (the decode rows carry KV history, so paged KV is required).
            return self._varlen_available(attention_preference, needs_paged_kv=True)
        if text.mode == ForwardMode.DECODE and any(length != 1 for length in lengths):
            return False
        if text.mode == ForwardMode.EXTEND:
            initial_extend = self._is_initial_extend(text)
            if len(set(lengths)) != 1:
                return self._varlen_available(
                    attention_preference,
                    needs_paged_kv=True,
                ) or self._varlen_available(
                    attention_preference,
                    needs_paged_kv=not initial_extend,
                )
            if not initial_extend and lengths[0] != 1:
                return self._multi_token_paged_available(attention_preference)
        return self._paged_available(attention_preference)

    def mixed_capable(
        self,
        ops,
        *,
        attention_preference: str | None,
    ) -> bool:
        """Whether a heterogeneous extend+decode group can fuse into one forward."""

        from collections.abc import Sequence as _Seq

        modes: list[ForwardMode] = []
        total_tokens = 0
        for op in ops:
            kind = op.get("kind")
            if not isinstance(kind, str):
                return False
            mode = (
                ForwardMode.EXTEND if kind == "prefill_und"
                else ForwardMode.DECODE if kind == "decode_und"
                else None
            )
            if mode is None:
                return False
            if _has_values(op.get("spec_token_ids")):
                return False
            tokens = op.get("token_ids") or []
            if not isinstance(tokens, _Seq) or isinstance(tokens, (str, bytes, bytearray)) or len(tokens) <= 0:
                return False
            total_tokens += len(tokens)
            if mode is ForwardMode.DECODE and len(tokens) != 1:
                return False
            modes.append(mode)
        if not modes or ForwardMode.EXTEND not in modes or ForwardMode.DECODE not in modes:
            return False
        max_tokens = max(0, get_worker_config().mixed_text_max_tokens)
        if max_tokens <= 0 or total_tokens > max_tokens:
            return False
        return self._varlen_available(attention_preference, needs_paged_kv=True)

    @staticmethod
    def _is_initial_extend(text: "TextBatch") -> bool:
        return text.mode == ForwardMode.EXTEND and all(int(pos[0]) == 0 for pos in text.pos_ranges)

    def _paged_available(self, preferred: str | None) -> bool:
        for _name, caps in self._attention_provider_caps(preferred):
            if not bool(caps.get("paged_kv", False)):
                continue
            if not self._head_dim_ok(caps):
                continue
            if not self.paged_storage_ok:
                continue
            multiple = int(caps.get("paged_block_size_multiple", 1) or 1)
            if self.block_size % max(1, multiple) == 0:
                return True
        return False

    def _multi_token_paged_available(self, preferred: str | None) -> bool:
        for name, caps in self._attention_provider_caps(preferred):
            if name == "flashinfer" or bool(caps.get("paged_decode_only", False)):
                continue
            if not bool(caps.get("paged_kv", False)):
                continue
            if not self._head_dim_ok(caps):
                continue
            if not self.paged_storage_ok:
                continue
            multiple = int(caps.get("paged_block_size_multiple", 1) or 1)
            if self.block_size % max(1, multiple) == 0:
                return True
        return False

    def _varlen_available(self, preferred: str | None, *, needs_paged_kv: bool = False) -> bool:
        if not get_worker_config().varlen_prefill:
            return False
        if needs_paged_kv and not self.paged_storage_ok:
            return False
        for _name, caps in self._attention_provider_caps(preferred):
            if not bool(caps.get("varlen_attention", False)):
                continue
            if needs_paged_kv:
                if not bool(caps.get("varlen_paged_kv", False)):
                    continue
                multiple = int(caps.get("paged_block_size_multiple", 1) or 1)
                if self.block_size % max(1, multiple) != 0:
                    continue
            elif bool(caps.get("requires_paged_varlen", False)):
                continue
            if not self._head_dim_ok(caps):
                continue
            return True
        return False

    def _attention_provider_caps(self, preferred: str | None):
        import uniserve_worker.ops as ops

        from .selector import AttentionBackendSelector

        yield from AttentionBackendSelector(ops.attention_dispatcher()).capability_rows(
            preferred or "auto"
        )

    def _head_dim_ok(self, caps) -> bool:
        return self.head_dim >= int(caps.get("min_head_dim", 1) or 1)


def _has_values(raw) -> bool:
    if raw is None:
        return False
    try:
        return len(raw) > 0
    except TypeError:
        return bool(raw)
