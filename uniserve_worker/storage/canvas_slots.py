"""Resident block-diffusion canvases indexed by scheduler request slot.

Between two denoising steps a generating canvas keeps its tokens, the argmax
canvases its stopping rule compares, and the self-conditioning embedding of
its latest sampling distribution (``uniserve.diffusion.canvas.CanvasState``,
one row per canvas). ``CanvasSlots`` keeps these rows for every request-pool
slot in one device bank per field, with a leading slot axis. A pass stages
the rows of its slots into contiguous row views (``stage``), the sampler
steps them in place, and ``commit`` writes them back; ``CanvasRunner`` runs
that numerical flow in chunks of at most ``step_rows`` canvases, whose
sampler scratch (``CanvasWorkspace``) this owner also holds. Slot ``0`` is
the padding sentinel, as in ``DecodeState``.

Every canvas uses the one block-diffusion sampling the deployment serves
(``WorkerConfig.canvas_sampling``): its stability threshold sizes the
argmax history, and a request that carries other sampling is refused
(``sampling``). The host records the step each slot ran last, so a call that
skips or repeats a step of its canvas is refused before any device work
(``advance``); ``reset`` forgets a slot's canvas when a request is admitted
to it or released from it.

A step may be queued behind the one before it, before that one's result is
known. Each slot therefore also keeps, on the device, whether its block
continues after the step it ran last (``live``): a queued step whose block
an earlier step stopped runs as a no-op (``CanvasRunner.step``), and its
result row reports ``STEP_SKIPPED``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import torch

from uniserve.diffusion import canvas as sampler
from uniserve.model import CanvasTokens, TokenDenoiser, VocabShard
from uniserve.nn import VocabParallelEmbedding
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.protocol.batch import CanvasSampling

# Sampler state fields kept per slot, each a bank with one row per slot and
# a staging area with one row per canvas of a pass.
FIELDS = ("canvas", "history", "self_conditioning")
# The first value of a canvas step's int64 result row: the step left its
# block running, stopped it, or was skipped since an earlier step had
# stopped it.
STEP_CONTINUED, STEP_STOPPED, STEP_SKIPPED = 0, 1, 2
# Bytes of one step chunk's FP32 logits and BF16 sampling weights, the two
# vocabulary-wide tensors of a step. A pass steps as many whole canvases at
# a time as fit (at least one), so its transient logits and its workspace
# stay bounded; a captured step keeps the head's transients of one chunk in
# its graph pool.
STEP_BYTES = 512 << 20


def step_rows(*, canvas_length: int, vocab_size: int, max_rows: int) -> int:
    """Canvases one sampler step chunk takes: those ``STEP_BYTES`` holds."""
    per_row = canvas_length * vocab_size * (4 + 2)
    return max(1, min(max_rows, STEP_BYTES // per_row))


def generating_denoiser(module) -> TokenDenoiser | None:
    """Return ``module`` when it generates canvases on this rank.

    The sampler embeds each step's distributions with the whole token
    embedding table, so a token denoiser generates canvases when it has
    declared them and its first pipeline stage, which holds an unsharded
    table, also projects the logits it samples from. Returns None otherwise;
    such a rank serves canvas readouts only.
    """
    if not isinstance(module, TokenDenoiser):
        return None
    mesh = module.backbone.mesh
    pipeline = mesh.get_group("pp" if "pp" in mesh.axes else ())
    embedding = module.backbone.embedding
    if pipeline.size != 1 or embedding is None or embedding.group.size != 1:
        return None
    return module


class DenoiserFields(TypedDict):
    """The ``CanvasSlots`` keyword arguments a generating denoiser implies."""

    tokens: CanvasTokens
    vocab_size: int
    hidden_size: int
    dtype: torch.dtype


def denoiser_fields(denoiser: TokenDenoiser) -> DenoiserFields:
    """The ``CanvasSlots`` dimensions a generating denoiser implies.

    ``denoiser`` is one ``generating_denoiser`` returns, so this rank holds
    its token embedding table and its vocabulary head.

    Raises:
        TypeError: When the rank lacks the token embedding table or a
            vocabulary head exposing its ``VocabShard``.
    """
    embedding, head = denoiser.backbone.embedding, denoiser.lm_head
    if not isinstance(embedding, VocabParallelEmbedding) or head is None:
        raise TypeError(
            "a generating denoiser holds its token embedding table and "
            "vocabulary head"
        )
    vocab = head.vocab
    if not isinstance(vocab, VocabShard):
        raise TypeError("the vocabulary head must expose a VocabShard")
    return {
        "tokens": denoiser.canvas,
        "vocab_size": vocab.size,
        "hidden_size": embedding.embedding_dim,
        "dtype": embedding.weight.dtype,
    }


class CanvasSlots:
    """Own the sampler state of every request slot's generating canvas."""

    # The step workspace; None once ``close`` released it.
    workspace: sampler.CanvasWorkspace | None

    def __init__(
        self,
        *,
        request_pool_size: int,
        max_rows: int,
        tokens: CanvasTokens,
        vocab_size: int,
        hidden_size: int,
        sampling: CanvasSampling,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Allocate the banks, the pass staging and the step workspace.

        Args:
            request_pool_size: Number of real request slots.
            max_rows: Most canvases one pass stages.
            tokens: The length and tokens of the canvases the model
                generates.
            vocab_size: Vocabulary size of the denoiser's logits.
            hidden_size: Width of one self-conditioning embedding.
            sampling: The block-diffusion sampling the deployment serves;
                its stability threshold is the argmax history kept.
            dtype: Dtype of the self-conditioning embeddings and sampling
                weights, the model's embedding dtype.
            device: Device of every tensor.

        Raises:
            ValueError: When a dimension is not positive, the sampling's
                canvas length is not the model's, or the sampler refuses
                the sampling's values.
        """
        if sampling.canvas_length != tokens.length:
            raise ValueError(
                "the served canvas length is not the model's canvas length"
            )
        fields = self.buffers(
            request_pool_size=request_pool_size,
            max_rows=max_rows,
            canvas_length=tokens.length,
            hidden_size=hidden_size,
            history_depth=sampling.stability_threshold,
            dtype=dtype,
        )
        self.request_pool_size = int(request_pool_size)
        self.max_rows = int(max_rows)
        self.tokens = tokens
        self.canvas_length = tokens.length
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.served = sampling
        self.history_depth = sampling.stability_threshold
        # The sampler's constants for the served sampling, with the model's
        # end-of-sequence and padding tokens; the sampler checks their domain.
        self.constants = sampler.CanvasSampling(
            steps=sampling.max_steps,
            entropy_bound=sampling.entropy_bound,
            t_min=sampling.t_min,
            t_max=sampling.t_max,
            confidence=sampling.confidence_threshold,
            stability=sampling.stability_threshold,
            eos_ids=tokens.eos_token_ids,
            pad_id=tokens.pad_token_id,
        )
        self.step_rows = step_rows(
            canvas_length=tokens.length,
            vocab_size=vocab_size,
            max_rows=max_rows,
        )
        self.device = torch.device(device)
        self._backing = TensorBuffers.allocate(fields, device=self.device)
        tensors = self._backing.view(fields)
        # A slot's rows are rewritten by its first step before any read, so
        # zeros only give the banks defined contents.
        for tensor in tensors.values():
            tensor.zero_()
        self.banks = {name: tensors[name] for name in FIELDS}
        # uint8 [slots + 1]: 1 while the slot's block continues after the
        # step it ran last. Step zero sets it, and a step that stops the
        # block clears it.
        self.live = tensors["live"]
        # One step chunk's sampler scratch, which every chunk of every pass
        # reuses in stream order.
        self.workspace = sampler.CanvasWorkspace.empty(
            self.step_rows,
            tokens.length,
            vocab_size,
            hidden_size,
            dtype=dtype,
            device=self.device,
        )
        # Per slot, the (block, step) it ran last; absent for a slot whose
        # canvas has not started since its request was admitted.
        self._last: dict[int, tuple[int, int]] = {}

    @classmethod
    def for_denoiser(
        cls,
        denoiser: TokenDenoiser,
        *,
        request_pool_size: int,
        max_rows: int,
        sampling: CanvasSampling,
        device: torch.device | str,
    ) -> CanvasSlots:
        """Allocate the slots of a generating denoiser's canvases.

        ``sampling`` is the deployment's served canvas sampling
        (``WorkerConfig.canvas_sampling``).
        """
        return cls(
            request_pool_size=request_pool_size,
            max_rows=max_rows,
            sampling=sampling,
            device=device,
            **denoiser_fields(denoiser),
        )

    @classmethod
    def denoiser_bytes(
        cls,
        denoiser: TokenDenoiser,
        *,
        request_pool_size: int,
        max_rows: int,
        history_depth: int,
        device_type: str,
    ) -> int:
        """Device bytes of the slots ``for_denoiser`` allocates.

        The banks and their staging (``buffers``) and one step chunk's
        ``CanvasWorkspace`` on a device of type ``device_type``. Startup
        sizing (``bootstrap.report``) charges them without an instance.
        """
        fields = denoiser_fields(denoiser)
        tokens, vocab_size = fields["tokens"], fields["vocab_size"]
        buffers = cls.buffers(
            request_pool_size=request_pool_size,
            max_rows=max_rows,
            canvas_length=tokens.length,
            hidden_size=fields["hidden_size"],
            history_depth=history_depth,
            dtype=fields["dtype"],
        )
        rows = step_rows(
            canvas_length=tokens.length,
            vocab_size=vocab_size,
            max_rows=max_rows,
        )
        return sum(
            config.nbytes for config in buffers.values()
        ) + sampler.CanvasWorkspace.nbytes(
            rows,
            tokens.length,
            vocab_size,
            fields["hidden_size"],
            dtype=fields["dtype"],
            device_type=device_type,
        )

    @staticmethod
    def buffers(
        *,
        request_pool_size: int,
        max_rows: int,
        canvas_length: int,
        hidden_size: int,
        history_depth: int,
        dtype: torch.dtype,
    ) -> dict[str, BufferConfig]:
        """Describe the banks and their staging, by name.

        Each field's bank has one row per slot and the sentinel, and its
        staging (``staged_<field>``) one row per canvas of a pass. A row of
        ``canvas`` holds int64 tokens ``[canvas]``, of ``history`` the int64
        argmax canvases ``[history_depth, canvas]``, and of
        ``self_conditioning`` the embeddings ``[canvas, hidden]``. ``live``
        holds one uint8 continuation flag per slot and the sentinel.

        Raises:
            ValueError: When a dimension is not positive or the history depth
                is negative.
        """
        if (
            min(request_pool_size, max_rows, canvas_length) < 1
            or hidden_size < 1
            or history_depth < 0
        ):
            raise ValueError("canvas state dimensions must be positive")
        counts = {"": int(request_pool_size) + 1, "staged_": int(max_rows)}
        return {
            f"{prefix}{name}": config
            for prefix, count in counts.items()
            for name, config in {
                "canvas": BufferConfig((count, canvas_length), torch.int64),
                "history": BufferConfig(
                    (count, history_depth, canvas_length), torch.int64
                ),
                "self_conditioning": BufferConfig(
                    (count, canvas_length, hidden_size), dtype
                ),
            }.items()
        } | {"live": BufferConfig((counts[""],), torch.uint8)}

    def close(self) -> None:
        """Release the banks and the workspace once every reader has retired."""
        self.banks = {}
        self.workspace = None
        self._backing.close()

    def sampling(self, admitted: CanvasSampling) -> sampler.CanvasSampling:
        """The sampler constants of a request's admitted canvas sampling.

        Every request carries the sampling the deployment serves, so these
        are ``constants``.

        Raises:
            WorkerError: ``invalid_descriptor`` when ``admitted`` is not the
                served sampling.
        """
        if admitted != self.served:
            raise invalid_descriptor(
                "the admitted canvas sampling is not the sampling this "
                "worker serves"
            )
        return self.constants

    def advance(self, slot: int, block: int, step: int) -> None:
        """Record that ``slot`` runs step ``step`` of its block ``block``.

        Step zero starts a canvas: the slot's first, or the one after the
        block it ran last. Any other step must follow the slot's last step
        of the same block.

        Raises:
            WorkerError: ``invalid_descriptor`` for a slot outside the pool
                or a step that does not continue the slot's canvas.
        """
        self._validate((slot,))
        last = self._last.get(slot)
        if step == 0:
            expected = last is None or block == last[0] + 1
        else:
            expected = last == (block, step - 1)
        if not expected:
            raise invalid_descriptor(
                f"canvas step {(block, step)} does not continue slot {slot}'s "
                f"canvas at {last}"
            )
        self._last[slot] = (block, step)

    def reset(self, slots: Sequence[int]) -> None:
        """Forget the canvases of ``slots``; their next step must be zero."""
        self._validate(slots)
        for slot in slots:
            self._last.pop(int(slot), None)

    def stage(self, slots: torch.Tensor, depth: int) -> dict[str, torch.Tensor]:
        """Gather the state of ``slots`` into contiguous row views.

        ``slots`` is a device int64 ``[rows]`` vector. Returns, per field,
        the staging's leading rows in slot order: ``canvas`` ``[rows,
        canvas]``, ``history`` ``[rows, depth, canvas]`` holding each slot's
        first ``depth`` argmax canvases, and ``self_conditioning`` ``[rows,
        canvas, hidden]``. The views stay valid until the next ``stage``.

        Raises:
            ValueError: When the rows exceed the staging or ``depth`` the
                history.
        """
        rows = int(slots.numel())
        if (
            not 0 < rows <= self.max_rows
            or not 0 <= depth <= self.history_depth
        ):
            raise ValueError("canvas rows exceed the staged sampler state")
        shapes = {
            "canvas": (rows, self.canvas_length),
            "history": (rows, depth, self.canvas_length),
            "self_conditioning": (rows, self.canvas_length, self.hidden_size),
        }
        staged = self._backing.view(
            {
                f"staged_{name}": BufferConfig(shape, self.banks[name].dtype)
                for name, shape in shapes.items()
            }
        )
        views = {name: staged[f"staged_{name}"] for name in FIELDS}
        for name, view in views.items():
            bank = self.banks[name]
            if name == "history":
                bank = bank[:, :depth]
            torch.index_select(bank, 0, slots, out=view)
        return views

    def commit(
        self, slots: torch.Tensor, views: dict[str, torch.Tensor]
    ) -> None:
        """Write staged row views back to the banks' rows of ``slots``."""
        for name, values in views.items():
            bank = self.banks[name]
            if name == "history":
                bank[slots, : values.shape[1]] = values
            else:
                bank.index_copy_(0, slots, values)

    def _validate(self, slots: Sequence[int]) -> None:
        if any(not 1 <= int(slot) <= self.request_pool_size for slot in slots):
            raise invalid_descriptor("canvas state slot is outside the pool")
