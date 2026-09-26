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

The host records the step each slot ran last, so a call that skips or
repeats a step of its canvas is refused before any device work
(``advance``); ``reset`` forgets a slot's canvas when a request is admitted
to it or released from it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from uniserve.diffusion import canvas as sampler
from uniserve.model import CanvasTokens, TokenDenoiser
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig
from uniserve_worker.errors import invalid_descriptor

# Sampler state fields kept per slot, each a bank with one row per slot and
# a staging area with one row per canvas of a pass.
FIELDS = ("canvas", "history", "self_conditioning")
# Bytes of one step chunk's FP32 logits and BF16 sampling weights, the two
# vocabulary-wide tensors of a step. A pass steps as many whole canvases at
# a time as fit, so its transient logits and its workspace stay bounded.
STEP_BYTES = 2 << 30
# Bytes of the cuBLASLt workspace the sampler's CUDA self-conditioning
# product uses for split-K partial sums: the most it takes for any step
# shape. The product runs without one on other devices.
SCRATCH_BYTES = 256 << 20


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


def denoiser_fields(denoiser: TokenDenoiser) -> dict[str, object]:
    """The ``CanvasSlots`` dimensions a generating denoiser implies."""
    embedding = denoiser.backbone.embedding
    return {
        "tokens": denoiser.canvas,
        "vocab_size": denoiser.lm_head.vocab.size,
        "hidden_size": embedding.embedding_dim,
        "dtype": embedding.weight.dtype,
    }


class CanvasSlots:
    """Own the sampler state of every request slot's generating canvas."""

    def __init__(
        self,
        *,
        request_pool_size: int,
        max_rows: int,
        tokens: CanvasTokens,
        vocab_size: int,
        hidden_size: int,
        history_depth: int,
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
            history_depth: Argmax canvases the stopping rule keeps, the
                largest stability a request may use: the stability of the
                canvas sampling the deployment serves.
            dtype: Dtype of the self-conditioning embeddings and sampling
                weights, the model's embedding dtype.
            device: Device of every tensor.

        Raises:
            ValueError: When a dimension is not positive or the history depth
                is negative.
        """
        fields = self.buffers(
            request_pool_size=request_pool_size,
            max_rows=max_rows,
            canvas_length=tokens.length,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            history_depth=history_depth,
            dtype=dtype,
            cuda=torch.device(device).type == "cuda",
        )
        self.request_pool_size = int(request_pool_size)
        self.max_rows = int(max_rows)
        self.tokens = tokens
        self.canvas_length = tokens.length
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.history_depth = int(history_depth)
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
        self.workspace = sampler.CanvasWorkspace(
            tensors["workspace_weights"],
            tensors["workspace_normalizer"],
            tensors["workspace_product"],
            tensors["workspace_scratch"],
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
        history_depth: int,
        device: torch.device | str,
    ) -> CanvasSlots:
        """Allocate the slots of a generating denoiser's canvases.

        ``history_depth`` is the deployment's canvas stability threshold
        (``WorkerConfig.canvas_history_depth``).
        """
        return cls(
            request_pool_size=request_pool_size,
            max_rows=max_rows,
            history_depth=history_depth,
            device=device,
            **denoiser_fields(denoiser),
        )

    @classmethod
    def denoiser_buffers(
        cls,
        denoiser: TokenDenoiser,
        *,
        request_pool_size: int,
        max_rows: int,
        history_depth: int,
        cuda: bool,
    ) -> dict[str, BufferConfig]:
        """``buffers`` of the slots ``for_denoiser`` allocates."""
        fields = denoiser_fields(denoiser)
        tokens = fields.pop("tokens")
        return cls.buffers(
            request_pool_size=request_pool_size,
            max_rows=max_rows,
            canvas_length=tokens.length,
            history_depth=history_depth,
            cuda=cuda,
            **fields,
        )

    @staticmethod
    def buffers(
        *,
        request_pool_size: int,
        max_rows: int,
        canvas_length: int,
        vocab_size: int,
        hidden_size: int,
        history_depth: int,
        dtype: torch.dtype,
        cuda: bool,
    ) -> dict[str, BufferConfig]:
        """Describe the banks, their staging and the workspace, by name.

        Each field's bank has one row per slot and the sentinel, and its
        staging (``staged_<field>``) one row per canvas of a pass. A row of
        ``canvas`` holds int64 tokens ``[canvas]``, of ``history`` the int64
        argmax canvases ``[history_depth, canvas]``, and of
        ``self_conditioning`` the embeddings ``[canvas, hidden]``. The
        ``workspace_*`` fields are one step chunk's ``CanvasWorkspace``,
        whose ``scratch`` holds ``SCRATCH_BYTES`` when the slots are on a
        ``cuda`` device and nothing otherwise.
        Startup sizing (``bootstrap.report``) charges them without an
        instance.

        Raises:
            ValueError: When a dimension is not positive or the history depth
                is negative.
        """
        if (
            min(request_pool_size, max_rows, canvas_length, vocab_size) < 1
            or hidden_size < 1
            or history_depth < 0
        ):
            raise ValueError("canvas state dimensions must be positive")
        counts = {"": int(request_pool_size) + 1, "staged_": int(max_rows)}
        fields = {
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
        }
        positions = (
            step_rows(
                canvas_length=canvas_length,
                vocab_size=vocab_size,
                max_rows=max_rows,
            )
            * canvas_length
        )
        return {
            **fields,
            "workspace_weights": BufferConfig((positions, vocab_size), dtype),
            "workspace_normalizer": BufferConfig((positions,), torch.float32),
            "workspace_product": BufferConfig(
                (positions, hidden_size), torch.float32
            ),
            "workspace_scratch": BufferConfig(
                (SCRATCH_BYTES if cuda else 0,), torch.uint8
            ),
        }

    def close(self) -> None:
        """Release the banks once every reader has retired."""
        self.banks = {}
        self._backing.close()

    def sampling(self, admitted) -> sampler.CanvasSampling:
        """The sampler constants of a request's admitted canvas sampling.

        ``admitted`` is the request's protocol ``CanvasSampling``; the
        end-of-sequence and padding tokens are the model's.

        Raises:
            WorkerError: ``invalid_descriptor`` when the admitted sampling
                has another canvas length, a stability beyond the kept
                history, or values the sampler refuses.
        """
        if admitted.canvas_length != self.canvas_length:
            raise invalid_descriptor(
                "the admitted canvas length is not the model's canvas length"
            )
        if admitted.stability_threshold > self.history_depth:
            raise invalid_descriptor(
                "the canvas stability threshold exceeds the kept history"
            )
        try:
            return sampler.CanvasSampling(
                steps=admitted.max_steps,
                entropy_bound=admitted.entropy_bound,
                t_min=admitted.t_min,
                t_max=admitted.t_max,
                confidence=admitted.confidence_threshold,
                stability=admitted.stability_threshold,
                eos_ids=self.tokens.eos_token_ids,
                pad_id=self.tokens.pad_token_id,
            )
        except ValueError as error:
            raise invalid_descriptor(str(error)) from error

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
