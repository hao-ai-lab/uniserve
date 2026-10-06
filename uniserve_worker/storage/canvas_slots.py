"""Resident block-diffusion canvases indexed by scheduler request slot.

Between two denoising steps a generating canvas keeps its tokens, the argmax
canvases its stopping rule compares, and the self-conditioning embedding of
its latest sampling distribution (``uniserve.diffusion.canvas.CanvasState``,
one row per canvas). ``CanvasSlots`` keeps these rows for every request-pool
slot in one device bank per field, with a leading slot axis. A pass gathers
the rows of its slots into its own contiguous input buffers (``gather``),
the sampler steps them in place, and ``commit`` writes them back.
Concurrent executions own separate input buffers and sampler workspaces while
sharing these banks for disjoint request slots. Slot ``0`` is the padding
sentinel, as in ``DecodeState``.

Every canvas uses the block-diffusion sampling the deployment serves
(``WorkerConfig.canvas_sampling``); its stability threshold sizes the
argmax history. The native executor checks the admitted sampling and
advances submitted canvas coordinates in the request pool.

A step may be queued behind the one before it, before that one's result is
known. Each slot therefore also keeps, on the device, whether its block
continues after the step it ran last (``live``): a queued step whose block
an earlier step stopped runs as a no-op (``CanvasRunner.step``), and its
result row reports ``STEP_SKIPPED``.
"""

from __future__ import annotations

import torch

from uniserve.diffusion import canvas as sampler
from uniserve.model import CanvasTokens, TokenDenoiser
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.tensors import BufferConfig
from uniserve_worker.protocol.batch import CanvasSampling

# Sampler state fields kept per slot, each a bank with one row per slot.
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
        tokens: CanvasTokens,
        vocab_size: int,
        hidden_size: int,
        sampling: CanvasSampling,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Allocate the resident sampler state banks.

        Args:
            request_pool_size: Number of real request slots.
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
            canvas_length=tokens.length,
            hidden_size=hidden_size,
            history_depth=sampling.stability_threshold,
            dtype=dtype,
        )
        self.request_pool_size = int(request_pool_size)
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

    @classmethod
    def for_denoiser(
        cls,
        denoiser: TokenDenoiser,
        *,
        request_pool_size: int,
        sampling: CanvasSampling,
        device: torch.device | str,
    ) -> CanvasSlots:
        """Allocate the slots of a generating denoiser's canvases.

        ``sampling`` is the deployment's served canvas sampling
        (``WorkerConfig.canvas_sampling``).
        """
        return cls(
            request_pool_size=request_pool_size,
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
        history_depth: int,
    ) -> int:
        """Device bytes of the slots ``for_denoiser`` allocates.

        Startup sizing charges the banks without an instance. Per-execution
        input buffers and sampler workspaces are charged by ``CanvasRunner``.
        """
        fields = denoiser_fields(denoiser)
        tokens = fields.pop("tokens")
        fields.pop("vocab_size")
        buffers = cls.buffers(
            request_pool_size=request_pool_size,
            canvas_length=tokens.length,
            history_depth=history_depth,
            **fields,
        )
        return sum(config.nbytes for config in buffers.values())

    @staticmethod
    def buffers(
        *,
        request_pool_size: int,
        canvas_length: int,
        hidden_size: int,
        history_depth: int,
        dtype: torch.dtype,
    ) -> dict[str, BufferConfig]:
        """Describe the resident banks by name.

        Each field's bank has one row per slot and the sentinel. A row of
        ``canvas`` holds int64 tokens ``[canvas]``, of ``history`` the int64
        argmax canvases ``[history_depth, canvas]``, and of
        ``self_conditioning`` the embeddings ``[canvas, hidden]``. ``live``
        holds one uint8 continuation flag per slot and the sentinel.

        Raises:
            ValueError: When a dimension is not positive or the history depth
                is negative.
        """
        if (
            min(request_pool_size, canvas_length) < 1
            or hidden_size < 1
            or history_depth < 0
        ):
            raise ValueError("canvas state dimensions must be positive")
        count = int(request_pool_size) + 1
        return {
            "canvas": BufferConfig((count, canvas_length), torch.int64),
            "history": BufferConfig(
                (count, history_depth, canvas_length), torch.int64
            ),
            "self_conditioning": BufferConfig(
                (count, canvas_length, hidden_size), dtype
            ),
            "live": BufferConfig((count,), torch.uint8),
        }

    def close(self) -> None:
        """Release the banks once every reader has retired."""
        self.banks = {}
        self._backing.close()

    def gather(
        self, slots: torch.Tensor, views: dict[str, torch.Tensor]
    ) -> None:
        """Gather the state of ``slots`` into caller-owned contiguous views.

        ``slots`` is a device int64 ``[rows]`` vector. The views hold, per
        field, the rows in slot order: ``canvas`` ``[rows,
        canvas]``, ``history`` ``[rows, depth, canvas]`` holding each slot's
        first ``depth`` argmax canvases, and ``self_conditioning`` ``[rows,
        canvas, hidden]``. Concurrent callers must supply distinct storage
        and own disjoint real request slots until their commits complete.
        """
        for name, view in views.items():
            bank = self.banks[name]
            if name == "history":
                bank = bank[:, : view.shape[1]]
            torch.index_select(bank, 0, slots, out=view)

    def commit(
        self,
        slots: torch.Tensor,
        views: dict[str, torch.Tensor],
        *,
        live: torch.Tensor,
    ) -> None:
        """Commit active rows and continuation flags; zero slots never write."""
        from uniserve_kernels.diffusion.canvas import commit_rows

        for name, values in (*views.items(), ("live", live)):
            bank = self.live if name == "live" else self.banks[name]
            if name == "history":
                bank = bank[:, : values.shape[1]]
            if bank.is_cuda:
                commit_rows(bank, slots, values)
            else:
                selected = slots > 0
                bank[slots[selected]] = values[selected]
