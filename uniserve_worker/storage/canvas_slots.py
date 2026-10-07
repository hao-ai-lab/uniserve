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

from uniserve.model import TokenDenoiser
from uniserve_worker._uniserve_ipc import CanvasSlots

# Sampler state fields kept per slot, each a bank with one row per slot.
FIELDS = ("canvas", "history", "self_conditioning")
# The first value of a canvas step's int64 result row: the step left its
# block running, stopped it, or was skipped since an earlier step had
# stopped it.
STEP_CONTINUED, STEP_STOPPED, STEP_SKIPPED = 0, 1, 2


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


def gather(
    banks: dict[str, torch.Tensor],
    slots: torch.Tensor,
    views: dict[str, torch.Tensor],
) -> None:
    """Gather slot state into caller-owned contiguous numerical views."""
    for name, view in views.items():
        bank = banks[name]
        if name == "history":
            bank = bank[:, : view.shape[1]]
        torch.index_select(bank, 0, slots, out=view)


def commit(
    banks: dict[str, torch.Tensor],
    live_bank: torch.Tensor,
    slots: torch.Tensor,
    views: dict[str, torch.Tensor],
    live: torch.Tensor,
) -> None:
    """Scatter active rows and flags, leaving padding slot zero unchanged."""
    from uniserve_kernels.diffusion.canvas import commit_rows

    for name, values in (*views.items(), ("live", live)):
        bank = live_bank if name == "live" else banks[name]
        if name == "history":
            bank = bank[:, : values.shape[1]]
        if bank.is_cuda:
            commit_rows(bank, slots, values)
        else:
            selected = slots > 0
            bank[slots[selected]] = values[selected]


__all__ = ["CanvasSlots", "generating_denoiser"]
