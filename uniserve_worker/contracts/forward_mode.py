"""The :class:`ForwardMode` enum and op-kind -> mode resolution.

Defines the single most-imported forward-mode type in the worker. Imports only
``foundation`` and the torch-free ``contracts.op_kinds`` table.
"""
from __future__ import annotations

from enum import Enum

from ..foundation.errors import invalid_descriptor
from .op_kinds import OP_KIND_TABLE

__all__ = ["ForwardMode", "mode_for_op"]


class ForwardMode(str, Enum):
    """Forward-pass execution mode, one per batch or per emission stage.

    Text modes (share the ``forward`` model hook):

    - EXTEND: prefill — process a variable-length token sequence and populate
      the KV cache.
    - DECODE: autoregressive generation — advance one (or a few) tokens per
      request against an already-populated KV cache.
    - VERIFY_DRAFT: speculative-decoding verification — the target model
      evaluates a draft sequence in one shot to accept or reject candidates.

    Vision / diffusion modes:

    - ENCODE: run an encoder (ViT or VAE) to produce embeddings or latents
      from a non-text input.
    - DENOISE: one diffusion denoising step in latent space (predict velocity).
    - COMMIT: VAE-decode the final latent into pixel space.

    Composite:

    - MIXED: a batch containing ops of more than one mode; the execution layer
      dispatches each row to its corresponding sub-path.

    Emission stages (run on a downstream worker, not the model worker):

    - EMIT_TOKEN: turn a logits handle into a sampled token (separated from
      DECODE when the sampler runs on its own worker).
    - EMIT_FRAME: encode raw pixel output into a deliverable frame format
      (separated from COMMIT when postprocessing runs on its own worker).
    """

    EXTEND = "extend"
    DECODE = "decode"
    VERIFY_DRAFT = "verify_draft"
    ENCODE = "encode"
    DENOISE = "denoise"
    COMMIT = "commit"
    MIXED = "mixed"
    EMIT_TOKEN = "emit_token"
    EMIT_FRAME = "emit_frame"


def mode_for_op(kind: str) -> "ForwardMode":
    spec = OP_KIND_TABLE.get(kind)
    if spec is None:
        raise invalid_descriptor(f"unsupported op kind {kind!r}")
    return ForwardMode(spec.mode)


# Every op-kind spec must name a real ``ForwardMode``; this couples the
# contract-side mode strings (kept torch-free in ``contracts.op_kinds``) to this
# enum so a typo there fails at import rather than at first dispatch.
assert all(spec.mode in ForwardMode._value2member_map_ for spec in OP_KIND_TABLE.values())
