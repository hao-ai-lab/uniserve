"""Denoising passes over discrete token canvases that read a prefix cache.

A token denoiser refines a block of token ids, the canvas, conditioned on a
prompt whose keys and values a causal pass of the same backbone already
cached. Each pass embeds the canvas, mixes in the previous pass's soft
embeddings (self-conditioning), and traverses the backbone with canvas rows
that attend non-causally to their cached prefix and to the whole canvas
without writing the cache. Sampling, acceptance and commitment of canvas
tokens belong to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from uniserve.nn.attention import AttentionBatch

from .logits import Logits, project_logits
from .transformer import TransformerDecoder


@dataclass(frozen=True, slots=True)
class CanvasInput:
    """Borrowed tensors for one denoising pass over packed token canvases.

    ``input_ids`` and ``positions`` cover every canvas row back to back
    (``[rows * canvas]``); positions continue after each row's prefix.
    ``attention`` lets each canvas token read its row's cached prefix and
    every token of its own canvas. Its entries carry no causal rows, and the
    caller keeps the prefix pages read-only: the pass must not write them.
    ``self_conditioning`` holds the soft embeddings of the previous pass,
    ``[rows * canvas, hidden]`` in the token-embedding scale, or None on a
    row's first pass.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    attention: AttentionBatch
    self_conditioning: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            self.input_ids.ndim != 1
            or self.positions.shape[-1] != self.input_ids.shape[0]
        ):
            raise ValueError(
                "canvas tokens and positions must cover the same packed rows"
            )
        if self.self_conditioning is not None and (
            self.self_conditioning.ndim != 2
            or self.self_conditioning.shape[0] != self.input_ids.shape[0]
        ):
            raise ValueError(
                "self-conditioning embeddings must cover every canvas token"
            )
        # Canvas tokens see each other in both directions. Inputs that name
        # causality per row must not restrict any canvas row.
        for entry in self.attention.entries.values():
            causal = getattr(entry, "causal", False)
            if any(causal if isinstance(causal, tuple) else (causal,)):
                raise ValueError("canvas rows attend without causal masks")

    @property
    def batch_size(self) -> int:
        """Number of canvas rows in the packed pass."""
        queries = self.attention.queries
        if queries is None:
            raise ValueError("canvas passes require packed canvas rows")
        return queries.batch_size


class SelfConditioning(nn.Module):
    """Mix the previous pass's soft embeddings into canvas token embeddings.

    ``forward(embeddings, soft)`` computes
    ``post_norm(embeddings + mlp(pre_norm(soft)))`` over ``[tokens, hidden]``
    rows. The model supplies the three modules. ``pre_norm`` and ``mlp``
    must map zero rows to zero (RMS normalization and bias-free gated
    projections whose activation vanishes at zero do), so a first pass with
    no soft embeddings equals the all-zero signal exactly; ``forward``
    evaluates that case as ``post_norm(embeddings)``.
    """

    def __init__(
        self, pre_norm: nn.Module, mlp: nn.Module, post_norm: nn.Module
    ) -> None:
        super().__init__()
        self.pre_norm, self.mlp, self.post_norm = pre_norm, mlp, post_norm

    def forward(
        self, embeddings: torch.Tensor, soft: torch.Tensor | None = None
    ) -> torch.Tensor:
        if soft is None:
            return self.post_norm(embeddings)
        signal = self.mlp(self.pre_norm(soft.to(embeddings.dtype)))
        return self.post_norm(embeddings + signal)


class TokenDenoiser(nn.Module):
    """One denoising pass of a discrete token canvas over a read-only prefix.

    The denoiser shares the causal model's backbone and vocabulary head.
    ``forward(inputs)`` returns the final-normalized canvas rows
    ``[rows * canvas, hidden]`` on the last pipeline stage (earlier stages
    return the activations they forward). ``compute_logits`` projects
    caller-selected rows through the head. Concrete models supply only the
    ``SelfConditioning`` modules and the head's mathematics.
    """

    # Pipeline binding drops the head outside the last stage.
    lm_head: nn.Module | None

    def __init__(
        self,
        backbone: TransformerDecoder,
        lm_head: nn.Module,
        self_conditioning: SelfConditioning,
    ):
        super().__init__()
        self.backbone, self.lm_head = backbone, lm_head
        self.self_conditioning = self_conditioning

    @property
    def cache_config(self):
        return self.backbone.cache_config

    def forward(self, inputs: CanvasInput) -> torch.Tensor:
        embeddings = None
        if self.backbone._pipeline.rank == 0:
            embeddings = self.self_conditioning(
                self.backbone.embed_input_ids(inputs.input_ids),
                inputs.self_conditioning,
            )
        return self.backbone(embeddings, inputs.positions, inputs.attention)

    def compute_logits(
        self, hidden: torch.Tensor, *, token_indices: torch.Tensor
    ) -> Logits | None:
        """Project caller-selected canvas rows on the final pipeline stage.

        Returns None on other stages; see ``project_logits`` for the
        selection contract.
        """
        pipeline = self.backbone._pipeline
        if pipeline.rank != pipeline.size - 1:
            return None
        head = self.lm_head
        assert head is not None
        return project_logits(head, hidden, token_indices)
