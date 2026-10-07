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


@dataclass(frozen=True, slots=True)
class CanvasTokens:
    """The block of text one generating canvas holds.

    ``length`` is the canvas length in tokens. A canvas's text ends at its
    first ``eos_token_ids`` token; the tokens after it are ``pad_token_id``.
    """

    length: int
    pad_token_id: int
    eos_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.length < 1
            or self.pad_token_id < 0
            or not self.eos_token_ids
            or min(self.eos_token_ids) < 0
        ):
            raise ValueError(
                "a canvas has a positive length and token-id pad and EOS"
            )


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
    return the activations they forward). A pass that reads only some rows
    splits instead: ``attend(inputs)`` evaluates every canvas token up to the
    final layer's attention output, and ``finish`` completes the rest for
    the selected rows alone (``TransformerDecoder.attend``). Its rows equal
    ``forward``'s up to the rounding that fewer rows allow.
    ``compute_logits`` projects caller-selected rows through the head.
    ``canvas`` declares the canvases the denoiser generates text in.
    Concrete models supply only the ``SelfConditioning`` modules, the head's
    mathematics and the canvas tokens.
    """

    # Pipeline binding drops the head outside the last stage.
    lm_head: nn.Module | None

    def __init__(
        self,
        backbone: TransformerDecoder,
        lm_head: nn.Module,
        self_conditioning: SelfConditioning,
        canvas: CanvasTokens,
    ):
        super().__init__()
        self.backbone, self.lm_head = backbone, lm_head
        self.self_conditioning = self_conditioning
        self.canvas = canvas

    @property
    def cache_config(self):
        return self.backbone.cache_config

    def forward(self, inputs: CanvasInput) -> torch.Tensor:
        return self.backbone(
            self._embeddings(inputs), inputs.positions, inputs.attention
        )

    def attend(self, inputs: CanvasInput) -> tuple[torch.Tensor, ...]:
        """Evaluate every canvas token up to the final layer's attention.

        Returns the backbone's per-token state of every canvas token on the
        last pipeline stage, for ``finish``; earlier stages return the
        activations they forward.
        """
        return self.backbone.attend(
            self._embeddings(inputs), inputs.positions, inputs.attention
        )

    def finish(
        self, state: tuple[torch.Tensor, ...], rows: torch.Tensor
    ) -> torch.Tensor:
        """Return the final-normalized ``[rows, hidden]`` outputs of ``rows``.

        ``state`` is ``attend``'s per-token state and ``rows`` the int64
        packed canvas token rows to complete, on the last pipeline stage.
        """
        return self.backbone.finish(state, rows)

    def _embeddings(self, inputs: CanvasInput) -> torch.Tensor | None:
        """Self-conditioned canvas embeddings, on the first stage."""
        if self.backbone._pipeline.rank != 0:
            return None
        return self.self_conditioning(
            self.backbone.embed_input_ids(inputs.input_ids),
            inputs.self_conditioning,
        )

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
