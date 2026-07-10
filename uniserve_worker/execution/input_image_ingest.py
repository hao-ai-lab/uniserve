"""System-owned ingest of external understanding images into a text cache.

The input-image sibling of
:meth:`~uniserve_worker.execution.interleaved_image_commit.GeneratedImageCommitDriver.append_generated_image`:
encode preprocessed image patches through the owner's understanding feature
extractor and append the resulting vision tokens into the owner's conditional
paged text cache at a single shared temporal RoPE index. The image's
begin/end marker tokens are ordinary prompt tokens owned by the frontend
prompt stream, so this driver appends *only* the patch block; the engine
resumes prefill with the end marker at the next temporal index.

Kept separate from the generated-image commit driver: commit also owns
marker emission, logits handoff, and scratch/cache release, none of which
apply to input ingestion.
"""
from __future__ import annotations

from typing import Any, Protocol

import torch

from ..foundation.errors import model_execution_error
from ..nn.vision import build_abs_positions_from_grid_hw
from .interleaved_text_stepper import TextCache

__all__ = [
    'InputImageIngestOwner',
    'InputImageIngestDriver',
]


class InputImageIngestOwner(Protocol):
    """Collaborator surface for ingesting understanding images."""

    device: Any

    def interleaved_text_forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        indexes: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        attention_mask: Any = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        text_only_rope: bool = False,
        causal_paged_update: bool = False,
    ) -> Any: ...
    def interleaved_image_features(
        self, image_input: torch.Tensor, *, grid_hw: torch.Tensor, gen_model: bool = ...
    ) -> torch.Tensor: ...
    def interleaved_image_downsample_ratio(self) -> float: ...


class InputImageIngestDriver:
    """Append an external image's vision tokens into a paged text cache."""

    def __init__(self, owner: InputImageIngestOwner) -> None:
        self.owner = owner

    def ingest_understanding_image(
        self,
        cache: TextCache,
        flattened_patches: torch.Tensor,
        grid_hw: torch.Tensor,
        *,
        t_index: int,
    ) -> int:
        """Encode + append one image's patch block at temporal index ``t_index``.

        Returns the number of vision tokens appended. The patch block is
        bidirectional within itself (all patches share ``t_index``) and attends
        to the whole existing prefix, matching the block-causal semantics the
        reference pipeline builds from its expanded placeholder stream.
        """
        vit_embeds = self.encode_understanding_image(flattened_patches, grid_hw)
        return self.ingest_understanding_embeddings(
            cache,
            vit_embeds,
            grid_hw,
            t_index=t_index,
        )

    def encode_understanding_image(
        self,
        flattened_patches: torch.Tensor,
        grid_hw: torch.Tensor,
    ) -> torch.Tensor:
        """Produce the reusable vision-encoder output for one image."""
        owner = self.owner
        return owner.interleaved_image_features(
            flattened_patches.to(owner.device),
            grid_hw=grid_hw.to(owner.device),
        )

    def ingest_understanding_embeddings(
        self,
        cache: TextCache,
        vit_embeds: torch.Tensor,
        grid_hw: torch.Tensor,
        *,
        t_index: int,
    ) -> int:
        """Append reusable vision embeddings into one request's paged text cache."""
        owner = self.owner
        if cache.past is None:
            raise model_execution_error(
                "input-image ingest requires an initialized paged text cache"
            )
        device = owner.device
        vit_embeds = vit_embeds.to(device).unsqueeze(0)
        num_tokens = int(vit_embeds.shape[1])

        merge = int(1 / owner.interleaved_image_downsample_ratio())
        abs_w, abs_h = build_abs_positions_from_grid_hw(
            grid_hw[:1].to(device) // merge, device=device
        )
        t_indexes = torch.full((num_tokens,), int(t_index), dtype=torch.long, device=device)
        indexes = torch.stack(
            [t_indexes, abs_h.to(torch.long), abs_w.to(torch.long)], dim=0
        )

        past_len = cache.past.get_seq_length()
        # Patches attend to the full prefix and to each other: an all-zeros
        # additive mask over [num_tokens, past + num_tokens].
        mask = torch.zeros(1, 1, num_tokens, past_len + num_tokens, device=device)

        outputs = owner.interleaved_text_forward(
            inputs_embeds=vit_embeds,
            indexes=indexes,
            attention_mask={"full_attention": mask},
            past_key_values=cache.past,
            use_cache=True,
        )
        cache.past = outputs.past_key_values
        cache.t_index = int(t_index)
        cache.last_logits = outputs.logits
        return num_tokens
