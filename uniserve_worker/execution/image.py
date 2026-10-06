"""Numerical image encoder rows and visual/latent KV input construction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from uniserve.media import image
from uniserve.processing import (
    FeatureInjection,
    FeatureLayout,
    PatchTransform,
    PositionLayout,
)
from uniserve_worker.errors import invalid_descriptor
from uniserve_worker.model_executor.diffusion_inputs import DiffusionRow
from uniserve_worker.model_executor.image_inputs import (
    PreparedImage,
    VisionRow,
    patch_grid_shape,
)
from uniserve_worker.model_executor.input_batch import TokenRow
from uniserve_worker.protocol.call import (
    ForwardMode,
    MediaCall,
)
from uniserve_worker.sampling.metadata import TokenSelection

if TYPE_CHECKING:
    from uniserve_worker.execution.model_executor import ModelExecutor


def encode_row(
    mode: MediaCall,
    prepared: PreparedImage,
) -> VisionRow:
    """Build a vision- or latent-encoder row from prepared image tensors."""
    return VisionRow(
        forward_mode=mode,
        encode_pixels=prepared.pixels,
        encode_grid=prepared.grid,
        encode_grid_shape=prepared.grid_shape,
    )


def vision_state_row(
    features: torch.Tensor,
    height: int,
    width: int,
    conditioning_position: int,
    *,
    seq_len: int,
    slot: int,
    input_images: int | None,
    close_image: bool,
    logits: bool,
    model_runner: ModelExecutor,
) -> TokenRow:
    """Build the prefill row that writes one image's vision features into KV.

    ``features`` are ``[tokens, hidden]`` (or with a leading singleton batch
    axis). The row is non-causal: its features attend to each other in both
    directions and to the ``seq_len`` KV tokens before it in the request's
    slot, which are the request's visible KV and, in a context prefill, the
    rows of its call before this one. It writes its KV after them and
    selects last logits when ``logits`` is set (only the K/V cache
    otherwise). A framed layout adds start and end marker tokens;
    ``close_image`` adds the end marker in any layout.

    Raises:
        WorkerError: When the model declares no feature injection, the
            features are not ``[tokens, hidden]``, or the positional layout
            cannot represent their patch grid.
    """
    injection = model_runner.image_processor().feature_injection
    if injection is None:
        raise invalid_descriptor(
            "vision state preparation requires declared feature injection"
        )

    # Accept a leading singleton batch axis; the row layout is [tokens, hidden].
    embeddings = (
        features.squeeze(0)
        if features.ndim == 3 and int(features.shape[0]) == 1
        else features
    )
    if embeddings.ndim != 2 or int(embeddings.shape[0]) < 1:
        raise invalid_descriptor(
            "vision features must have shape [tokens, hidden]"
        )

    # The sequence is [start marker?, feature tokens..., end marker?]; marker
    # slots keep placeholder token ids while feature slots carry embeddings.
    leading = injection.layout is FeatureLayout.FRAMED
    trailing = leading or close_image
    query = int(leading) + int(embeddings.shape[0]) + int(trailing)
    token_ids = torch.ones(query, dtype=torch.long)
    token_embeddings = embeddings.new_zeros((query, int(embeddings.shape[1])))
    embedding_mask = torch.zeros(
        query, dtype=torch.bool, device=embeddings.device
    )
    begin = int(leading)
    token_embeddings[begin : begin + int(embeddings.shape[0])] = embeddings
    embedding_mask[begin : begin + int(embeddings.shape[0])] = True
    if leading:
        token_ids[0] = _feature_token_id(injection, start=True)
    if trailing:
        token_ids[-1] = _feature_token_id(injection, start=False)

    # A row that closes the image carries generated-image feedback; any
    # other vision row carries one of the request's input images, whose
    # patch grid follows the pixel budget those images share.
    positions = _vision_positions(
        injection.positions,
        int(embeddings.shape[0]),
        conditioning_position,
        height=height,
        width=width,
        leading=leading,
        trailing=trailing,
        close_image=close_image,
        input_images=input_images,
        model_runner=model_runner,
    )
    return TokenRow(
        forward_mode=ForwardMode.PREFILL,
        token_ids=token_ids,
        token_embeddings=token_embeddings,
        token_embedding_mask=embedding_mask,
        positions=positions,
        # A row that samples no token only writes the K/V cache; its
        # outcome reads no output.
        selection=TokenSelection.LAST_LOGITS
        if logits
        else TokenSelection.CACHE,
        request_pool_idx=slot,
        seq_len=seq_len,
        write_kv=True,
        causal=False,
    )


def _feature_token_id(injection: FeatureInjection, *, start: bool) -> int:
    """Read a marker token id already resolved on the feature injection."""
    value = injection.start_token_id if start else injection.end_token_id
    if value is None:
        raise invalid_descriptor(
            "feature injection requires a resolved marker token id"
        )
    return value


def _vision_positions(
    layout: PositionLayout,
    feature_tokens: int,
    conditioning_position: int,
    *,
    height: int,
    width: int,
    leading: bool,
    trailing: bool,
    close_image: bool,
    input_images: int | None,
    model_runner: ModelExecutor,
) -> torch.Tensor:
    """Build position ids for a vision row's marker and feature slots.

    Returns ``[query]`` positions for ``PositionLayout.TEMPORAL``, all at
    ``conditioning_position``, and for ``PositionLayout.SEQUENTIAL``,
    consecutive from ``conditioning_position``. Otherwise returns
    ``[3, query]`` rows of
    temporal, height and width coordinates: feature slots take their raster
    grid coordinates and marker slots zero spatial coordinates. The raster
    grid is the patch grid of a ``height`` x ``width`` canvas under the
    pixel bound for ``input_images`` (see ``patch_grid_shape``).
    """
    query = int(leading) + feature_tokens + int(trailing)
    if layout is PositionLayout.TEMPORAL:
        return torch.full(
            (query,), int(conditioning_position), dtype=torch.long
        )
    if layout is PositionLayout.SEQUENTIAL:
        start = int(conditioning_position)
        return torch.arange(start, start + query, dtype=torch.long)

    transform = model_runner.image_processor().vit
    if not isinstance(transform, PatchTransform):
        raise invalid_descriptor(
            "temporal-spatial feature injection requires a patch image "
            "transform"
        )

    # The encoder may pool patches, so the feature count can be a square
    # downscale of the raw patch grid; recover the per-axis grid factor.
    raw_height, raw_width = patch_grid_shape(
        transform, height, width, input_images
    )
    factor_squared, remainder = divmod(raw_height * raw_width, feature_tokens)
    factor = math.isqrt(factor_squared)
    if remainder or factor < 1 or factor * factor != factor_squared:
        raise invalid_descriptor(
            "vision feature count does not align with its patch grid"
        )
    grid_height, grid_width = raw_height // factor, raw_width // factor
    if grid_height * grid_width != feature_tokens:
        raise invalid_descriptor("vision output grid is not integral")

    temporal = torch.full(
        (query,),
        int(conditioning_position + (1 if close_image else 0)),
        dtype=torch.long,
    )
    # Raster-order [feature_tokens] grid coordinates for the feature slots.
    y = torch.arange(grid_height, dtype=torch.long).repeat_interleave(
        grid_width
    )
    x = torch.arange(grid_width, dtype=torch.long).repeat(grid_height)
    spatial_y = torch.zeros(query, dtype=torch.long)
    spatial_x = torch.zeros(query, dtype=torch.long)
    begin = int(leading)
    spatial_y[begin : begin + feature_tokens] = y
    spatial_x[begin : begin + feature_tokens] = x
    if trailing and close_image:
        temporal[-1] = conditioning_position + 2

    return torch.stack((temporal, spatial_y, spatial_x))


def latent_state_row(
    latent: torch.Tensor,
    height: int,
    width: int,
    conditioning_position: int,
    *,
    slot: int,
    seq_len: int,
    model_runner: ModelExecutor,
) -> DiffusionRow:
    """Build the denoiser row that writes an image latent into KV.

    The row runs the latent at timestep zero between the builder's two
    framing tokens, non-causally, at the request's cache coordinates. It
    requires an image builder with two framing tokens and a latent whose
    token count matches the declared image size.
    """
    builder = model_runner.image_builder
    if builder is None or builder.framing != 2:
        raise invalid_descriptor(
            "latent feature export requires framed image conditioning"
        )

    size = image.Config(height, width)
    image_tokens = builder.denoiser.latent_shape("image", size)[0]
    if latent.reshape(-1, latent.shape[-1]).shape[0] != image_tokens:
        raise invalid_descriptor(
            "state latent does not match the declared image dimensions"
        )

    query = builder.sequence_length(size)
    positions = builder.positions(
        size, conditioning_position + 1, device=latent.device
    )
    # Frame markers bound the image span: the first sits at the conditioning
    # position, the last advances past the rope range of the image tokens.
    positions[0, 0] = conditioning_position
    positions[0, -1] = conditioning_position + builder.rope_advance

    return DiffusionRow(
        forward_mode=MediaCall.DENOISING,
        positions=positions,
        timestep=latent.new_zeros(1),
        latent=latent,
        image_tokens=query,
        image_height=height,
        image_width=width,
        request_pool_idx=slot,
        seq_len=seq_len,
        write_kv=True,
        causal=False,
    )


__all__ = ["encode_row", "vision_state_row", "latent_state_row"]
