"""Qwen3-VL multimodal rotary (M-RoPE) positions of a presentation.

The language model rotates every token by three coordinates, (temporal,
height, width). Text advances all three together; a vision block places its
merged tokens on their grid around a shared origin. ``rope_index`` derives
the coordinates from the token ids and the vision grids, as
``Qwen3VLModel.get_rope_index`` in Transformers does, for one sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby

import torch

# Token type codes of one presentation: text, image and video placeholders.
_TEXT, _IMAGE, _VIDEO = 0, 1, 2


def rope_index(
    input_ids: Sequence[int] | torch.Tensor,
    *,
    image_grids: Sequence[tuple[int, int, int]] = (),
    video_grids: Sequence[tuple[int, int, int]] = (),
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Return the ``[3, tokens]`` int64 M-RoPE positions of one sequence.

    Every maximal run of ``image_token_id`` placeholders is one image block
    and consumes the next entry of ``image_grids``; video placeholders
    consume ``video_grids`` likewise. A video grid ``(time, height, width)``
    covers ``time`` blocks of one time step each, one per placeholder run:
    Qwen3-VL presents every temporal patch of a video as its own block
    behind a text timestamp, so the timestamp text, not the temporal
    coordinate, orders them.

    Text tokens take consecutive coordinates on all three axes, continuing
    from the running origin. A block of grid ``(t, h, w)`` patches has
    ``t * (h / m) * (w / m)`` merged tokens, ``m = spatial_merge_size``;
    the token at merged coordinates ``(i, j, k)`` takes ``origin + (i, j,
    k)`` in raster order. The origin then advances by ``max(h, w) / m``.

    Args:
        input_ids: The presentation's token ids, a sequence or a 1-D tensor.
        image_grids: Patch grids of the image blocks, in presentation order.
        video_grids: Patch grids of the videos, in presentation order.
        image_token_id: The image placeholder token.
        video_token_id: The video placeholder token.
        spatial_merge_size: Patches merged per token on each spatial axis.

    Returns:
        A CPU int64 tensor ``[3, len(input_ids)]``.

    Raises:
        ValueError: A placeholder run does not hold its grid's merged token
            count, a grid is missing for a run, or a grid is left unused.
    """
    ids = torch.as_tensor(input_ids, dtype=torch.int64).cpu()
    if ids.ndim != 1:
        raise ValueError("M-RoPE positions describe one token sequence")
    merge = spatial_merge_size
    types = torch.full_like(ids, _TEXT)
    types[ids == image_token_id] = _IMAGE
    types[ids == video_token_id] = _VIDEO

    # A video contributes one single-step grid per temporal patch.
    grids = {
        _IMAGE: iter(tuple(image_grids)),
        _VIDEO: iter(
            tuple(
                (1, height, width)
                for time, height, width in video_grids
                for _ in range(time)
            )
        ),
    }

    parts = []
    origin = 0
    for kind, run in groupby(types.tolist()):
        length = sum(1 for _ in run)
        if kind == _TEXT:
            parts.append(
                torch.arange(origin, origin + length).expand(3, length)
            )
            origin += length
            continue

        grid = next(grids[kind], None)
        if grid is None:
            raise ValueError("a vision placeholder run has no grid")
        time, height, width = grid
        if height % merge or width % merge:
            raise ValueError("vision grids must hold whole merge blocks")
        rows, columns = height // merge, width // merge
        if length != time * rows * columns:
            raise ValueError(
                "a vision placeholder run must hold its grid's merged tokens"
            )

        # [3, time, rows, columns] coordinates around the shared origin.
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(time),
                torch.arange(rows),
                torch.arange(columns),
                indexing="ij",
            )
        )
        parts.append(coordinates.reshape(3, -1) + origin)
        origin += max(height, width) // merge

    if any(next(remaining, None) is not None for remaining in grids.values()):
        raise ValueError("every vision grid requires a placeholder run")
    if not parts:
        return torch.zeros((3, 0), dtype=torch.int64)
    return torch.cat(parts, dim=1)
