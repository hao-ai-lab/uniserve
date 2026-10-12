"""Independent dense equations for segment-wise sparse-attention domains."""

import torch
import torch.nn.functional as F

# A segment domain of 128-row tiles: two dense tiles, an empty one, a segment
# of three tiles, another dense tile, a segment of four tiles and an empty
# alignment tile, each segment keeping its own key tiles.
SEGMENT_TILE = 128
SEGMENT_VALID = (128, 40, 0, 128, 128, 96, 77, 128, 128, 128, 64, 0)
SEGMENT_INDICES = (-1, -1, -1, 0, 0, 0, -1, 1, 1, 1, 1, -1)
SEGMENT_KEEP = (2, 1)


def segment_tables(device):
    """Return the domain's ``(valid, segments, starts, keep)`` int32 tables."""
    return layout_tables(SEGMENT_VALID, SEGMENT_INDICES, SEGMENT_KEEP, device)


def layout_tables(valid, indices, keep, device):
    """Return ``(valid, segments, starts, keep)`` int32 tables of a layout.

    ``valid`` lists each tile's valid rows, ``indices`` its segment (-1 for a
    dense or empty tile) and ``keep`` the key tiles a video query keeps of
    each segment.
    """
    segments = torch.tensor(indices, dtype=torch.int32)
    starts = torch.zeros(len(valid), dtype=torch.int32)
    kept = torch.zeros(len(valid), dtype=torch.int32)
    for segment, count in enumerate(keep):
        starts[segment] = int((segments < segment).sum())
        kept[segment] = count
    tables = (torch.tensor(valid, dtype=torch.int32), segments, starts, kept)
    return tuple(value.to(device) for value in tables)


def segment_reference(
    q,
    k,
    v,
    gate,
    *,
    valid=SEGMENT_VALID,
    indices=SEGMENT_INDICES,
    keep=SEGMENT_KEEP,
    tile=SEGMENT_TILE,
    scale=None,
):
    """Evaluate segment selection, fine attention and compression in FP64.

    Dense tiles attend every live tile; video tiles attend the live dense
    tiles and each segment's top-scoring tiles; each row adds its gate times
    the softmax over live tiles of their mean values. ``scale`` is the fine
    attention's softmax scale, the inverse square root of the head width by
    default. Returns the result and the rows that attend (valid rows of live
    tiles).
    """
    tiles = len(valid)
    heads, width = q.shape[1], q.shape[2]
    valid = torch.tensor(valid, device=q.device)
    segments = torch.tensor(indices, device=q.device)
    live = torch.arange(tiles * tile, device=q.device) % tile < (
        valid.repeat_interleave(tile)
    )
    means = []
    for value in (q, k, v):
        rows = value.double().view(tiles, tile, heads, width)
        means.append(
            (
                rows.masked_fill(~live.view(tiles, tile, 1, 1), 0).sum(1)
                / valid.clamp_min(1).view(-1, 1, 1)
            ).transpose(0, 1)
        )
    scores = means[0] @ means[1].transpose(-1, -2) / width**0.5
    occupied = valid > 0
    dense, video = occupied & (segments < 0), occupied & (segments >= 0)
    mask = torch.zeros(heads, tiles, tiles, dtype=torch.bool, device=q.device)
    mask[:, dense] = occupied
    mask[:, video] = dense
    for segment, kept in enumerate(keep):
        members = torch.nonzero(segments == segment).flatten()
        best = scores[:, :, members].topk(kept, dim=-1).indices
        chosen = torch.zeros_like(mask)
        chosen.scatter_(-1, members[best], True)
        mask |= chosen & video.view(1, -1, 1)
    rows = mask.repeat_interleave(tile, 1).repeat_interleave(tile, 2)
    rows &= live.view(1, 1, -1)
    attending = live & occupied.repeat_interleave(tile)
    fine = torch.zeros(
        heads, tiles * tile, width, dtype=torch.float64, device=q.device
    )
    fine[:, attending] = F.scaled_dot_product_attention(
        q.transpose(0, 1).double()[:, attending],
        k.transpose(0, 1).double(),
        v.transpose(0, 1).double(),
        attn_mask=rows[:, attending],
        scale=scale,
    )
    compressed = (
        scores.masked_fill(~occupied.view(1, 1, -1), -torch.inf).softmax(-1)
        @ means[2]
    )
    result = fine.transpose(0, 1) + gate.double() * compressed.transpose(
        0, 1
    ).repeat_interleave(tile, dim=0)
    return result.to(q.dtype), attending
