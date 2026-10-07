"""Independent dense equations for a four-tile sparse-attention test domain."""

import torch
import torch.nn.functional as F


def reference_attention(q, k, v, gate, valid, *, scale=128**-0.5):
    """Evaluate tile selection, fine attention and compression densely.

    ``scale`` multiplies both the pooled tile scores and the fine attention
    logits, as the softmax scale of the sparse layer does.
    """
    live = torch.arange(256, device=q.device) % 64 < valid.repeat_interleave(64)
    means = []
    for tensor in (q, k, v):
        tiles = tensor.double().reshape(4, 64, q.shape[1], 128)
        mask = live.reshape(4, 64, 1, 1)
        means.append(
            (
                tiles.masked_fill(~mask, 0).sum(1)
                / valid.clamp_min(1).view(4, 1, 1)
            ).permute(1, 0, 2)
        )
    scores = means[0] @ means[1].transpose(-1, -2) * scale
    selection = scores[:, 1:3, 1:3].argmax(-1) + 1
    mask = torch.zeros(q.shape[1], 256, 256, device=q.device, dtype=torch.bool)
    mask[:, :64, :192] = True
    mask[:, 64:192, :64] = True
    for head in range(q.shape[1]):
        for tile in range(1, 3):
            selected = int(selection[head, tile - 1])
            mask[
                head,
                tile * 64 : (tile + 1) * 64,
                selected * 64 : (selected + 1) * 64,
            ] = True
    mask[:, 192:, :64] = True
    mask &= live.view(1, 1, -1)
    fine = F.scaled_dot_product_attention(
        q.transpose(0, 1).double(),
        k.transpose(0, 1).double(),
        v.transpose(0, 1).double(),
        attn_mask=mask,
        scale=scale,
    )
    compression = (
        scores.masked_fill(valid.view(1, 1, -1) == 0, -torch.inf).softmax(-1)
        @ means[2]
    )
    compression[:, valid == 0] = 0
    result = fine.transpose(0, 1) + gate.double() * compression.transpose(
        0, 1
    ).repeat_interleave(64, dim=0)
    return result.to(q.dtype), live


# A region domain of 128-row tiles: two dense tiles, an empty one, a region
# of three tiles, another dense tile, a region of four tiles and an empty
# alignment tile, each region keeping its own key tiles.
REGION_TILE = 128
REGION_VALID = (128, 40, 0, 128, 128, 96, 77, 128, 128, 128, 64, 0)
REGION_INDICES = (-1, -1, -1, 0, 0, 0, -1, 1, 1, 1, 1, -1)
REGION_KEEP = (2, 1)


def region_tables(device):
    """Return the domain's ``(valid, regions, starts, keep)`` int32 tables."""
    regions = torch.tensor(REGION_INDICES, dtype=torch.int32)
    starts = torch.zeros(len(REGION_VALID), dtype=torch.int32)
    keep = torch.zeros(len(REGION_VALID), dtype=torch.int32)
    for region, kept in enumerate(REGION_KEEP):
        starts[region] = int((regions < region).sum())
        keep[region] = kept
    valid = torch.tensor(REGION_VALID, dtype=torch.int32)
    return tuple(value.to(device) for value in (valid, regions, starts, keep))


def region_reference(q, k, v, gate):
    """Evaluate region selection, fine attention and compression in FP64.

    Dense tiles attend every live tile; video tiles attend the live dense
    tiles and each region's top-scoring tiles; each row adds its gate times
    the softmax over live tiles of their mean values. Returns the result and
    the rows that attend (valid rows of live tiles).
    """
    tiles, tile = len(REGION_VALID), REGION_TILE
    heads, width = q.shape[1], q.shape[2]
    valid = torch.tensor(REGION_VALID, device=q.device)
    regions = torch.tensor(REGION_INDICES, device=q.device)
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
    dense, video = occupied & (regions < 0), occupied & (regions >= 0)
    mask = torch.zeros(heads, tiles, tiles, dtype=torch.bool, device=q.device)
    mask[:, dense] = occupied
    mask[:, video] = dense
    for region, kept in enumerate(REGION_KEEP):
        members = torch.nonzero(regions == region).flatten()
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
    )
    compressed = (
        scores.masked_fill(~occupied.view(1, 1, -1), -torch.inf).softmax(-1)
        @ means[2]
    )
    result = fine.transpose(0, 1) + gate.double() * compressed.transpose(
        0, 1
    ).repeat_interleave(tile, dim=0)
    return result.to(q.dtype), attending
