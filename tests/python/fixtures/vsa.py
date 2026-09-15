"""Independent dense equations for a four-tile sparse-attention test domain."""

import torch
import torch.nn.functional as F


def reference_attention(q, k, v, gate, valid):
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
    scores = means[0] @ means[1].transpose(-1, -2) / 128**0.5
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
