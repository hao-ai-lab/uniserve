"""FastH3 tile-64 attention and its startup-qualified SM100a boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch.nn import functional as F

from ...nn.mesh import DeviceMesh
from . import vsa_sm100

__all__ = ["H3VsaAttention", "H3VsaMetadata", "build_vsa_metadata"]

TILE = 64
SPARSITY = 0.9
H3AttentionMode = Literal["sparse_kernel", "sparse_oracle", "dense_oracle"]


@dataclass(frozen=True, slots=True)
class H3VsaMetadata:
    padded_rows: int
    prefix_tiles: int
    video_tiles: int
    valid_tiles: int
    valid_sizes: torch.Tensor


def build_vsa_metadata(
    *,
    padded_rows: int,
    prefix_tiles: int,
    video_tiles: int,
    valid_sizes: torch.Tensor,
    device: torch.device,
) -> H3VsaMetadata:
    if padded_rows % (TILE * 2):
        raise ValueError("H3 VSA transport requires an even tile-64 count")
    total_tiles = padded_rows // TILE
    if valid_sizes.shape != (total_tiles,):
        raise ValueError("H3 VSA tile-valid metadata does not match padded rows")
    if prefix_tiles + video_tiles > total_tiles:
        raise ValueError("H3 VSA segment tile counts exceed transport geometry")
    valid = valid_sizes.to(device=device, dtype=torch.int32)
    expected = torch.cat(
        (
            torch.ones(prefix_tiles + video_tiles, dtype=torch.bool),
            torch.zeros(total_tiles - prefix_tiles - video_tiles, dtype=torch.bool),
        )
    ).to(device)
    if not bool(torch.equal(valid > 0, expected)):
        raise ValueError("H3 VSA valid sizes do not describe prefix/video/partner tiles")
    return H3VsaMetadata(
        padded_rows=padded_rows,
        prefix_tiles=prefix_tiles,
        video_tiles=video_tiles,
        valid_tiles=prefix_tiles + video_tiles,
        valid_sizes=valid,
    )


def _resolve_kernel() -> Callable[..., torch.Tensor]:
    """Resolve the FastH3 SM100a operation once at startup."""

    if not vsa_sm100.available():
        raise RuntimeError(
            "FastH3 requires the SM100a H3 VSA kernel"
        ) from vsa_sm100.import_error()
    return vsa_sm100.block_sparse_attention


class H3VsaAttention:
    """Checkpoint VSA: sparse top-k attention plus trained dense compression."""

    def __init__(
        self,
        mesh: DeviceMesh,
        metadata: H3VsaMetadata,
        *,
        mode: H3AttentionMode,
    ) -> None:
        if mode not in {"sparse_kernel", "sparse_oracle", "dense_oracle"}:
            raise ValueError(f"unknown H3 attention mode {mode!r}")
        self.mesh = mesh
        self.metadata = metadata
        self.mode = mode
        self.kernel = _resolve_kernel() if mode == "sparse_kernel" else None
        self.local_heads = 56 // mesh.size("sp")
        self.keep_video_tiles = max(
            1, math.ceil((1.0 - SPARSITY) * metadata.video_tiles)
        )
        self.topk_position_offsets = torch.arange(
            self.keep_video_tiles,
            dtype=torch.long,
            device=metadata.valid_sizes.device,
        ).view(1, 1, -1)
        self.topk_positions = torch.empty_like(self.topk_position_offsets)

    def _block_means(
        self,
        value: torch.Tensor,
        valid_sizes: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tiles, heads = value.shape[0] // TILE, value.shape[1]
        blocked = value.reshape(tiles, TILE, heads, -1)
        if output is None:
            output = blocked.sum(dim=1, dtype=torch.float32)
        else:
            torch.sum(
                blocked,
                dim=1,
                dtype=torch.float32,
                out=output,
            )
        output.div_(valid_sizes.clamp_min(1).view(tiles, 1, 1))
        return output.permute(1, 0, 2)

    def _compression(
        self,
        q_mean: torch.Tensor,
        k_mean: torch.Tensor,
        v_mean: torch.Tensor,
        gate: torch.Tensor,
        valid_sizes: torch.Tensor,
        scores: torch.Tensor | None = None,
        compressed: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if scores is None:
            scores = torch.matmul(q_mean, k_mean.transpose(-1, -2))
            scores.mul_(q_mean.shape[-1] ** -0.5)
        scores.masked_fill_(valid_sizes.view(1, 1, -1) == 0, -torch.inf)
        scores.sub_(scores.amax(dim=-1, keepdim=True)).exp_()
        scores.div_(scores.sum(dim=-1, keepdim=True))
        if compressed is None:
            compressed = torch.matmul(scores, v_mean)
        else:
            torch.matmul(scores, v_mean, out=compressed)
        compressed.masked_fill_(valid_sizes.view(1, -1, 1) == 0, 0)
        repeated = (
            compressed.permute(1, 0, 2)
            .unsqueeze(1)
            .expand(-1, TILE, -1, -1)
            .reshape_as(gate)
        )
        if output is None:
            return repeated * gate
        output.addcmul_(gate, repeated)
        return output

    def block_map_from_scores(
        self,
        scores: torch.Tensor,
        prefix_key_indices: torch.Tensor,
        dense_key_indices: torch.Tensor,
        prefix_count: torch.Tensor,
        counts: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        topk_values: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_indices_i32: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the exempt-prefix map from checkpoint compression scores."""

        heads, tiles, _ = scores.shape
        prefix = self.metadata.prefix_tiles
        video_end = prefix + self.metadata.video_tiles
        width = self.metadata.valid_tiles
        if indices is None:
            indices = torch.empty(
                (heads, tiles, width), dtype=torch.int32, device=scores.device
            )
        if counts is None:
            counts = torch.empty(
                (heads, tiles), dtype=torch.int32, device=scores.device
            )
        indices.zero_()
        counts.zero_()
        if prefix:
            indices[:, :prefix, : self.metadata.valid_tiles] = dense_key_indices
            counts[:, :prefix] = prefix_count + self.metadata.video_tiles
        if self.metadata.video_tiles:
            video_scores = scores[:, prefix:video_end, prefix:video_end]
            if topk_values is None or topk_indices is None:
                selected = torch.topk(
                    video_scores,
                    self.keep_video_tiles,
                    dim=-1,
                    sorted=True,
                ).indices.to(torch.int32)
            else:
                torch.topk(
                    video_scores,
                    self.keep_video_tiles,
                    dim=-1,
                    sorted=True,
                    out=(topk_values, topk_indices),
                )
                if topk_indices_i32 is None:
                    selected = topk_indices.to(torch.int32)
                else:
                    topk_indices_i32.copy_(topk_indices)
                    selected = topk_indices_i32
            selected.add_(prefix)
            if prefix:
                indices[:, prefix:video_end, :prefix] = prefix_key_indices
            torch.add(
                self.topk_position_offsets,
                prefix_count,
                out=self.topk_positions,
            )
            indices[:, prefix:video_end].scatter_(
                2,
                self.topk_positions.expand(heads, self.metadata.video_tiles, -1),
                selected,
            )
            counts[:, prefix:video_end] = prefix_count + self.keep_video_tiles
        # The final partner tile is transport-only. Give its ignored queries a
        # legal one-block map so the kernel need not special-case zero counts.
        if video_end < tiles:
            counts[:, video_end:] = 1
        return counts, indices

    def _sparse_oracle(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        counts: torch.Tensor,
        indices: torch.Tensor,
        valid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(query)
        for tile in range(self.metadata.valid_tiles):
            query_size = int(valid_sizes[tile])
            if query_size == 0:
                continue
            query_rows = query[tile * TILE : tile * TILE + query_size]
            per_head: list[torch.Tensor] = []
            for head in range(query.shape[1]):
                selected = indices[head, tile, : int(counts[head, tile])].long()
                key_parts = [
                    key[
                        int(index) * TILE : int(index) * TILE
                        + int(valid_sizes[int(index)])
                    , head]
                    for index in selected
                ]
                value_parts = [
                    value[
                        int(index) * TILE : int(index) * TILE
                        + int(valid_sizes[int(index)])
                    , head]
                    for index in selected
                ]
                attended = F.scaled_dot_product_attention(
                    query_rows[:, head].unsqueeze(0).unsqueeze(0),
                    torch.cat(key_parts).unsqueeze(0).unsqueeze(0),
                    torch.cat(value_parts).unsqueeze(0).unsqueeze(0),
                    dropout_p=0.0,
                    is_causal=False,
                )
                per_head.append(attended[0, 0])
            output[tile * TILE : tile * TILE + query_size] = torch.stack(
                per_head, dim=1
            )
        return output

    @staticmethod
    def _dense_oracle(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_sizes: torch.Tensor,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        row_offsets = torch.arange(TILE, device=query.device).view(1, TILE)
        valid_rows = (row_offsets < valid_sizes.view(-1, 1)).reshape(-1)
        query_valid = query[valid_rows].transpose(0, 1).unsqueeze(0)
        key_valid = key[valid_rows].transpose(0, 1).unsqueeze(0)
        value_valid = value[valid_rows].transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            query_valid,
            key_valid,
            value_valid,
            dropout_p=0.0,
            is_causal=False,
        )[0].transpose(0, 1)
        if output is None:
            output = torch.zeros_like(query)
        else:
            output.zero_()
        output[valid_rows] = attended
        return output

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        valid_sizes: torch.Tensor | None = None,
        prefix_key_indices: torch.Tensor | None = None,
        dense_key_indices: torch.Tensor | None = None,
        prefix_count: torch.Tensor | None = None,
        *,
        output: torch.Tensor | None = None,
        tile_scores: torch.Tensor | None = None,
        block_counts: torch.Tensor | None = None,
        block_indices: torch.Tensor | None = None,
        pooled_query: torch.Tensor | None = None,
        pooled_key: torch.Tensor | None = None,
        pooled_value: torch.Tensor | None = None,
        compressed_tiles: torch.Tensor | None = None,
        topk_values: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_indices_i32: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_sizes = self.metadata.valid_sizes if valid_sizes is None else valid_sizes
        if self.mode == "dense_oracle":
            return self._dense_oracle(query, key, value, valid_sizes, output)
        if prefix_key_indices is None or dense_key_indices is None or prefix_count is None:
            prefix_key_indices = torch.arange(
                self.metadata.prefix_tiles, dtype=torch.int32, device=query.device
            )
            dense_key_indices = torch.arange(
                self.metadata.valid_tiles, dtype=torch.int32, device=query.device
            )
            prefix_count = torch.tensor(
                self.metadata.prefix_tiles, dtype=torch.int32, device=query.device
            )
        q_mean = self._block_means(query, valid_sizes, pooled_query)
        k_mean = self._block_means(key, valid_sizes, pooled_key)
        v_mean = self._block_means(value, valid_sizes, pooled_value)
        if tile_scores is None:
            selection_scores = torch.matmul(q_mean, k_mean.transpose(-1, -2))
        else:
            torch.matmul(
                q_mean,
                k_mean.transpose(-1, -2),
                out=tile_scores,
            )
            selection_scores = tile_scores
        selection_scores.mul_(query.shape[-1] ** -0.5)
        counts, indices = self.block_map_from_scores(
            selection_scores,
            prefix_key_indices,
            dense_key_indices,
            prefix_count,
            counts=block_counts,
            indices=block_indices,
            topk_values=topk_values,
            topk_indices=topk_indices,
            topk_indices_i32=topk_indices_i32,
        )
        if self.mode == "sparse_oracle":
            sparse = self._sparse_oracle(
                query, key, value, counts, indices, valid_sizes
            )
        else:
            if self.kernel is None:
                raise RuntimeError("the H3 sparse-kernel route lost its resolved operation")
            sparse = self.kernel(
                query,
                key,
                value,
                mask_block_count=counts,
                mask_block_indices=indices,
                valid_sizes=valid_sizes,
                tile_size=TILE,
            )
        if output is None:
            output = sparse
        else:
            output.copy_(sparse)
        return self._compression(
            q_mean,
            k_mean,
            v_mean,
            gate,
            valid_sizes,
            selection_scores,
            compressed_tiles,
            output,
        )
