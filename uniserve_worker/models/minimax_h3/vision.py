# SPDX-License-Identifier: Apache-2.0
"""H3 Qwen3-VL vision tower and DeepStack mergers.

Ported from FastVideo a943220c115228ade5d57b3bab9a6a87fd600a10.
The tower is replicated across language TP ranks; outputs have full language width.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3VLVisionConfig


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class H3VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels, self.hidden_size, kernel_size=kernel, stride=kernel, bias=True
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(pixels.to(self.proj.weight.dtype)).view(-1, self.hidden_size)


class H3VisionRotaryEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        # Keep a real CPU constant even under meta parameter construction. The
        # shared loader moves buffers after loading; CPU evaluation also matches
        # the reference frequencies without device-dependent power rounding.
        exponents = torch.arange(0, dimension, 2, dtype=torch.float32, device="cpu") / dimension
        inv_freq = 1.0 / (10000.0**exponents)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, sequence_length: int) -> torch.Tensor:
        positions = torch.arange(
            sequence_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(positions, self.inv_freq)


class H3VisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.view(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = F.gelu(self.linear_fc1(hidden_states))
        return self.linear_fc2(hidden_states)


class H3VisionMLP(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states), approximate="tanh"))


class H3VisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .view(sequence_length, 3, self.num_heads, self.head_dim)
            .unbind(1)
        )
        cos, sin = position_embeddings
        query_dtype, key_dtype = query.dtype, key.dtype
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        query = (query.float() * cos + _rotate_half(query.float()) * sin).to(query_dtype)
        key = (key.float() * cos + _rotate_half(key.float()) * sin).to(key_dtype)

        query_chunks = query.split(sequence_lengths, dim=0)
        key_chunks = key.split(sequence_lengths, dim=0)
        value_chunks = value.split(sequence_lengths, dim=0)
        outputs = []
        for query_chunk, key_chunk, value_chunk in zip(
            query_chunks, key_chunks, value_chunks, strict=True
        ):
            output = F.scaled_dot_product_attention(
                query_chunk.transpose(0, 1).unsqueeze(0),
                key_chunk.transpose(0, 1).unsqueeze(0),
                value_chunk.transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            outputs.append(output.transpose(1, 2))
        hidden_states = torch.cat(outputs, dim=1).reshape(sequence_length, -1)
        return self.proj(hidden_states)


class H3VisionBlock(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = H3VisionAttention(config)
        self.mlp = H3VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths: list[int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), sequence_lengths, position_embeddings
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class H3VisionModel(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = H3VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = H3VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(H3VisionBlock(config) for _ in range(config.depth))
        self.merger = H3VisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            H3VisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        )

    def _rotary_positions(self, grid_thw: torch.Tensor) -> torch.Tensor:
        max_height_width = int(grid_thw[:, 1:].max().item())
        frequency_table = self.rotary_pos_emb(max_height_width)
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        position_ids = torch.empty(
            (total_tokens, 2), dtype=torch.long, device=frequency_table.device
        )
        offset = 0
        merge = self.spatial_merge_size
        for frames_tensor, height_tensor, width_tensor in grid_thw:
            frames, height, width = int(frames_tensor), int(height_tensor), int(width_tensor)
            merged_height, merged_width = height // merge, width // merge
            block_rows = torch.arange(merged_height, device=frequency_table.device)
            block_cols = torch.arange(merged_width, device=frequency_table.device)
            intra_rows = torch.arange(merge, device=frequency_table.device)
            intra_cols = torch.arange(merge, device=frequency_table.device)
            rows = block_rows[:, None, None, None] * merge + intra_rows[None, None, :, None]
            cols = block_cols[None, :, None, None] * merge + intra_cols[None, None, None, :]
            rows = rows.expand(merged_height, merged_width, merge, merge).reshape(-1)
            cols = cols.expand(merged_height, merged_width, merge, merge).reshape(-1)
            coordinates = torch.stack((rows, cols), dim=-1).repeat(frames, 1)
            position_ids[offset : offset + coordinates.shape[0]] = coordinates
            offset += coordinates.shape[0]
        return frequency_table[position_ids].flatten(1)

    def _interpolate_position_embeddings(self, grid_thw: torch.Tensor) -> torch.Tensor:
        index_lists: list[list[int]] = [[] for _ in range(4)]
        weight_lists: list[list[float]] = [[] for _ in range(4)]
        merge = self.spatial_merge_size
        patch_counts: list[int] = []
        grids: list[tuple[int, int, int]] = []
        for frames_tensor, height_tensor, width_tensor in grid_thw:
            frames, height, width = int(frames_tensor), int(height_tensor), int(width_tensor)
            grids.append((frames, height, width))
            patch_counts.append(height * width)
            height_positions = torch.linspace(0, self.num_grid_per_side - 1, height)
            width_positions = torch.linspace(0, self.num_grid_per_side - 1, width)
            height_floor = height_positions.int()
            width_floor = width_positions.int()
            height_ceil = (height_floor + 1).clip(max=self.num_grid_per_side - 1)
            width_ceil = (width_floor + 1).clip(max=self.num_grid_per_side - 1)
            delta_height = height_positions - height_floor
            delta_width = width_positions - width_floor
            base_height = height_floor * self.num_grid_per_side
            base_height_ceil = height_ceil * self.num_grid_per_side
            indices = (
                (base_height[:, None] + width_floor[None]).flatten(),
                (base_height[:, None] + width_ceil[None]).flatten(),
                (base_height_ceil[:, None] + width_floor[None]).flatten(),
                (base_height_ceil[:, None] + width_ceil[None]).flatten(),
            )
            weights = (
                ((1 - delta_height)[:, None] * (1 - delta_width)[None]).flatten(),
                ((1 - delta_height)[:, None] * delta_width[None]).flatten(),
                (delta_height[:, None] * (1 - delta_width)[None]).flatten(),
                (delta_height[:, None] * delta_width[None]).flatten(),
            )
            for index in range(4):
                index_lists[index].extend(indices[index].tolist())
                weight_lists[index].extend(weights[index].tolist())

        index_tensor = torch.tensor(
            index_lists, dtype=torch.long, device=self.pos_embed.weight.device
        )
        weight_tensor = torch.tensor(
            weight_lists, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        embeddings = self.pos_embed(index_tensor) * weight_tensor[:, :, None]
        embeddings = (embeddings[0] + embeddings[1] + embeddings[2] + embeddings[3]).split(
            patch_counts
        )
        permuted = []
        for embedding, (frames, height, width) in zip(embeddings, grids, strict=True):
            embedding = embedding.repeat(frames, 1)
            embedding = embedding.view(frames, height // merge, merge, width // merge, merge, -1)
            permuted.append(embedding.permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        return torch.cat(permuted)

    def forward(
        self, pixels: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(pixels)
        hidden_states = hidden_states + self._interpolate_position_embeddings(grid_thw)
        rotary = self._rotary_positions(grid_thw).reshape(hidden_states.shape[0], -1)
        embedding = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (embedding.cos(), embedding.sin())
        sequence_lengths = [
            int(value)
            for value in torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        ]
        deepstack_features = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, sequence_lengths, position_embeddings)
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(self.deepstack_merger_list[merger_index](hidden_states))
        return self.merger(hidden_states), deepstack_features
