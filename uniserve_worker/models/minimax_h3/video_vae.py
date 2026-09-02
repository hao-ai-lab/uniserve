"""Fixed-profile temporal video decode over the checkpoint MiniMax H3 VAE."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from ...execution.fixed_graph import StaticCudaGraph
from ...nn.quant.nvfp4 import replace_nvfp4_linears
from .precision import VideoVAELinearPrecision
from .video_vae_decoder import MiniMaxH3VideoDecoder

__all__ = ["MiniMaxH3VideoVAE"]


class MiniMaxH3VideoVAE(nn.Module):
    """Own one resident checkpoint VAE and decode temporal segments."""

    def __init__(
        self, vae: MiniMaxH3VideoDecoder, *, linear_precision: VideoVAELinearPrecision
    ) -> None:
        super().__init__()
        self.vae = vae.float()
        self.linear_precision = linear_precision
        self.autocast_dtype = torch.float16 if linear_precision == "fp16" else torch.bfloat16
        if linear_precision in {"fp16", "bf16"}:
            for module in self.vae.modules():
                if isinstance(module, (nn.Linear, nn.Conv3d)):
                    module.to(dtype=self.autocast_dtype)
        else:
            replaced = replace_nvfp4_linears(self.vae.decoder)
            if replaced != 217:
                raise RuntimeError(
                    f"MiniMax H3 video decoder expected 217 aligned linear layers, got {replaced}"
                )
        required = (
            "tokens_chunk_size",
            "token_overlap",
            "frame_pre_padding",
            "frame_overlap",
        )
        missing = [name for name in required if not hasattr(vae, name)]
        if missing:
            raise TypeError(f"MiniMax H3 video VAE is missing {missing!r}")
        mean = (
            0.858090341091156,
            -0.9606591463088989,
            1.0661640167236328,
            -0.5090325474739075,
            -0.2727581858634949,
            -1.3675414323806763,
            -0.2553254961967468,
            -0.26907554268836975,
            -0.5376840829849243,
            -0.0464097298681736,
            0.6657370328903198,
            0.19690127670764923,
            -0.5460608005523682,
            -0.4035342037677765,
            -0.23683024942874908,
            0.25928452610969543,
            -0.30133944749832153,
            0.211341992020607,
            -1.1206848621368408,
            0.3581933379173279,
            -0.04225143790245056,
            0.2604829967021942,
            0.22864092886447906,
            0.7056031823158264,
        )
        std = (
            1.2223774194717407,
            1.2767263650894165,
            1.6831774711608887,
            1.7549455165863037,
            1.5636216402053833,
            2.194143533706665,
            0.9653137922286987,
            1.0569885969161987,
            0.841948926448822,
            0.7729952931404114,
            1.8955937623977661,
            0.946841835975647,
            0.7996809482574463,
            0.44988900423049927,
            0.7197399735450745,
            0.6936293244361877,
            2.961095094680786,
            2.7694199085235596,
            3.0496184825897217,
            2.1088054180145264,
            3.276226282119751,
            3.1627357006073,
            2.2816812992095947,
            2.6127843856811523,
        )
        if len(mean) != 24 or len(std) != 24:
            raise ValueError("MiniMax H3 video VAE must declare 24-channel latent statistics")
        self.register_buffer(
            "latents_mean",
            torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 24, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 24, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor(
                (0.485, 0.456, 0.406),
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(
                (0.229, 0.224, 0.225),
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.decode_graph = StaticCudaGraph[torch.Tensor](self.device)
        self.register_buffer("decode_graph_input", None, persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: torch.device,
        local_files_only: bool = False,
        linear_precision: VideoVAELinearPrecision,
    ) -> "MiniMaxH3VideoVAE":
        del local_files_only
        vae = MiniMaxH3VideoDecoder(parameter_device="meta", buffer_device=device)
        component = Path(checkpoint) / "vae"
        indexes = tuple(component.glob("*.safetensors.index.json"))
        if len(indexes) != 1:
            raise RuntimeError("H3 video decoder requires one safetensor index")
        weight_map = json.loads(indexes[0].read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError("H3 video decoder checkpoint index has no weight map")
        targets = dict(vae.named_parameters())
        by_file: dict[Path, list[str]] = {}
        for name in targets:
            filename = weight_map.get(name)
            if not isinstance(filename, str):
                raise KeyError(f"H3 video decoder is missing checkpoint tensor {name!r}")
            by_file.setdefault(component / filename, []).append(name)
        from safetensors.torch import safe_open

        for path, names in sorted(by_file.items()):
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in names:
                    value = source.get_tensor(name).to(device=device, dtype=torch.float32)
                    owner: nn.Module = vae
                    fields = name.split(".")
                    for field in fields[:-1]:
                        owner = owner[int(field)] if field.isdigit() else getattr(owner, field)
                    setattr(owner, fields[-1], nn.Parameter(value, requires_grad=False))
        return cls(vae, linear_precision=linear_precision)

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    def _decode_segment(self, latents: torch.Tensor) -> torch.Tensor:
        span = int(self.vae.tokens_chunk_size) + int(self.vae.token_overlap)
        clip = self._decode_spatial_tiles(latents[:, :, :span])
        return clip[:, :, int(self.vae.frame_pre_padding) :].contiguous()

    def _decode_spatial_tiles(self, latents: torch.Tensor) -> torch.Tensor:
        if not bool(self.vae.use_tiling):
            return self.vae(self.vae.post_quant_conv(latents))

        ratio = int(self.vae.spatial_compression_ratio)
        height = int(latents.shape[-2]) * ratio
        width = int(latents.shape[-1]) * ratio
        y_indices, y_lengths, y_overlaps = self.vae._split_tiles(
            height,
            int(self.vae.tile_sample_min_height),
            int(self.vae.tile_sample_min_overlap_height),
        )
        x_indices, x_lengths, x_overlaps = self.vae._split_tiles(
            width,
            int(self.vae.tile_sample_min_width),
            int(self.vae.tile_sample_min_overlap_width),
        )
        tiles = torch.cat(
            tuple(
                latents[
                    ...,
                    y_pos // ratio : y_pos // ratio + y_length // ratio,
                    x_pos // ratio : x_pos // ratio + x_length // ratio,
                ]
                for y_pos, y_length in zip(y_indices, y_lengths, strict=True)
                for x_pos, x_length in zip(x_indices, x_lengths, strict=True)
            ),
            dim=0,
        )
        decoded = self.vae(self.vae.post_quant_conv(tiles))
        flat_tiles = decoded.split(1, dim=0)
        columns = len(x_indices)
        rows = [
            list(flat_tiles[start : start + columns])
            for start in range(0, len(flat_tiles), columns)
        ]
        return self.vae._stitch_tiles(rows, y_overlaps, x_overlaps)

    def _decode_normalized_segment(
        self,
        normalized_latents: torch.Tensor,
    ) -> torch.Tensor:
        latents = normalized_latents.to(device=self.device, dtype=torch.float32)
        latents = latents * self.latents_std + self.latents_mean
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.device.type == "cuda",
        ):
            return self._decode_segment(latents).to(torch.float16)

    @torch.inference_mode()
    def decode_segment(
        self,
        normalized_latents: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_latents.shape != (1, 24, 7, 48, 84):
            raise ValueError("an H3 video decode unit must have shape [1, 24, 7, 48, 84]")
        if self.decode_graph_input is None:
            raise RuntimeError("the H3 video decoder graph has not been captured")
        self.decode_graph_input.copy_(normalized_latents)
        return self.decode_graph.replay()

    @torch.inference_mode()
    def assemble_segment(
        self,
        segment: torch.Tensor,
        previous_overlap: torch.Tensor | None,
        *,
        final_unit: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Join one decoded segment and return RGB24 frames plus its successor overlap."""

        if segment.shape != (1, 3, 25, 768, 1344):
            raise ValueError("an H3 decoded video segment must have shape [1, 3, 25, 768, 1344]")
        body_frames = int(self.vae.tokens_chunk_size) * int(
            self.vae.temporal_compression_ratio
        ) - int(self.vae.frame_pre_padding)
        body = segment[:, :, :body_frames]
        if previous_overlap is not None:
            body = self.vae._blend(
                previous_overlap,
                body,
                int(self.vae.frame_overlap),
                dim=-3,
            )
        next_overlap = segment[:, :, body_frames + int(self.vae.frame_pre_padding) :].contiguous()
        if final_unit:
            body = torch.cat((body, next_overlap[:, :, :5]), dim=2)
        pixels = (body.float() * self.pixel_std + self.pixel_mean).clamp_(0.0, 1.0)
        rgb24 = pixels[0].permute(1, 2, 3, 0).mul_(255.0).round_().to(torch.uint8).contiguous()
        return rgb24, next_overlap[:, :, :5].contiguous()

    @torch.inference_mode()
    def capture_decoder(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        """Warm and capture the fixed-shape segment decoder."""

        if normalized_latents.shape != (1, 24, 7, 48, 84):
            raise ValueError("an H3 video decode unit must have shape [1, 24, 7, 48, 84]")
        self.decode_graph_input = torch.empty_like(normalized_latents, device=self.device)
        self.decode_graph_input.copy_(normalized_latents)
        return self.decode_graph.capture(
            lambda: self._decode_normalized_segment(self.decode_graph_input),
            warmup=lambda: self._decode_normalized_segment(self.decode_graph_input),
        )
