"""Decode complete H3 video latents through the public Python library."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uniserve.attention import FlashInferTuningConfig, resolve_attention_selection
from uniserve.distributed.parallel import ParallelConfig
from uniserve.distributed.process_groups import (
    ProcessGroups,
    initialize_model_parallel,
    initialize_process_groups,
)
from uniserve.loading import load_model
from uniserve.model.limits import ModelLimits
from uniserve.model.media import VideoSize
from uniserve.nn.attention import bind_dense_attention_modules
from uniserve.nn.layer import LayerConfig
from uniserve.runtime.tensor_buffers import TensorBuffers
from uniserve.runtime.tensors import bind_scratch, bind_state, merge_buffers, prepare_constants
from uniserve_models import resolve_model
from uniserve_models.minimax_h3 import MiniMaxH3Model


@torch.inference_mode()
def decode_video(
    checkpoint: str, latents: torch.Tensor, *, frames: int, device: str = "cuda:0"
) -> torch.Tensor:
    """Return CPU uint8 [frames, 768, 1344, 3] RGB pixels from H3 video rows.

    Input is the complete, final FP32 video modality in H3 packed row order,
    shape [((frames - 5) // 17 * 5 + 2) * 24 * 42, 96]. It can reside on CPU
    or CUDA. All sequence shards must be joined in logical order beforehand.
    The checkpoint must be the supported full FastH3 VSA checkpoint. Legal
    frame counts are at least 22 and have the form 17 * n + 5.
    """

    if frames < 22 or frames % 17 != 5:
        raise ValueError("H3 frames must be at least 22 and have the form 17 * n + 5")
    expected = (((frames - 5) // 17 * 5 + 2) * 24 * 42, 96)
    if latents.dtype != torch.float32 or tuple(latents.shape) != expected:
        raise ValueError(f"H3 video latents must be float32 with shape {expected}")
    with initialize_process_groups(rank=0, local_rank=0, world_size=1, device=device) as groups:
        # All numerical views and native graph outputs leave scope before the
        # process-group owner closes. Only the independent CPU pixels escape.
        return _decode(groups, checkpoint, latents, frames)


def _decode(
    groups: ProcessGroups, checkpoint: str, latents: torch.Tensor, frames: int
) -> torch.Tensor:
    device = groups.local_device
    parallel = {"video_decoder": ParallelConfig(), "video_output": ParallelConfig()}
    meshes = initialize_model_parallel(
        groups, {name: ((0,), value) for name, value in parallel.items()}
    )
    source = resolve_model(
        checkpoint, components=frozenset(parallel), quantization={"mode": "quality"}
    )
    layers = source.configure_layers(
        {name: LayerConfig(mesh.get_group("tp"), None) for name, mesh in meshes.items()}
    )
    loaded = load_model(
        source.model_class,
        source.config,
        sources=source.weights,
        device=device,
        dtype=torch.bfloat16,
        parallel=parallel,
        meshes=meshes,
        layers=layers,
        limits=ModelLimits(text_tokens=64, video_frames=frames),
    )
    model = loaded.model
    if not isinstance(model, MiniMaxH3Model):
        raise TypeError("this example decodes the FastH3 video layout")
    bind_dense_attention_modules(
        model, resolve_attention_selection("auto", tuning=FlashInferTuningConfig(), block_size=256)
    )
    shape = VideoSize(frames)
    backing = TensorBuffers.allocate(model.video_output.state_buffers(shape), device)
    scratch = TensorBuffers.allocate(
        merge_buffers(
            (
                model.video_decoder.workspace_buffers(shape),
                model.video_output.workspace_buffers(shape),
            )
        ),
        device,
    )
    try:
        decoder_constants = prepare_constants(
            model.video_decoder, VideoSize(shape.frames), device=device
        )
        decoder_scratch = bind_scratch(
            model.video_decoder.workspace_buffers(VideoSize(shape.frames)), scratch
        )
        pixel_constants = prepare_constants(
            model.video_output, VideoSize(shape.frames), device=device
        )
        pixel_state = bind_state(model.video_output.state_buffers(VideoSize(shape.frames)), backing)
        pixel_scratch = bind_scratch(
            model.video_output.workspace_buffers(VideoSize(shape.frames)), scratch
        )
        source = latents.to(device)
        pixels = []
        for window in model.decode_windows(model.video_info(frames)):
            decoded = model.video_decoder.decode(
                (source,),
                shape,
                (window,),
                constants=decoder_constants,
                scratch=decoder_scratch,
            ).values["video"][0]
            if decoded is None:
                raise RuntimeError("local video decoder did not produce a tensor")
            # The numerical decoder already applies its native temporal crop.
            # Remove only the leading result-unit dimension for postprocessing.
            segment = decoded[0]
            output = model.postprocess_video(
                (segment,),
                (window,),
                state=pixel_state,
                constants=pixel_constants,
                scratch=pixel_scratch,
            ).values["video"][0]
            if output is None:
                raise RuntimeError("local video postprocessing did not produce pixels")
            # The next call reuses RGB scratch. A blocking CPU copy completes
            # its reader and preserves this window independently of that reuse.
            pixels.append(output.cpu())
        return torch.cat(pixels)
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    latents = torch.load(args.latents, map_location="cpu", weights_only=True)
    if not isinstance(latents, torch.Tensor):
        raise TypeError("the latent file must contain one complete video tensor")
    pixels = decode_video(args.checkpoint, latents, frames=args.frames, device=args.device)
    torch.save(pixels, args.output)
    print(f"Saved {tuple(pixels.shape)} {pixels.dtype} RGB pixels to {args.output}")


if __name__ == "__main__":
    main()
