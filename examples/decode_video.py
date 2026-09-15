"""Decode complete H3 video latents through the public Python library."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uniserve.model import VideoDecoder, VideoPostprocessor
from uniserve.runtime import ExecutionContext, TensorBuffers
from uniserve_models.loading import load_model, read_config


@torch.inference_mode()
def decode_video(
    checkpoint: str, latents: torch.Tensor, *, frames: int, device: str = "cuda:0"
) -> torch.Tensor:
    """Return CPU uint8 [frames, 768, 1344, 3] pixels from complete H3 video rows.

    Input is the final FP32 video modality in tile-major canonical order, with
    shape [((frames - 5) // 17 * 5 + 2) * 24 * 42, 96]. Sequence shards must
    already be joined in logical order. Frames have the form 17 * n + 5, n >= 1.
    Only the video decoder and postprocessor checkpoint modules are loaded.
    """

    if frames < 22 or frames % 17 != 5:
        raise ValueError("H3 frames must be at least 22 and have the form 17 * n + 5")
    expected = (((frames - 5) // 17 * 5 + 2) * 24 * 42, 96)
    if latents.dtype != torch.float32 or tuple(latents.shape) != expected:
        raise ValueError(f"H3 video latents must be float32 with shape {expected}")
    config = read_config(checkpoint, modules=frozenset({"video_decoder", "video_postprocessor"}))
    model = load_model(config, device=device, precision="quality").model
    decoder, postprocessor = model.video_decoder, model.video_postprocessor
    if not isinstance(decoder, VideoDecoder) or not isinstance(postprocessor, VideoPostprocessor):
        raise TypeError("video reconstruction requires decoder and postprocessor capabilities")
    requirements = postprocessor.state_buffers(frames)
    with (
        ExecutionContext(decoder) as decoding,
        ExecutionContext(postprocessor) as pixels,
        TensorBuffers.allocate(requirements, device=device) as backing,
    ):
        decoding.prepare(frames)
        pixels.prepare(frames)
        state = backing.view(requirements)
        source = latents.to(device)
        outputs = []
        for interval in decoder.frame_slices(frames):
            with decoding.activate():
                decoded = decoder.decode(
                    (source,),
                    frames=(interval,),
                    num_frames=(frames,),
                    constants=decoding.constants,
                    workspace=decoding.workspace,
                )
            if decoded[0] is None:
                raise RuntimeError("local video reconstruction returned no tensor")
            with pixels.activate():
                output = postprocessor(
                    decoded,
                    frames=(interval,),
                    num_frames=(frames,),
                    state=state,
                    constants=pixels.constants,
                    workspace=pixels.workspace,
                )[0]
            # Each invocation borrows RGB workspace. Complete an independent
            # CPU copy before the next window can reuse that backing.
            outputs.append(output.tensor.to("cpu", copy=True))
        return torch.cat(outputs)


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
