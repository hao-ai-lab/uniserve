"""Decode complete H3 video latents through the public Python library."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from uniserve_worker.bootstrap.distributed import initialize_entries, initialize_process_groups
from uniserve_worker.config import WorkerConfig
from uniserve_worker.execution.model_runner import ModelRunner
from uniserve_worker.loader import LoadRequest, load_model
from uniserve_worker.modeling.batch import DecodeBatch
from uniserve_worker.modeling.components import Call
from uniserve_worker.modeling.decoder import DecoderMixin
from uniserve_worker.modeling.geometry import MediaShape
from uniserve_worker.modeling.video import VideoMixin
from uniserve_worker.nn.parallel import ComponentConfig
from uniserve_worker.runtime.process_groups import ProcessGroups
from uniserve_worker.runtime.tensor_buffers import TensorBuffers
from uniserve_worker.runtime.tensors import bind_scratch, bind_state, prepare_constants


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
    bindings = initialize_entries(
        groups,
        {
            "video_decoder": ComponentConfig((0,), distribution="temporal_units"),
            "output": ComponentConfig((0,)),
        },
    )
    device = groups.local_device
    loaded = load_model(
        LoadRequest(
            model_path=checkpoint,
            execution=WorkerConfig(device=str(device), model_dtype="bfloat16"),
            bindings=bindings,
            max_text_rows=64,
            max_video_seconds=frames / 24,
            pipeline_depth=8,
            quantization_config={"mode": "quality"},
        )
    )
    model = loaded.model
    if not isinstance(model, DecoderMixin) or not isinstance(model, VideoMixin):
        raise TypeError("video decoding requires decoder and video capabilities")
    shape = MediaShape(768, 1344, frames=frames)
    runner = ModelRunner(
        model, loaded.worker_config, bindings=loaded.bindings, schedule=loaded.schedule
    )
    try:
        runner.prepare_fixed_modules()
        backing = TensorBuffers.allocate(runner.tensor_resources.state, device)
        scratch = runner.scratch
        if scratch is None:
            raise RuntimeError("video decoding requires declared numerical scratch")
        decoder_constants = prepare_constants(model, Call.DECODE_VIDEO, shape, device=device)
        decoder_scratch = bind_scratch(model, Call.DECODE_VIDEO, shape, scratch)
        pixel_constants = prepare_constants(model, Call.POSTPROCESS_VIDEO, shape, device=device)
        pixel_state = bind_state(model, Call.POSTPROCESS_VIDEO, shape, backing)
        pixel_scratch = bind_scratch(model, Call.POSTPROCESS_VIDEO, shape, scratch)
        source = latents.to(device)
        pixels = []
        for window in model.decode_windows(model.output_geometry(frames)):
            decoded = runner.run_decoder(
                "video",
                DecodeBatch((source,), (shape,), (window,)),
                constants=decoder_constants,
                scratch=decoder_scratch,
            ).values[0]
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
        runner.close()


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
