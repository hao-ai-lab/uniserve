"""Captured video reconstruction consumes the supplied overlap.

The overlap is consumed on every replay.
"""

import pytest
import torch

from uniserve.media import image
from uniserve.model.video import VideoPostprocessor
from uniserve.runtime import CUDAGraph, ExecutionContext
from uniserve.tensors import OutputLayout, TensorOutput

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


class Reconstruction(VideoPostprocessor):
    def reconstruction_slices(self, frames, num_frames):
        return slice(0, 3), slice(4, 6)


def test_video_postprocess_replay_uses_current_overlap_and_pixels():
    device = torch.device("cuda", 0)
    model = Reconstruction(
        torch.tensor([0, 0.5], dtype=torch.float16, device=device),
        frame_size=image.Config(2, 3),
        frame_rate=24,
    )
    frames = torch.tensor(
        [1, 1, 0.5, 0, 0.25, 1], dtype=torch.float16, device=device
    )
    segment = frames.view(1, 1, 6, 1, 1).expand(1, 3, 6, 2, 3).contiguous()
    overlap = torch.ones((1, 3, 2, 2, 3), dtype=torch.float16, device=device)
    saved_overlap = overlap.clone()
    pixels = torch.empty((5, 2, 3, 3), dtype=torch.uint8, device=device)
    constants = {
        "pixel_mean": torch.zeros((1, 3, 1, 1, 1), device=device),
        "pixel_std": torch.ones((1, 3, 1, 1, 1), device=device),
    }

    def forward():
        result = model(
            (
                TensorOutput(
                    segment,
                    OutputLayout(
                        segment.shape,
                        segment.dtype,
                        tuple(slice(0, n) for n in segment.shape),
                    ),
                ),
            ),
            frames=(slice(3, 8),),
            num_frames=(8,),
            state={"video_overlap": overlap},
            constants=constants,
            workspace={"rgb_frames": pixels},
        )
        return result

    context = ExecutionContext(model)
    forward()  # Warm the reconstruction kernels before capture.
    overlap.copy_(saved_overlap)
    graph = CUDAGraph(context=context)
    try:
        graph.capture(
            forward,
            restore=lambda: overlap.copy_(saved_overlap),
        )
        result = graph.replay()
        expected = (
            torch.tensor(
                [255, 255, 128, 64, 255], dtype=torch.uint8, device=device
            )
            .view(5, 1, 1, 1)
            .expand_as(pixels)
        )
        first = result[0].tensor.clone()
        torch.testing.assert_close(first, expected, rtol=0, atol=0)

        overlap.zero_()
        segment[:, :, 4:].zero_()
        result = graph.replay()
        expected = (
            torch.tensor([0, 128, 128, 0, 0], dtype=torch.uint8, device=device)
            .view(5, 1, 1, 1)
            .expand_as(pixels)
        )
        torch.testing.assert_close(result[0].tensor, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            overlap, torch.zeros_like(overlap), rtol=0, atol=0
        )
        # Retained copies remain valid after the graph reuses its scratch
        # result.
        torch.testing.assert_close(
            first[:, 0, 0, 0],
            frames[[0, 1, 2, 4, 5]].mul(255).round().to(torch.uint8),
            rtol=0,
            atol=0,
        )
    finally:
        torch.cuda.synchronize(device)
        graph.close()
        context.close()
