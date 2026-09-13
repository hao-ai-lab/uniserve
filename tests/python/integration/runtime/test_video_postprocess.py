"""Captured video reconstruction consumes the supplied overlap on every replay."""

import pytest
import torch

from uniserve_worker.execution.cuda_graph import CudaGraph
from uniserve_worker.modeling.batch import TensorOutput
from uniserve_worker.modeling.geometry import DecodeWindow
from uniserve_worker.modeling.resources import TensorAlias, TensorNeeds, TensorSchema
from uniserve_worker.modeling.video import VideoMixin

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_video_postprocess_replay_uses_current_overlap_and_pixels():
    device = torch.device("cuda", 0)
    model = VideoMixin()
    window = DecodeWindow(2, 5, 3, 8, 3, 2, 1, final=True)
    frames = torch.tensor([1, 1, 0.5, 0, 0.25, 1], dtype=torch.float16, device=device)
    segment = frames.view(1, 1, 6, 1, 1).expand(1, 3, 6, 2, 3).contiguous()
    overlap = torch.ones((1, 3, 2, 2, 3), dtype=torch.float16, device=device)
    saved_overlap = overlap.clone()
    pixels = torch.empty((5, 2, 3, 3), dtype=torch.uint8, device=device)
    constants = {
        "pixel_mean": torch.zeros((1, 3, 1, 1, 1), device=device),
        "pixel_std": torch.ones((1, 3, 1, 1, 1), device=device),
    }
    needs = TensorNeeds(
        scratch={"rgb_frames": TensorSchema((5, 2, 3, 3), torch.uint8)},
        outputs={
            "video": TensorSchema(
                (5, 2, 3, 3), torch.uint8, alias=TensorAlias("scratch", "rgb_frames")
            )
        },
    )

    def forward():
        result = model.postprocess_video(
            (segment,),
            (window,),
            state={"video_overlap": overlap},
            constants=constants,
            scratch={"rgb_frames": pixels},
        )
        result.validate(needs, state={"video_overlap": overlap}, scratch={"rgb_frames": pixels})
        return result

    graph = CudaGraph[TensorOutput](device=device, stream=torch.cuda.Stream(device=device))
    try:
        graph.capture(
            forward,
            keepalive=(segment, overlap, pixels, constants),
            restore=lambda: overlap.copy_(saved_overlap),
        )
        graph.replay()
        expected = (
            torch.tensor([255, 255, 128, 64, 255], dtype=torch.uint8, device=device)
            .view(5, 1, 1, 1)
            .expand_as(pixels)
        )
        assert graph.output is not None
        first = graph.output.values["video"][0].clone()
        torch.testing.assert_close(first, expected, rtol=0, atol=0)

        overlap.zero_()
        segment[:, :, 4:].zero_()
        graph.replay()
        expected = (
            torch.tensor([0, 128, 128, 0, 0], dtype=torch.uint8, device=device)
            .view(5, 1, 1, 1)
            .expand_as(pixels)
        )
        torch.testing.assert_close(graph.output.values["video"][0], expected, rtol=0, atol=0)
        torch.testing.assert_close(overlap, torch.zeros_like(overlap), rtol=0, atol=0)
        # Retained copies remain valid after the graph reuses its scratch result.
        torch.testing.assert_close(
            first[:, 0, 0, 0],
            frames[[0, 1, 2, 4, 5]].mul(255).round().to(torch.uint8),
            rtol=0,
            atol=0,
        )
    finally:
        torch.cuda.synchronize(device)
        graph.close()
