"""The MiniMax-H3 base checkpoint serves text-to-video on every named canvas.

Serves the diffusers root's ``transformer`` on the published Ulysses-4
deployment and requests one five-second ``t2va`` video at each named aspect
ratio of the canvas rule. Every response is a complete MP4 at the ratio's
canvas; the capabilities report the handshake's schedule and canvases, and
the request body without a task is refused.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    t2va_request,
)
from uniserve_eval.transport.video import inspect_video_bytes

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]

# The canvas rule's named aspect ratios and the canvases they resolve to.
CANVASES = {
    "21:9": (1536, 672),
    "16:9": (1344, 768),
    "4:3": (1024, 768),
    "1:1": (768, 768),
    "3:4": (768, 1024),
    "9:16": (768, 1344),
}


def test_base_text_to_video_on_every_named_canvas(tmp_path: Path) -> None:
    model = os.environ.get("UNISERVE_MINIMAX_H3_MODEL")
    if not model or not Path(model).is_dir():
        pytest.fail(
            "UNISERVE_MINIMAX_H3_MODEL must name a MiniMax-H3 diffusers root"
        )
    deployment = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "minimax_h3"
        / "ulysses4.json"
    )
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        model,
        "--workers",
        str(deployment),
        "--worker-python",
        sys.executable,
        "--served-model-name",
        "MiniMax-H3",
        "--port",
        str(port),
        # Five seconds bound the layouts startup prepares: two frame counts
        # at each of the six canvases and one text capacity.
        "--max-video-seconds",
        "5",
        "--max-model-len",
        "1024",
    ]
    prompt = (
        "A red fox trots across a snowy meadow at sunrise, its breath "
        "steaming, while birds call from the pines."
    )
    with (
        server_process(command, base, tmp_path / "server.log", timeout_s=3600),
        httpx.Client(base_url=base, timeout=1800) as client,
    ):
        video = client.get("/v1/capabilities").json()["video"]
        # The base DiT generates from the prompt alone and from keyframes.
        assert video["tasks"] == ["t2va", "fl2va"]
        assert video["schedule"] == {
            "num_inference_steps": 50,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        }
        assert video["canvas"]["canvases"] is None
        assert set(video["canvas"]["aspect_ratios"]) == set(CANVASES)

        legacy = client.post(
            "/v1/videos/sync",
            json={"model": "MiniMax-H3", "prompt": prompt, "seconds": 5},
        )
        assert legacy.status_code == 400
        assert "task" in legacy.json()["error"]["message"]

        for ratio, (width, height) in CANVASES.items():
            response = client.post(
                "/v1/videos/sync",
                json=t2va_request(
                    "MiniMax-H3", prompt, 5.0, 42, aspect_ratio=ratio
                ),
            )
            assert response.status_code == 200, (ratio, response.text)
            media = inspect_video_bytes(
                response.content,
                declared_mime=response.headers["content-type"],
            )
            (tmp_path / f"{ratio.replace(':', 'x')}.mp4").write_bytes(
                response.content
            )
            assert (media.width, media.height) == (width, height), ratio
            # Five seconds at 24 fps align up to 124 frames.
            assert media.frame_count == 124, ratio
            assert media.fps_numerator == 24 * media.fps_denominator
            assert media.video_codec == "h264"
            assert (media.audio_codec, media.audio_channels) == ("aac", 2)
            assert media.audio_sample_rate == 32_000
            assert abs(media.audio_duration_s - 124 / 24) <= 1 / 24
            assert media.video_variance > 0 and media.audio_rms > 0, ratio
