"""The MiniMax-H3 base checkpoint serves keyframe and reference requests.

Serves the diffusers root on the published Ulysses-4 deployments and sends
the protocol's conditioned workloads (``tools/minimax_h3/workloads.py``):
the official first-frame request (W3) on the ``denoiser`` deployment, and
the image-plus-audio (W4) and official video-plus-audio (W5) reference
requests on the ``reference_denoiser`` deployment. Condition media are read
from ``file://`` URIs under ``--media-directory``. Every response is a
complete MP4 at the canvas and frame count the request resolves to.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import httpx
import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
)
from uniserve_eval.transport.video import inspect_video_bytes

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]

ROOT = Path(__file__).resolve().parents[3]


def _workloads():
    """Load the protocol's workload definitions from the tools tree."""
    path = ROOT / "tools" / "minimax_h3" / "workloads.py"
    spec = importlib.util.spec_from_file_location("minimax_h3_workloads", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WORKLOADS


def _paths() -> tuple[str, Path]:
    model = os.environ.get("UNISERVE_MINIMAX_H3_MODEL")
    inputs = os.environ.get("UNISERVE_MINIMAX_H3_INPUTS")
    if not model or not Path(model).is_dir():
        pytest.fail(
            "UNISERVE_MINIMAX_H3_MODEL must name a MiniMax-H3 diffusers root"
        )
    if not inputs or not (Path(inputs) / "prompts.json").is_file():
        pytest.fail(
            "UNISERVE_MINIMAX_H3_INPUTS must name the protocol inputs "
            "directory holding prompts.json and media/"
        )
    return model, Path(inputs).resolve()


def _command(model, deployment, port, inputs, options) -> list[str]:
    return [
        str(require_uniserve_binary()),
        "serve",
        model,
        "--workers",
        str(ROOT / "configs" / "minimax_h3" / deployment),
        "--worker-python",
        sys.executable,
        "--served-model-name",
        "MiniMax-H3",
        "--port",
        str(port),
        "--media-directory",
        str(inputs),
        *options,
    ]


def _check_media(response, *, width, height, frames) -> None:
    assert response.status_code == 200, response.text
    media = inspect_video_bytes(
        response.content, declared_mime=response.headers["content-type"]
    )
    assert (media.width, media.height) == (width, height)
    assert media.frame_count == frames
    assert media.fps_numerator == 24 * media.fps_denominator
    assert media.video_codec == "h264"
    assert (media.audio_codec, media.audio_channels) == ("aac", 2)
    assert media.audio_sample_rate == 32_000
    assert abs(media.audio_duration_s - frames / 24) <= 1 / 24
    assert media.video_variance > 0 and media.audio_rms > 0


def test_base_first_frame_request(tmp_path: Path) -> None:
    model, inputs = _paths()
    workload = _workloads()["fl2va_first_8s"]
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    # The official first-frame presentation holds 1935 text rows and its
    # keyframe 1008 condition rows, within the default condition capacity.
    options = [
        "--max-video-seconds",
        "8",
        "--max-model-len",
        "2048",
        "--video-text-capacities",
        "2048",
    ]
    with (
        server_process(
            _command(model, "ulysses4.json", port, inputs, options),
            base,
            tmp_path / "server.log",
            timeout_s=3600,
        ),
        httpx.Client(base_url=base, timeout=3600) as client,
    ):
        video = client.get("/v1/capabilities").json()["video"]
        assert set(video["tasks"]) == {"t2va", "fl2va"}

        body = {"model": "MiniMax-H3", **workload.request_body(inputs, 42)}
        response = client.post("/v1/videos/sync", json=body)
        (tmp_path / "fl2va.mp4").write_bytes(response.content)
        # The 16:9 keyframe resolves the 1344x768 canvas; eight seconds
        # are 192 frames.
        _check_media(response, width=1344, height=768, frames=192)


def test_base_reference_requests(tmp_path: Path) -> None:
    model, inputs = _paths()
    workloads = _workloads()
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    # The official video-plus-audio presentation holds 6913 text rows and
    # its references about 38k condition rows.
    options = [
        "--max-video-seconds",
        "5",
        "--max-model-len",
        "8192",
        "--video-text-capacities",
        "8192",
        "--max-condition-rows",
        "40960",
    ]
    with (
        server_process(
            _command(model, "ulysses4-reference.json", port, inputs, options),
            base,
            tmp_path / "server.log",
            timeout_s=3600,
        ),
        httpx.Client(base_url=base, timeout=3600) as client,
    ):
        video = client.get("/v1/capabilities").json()["video"]
        assert video["tasks"] == ["ref2va"]

        for name in ("ref2va_image_audio_5s", "ref2va_video_audio_5s"):
            body = {
                "model": "MiniMax-H3",
                **workloads[name].request_body(inputs, 42),
            }
            response = client.post("/v1/videos/sync", json=body)
            (tmp_path / f"{name}.mp4").write_bytes(response.content)
            # References leave the auto canvas at 16:9; five seconds align
            # up to 124 frames.
            _check_media(response, width=1344, height=768, frames=124)
