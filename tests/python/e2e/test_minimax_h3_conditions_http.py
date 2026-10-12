"""MiniMax-H3 checkpoints serve keyframe and reference requests.

Serves the published Ulysses-4 deployments and sends the seed-42 requests
of the packaged conditioned workloads (``uniserve_eval/workloads/minimax_h3``):
the official first-frame request (W3) on the base release's ``transformer``
deployment and on a FastH3 DMD student of it, and the image-plus-audio (W4)
and official video-plus-audio (W5) reference requests on the
``transformer_ref`` deployment, with the base release's reference DiT and
with a FastH3 OmniRef export. An OmniRef deployment that states a smaller
condition capacity than the image-plus-audio request takes refuses it,
naming its rows. Condition media are read from
``file://`` URIs under ``--media-directory``. Every served response is a
complete MP4 at the canvas and frame count the request resolves to.
"""

from __future__ import annotations

import json
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
WORKLOADS = ROOT / "uniserve_eval" / "workloads" / "minimax_h3"


def _request(workload: str, inputs: Path) -> dict:
    """The seed-42 request of a packaged conditioned workload.

    A workload's warmup manifest holds its seed-42 request: the prompt, the
    duration and the condition media in request order, named relative to the
    inputs directory. Conditioned workloads resolve their canvas from
    ``auto``.
    """
    manifest = WORKLOADS / f"{workload}-warmup.jsonl"
    row = json.loads(manifest.read_text().splitlines()[0])
    conditions = [
        {
            **{
                key: value for key, value in condition.items() if key != "media"
            },
            "uri": f"file://{(inputs / condition['media']).resolve()}",
        }
        for condition in row["conditions"]
    ]
    return {
        "model": "MiniMax-H3",
        "task": row["metadata"]["task"],
        "prompt": row["prompt"],
        "conditions": conditions,
        "target": {
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": row["seconds"],
        },
        "seed": row["seed"],
    }


def _root() -> str:
    model = os.environ.get("UNISERVE_MINIMAX_H3_MODEL")
    if not model or not Path(model).is_dir():
        pytest.fail(
            "UNISERVE_MINIMAX_H3_MODEL must name a MiniMax-H3 diffusers root"
        )
    return model


def _inputs() -> Path:
    inputs = os.environ.get("UNISERVE_MINIMAX_H3_INPUTS")
    if not inputs or not (Path(inputs) / "media").is_dir():
        pytest.fail(
            "UNISERVE_MINIMAX_H3_INPUTS must name the protocol inputs "
            "directory holding media/"
        )
    return Path(inputs).resolve()


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
    model, inputs = _root(), _inputs()
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    # The official first-frame presentation holds 1935 text rows and its
    # keyframe 1008 condition rows; without a stated condition capacity the
    # deployment provisions the two keyframes a request may bring.
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

        response = client.post("/v1/videos/sync", json=_request("w3", inputs))
        (tmp_path / "fl2va.mp4").write_bytes(response.content)
        # The 16:9 keyframe resolves the 1344x768 canvas; eight seconds
        # are 192 frames.
        _check_media(response, width=1344, height=768, frames=192)


def test_fasth3_first_frame_request(tmp_path: Path) -> None:
    """A FastH3 DMD student serves its partition's keyframe requests."""
    model = os.environ.get("UNISERVE_H3_MODEL")
    if not model or not Path(model).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name a FastH3 DMD export directory; "
            "docs/minimax_h3/minimax_h3.md lists them"
        )
    inputs = _inputs()
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    # The student generates 16:9 at 768p; eight seconds admit W3.
    options = ["--video-aspect-ratios", "16:9", "--max-video-seconds", "8"]
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
        assert video["tasks"] == ["t2va", "fl2va"]

        response = client.post("/v1/videos/sync", json=_request("w3", inputs))
        (tmp_path / "fl2va.mp4").write_bytes(response.content)
        _check_media(response, width=1344, height=768, frames=192)


def _reference_requests(command, base, tmp_path, *, inputs):
    """Serve ``command`` and check W4 and W5 against it."""
    with (
        server_process(command, base, tmp_path / "server.log", timeout_s=3600),
        httpx.Client(base_url=base, timeout=3600) as client,
    ):
        video = client.get("/v1/capabilities").json()["video"]
        assert video["tasks"] == ["ref2va"]

        for name in ("w4", "w5"):
            response = client.post(
                "/v1/videos/sync", json=_request(name, inputs)
            )
            (tmp_path / f"{name}.mp4").write_bytes(response.content)
            # References leave the auto canvas at 16:9; five seconds align
            # up to 124 frames.
            _check_media(response, width=1344, height=768, frames=124)


# The official video-plus-audio presentation holds 6913 text rows and its
# references about 38k condition rows.
REFERENCE_OPTIONS = [
    "--max-video-seconds",
    "5",
    "--max-model-len",
    "8192",
    "--video-text-capacities",
    "8192",
    "--max-condition-rows",
    "40960",
]


def test_base_reference_requests(tmp_path: Path) -> None:
    model, inputs = _root(), _inputs()
    port = find_free_port()
    _reference_requests(
        _command(
            model, "ulysses4-reference.json", port, inputs, REFERENCE_OPTIONS
        ),
        f"http://127.0.0.1:{port}",
        tmp_path,
        inputs=inputs,
    )


# The OmniRef reference DiT packs each condition into whole 128-row tiles,
# a video's token grid in tiles of 4 latent frames by 4 by 8 tokens, so the
# same video-plus-audio request takes 47,104 rows there: its 37 x 24 x 42
# reference video fills 10 x 6 x 6 tiles and its soundtrack and audio
# reference 4 tiles each, against 38,124 rows packed densely.
OMNIREF_OPTIONS = [
    "--max-video-seconds",
    "5",
    "--max-model-len",
    "8192",
    "--video-text-capacities",
    "8192",
    "--max-condition-rows",
    "49152",
]


def _omniref() -> str:
    """The OmniRef export the environment names.

    The Hugging Face cache supplies the base revision it pins.
    """
    export = os.environ.get("UNISERVE_MINIMAX_H3_OMNIREF")
    if not export:
        pytest.fail(
            "UNISERVE_MINIMAX_H3_OMNIREF must name a FastH3 OmniRef export"
        )
    return export


def test_omniref_reference_requests(tmp_path: Path) -> None:
    inputs = _inputs()
    export = _omniref()
    port = find_free_port()
    _reference_requests(
        _command(
            export,
            "ulysses4-reference.json",
            port,
            inputs,
            OMNIREF_OPTIONS,
        ),
        f"http://127.0.0.1:{port}",
        tmp_path,
        inputs=inputs,
    )


# The image-plus-audio request (W4) packs 7,808 condition rows on OmniRef when
# references are enlarged to the released pipeline's encoding size: its 16:9
# reference image, enlarged to a 2048-pixel short edge, fills 57 whole 128-row
# tiles (7,296 rows of video latents) and the five-second voice four. This
# deployment states a 4,096-row condition capacity, below that request's.
SMALL_CAPACITY_OPTIONS = [
    "--mm-processor-kwargs",
    '{"enlarge_references": true}',
    "--max-video-seconds",
    "5",
    "--max-model-len",
    "8192",
    "--video-text-capacities",
    "8192",
    "--max-condition-rows",
    "4096",
]


def test_omniref_refuses_conditions_above_capacity(tmp_path: Path) -> None:
    # The latent encoder's declared bound is the condition capacity, so the
    # engine refuses the request at admission, naming the product, the bound
    # and the rows the request needs of it.
    inputs = _inputs()
    export = _omniref()
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    command = _command(
        export,
        "ulysses4-reference.json",
        port,
        inputs,
        SMALL_CAPACITY_OPTIONS,
    )
    with (
        server_process(command, base, tmp_path / "server.log", timeout_s=3600),
        httpx.Client(base_url=base, timeout=3600) as client,
    ):
        refused = client.post("/v1/videos/sync", json=_request("w4", inputs))
        assert refused.status_code == 400, refused.text
        error = refused.json()["error"]
        assert error["type"] == "invalid_request_error", error
        message = error["message"]
        assert "condition_video_latents" in message, message
        assert "4096" in message and "7296" in message, message
