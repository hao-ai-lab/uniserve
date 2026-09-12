"""CPU HTTP → real Rust admission/engine/IPC → H3 transformer-input evidence."""

import base64
import io
import os
import shlex
import sys
import time
from pathlib import Path

import httpx
import pytest
import torch
from PIL import Image

from tests.python.e2e.http_helpers import find_free_port, server_process
from tests.python.unit.models.test_h3_image_conditioning import make_encoder

pytestmark = pytest.mark.e2e
ROOT = Path("/mnt/lustre/vlm-k1kong/models/MiniMax-H3")


def test_http_reference_reaches_transformer_and_preserves_text_bytes(tmp_path):
    binary = Path(
        os.environ.get("UNISERVE_REFERENCE_CPU_BINARY", "target/debug/examples/h3_reference_cpu")
    ).resolve()
    if not binary.is_file():
        pytest.fail(
            "build the CPU fixture: cargo build -p uniserve-server --example h3_reference_cpu"
        )
    wrapper = tmp_path / "worker-python"
    wrapper.write_text(
        "#!/bin/sh\nshift 2\nexec "
        + shlex.quote(sys.executable)
        + ' -m tests.python.fixtures.h3_reference_worker "$@"\n'
    )
    wrapper.chmod(0o700)
    port = find_free_port()
    origin = f"http://127.0.0.1:{port}"
    image = torch.arange(64 * 64 * 3).remainder(256).to(torch.uint8).reshape(64, 64, 3)
    stream = io.BytesIO()
    Image.fromarray(image.numpy()).save(stream, format="PNG")
    reference = {
        "type": "image",
        "task": "reference",
        "role": "reference",
        "source": {"type": "base64", "value": base64.b64encode(stream.getvalue()).decode()},
    }
    payload = {
        "model": "minimax-h3-ref",
        "prompt": "A calm sea",
        "seconds": 1.5,
        "seed": 123,
        "steps": 2,
    }
    observations = []
    # Each numerical fixture stops at the transformer boundary, so it owns one
    # request lifetime. Run fixtures serially, with identical synthetic weights.
    for index, addition in enumerate(({"references": [reference]}, {}, {"references": []})):
        directory = tmp_path / str(index)
        directory.mkdir()
        with (
            server_process(
                [str(binary), str(ROOT), str(wrapper), str(port)],
                origin,
                directory / "server.log",
                timeout_s=90,
                env={"UNISERVE_INPUT_EVIDENCE": str(directory), "OMP_NUM_THREADS": "2"},
            ),
            httpx.Client(base_url=origin, timeout=30) as client,
        ):
            response = client.post("/v1/videos", json={**payload, **addition})
            assert response.status_code in (200, 202), response.text
            evidence = directory / "0.pt"
            job_status = {}
            deadline = time.monotonic() + 30
            while not evidence.exists() and time.monotonic() < deadline:
                job_status = client.get(f"/v1/videos/{response.json()['id']}").json()
                if job_status["status"] == "failed":
                    break
                time.sleep(0.1)
            assert evidence.exists(), (
                job_status.get("error"),
                (directory / "server.log").read_text(),
            )
            observations.append(torch.load(evidence, weights_only=True))
            # The test ends at transformer inputs. The meta-weight job must fail,
            # never publish a fabricated MP4, before the next serial request.
            job = response.json()["id"]
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                status = client.get(f"/v1/videos/{job}").json()["status"]
                if status == "failed":
                    break
                time.sleep(0.1)
            assert status == "failed"

            if index == 0:
                invalid_bundles = (
                    ([reference, reference], "at most 1"),
                    ([{**reference, "type": "audio"}], "only image"),
                    ([{**reference, "task": "first_frame"}], "task=reference"),
                    ([{**reference, "include_audio": False}], "forbids include_audio"),
                    ([{**reference, "extra": True}], "unknown field"),
                    (reference, "sequence"),
                    (None, "sequence"),
                )
                invalid_sources = (
                    ("url", "https://example.invalid/image.png", "inline base64"),
                    ("base64", "%%%%", "standard base64"),
                    ("base64", "YWJj", "PNG or JPEG"),
                )
                invalid_bundles += tuple(
                    ([{**reference, "source": {"type": kind, "value": value}}], rule)
                    for kind, value, rule in invalid_sources
                )
                for bundle, rule in invalid_bundles:
                    rejected = client.post("/v1/videos", json={**payload, "references": bundle})
                    assert rejected.status_code == 400, rejected.text
                    assert rule in rejected.text

    conditioned, plain, empty = observations
    # The fixture's projection and conditioner are tiled identity weights. Compare
    # the HTTP/IPC result with the real encoder on the original RGB image, and
    # with its unchanged language-only computation for the no-reference path.
    with torch.inference_mode():
        encoder = make_encoder()
        tokens = torch.tensor(
            [encoder.processor.tokenizer(payload["prompt"], add_special_tokens=False)["input_ids"]]
        )
        text_states = encoder.language_model(tokens, torch.arange(tokens.numel()))
        image_states, _ = encoder.numerical_entry(tokens, image.unsqueeze(0))
        for observed, expected in ((plain, text_states), (conditioned, image_states)):
            tiled = expected.repeat(1, 1, 5376 // 16)
            actual = observed["text"][:, : expected.shape[1]]
            assert torch.equal(actual.contiguous().view(torch.uint8), tiled.view(torch.uint8))
            assert not observed["text"][:, expected.shape[1] :].count_nonzero()
    for name in ("text", "video", "audio", "reference", "positions"):
        assert torch.equal(
            plain[name].contiguous().view(torch.uint8), empty[name].contiguous().view(torch.uint8)
        )
    assert plain["reference_indices"].numel() == 0
    assert conditioned["tags"].tolist() == [1] * 6 + [0] * 66 + [1] * 3
    assert conditioned["text_indices"].numel() == 128
    # FastVideo a943220c115228ade5d57b3bab9a6a87fd600a10: VAE image
    # rows follow the padded presentation, before target audio and video.
    assert conditioned["reference_indices"].tolist() == list(range(128, 132))
    assert conditioned["audio_indices"][0] == 192
    assert conditioned["reference"].shape == (4, 96)
    assert conditioned["reference"].count_nonzero()
    assert torch.equal(conditioned["positions"][128:132, 0], torch.full((4,), 75.0))
    assert conditioned["positions"][192, 0] == 76
    for name in ("video", "audio"):
        assert torch.equal(plain[name].view(torch.uint8), conditioned[name].view(torch.uint8))
