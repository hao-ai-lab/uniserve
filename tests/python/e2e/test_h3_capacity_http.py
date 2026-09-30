"""Startup-captured capacity layouts serve every admitted H3 request shape."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import httpx
import pytest
from transformers import AutoTokenizer

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
)
from tests.python.e2e.test_h3_parallel_http import _assert_media_values_close
from uniserve_eval.datasets.minimax_h3 import _PROMPT
from uniserve_eval.transport.video import inspect_video_bytes

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]

# Every output length the [4, 15] second API range produces at 24 fps.
FRAME_COUNTS = tuple(107 + 17 * index for index in range(16))
# Prompt lengths around and between the text capacities below.
PROMPT_TOKENS = (1, 777, 1024, 1025, 3000, 16384, 9000, 64)
TEXT_CAPACITIES = "1024,16384"


def _prompt(tokenizer, target: int) -> str:
    """The benchmark prompt, cut or extended to exactly ``target`` tokens."""
    filler = tokenizer.encode(
        " A coherent continuation preserves the scene, motion, lighting, and "
        "sound.",
        add_special_tokens=False,
    )
    ids = tokenizer.encode(_PROMPT, add_special_tokens=False)[:target]
    while len(ids) < target:
        ids.extend(filler[: target - len(ids)])
    text = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert len(tokenizer.encode(text, add_special_tokens=False)) == target
    return text


def _graph_modes(client: httpx.Client) -> dict[str, float]:
    """Sum the worker's graph replay, capture and eager call counters."""
    response = client.get("/metrics")
    response.raise_for_status()
    result: dict[str, float] = {}
    for line in response.text.splitlines():
        match = re.match(
            r"uniserve:worker_cuda_graph_runtime_mode_counts_total"
            r"\{([^}]*)\} (\S+)",
            line,
        )
        if match:
            mode = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))["mode"]
            result[mode] = result.get(mode, 0.0) + float(match.group(2))
    return result


def _settled(client: httpx.Client) -> dict[str, float]:
    # Worker statistics reach the registry with the engine's periodic
    # snapshot; read them once they stop moving.
    current, stable = _graph_modes(client), 0
    deadline = time.monotonic() + 120
    while stable < 3:
        if time.monotonic() > deadline:
            pytest.fail("worker graph counters did not settle")
        time.sleep(1)
        value = _graph_modes(client)
        stable = stable + 1 if value == current else 0
        current = value
    return current


def test_every_admitted_duration_and_prompt_length_replays(
    tmp_path: Path,
) -> None:
    """Serve the complete admitted range from startup-captured graphs.

    Startup captures every admitted duration at two text capacities. Each
    of the 16 output lengths is then requested once, at a fractional
    duration, with prompt lengths at, around and between the capacities.
    Every request returns complete media of its aligned length, and serving
    captures nothing. A denoiser that captures refuses to step eagerly, so a
    served request is one whose eight denoising steps replayed. Two requests
    of different layouts then run concurrently on the two request slots, and
    each matches its own isolated execution.
    """
    model = os.environ.get("UNISERVE_H3_MODEL")
    if not model or not Path(model).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name a supported FastH3 checkpoint "
            "directory; docs/fast_h3/fast_h3.md lists them"
        )
    deployment = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "fast_h3"
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
        "FastH3",
        "--port",
        str(port),
        "--max-running-requests",
        "2",
        "--max-model-len",
        "16384",
        "--max-video-seconds",
        "15",
        "--video-text-capacities",
        TEXT_CAPACITIES,
        "--graph-policy",
        "full",
    ]
    tokenizer = AutoTokenizer.from_pretrained(Path(model) / "tokenizer")

    with (
        server_process(
            command, base, tmp_path / "capacity.log", timeout_s=2400
        ),
        httpx.Client(base_url=base, timeout=900) as client,
    ):
        capabilities = client.get("/v1/capabilities").json()["video"]
        assert (
            capabilities["min_seconds"],
            capabilities["max_seconds"],
            capabilities["model_max_seconds"],
        ) == (4.0, 15.0, 15.0)

        for index, frames in enumerate(FRAME_COUNTS):
            # Eight frames short of the output length, which extends to it.
            seconds = (frames - 8) / 24
            tokens = PROMPT_TOKENS[index % len(PROMPT_TOKENS)]
            before = _settled(client)
            response = client.post(
                "/v1/videos/sync",
                json={
                    "model": "FastH3",
                    "prompt": _prompt(tokenizer, tokens),
                    "seconds": seconds,
                    "seed": 2000 + index,
                },
            )
            response.raise_for_status()
            media = inspect_video_bytes(
                response.content,
                declared_mime=response.headers["content-type"],
            )
            assert media.frame_count == frames, (seconds, tokens)
            assert (media.width, media.height) == (1344, 768)
            assert (media.audio_channels, media.audio_sample_rate) == (
                2,
                32000,
            )
            after = _settled(client)
            assert after.get("graph_capture", 0.0) == before.get(
                "graph_capture", 0.0
            ), (seconds, tokens)
            # Eight denoising steps, the text encoder and the refiner.
            assert (
                after["graph_replay"] - before.get("graph_replay", 0.0) >= 10
            ), (seconds, tokens)

        # Two layouts in flight together keep their own slots and state.
        payloads = [
            {
                "model": "FastH3",
                "prompt": _prompt(tokenizer, 500),
                "seconds": 7.3,
                "seed": 3001,
            },
            {
                "model": "FastH3",
                "prompt": _prompt(tokenizer, 9000),
                "seconds": 12.2,
                "seed": 3002,
            },
        ]
        ids = [
            client.post("/v1/videos", json=payload).json()["id"]
            for payload in payloads
        ]
        for payload, job_id in zip(payloads, ids, strict=True):
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                job = client.get(f"/v1/videos/{job_id}").json()
                assert job["status"] != "failed", job
                if job["status"] == "completed":
                    break
                time.sleep(0.2)
            else:
                pytest.fail("concurrent video did not complete")
            content = client.get(f"/v1/videos/{job_id}/content")
            isolated = client.post("/v1/videos/sync", json=payload)
            isolated.raise_for_status()
            _assert_media_values_close(content.content, isolated.content)
