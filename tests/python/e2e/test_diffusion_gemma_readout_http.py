"""System One readouts served from the published DiffusionGemma checkpoints.

Each precision's checkpoint is served by ``uniserve serve`` on one GPU and
answers ``POST /v1/systemone`` requests end to end: the server plans the
readout prompts, the engine prefills them into the paged KV cache (image
features included) and denoises every canvas once, and the answers come
back in the official shape. The tests check what a client observes:

- every golden readout plan of ``diffusion_gemma_readout_prompts.json``
  without remote images answers every question and charges exactly the
  plan's prompt tokens;
- a fact stated at the start of the prompt is read back at prompt lengths
  around the 1023-token sliding window and beyond it, where only the full
  attention layers still see it;
- attached images, one or eight, are read in ``x_images`` order;
- a stream of distinct images longer than the encoder cache retains is read
  in full, each image answered after the cache fills.

The checkpoint directories come from ``UNISERVE_DIFFUSION_GEMMA_MODEL``
(BF16) and ``UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL`` (NVFP4). The worker
interpreter is ``UNISERVE_WORKER_PYTHON``, else the repository's
``.venv/bin/python``.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from PIL import Image

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
)
from uniserve_models import loading as models
from uniserve_worker.config.execution import WorkerConfig

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu,
    pytest.mark.model("diffusion_gemma"),
]

SERVED_MODEL = "diffusion-gemma"
CHECKPOINTS = {
    "bf16": "UNISERVE_DIFFUSION_GEMMA_MODEL",
    "nvfp4": "UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL",
}
FIXTURES = Path(__file__).parents[1] / "fixtures"
FIXTURE = FIXTURES / "diffusion_gemma_readout_prompts.json"
COLORS = {
    "red": (220, 30, 30),
    "green": (30, 170, 60),
    "blue": (30, 60, 220),
    "yellow": (235, 210, 40),
}


def _checkpoint(precision: str) -> Path:
    variable = CHECKPOINTS[precision]
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"{variable} is required for the DiffusionGemma GPU gate")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"configured checkpoint directory is missing: {path}")
    return path


@pytest.fixture(scope="module", params=sorted(CHECKPOINTS))
def served(request, tmp_path_factory) -> Iterator[tuple[str, Path]]:
    """Serve one precision's checkpoint; yield its URL and directory."""
    checkpoint = _checkpoint(request.param)
    try:
        binary = require_uniserve_binary()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    python = os.environ.get(
        "UNISERVE_WORKER_PYTHON", str(Path.cwd() / ".venv" / "bin" / "python")
    )
    args = [
        str(binary),
        "serve",
        str(checkpoint),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-python",
        python,
        "--max-model-len",
        "8192",
    ]
    log = tmp_path_factory.mktemp(request.param) / "server.log"
    with server_process(args, base_url, log, timeout_s=1800.0):
        yield base_url, checkpoint


def _readout(base_url: str, body: dict) -> dict:
    response = httpx.post(f"{base_url}/v1/systemone", json=body, timeout=600.0)
    assert response.status_code == 200, response.text
    return response.json()


def _distribution(answer: dict) -> dict[str, float]:
    """The normalized key distribution of one answer."""
    if answer["type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return answer["probabilities"]


def _assert_well_formed(answer: dict) -> None:
    distribution = _distribution(answer)
    assert all(0.0 <= value <= 1.0 for value in distribution.values())
    assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-6)
    assert 0.0 < answer["x_candidate_mass"] <= 1.0 + 1e-6


def _choice(answer: dict) -> str:
    distribution = _distribution(answer)
    return max(distribution, key=distribution.get)


def test_golden_plans_answer_every_question(served):
    base_url, _ = served
    fixture = json.loads(FIXTURE.read_text())
    cases = [
        case
        for case in fixture["cases"]
        if (case["layout"], case["canvas"]) == ("joint", "full")
        and "x_images" not in case["request"]
    ]
    assert cases
    for case in cases:
        body = {**case["request"], "model": SERVED_MODEL}

        response = _readout(base_url, body)

        assert response["model"] == SERVED_MODEL
        assert response["usage"] == {
            "input_tokens": case["input_tokens"],
            "output_tokens": 0,
        }, case["name"]
        assert list(response["answers"]) == list(case["request"]["questions"])
        for answer in response["answers"].values():
            _assert_well_formed(answer)


@pytest.mark.parametrize("length", [1022, 1023, 1024, 1025, 2000])
def test_a_fact_before_the_sliding_window_is_read_back(served, length):
    """The canvas reads a fact from the prompt's start at every length.

    One filler token per `` x`` pads the state until the prompt holds
    ``length`` tokens, which the response's usage confirms.
    """
    base_url, _ = served
    question = {
        "color": {
            "type": "choice",
            "instructions": "What is the secret color named at the start "
            "of the state?",
            "criteria": dict.fromkeys(COLORS),
        }
    }

    def body(filler: int) -> dict:
        return {
            "model": SERVED_MODEL,
            "state": "The secret color is green." + " x" * filler,
            "questions": question,
        }

    base = _readout(base_url, body(0))["usage"]["input_tokens"]
    response = _readout(base_url, body(length - base))

    assert response["usage"]["input_tokens"] == length
    answer = response["answers"]["color"]
    _assert_well_formed(answer)
    assert _choice(answer) == "green"


def _png(color: tuple[int, int, int], size: tuple[int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@pytest.mark.parametrize("count", [1, 8])
def test_attached_images_are_read_in_order(served, count):
    """The last attached image's color is read, and usage counts its tokens.

    Every image before the last has another color, so an answer read from
    the wrong placement names a different color.
    """
    base_url, checkpoint = served
    names = list(COLORS)
    colors = [names[index % 3] for index in range(count - 1)] + ["yellow"]
    sizes = [(96 + 48 * index, 64 + 16 * index) for index in range(count)]
    vit = models.read_config(checkpoint).image_processor.vit
    soft = sum(vit.tokens(height, width, 1) for width, height in sizes)
    body = {
        "model": SERVED_MODEL,
        "state": "The attached images are plain color swatches.",
        "questions": {
            "last": {
                "type": "choice",
                "instructions": f"Which color fills Image {count}?",
                "criteria": dict.fromkeys(COLORS),
            }
        },
        "x_images": [
            _png(COLORS[color], size)
            for color, size in zip(colors, sizes, strict=True)
        ],
    }
    without_images = {
        key: value for key, value in body.items() if key != "x_images"
    }

    response = _readout(base_url, body)
    text_only = _readout(base_url, without_images)

    answer = response["answers"]["last"]
    _assert_well_formed(answer)
    assert _choice(answer) == "yellow"
    # Each image adds its soft tokens between its two marker tokens, and
    # the prompt names the images in one line of text.
    added = (
        response["usage"]["input_tokens"] - text_only["usage"]["input_tokens"]
    )
    assert added >= soft + 2 * count


def test_distinct_images_beyond_the_encoder_cache_are_read(served):
    """Every image of a stream longer than the encoder cache is read.

    The served worker keeps the default encoder cache budget. Each request
    attaches one image no earlier request used, so every image is encoded
    and, past the budget, encoded while the cache holds its full set of
    retained features.
    """
    base_url, _ = served
    entries = WorkerConfig().encoder_cache_entries
    names = list(COLORS)
    with httpx.Client(timeout=600.0) as client:
        for index in range(entries + 8):
            # The size identifies the image; the color is the expected answer.
            color = names[index % len(names)]
            size = (64 + index % 64, 64 + index // 64)
            body = {
                "model": SERVED_MODEL,
                "state": "The attached image is a plain color swatch.",
                "questions": {
                    "color": {
                        "type": "choice",
                        "instructions": "Which color fills Image 1?",
                        "criteria": dict.fromkeys(COLORS),
                    }
                },
                "x_images": [_png(COLORS[color], size)],
            }
            response = client.post(f"{base_url}/v1/systemone", json=body)
            assert response.status_code == 200, (index, response.text)
            answer = response.json()["answers"]["color"]
            _assert_well_formed(answer)
            assert _choice(answer) == color, (index, answer)
