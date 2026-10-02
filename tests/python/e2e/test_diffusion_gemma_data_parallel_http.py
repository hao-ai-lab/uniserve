"""DiffusionGemma served by one data-parallel replica per visible GPU.

``uniserve serve --data-parallel-size N`` runs a full model replica on each
of the N visible GPUs, each with its own scheduler and KV cache, and routes
every request to the replica with the fewest requests in flight; with
``--expert-parallel`` the replicas also shard the routed experts and exchange
tokens at every expert layer. On both topologies the tests check what a
client observes:

- readouts sent together are all answered correctly, and every replica
  admits some of them;
- a seeded chat request sent to every replica at once is answered by each,
  with the same reply wherever the replicas are independent.

The checkpoint directories come from ``UNISERVE_DIFFUSION_GEMMA_MODEL``
(BF16) and ``UNISERVE_DIFFUSION_GEMMA_NVFP4_MODEL`` (NVFP4). The worker
interpreter is ``UNISERVE_WORKER_PYTHON``, else the repository's
``.venv/bin/python``. At least two GPUs must be visible.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
import torch

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
)

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
# One series per replica: requests each replica's scheduler admitted.
# Each replica keeps every expert, or the replicas shard the experts.
TOPOLOGIES = {
    "replicas": (),
    "experts": ("--expert-parallel",),
    "dwdp": ("--expert-parallel", "--expert-exchange", "dwdp"),
}
ADMITTED = re.compile(
    r'^uniserve:num_requests_admitted_total\{[^}]*engine="(\d+)"[^}]*\} (\S+)$',
    re.MULTILINE,
)


def _checkpoint(precision: str) -> Path:
    variable = CHECKPOINTS[precision]
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"{variable} is required for the DiffusionGemma GPU gate")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"configured checkpoint directory is missing: {path}")
    return path


@pytest.fixture(
    scope="module",
    params=[
        (precision, topology)
        for precision in sorted(CHECKPOINTS)
        for topology in TOPOLOGIES
    ],
    ids="-".join,
)
def served(request, tmp_path_factory) -> Iterator[tuple[str, int, str]]:
    """Serve one replica per visible GPU.

    Yields the URL, the replica count and the topology (a key of
    ``TOPOLOGIES``).
    """
    replicas = torch.cuda.device_count()
    if replicas < 2:
        pytest.fail(f"data-parallel serving needs two GPUs, found {replicas}")
    precision, topology = request.param
    checkpoint = _checkpoint(precision)
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
        "--data-parallel-size",
        str(replicas),
        *TOPOLOGIES[topology],
    ]
    log = tmp_path_factory.mktemp(f"{precision}-{topology}") / "server.log"
    with server_process(args, base_url, log, timeout_s=1800.0):
        yield base_url, replicas, topology


def _post(base_url: str, route: str, body: dict) -> dict:
    response = httpx.post(f"{base_url}{route}", json=body, timeout=600.0)
    assert response.status_code == 200, response.text
    return response.json()


def _admitted(base_url: str) -> dict[int, float]:
    """Requests admitted so far by each replica, from ``/metrics``."""
    text = httpx.get(f"{base_url}/metrics", timeout=30.0).text
    return {
        int(engine): float(value) for engine, value in ADMITTED.findall(text)
    }


def test_concurrent_readouts_are_answered_by_every_replica(served):
    base_url, replicas, _ = served
    question = {
        "color": {
            "type": "choice",
            "instructions": "What is the secret color named at the start "
            "of the state?",
            "criteria": dict.fromkeys(("red", "green", "blue", "yellow")),
        }
    }
    # Distinct fillers give every request its own prompt.
    bodies = [
        {
            "model": SERVED_MODEL,
            "state": "The secret color is green." + " x" * (40 + index),
            "questions": question,
        }
        for index in range(4 * replicas)
    ]

    with ThreadPoolExecutor(len(bodies)) as pool:
        responses = list(
            pool.map(
                lambda body: _post(base_url, "/v1/systemone", body), bodies
            )
        )

    for response in responses:
        probabilities = response["answers"]["color"]["probabilities"]
        assert max(probabilities, key=probabilities.get) == "green"

    # Replicas publish their scheduler counters once a second.
    deadline = time.monotonic() + 30.0
    admitted = _admitted(base_url)
    while time.monotonic() < deadline and (
        len(admitted) < replicas or sum(admitted.values()) < len(bodies)
    ):
        time.sleep(1.0)
        admitted = _admitted(base_url)
    assert sorted(admitted) == list(range(replicas))
    assert all(count > 0 for count in admitted.values()), admitted


def test_every_replica_answers_a_seeded_request(served):
    """One seeded request sent once per replica at the same time.

    Each copy is routed while the others are in flight, so every replica
    serves one. Independent replicas each serve their copy alone with the
    same model, so their replies are identical. Expert-parallel replicas
    take every expert layer in steps together, and each pads its forward to
    the graph of the step's agreed exchange capacity, as SGLang
    (``prepare_mlp_sync_batch_raw``) and vLLM (``coordinate_batch_across_dp``)
    pad data-parallel ranks to their largest: a copy that starts while the
    other replicas run larger steps runs in a larger graph, so a reply's
    numerics follow what the other replicas run alongside it. There each
    replica answers with a complete reply of its own.
    """
    base_url, replicas, topology = served
    body = {
        "model": SERVED_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "A shop has 17 apples and sells 8, then receives "
                "6. How many apples does it have?",
            }
        ],
        "max_completion_tokens": 256,
        "seed": 11,
    }

    with ThreadPoolExecutor(replicas) as pool:
        replies = list(
            pool.map(
                lambda _: _post(base_url, "/v1/chat/completions", body),
                range(replicas),
            )
        )

    for reply in replies:
        (choice,) = reply["choices"]
        assert choice["message"]["content"].strip()
        assert choice["finish_reason"] in ("stop", "length")
        assert 0 < reply["usage"]["completion_tokens"] <= 256
    if topology in {"replicas", "dwdp"}:
        first = replies[0]
        for reply in replies[1:]:
            assert reply["choices"] == first["choices"]
            assert reply["usage"] == first["usage"]
