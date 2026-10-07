"""A served model's startup kernel table names the kernel of every call.

``uniserve serve`` logs one ``uniserve-kernel-table`` line when startup
completes (see ``uniserve_worker.execution.kernel_table``). Startup runs
every staged call kind, image encoders and decoders included, so for an
image-understanding model the table already names the kernels that serve
image inputs: here BAGEL's SigLIP vision tower, its autoencoder's encoder
and decoder, the non-causal image-feature rows its prefill appends, and the
image latents its denoiser writes into the prompt. A call kind without a
native kernel would fail startup instead of its first request.

The checkpoint directory comes from ``UNISERVE_BAGEL_MODEL``; the worker
interpreter is ``UNISERVE_WORKER_PYTHON``, else the repository's
``.venv/bin/python``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
)

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("bagel")]

TAG = "uniserve-kernel-table "


def test_bagel_startup_table_names_every_image_kernel(tmp_path):
    checkpoint = os.environ.get("UNISERVE_BAGEL_MODEL")
    if not checkpoint or not Path(checkpoint).is_dir():
        pytest.fail("UNISERVE_BAGEL_MODEL must name the BAGEL checkpoint")
    try:
        binary = require_uniserve_binary()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    port = find_free_port()
    python = os.environ.get(
        "UNISERVE_WORKER_PYTHON", str(Path.cwd() / ".venv" / "bin" / "python")
    )
    log = tmp_path / "server.log"
    with server_process(
        [
            str(binary),
            "serve",
            checkpoint,
            "--served-model-name",
            "bagel",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--worker-python",
            python,
        ],
        f"http://127.0.0.1:{port}",
        log,
        timeout_s=1800.0,
    ):
        pass

    tables = [
        json.loads(line.split(TAG, 1)[1])
        for line in log.read_text(errors="replace").splitlines()
        if TAG in line
    ]
    (startup,) = (table for table in tables if table["stage"] == "startup")
    attention = {
        runner["runner"]: [
            site for site in runner["call_sites"] if site["op"] == "attention"
        ]
        for runner in startup["runners"]
    }

    def sites(kind):
        return [
            site
            for runner, values in attention.items()
            if runner.endswith(f"[{kind}]")
            for site in values
        ]

    # Every SigLIP layer, the autoencoder's encoder mid-block and its
    # decoder mid-block (each runner also holds the other half of the
    # autoencoder, which it never calls) have chosen a kernel.
    vision = sites("vision_encoding")
    assert vision and all(site["provider"] is not None for site in vision)
    for kind, half in (
        ("latent_encoding", "encoder.middle"),
        ("image_decoding", "decoder.decoder.middle"),
    ):
        called = [
            site
            for site in sites(kind)
            if any(layer.startswith(half) for layer in site["layers"])
        ]
        assert called and all(site["provider"] for site in called), kind

    # Prefill replays graphs for causal text and for non-causal image rows,
    # and the denoiser writes an input image's latent into the prompt.
    def inputs(kind):
        return {
            name
            for runner, values in attention.items()
            if kind in runner
            for site in values
            for name in site.get("inputs", {})
        }

    assert {
        "paged attention: causal rows",
        "paged attention: non-causal rows",
    } <= inputs("prefill")
    assert "paged attention: non-causal rows" in inputs("denoising")
