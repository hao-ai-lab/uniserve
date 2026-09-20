"""Complete media and cancellation across statically assigned H3 components."""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import replace
from itertools import zip_longest
from pathlib import Path

import av
import httpx
import numpy as np
import pytest
from transformers import AutoTokenizer

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
    written_deployment,
)
from uniserve_eval.config import load_config
from uniserve_eval.datasets.minimax_h3 import MiniMaxH3Dataset
from uniserve_eval.transport.video import inspect_video_bytes

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]


def _assert_media_values_close(actual: bytes, expected: bytes) -> None:
    """Compare decoded media within the BF16 model-composition error budget."""
    for kind in ("video", "audio"):
        with (
            av.open(io.BytesIO(actual)) as observed,
            av.open(io.BytesIO(expected)) as reference,
        ):
            for frame, wanted in zip_longest(
                observed.decode(**{kind: 0}), reference.decode(**{kind: 0})
            ):
                assert frame is not None and wanted is not None, (
                    f"{kind}: frame count mismatch"
                )
                assert frame.time == wanted.time, (
                    f"{kind}: presentation time mismatch"
                )
                if kind == "video":
                    values = (
                        frame.to_ndarray(format="rgb24").astype(np.float32)
                        / 255
                    )
                    reference_values = (
                        wanted.to_ndarray(format="rgb24").astype(np.float32)
                        / 255
                    )
                else:
                    values = frame.to_ndarray()
                    reference_values = wanted.to_ndarray()
                np.testing.assert_allclose(
                    values,
                    reference_values,
                    rtol=2e-2,
                    atol=2e-2,
                    equal_nan=False,
                )


@pytest.mark.parametrize(
    ("parallel_kind", "precision", "grouping"),
    [
        (kind, precision, "whole")
        for kind in (
            "ulysses2",
            "ulysses4",
            "gather2",
            "gather4",
            "ring2",
            "ring4",
            "hybrid",
            "attention2d",
            "tensor2_ulysses2",
            "tensor2",
            "tensor4",
            "pipeline2",
            "pipeline4",
            "local",
        )
        for precision in ("quality", "balanced", "performance", "maximum")
    ]
    + [("ulysses4", "balanced", grouping) for grouping in ("split", "mixed")],
)
def test_component_bindings_release_cancelled_requests(
    tmp_path: Path, parallel_kind: str, precision: str, grouping: str
) -> None:
    model_value = os.environ.get("UNISERVE_H3_MODEL")
    if not model_value or not Path(model_value).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name a supported FastH3 checkpoint "
            "directory; docs/fast_h3/fast_h3.md lists them"
        )
    # A layout that holds the whole denoiser on one device cannot also hold a
    # 15-second activation working set: the single-device warmup exhausts a
    # 184 GiB device with 178 GB allocated. That layout is qualified at a
    # five-second maximum; every other layout takes both shapes.
    shapes = (
        ((5, 1000, 124),)
        if parallel_kind == "local"
        else ((5, 1000, 124), (15, 16384, 362))
    )
    max_video_seconds = str(max(seconds for seconds, _, _ in shapes))
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    worker_config = {
        "devices": [0, 1, 2, 3],
        "denoiser": {
            "ranks": [3, 1],
            "parallel_config": {"tensor_parallel_size": 2},
        },
        "text_encoder": {"ranks": [0]},
        "video_decoder": {
            "ranks": [2, 0],
            "distribution": "temporal_units",
            "units_per_rank": 1,
        },
        "audio_decoder": {"ranks": [1]},
    }
    if parallel_kind in ("local", "ulysses2", "ulysses4"):
        degree = {"local": 1, "ulysses2": 2, "ulysses4": 4}[parallel_kind]
        ranks = list(range(degree))
        sequence = (
            {"kind": "local"}
            if degree == 1
            else {"kind": "ulysses", "ulysses_degree": degree}
        )
        worker_config = {
            "devices": [3, 1, 2, 0][:degree],
            "denoiser": {
                "ranks": ranks,
                "parallel_config": {"sequence_parallel": sequence},
            },
            "text_encoder": {
                "ranks": ranks,
                "parallel_config": {"tensor_parallel_size": degree},
            },
            "video_decoder": {
                "ranks": ranks,
                "distribution": "temporal_units",
                "units_per_rank": 1,
            },
            "audio_decoder": {"ranks": [0]},
        }
    elif parallel_kind in ("pipeline2", "pipeline4"):
        degree = 2 if parallel_kind == "pipeline2" else 4
        worker_config["denoiser"]["ranks"] = [3, 1, 2, 0][:degree]
        worker_config["denoiser"]["parallel_config"] = {
            "pipeline_parallel_size": degree
        }
    elif parallel_kind in ("gather2", "gather4"):
        degree = 2 if parallel_kind == "gather2" else 4
        worker_config["denoiser"]["ranks"] = [3, 1, 2, 0][:degree]
        worker_config["denoiser"]["parallel_config"] = {
            "sequence_parallel": {
                "kind": "allgather",
                "allgather_degree": degree,
            }
        }
    elif parallel_kind in ("ring2", "ring4"):
        degree = 4 if parallel_kind == "ring4" else 2
        worker_config["denoiser"]["ranks"] = [3, 1, 2, 0][:degree]
        worker_config["denoiser"]["parallel_config"] = {
            "sequence_parallel": {"kind": "ring", "ring_degree": degree}
        }
    elif parallel_kind == "attention2d":
        worker_config["denoiser"] = {
            "ranks": [3, 1, 2, 0],
            "parallel_config": {
                "sequence_parallel": {
                    "kind": "attention2d",
                    "attn2d_row_size": 2,
                    "attn2d_col_size": 2,
                }
            },
        }
    elif parallel_kind == "hybrid":
        worker_config["denoiser"] = {
            "ranks": [3, 1, 2, 0],
            "parallel_config": {
                "sequence_parallel": {
                    "kind": "hybrid",
                    "ulysses_degree": 2,
                    "ring_degree": 2,
                }
            },
        }
    elif parallel_kind == "tensor2_ulysses2":
        worker_config["denoiser"] = {
            "ranks": [3, 1, 2, 0],
            "parallel_config": {
                "tensor_parallel_size": 2,
                "sequence_parallel": {"kind": "ulysses", "ulysses_degree": 2},
            },
        }
    elif parallel_kind in ("tensor2", "tensor4"):
        degree = 2 if parallel_kind == "tensor2" else 4
        worker_config["denoiser"]["ranks"] = [3, 1, 2, 0][:degree]
        worker_config["denoiser"]["parallel_config"] = {
            "tensor_parallel_size": degree
        }
    else:
        raise ValueError(f"unsupported H3 test layout {parallel_kind!r}")
    devices = worker_config.pop("devices")
    groups = {
        "whole": [tuple(worker_config)],
        "split": [(name,) for name in worker_config],
        "mixed": [
            ("denoiser", "text_encoder"),
            ("video_decoder", "audio_decoder"),
        ],
    }[grouping]
    workers = []
    owners = {}
    for names in groups:
        members = sorted(
            {rank for name in names for rank in worker_config[name]["ranks"]}
        )
        worker_id = names[0]
        components = {
            name: {
                **worker_config[name],
                "ranks": [
                    members.index(rank) for rank in worker_config[name]["ranks"]
                ],
            }
            for name in names
        }
        workers.append(
            {
                "id": worker_id,
                "ranks": [
                    {"node": "localhost", "device": f"cuda:{devices[rank]}"}
                    for rank in members
                ],
                "components": components,
                "queue_depth": 6,
            }
        )
        owners.update(dict.fromkeys(names, worker_id))
    # Media units are encoded and the artifact assembled on a host worker:
    # one host rank encodes every unit of a round and muxes.
    decoder_units = len(worker_config["video_decoder"]["ranks"])
    workers.append(
        {
            "id": "host",
            "ranks": [{"node": "localhost", "device": "cpu"}],
            "components": {
                "video_encoder": {
                    "ranks": [0],
                    "distribution": "temporal_units",
                    "units_per_rank": decoder_units,
                },
                "muxer": {"ranks": [0]},
            },
            "queue_depth": 6,
        }
    )
    owners.update({"video_encoder": "host", "muxer": "host"})
    edges = {
        (owners[source], owners[destination])
        for source, destination in (
            ("text_encoder", "denoiser"),
            ("denoiser", "video_decoder"),
            ("denoiser", "audio_decoder"),
            ("video_decoder", "video_encoder"),
            ("video_encoder", "muxer"),
            ("audio_decoder", "muxer"),
        )
    }
    command = [
        str(require_uniserve_binary()),
        "serve",
        model_value,
        "--served-model-name",
        "MiniMax-H3",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-python",
        sys.executable,
        "--workers",
        str(written_deployment(tmp_path, workers)),
        "--transfer",
        ",".join(
            # Device products cross workers over VMM handles; the host
            # worker's ranks have no device, so their edges carry host
            # products over shared memory only.
            f"{source}->{destination}="
            + ("shm" if "host" in (source, destination) else "cuda_vmm+shm")
            for source, destination in sorted(edges)
        ),
        "--queue-depth",
        "6",
        "--max-batch",
        "2",
        "--max-running-requests",
        "2",
        "--max-num-batched-tokens",
        "2",
        "--chunked-prefill-size",
        "1",
        "--max-model-len",
        "16384",
        "--max-video-seconds",
        max_video_seconds,
        # One and two device layouts hold the whole denoiser on each of their
        # ranks and split the 15-second sequence over fewer of them, so their
        # warmup reaches a larger share of the device than the four-device
        # deployment the default grant is sized for.
        "--mem-fraction-static",
        "0.92",
        "--dtype",
        "bfloat16",
        "--quantization-config",
        json.dumps({"mode": precision}),
    ]
    tokenizer = AutoTokenizer.from_pretrained(Path(model_value) / "tokenizer")
    point = load_config().benchmarks["minimax-h3-5s-1k"]
    payloads = []
    for seconds, tokens, _frames in shapes:
        case = replace(
            point,
            load=replace(point.load, num_prompts=1),
            video=replace(point.video, seconds=seconds, prompt_tokens=tokens),
        )
        prompt = MiniMaxH3Dataset(case).load(tokenizer)[0].prompt
        payloads.append(
            {
                "model": "MiniMax-H3",
                "prompt": prompt,
                "seconds": seconds,
                "seed": 1000,
            }
        )
    with server_process(
        command,
        base_url,
        tmp_path / "h3-components.log",
        timeout_s=600,
        env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    ):
        # Warm both decode geometries before exercising cancellation. Three
        # disconnects exceed the two provisioned slots and require their reuse.
        for payload, (_seconds, _tokens, frames) in zip(
            payloads, shapes, strict=True
        ):
            response = httpx.post(
                f"{base_url}/v1/videos/sync",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            media = inspect_video_bytes(
                response.content, declared_mime=response.headers["content-type"]
            )
            assert media.frame_count == frames
            assert (media.width, media.height) == (1344, 768)
            assert (media.audio_channels, media.audio_sample_rate) == (2, 32000)
            (tmp_path / f"{payload['seconds']}s.mp4").write_bytes(
                response.content
            )
            print(
                f"{parallel_kind}/{precision}: {payload['seconds']}s "
                "complete media passed",
                flush=True,
            )

        payload = payloads[0]
        for _ in range(3):
            with httpx.Client(timeout=httpx.Timeout(30, read=0.2)) as client:
                with pytest.raises(httpx.ReadTimeout):
                    client.post(f"{base_url}/v1/videos/sync", json=payload)
            response = httpx.post(
                f"{base_url}/v1/videos/sync", json=payload, timeout=600
            )
            response.raise_for_status()
            media = inspect_video_bytes(
                response.content, declared_mime=response.headers["content-type"]
            )
            assert media.frame_count == 124
            assert (media.audio_channels, media.audio_sample_rate) == (2, 32000)


def test_video_jobs_retain_content_and_cancel_active_work(
    tmp_path: Path,
) -> None:
    """Exercise async ownership and reuse through the HTTP contract.

    The exercise runs on one deployment.
    """
    import time

    model = os.environ.get("UNISERVE_H3_MODEL")
    if not model or not Path(model).is_dir():
        pytest.fail(
            "UNISERVE_H3_MODEL must name a supported FastH3 checkpoint "
            "directory; docs/fast_h3/fast_h3.md lists them"
        )
    port = find_free_port()
    base = f"http://127.0.0.1:{port}"
    command = [
        str(require_uniserve_binary()),
        "serve",
        model,
        "--worker-ranks",
        "4",
        "--worker-python",
        sys.executable,
        "--served-model-name",
        "FastH3",
        "--port",
        str(port),
        "--graph-policy",
        "full",
    ]

    def completed(client, job_id):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            response = client.get(f"/v1/videos/{job_id}")
            response.raise_for_status()
            job = response.json()
            assert job["status"] != "failed", job
            if job["status"] == "completed":
                return job
            time.sleep(0.1)
        pytest.fail("video did not complete within the request deadline")

    with (
        server_process(
            command, base, tmp_path / "video-jobs.log", timeout_s=900
        ),
        httpx.Client(base_url=base, timeout=600) as client,
    ):
        caps = client.get("/v1/capabilities").json()
        assert caps["tasks"] == ["t2va"]
        assert caps["model"] == "FastH3"
        for payload in (
            {"input_reference": "image.png"},
            {"num_inference_steps": 8},
            {"seconds": 16},
        ):
            response = client.post(
                "/v1/videos",
                json={"model": "FastH3", "prompt": "A river", **payload},
            )
            assert response.status_code == 400
        payload = {
            "model": "FastH3",
            "prompt": "A river flows through a forest, with birds singing.",
            "seconds": 5,
            "seed": 1001,
        }
        response = client.post(
            "/v1/videos",
            files={name: (None, str(value)) for name, value in payload.items()},
        )
        response.raise_for_status()
        job_id = response.json()["id"]
        job = completed(client, job_id)
        assert job["seconds"] == 5 and job["actual_seconds"] == 124 / 24
        assert job["completed_steps"] == job["total_steps"] == 4
        assert job["expires_at"] > job["completed_at"]
        assert job_id in {
            item["id"] for item in client.get("/v1/videos").json()["data"]
        }
        first = client.get(f"/v1/videos/{job_id}/content")
        second = client.get(f"/v1/videos/{job_id}/content")
        assert first.content == second.content
        media = inspect_video_bytes(
            first.content, declared_mime=first.headers["content-type"]
        )
        assert media.frame_count == 124 and media.audio_channels == 2
        assert client.delete(f"/v1/videos/{job_id}").json()["deleted"]
        assert client.get(f"/v1/videos/{job_id}/content").status_code == 404

        # Concurrent request storage must preserve each prompt and seed. Compare
        # both artifacts with isolated executions through the same public route.
        concurrent = [
            {**payload, "seed": 1010},
            {
                **payload,
                "prompt": (
                    "A train crosses a bridge at sunrise, with birds singing."
                ),
                "seed": 1011,
            },
        ]
        ids = [
            client.post("/v1/videos", json=item).json()["id"]
            for item in concurrent
        ]
        for item, concurrent_id in zip(concurrent, ids, strict=True):
            completed(client, concurrent_id)
            content = client.get(f"/v1/videos/{concurrent_id}/content")
            isolated = client.post("/v1/videos/sync", json=item)
            isolated.raise_for_status()
            _assert_media_values_close(content.content, isolated.content)

        # Cancel more requests than the two resident slots, including a
        # genuinely active job.
        for seed in range(3):
            active = client.post(
                "/v1/videos", json={**payload, "seconds": 15, "seed": seed}
            ).json()["id"]
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                state = client.get(f"/v1/videos/{active}").json()
                if state["status"] == "in_progress":
                    break
                assert state["status"] != "failed", state
                time.sleep(0.05)
            else:
                pytest.fail("job never reached execution")
            queued = client.post(
                "/v1/videos", json={**payload, "seed": seed + 100}
            ).json()["id"]
            assert client.delete(f"/v1/videos/{active}").status_code == 200
            assert client.delete(f"/v1/videos/{queued}").status_code == 200
            assert client.get(f"/v1/videos/{active}").status_code == 404
        reused = client.post(
            "/v1/videos", json={**payload, "seed": 1002}
        ).json()["id"]
        completed(client, reused)
        assert client.get(f"/v1/videos/{reused}/content").status_code == 200
        sync = client.post("/v1/videos/sync", json={**payload, "seed": 1003})
        sync.raise_for_status()
        assert (
            inspect_video_bytes(
                sync.content, declared_mime=sync.headers["content-type"]
            ).frame_count
            == 124
        )

    # The eager policy covers decoding as well as the learned denoising steps.
    # Keep the exact checkpoint, capacity, prompt and seed from the full run.
    command[-1] = "off"
    with (
        server_process(
            command, base, tmp_path / "video-eager.log", timeout_s=900
        ),
        httpx.Client(base_url=base, timeout=600) as client,
    ):
        eager = client.post("/v1/videos/sync", json=payload)
        eager.raise_for_status()
        _assert_media_values_close(eager.content, first.content)
