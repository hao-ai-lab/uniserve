"""Complete media and cancellation across independently placed H3 components."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from transformers import AutoTokenizer

from tests.python.e2e.http_helpers import (
    find_free_port,
    require_uniserve_binary,
    server_process,
)
from uniserve_eval.config import load_config
from uniserve_eval.datasets.minimax_h3 import MiniMaxH3Dataset
from uniserve_eval.transport.video import inspect_video_bytes

pytestmark = [pytest.mark.e2e, pytest.mark.gpu, pytest.mark.model("minimax_h3")]


@pytest.mark.parametrize(
    ("parallel_kind", "precision"),
    [
        (kind, precision)
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
    ],
)
def test_component_placement_releases_cancelled_requests(
    tmp_path: Path, parallel_kind: str, precision: str
) -> None:
    model_value = os.environ.get("UNISERVE_H3_MODEL")
    if not model_value or not Path(model_value).is_dir():
        pytest.fail("UNISERVE_H3_MODEL must name the FastH3 Preview v0.2 checkpoint directory")
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    deployment = {
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
        "output": {"ranks": [2]},
    }
    if parallel_kind in ("local", "ulysses2", "ulysses4"):
        degree = {"local": 1, "ulysses2": 2, "ulysses4": 4}[parallel_kind]
        ranks = list(range(degree))
        sequence = (
            {"kind": "local"} if degree == 1 else {"kind": "ulysses", "ulysses_degree": degree}
        )
        deployment = {
            "devices": [3, 1, 2, 0][:degree],
            "denoiser": {"ranks": ranks, "parallel_config": {"sequence_parallel": sequence}},
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
            "output": {"ranks": [0]},
        }
    elif parallel_kind in ("pipeline2", "pipeline4"):
        degree = 2 if parallel_kind == "pipeline2" else 4
        deployment["denoiser"]["ranks"] = [3, 1, 2, 0][:degree]
        deployment["denoiser"]["parallel_config"] = {"pipeline_parallel_size": degree}
    elif parallel_kind in ("gather2", "gather4"):
        degree = 2 if parallel_kind == "gather2" else 4
        deployment["denoiser"]["ranks"] = [3, 1, 2, 0][:degree]
        deployment["denoiser"]["parallel_config"] = {
            "sequence_parallel": {"kind": "allgather", "allgather_degree": degree}
        }
    elif parallel_kind in ("ring2", "ring4"):
        degree = 4 if parallel_kind == "ring4" else 2
        deployment["denoiser"]["ranks"] = [3, 1, 2, 0][:degree]
        deployment["denoiser"]["parallel_config"] = {
            "sequence_parallel": {"kind": "ring", "ring_degree": degree}
        }
    elif parallel_kind == "attention2d":
        deployment["denoiser"] = {
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
        deployment["denoiser"] = {
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
        deployment["denoiser"] = {
            "ranks": [3, 1, 2, 0],
            "parallel_config": {
                "tensor_parallel_size": 2,
                "sequence_parallel": {"kind": "ulysses", "ulysses_degree": 2},
            },
        }
    elif parallel_kind in ("tensor2", "tensor4"):
        degree = 2 if parallel_kind == "tensor2" else 4
        deployment["denoiser"]["ranks"] = [3, 1, 2, 0][:degree]
        deployment["denoiser"]["parallel_config"] = {"tensor_parallel_size": degree}
    else:
        raise ValueError(f"unsupported H3 test layout {parallel_kind!r}")
    command = [
        str(require_uniserve_binary()),
        "serve",
        model_value,
        "--model-description",
        "minimax-h3",
        "--served-model-name",
        "MiniMax-H3",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--worker-python",
        str(Path.cwd() / ".venv" / "bin" / "python"),
        "--deployment",
        json.dumps(deployment),
        "--pipeline-depth",
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
        "15",
        "--dtype",
        "bfloat16",
        "--quantization-config",
        json.dumps({"mode": precision}),
    ]
    tokenizer = AutoTokenizer.from_pretrained(Path(model_value) / "tokenizer")
    point = load_config().benchmarks["minimax-h3-5s-1k"]
    payloads = []
    for seconds, tokens in ((5, 1000), (15, 16384)):
        case = replace(
            point,
            load=replace(point.load, num_prompts=1),
            video=replace(point.video, seconds=seconds, prompt_tokens=tokens),
        )
        prompt = MiniMaxH3Dataset(case).load(tokenizer)[0].prompt
        payloads.append({"model": "MiniMax-H3", "prompt": prompt, "seconds": seconds, "seed": 1000})
    with server_process(
        command,
        base_url,
        tmp_path / "h3-components.log",
        timeout_s=600,
        env={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    ):
        # Warm both decode geometries before exercising cancellation. Three
        # disconnects exceed the two provisioned slots and require their reuse.
        for payload, frames in zip(payloads, (124, 362), strict=True):
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
            (tmp_path / f"{payload['seconds']}s.mp4").write_bytes(response.content)
            print(
                f"{parallel_kind}/{precision}: {payload['seconds']}s complete media passed",
                flush=True,
            )

        payload = payloads[0]
        for _ in range(3):
            with httpx.Client(timeout=httpx.Timeout(30, read=0.2)) as client:
                with pytest.raises(httpx.ReadTimeout):
                    client.post(f"{base_url}/v1/videos/sync", json=payload)
            response = httpx.post(f"{base_url}/v1/videos/sync", json=payload, timeout=600)
            response.raise_for_status()
            media = inspect_video_bytes(
                response.content, declared_mime=response.headers["content-type"]
            )
            assert media.frame_count == 124
            assert (media.audio_channels, media.audio_sample_rate) == (2, 32000)
