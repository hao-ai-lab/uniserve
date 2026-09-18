"""Worker launch descriptors for configuration tests.

Production launches are described entirely by the engine's typed descriptor.
Tests build one the same way, overriding only the fields under test, so they
exercise the contract the worker actually receives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.config import WorkerProcessArgs

# Neutral values for every field the descriptor must carry. They mirror what
# the engine emits for a single-rank CPU worker.
DEFAULTS: dict[str, Any] = {
    "registration_address": "127.0.0.1:0",
    "channel_transport": "iceoryx2",
    "worker_id": "worker",
    "queue_depth": 2,
    "ipc_payload_cap": 65536,
    "model": "model",
    "device": "cpu",
    "rank": 0,
    "local_rank": 0,
    "world_size": 1,
    "entries": {"model": {"ranks": [0]}},
    "supported_ops": None,
    "transfer_backends": "local",
    "publish_backends": "local",
    "distributed_init_method": None,
    "distributed_backend": None,
    "mesh": None,
    "lane": [],
    "no_model": False,
    "allow_stub": False,
    "load_format": "auto",
    "download_dir": None,
    "load_threads": None,
    "checksum_manifest": None,
    "model_dtype": "bfloat16",
    "quantization_config": {},
    "kv_cache_dtype": None,
    "kv_memory_fraction": 0.70,
    "kv_token_capacity": None,
    "attention_backend": "auto",
    "block_size": 64,
    "max_batch_calls": 256,
    "max_batch_tokens": 8192,
    "max_model_len": 8192,
    "max_video_seconds": 15.0,
    "graph_policy": "off",
    "decode_graph_batch_sizes": None,
    "prefill_cuda_graph": False,
    "prefill_graph_token_sizes": None,
    "flow_graph_batch_sizes": None,
    "flow_graph_shapes": None,
    "video_graph_shapes": None,
    "flashinfer_workspace_size": 536870912,
    "flashinfer_use_tensor_core": None,
    "flashinfer_decode_backend": "fa2",
    "flashinfer_prefill_backend": "auto",
    "flashinfer_decode_split_tile_size": None,
    "flashinfer_prefill_split_tile_size": None,
    "flashinfer_disable_split_kv": False,
}


def descriptor_path(directory: Path, **overrides: Any) -> Path:
    """Write one launch descriptor and return its path."""
    payload = {**DEFAULTS, **overrides}
    path = Path(directory) / "launch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def worker_args(directory: Path, **overrides: Any) -> WorkerProcessArgs:
    """Resolve a worker configuration from a descriptor built for this test."""
    path = descriptor_path(directory, **overrides)
    return parse_worker_args(["--launch-descriptor", str(path)])
