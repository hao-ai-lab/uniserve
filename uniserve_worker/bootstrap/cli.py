"""Launch adapter for :class:`WorkerProcessArgs`.

The engine writes one typed launch descriptor per rank and passes its location
on the command line. Every tuning value lives in that descriptor, so the
launching side is the single source of defaults and this module never restates
one. Only the process identity travels on argv, which keeps a running worker
identifiable from the process table.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

from .config import WorkerProcessArgs

# Values the descriptor must carry. A launch that omits one is a contract
# violation rather than something to paper over with a local default.
REQUIRED_FIELDS = (
    "registration_address",
    "channel_transport",
    "acknowledgment_slot",
    "products_cross_hosts",
    "worker_id",
    "queue_depth",
    "ipc_payload_cap",
    "model",
    "device",
    "rank",
    "local_rank",
    "world_size",
    "entries",
    "transfer_backends",
    "publish_backends",
    "attention_backend",
    "block_size",
    "max_batch_calls",
    "max_batch_tokens",
    "max_model_len",
    "max_video_seconds",
    "model_dtype",
    "quantization_config",
    "kv_memory_fraction",
    "graph_policy",
    "prefill_cuda_graph",
    "flashinfer_workspace_size",
    "flashinfer_decode_backend",
    "flashinfer_prefill_backend",
    "flashinfer_disable_split_kv",
    "load_format",
    "no_model",
    "allow_stub",
)


def create_worker_cli_parser() -> argparse.ArgumentParser:
    """Build the worker launch parser over identity and the descriptor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-descriptor", required=True, type=Path)
    # Identity is repeated on argv so `ps` identifies a worker without reading
    # its descriptor; the descriptor remains the authority.
    parser.add_argument("--worker-id", default="worker")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    return parser


def read_launch_descriptor(path: Path) -> Namespace:
    """Load one launch descriptor into the namespace the config consumes."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"unreadable launch descriptor {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid launch descriptor {path}: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("launch descriptor must be a JSON object")

    missing = [name for name in REQUIRED_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"launch descriptor omits {sorted(missing)}")
    return Namespace(**payload)


def parse_worker_args(
    arguments: Sequence[str] | None = None,
) -> WorkerProcessArgs:
    """Resolve the validated worker-process configuration for this launch."""
    parser = create_worker_cli_parser()
    namespace = parser.parse_args(arguments)
    try:
        return WorkerProcessArgs.from_namespace(
            read_launch_descriptor(namespace.launch_descriptor)
        )
    except ValueError as error:
        parser.error(str(error))
