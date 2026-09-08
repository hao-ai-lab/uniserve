"""Command-line adapter for :class:`WorkerProcessArgs`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ..execution.batch import OpCode
from .capacity import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_BATCH_OPS
from .config import WorkerProcessArgs


def _json_object(value: str) -> dict[str, object]:
    """Parse a command-line JSON object and reject non-object values."""

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("quantization config must be a JSON object")
    return parsed


def create_worker_cli_parser() -> argparse.ArgumentParser:
    """Build the worker CLI parser with launch, params, resource, execution, and loading options."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--service-name", required=True)
    parser.add_argument("--worker-id", default="worker")
    parser.add_argument("--pipeline-depth", type=int, default=2)
    parser.add_argument("--ipc-payload-cap", type=int, required=True)
    parser.add_argument("--ipc-max-inflight", type=int, default=1)
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--supported-ops",
        default=",".join(value.value for value in OpCode),
        help="comma-separated operation kinds assigned to this pool",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mesh",
        default="",
        help=(
            "comma-separated params entries: "
            "tower=text:<device>;gen:<device> and "
            "tower-kv-capacity=<tokens>"
        ),
    )
    parser.add_argument(
        "--transfer-backends",
        default="local",
        help="comma-separated physical backends: local, shm, cuda_ipc",
    )
    parser.add_argument(
        "--publish-backends",
        default="local",
        help="comma-separated bound backends required for outbound products",
    )
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument(
        "--quantization-config",
        type=_json_object,
        default={},
        help=(
            'JSON quantization policy, for example {"mode":"balanced"}; '
            "an empty object selects the model-owned default"
        ),
    )
    parser.add_argument(
        "--load-format",
        default="auto",
        choices=("auto", "safetensors", "pt", "dummy", "sharded_state", "layered"),
    )
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--load-threads", type=int, default=None)
    parser.add_argument("--checksum-manifest", default=None)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max-batch-operations", type=int, default=DEFAULT_MAX_BATCH_OPS)
    parser.add_argument("--max-batch-tokens", type=int, required=True)
    parser.add_argument("--kv-token-capacity", type=int, default=None)
    parser.add_argument("--kv-cache-dtype", default=None)
    parser.add_argument("--kv-memory-fraction", type=float, default=0.70)
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="process rank",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="process world size",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument(
        "--entries",
        type=_json_object,
        default={"model": {"ranks": [0], "parallel_config": {}}},
        help="host-expanded component membership and logical parallel settings",
    )
    parser.add_argument("--distributed-backend", default=None)
    parser.add_argument("--distributed-init-method", default=None)
    parser.add_argument(
        "--lane",
        action="append",
        default=[],
        help="repeatable JSON lane descriptor with lane_id, sm_budget, and domains",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--graph-policy", choices=("off", "auto", "full"), default="auto")
    parser.add_argument("--decode-graph-batch-sizes", default=None)
    parser.add_argument(
        "--prefill-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--prefill-graph-token-sizes", default=None)
    parser.add_argument("--flow-graph-batch-sizes", default=None)
    parser.add_argument("--flow-graph-shapes", default=None)
    parser.add_argument(
        "--flashinfer-workspace-size",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument("--flashinfer-use-tensor-core", default=None)
    parser.add_argument("--flashinfer-decode-backend", default="fa2")
    parser.add_argument("--flashinfer-prefill-backend", default="auto")
    parser.add_argument(
        "--flashinfer-decode-split-tile-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--flashinfer-prefill-split-tile-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--flashinfer-disable-split-kv",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--flashinfer-fast-decode-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--allow-stub", action="store_true", default=False)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-video-seconds", type=float, default=15.0)
    return parser


def parse_worker_args(
    arguments: Sequence[str] | None = None,
) -> WorkerProcessArgs:
    """Parse CLI arguments and return their validated worker-process configuration."""

    parser = create_worker_cli_parser()
    namespace = parser.parse_args(arguments)
    try:
        return WorkerProcessArgs.from_namespace(namespace)
    except ValueError as error:
        parser.error(str(error))
