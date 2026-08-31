"""Command-line adapter for :class:`WorkerProcessArgs`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..server.worker_kind import WorkerKind
from .capacity import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_BATCH_OPS
from .config import WorkerProcessArgs


def create_worker_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-name", required=True)
    parser.add_argument("--pipeline-depth", type=int, default=2)
    parser.add_argument("--ipc-payload-cap", type=int, required=True)
    parser.add_argument("--ipc-max-inflight", type=int, default=1)
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--worker-kind",
        default=WorkerKind.FULL.value,
        choices=WorkerKind.values(),
        help="deployment role served by this worker process",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mesh",
        default="",
        help=(
            "comma-separated placement entries: "
            "tower=text:<device>;gen:<device> and "
            "tower-kv-capacity=<tokens>"
        ),
    )
    parser.add_argument(
        "--transfer-backend",
        default="local",
        help="data-plane backend: local, shm, or cuda_ipc",
    )
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument(
        "--linear-precision",
        choices=("fp8", "nvfp4"),
        default="fp8",
    )
    parser.add_argument(
        "--h3-transformer-attention-precision",
        choices=("fp8", "nvfp4"),
        default=None,
    )
    parser.add_argument(
        "--h3-transformer-mlp-precision",
        choices=("fp8", "nvfp4"),
        default=None,
    )
    parser.add_argument(
        "--h3-text-encoder-precision",
        choices=("bf16", "nvfp4"),
        default=None,
    )
    parser.add_argument(
        "--h3-video-vae-precision",
        choices=("fp16", "bf16", "nvfp4"),
        default=None,
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
        "--tp-rank",
        type=int,
        default=0,
        help="tensor-parallel rank",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="tensor-parallel world size",
    )
    parser.add_argument("--tp-backend", default=None)
    parser.add_argument("--tp-init-method", default=None)
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
    parser.add_argument("--media-spool", default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-video-seconds", type=float, default=15.0)
    parser.add_argument("--fixed-graph-cache-capacity", type=int, default=32)
    return parser


def parse_worker_args(
    arguments: Sequence[str] | None = None,
) -> WorkerProcessArgs:
    parser = create_worker_cli_parser()
    namespace = parser.parse_args(arguments)
    try:
        return WorkerProcessArgs.from_namespace(namespace)
    except ValueError as error:
        parser.error(str(error))
