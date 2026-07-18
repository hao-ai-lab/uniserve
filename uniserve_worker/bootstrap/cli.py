"""Command-line adapter for :class:`WorkerLaunchConfig`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..foundation.sizing import DEFAULT_BLOCK_SIZE
from ..server.worker_kind import WorkerKind
from .config import WorkerLaunchConfig


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
        choices=WorkerKind.wire_values(),
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
        help="data-plane backend: local, shm, cuda_ipc, or mooncake",
    )
    parser.add_argument(
        "--defer-sampling",
        action="store_true",
        default=False,
        help="publish logits for a separate sampler worker",
    )
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--kv-token-capacity", type=int, default=None)
    parser.add_argument("--kv-cache-dtype", default=None)
    parser.add_argument("--kv-memory-fraction", type=float, default=0.70)
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument(
        "--transformers-trust-remote-code",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--transformers-attn-implementation",
        default="uniserve",
    )
    parser.add_argument("--disable-model-arch", action="append", default=[])
    parser.add_argument(
        "--strict-model-imports",
        action="store_true",
        default=False,
    )
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
    parser.add_argument("--mooncake-device", default="")
    parser.add_argument("--mooncake-protocol", default="rdma")
    parser.add_argument("--torch-compile", action="store_true", default=False)
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", default=None)
    parser.add_argument(
        "--torch-compile-fullgraph",
        action="store_true",
        default=False,
    )
    parser.add_argument("--torch-compile-dynamic", default=None)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cuda-graph-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cuda-graph-warmup-batches", default=None)
    parser.add_argument(
        "--prefill-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--prefill-cuda-graph-warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--prefill-cuda-graph-warmup-tokens",
        default=None,
    )
    parser.add_argument("--mixed-text-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--varlen-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--green-contexts", action="store_true", default=False)
    parser.add_argument(
        "--logits-processor-chunk-size",
        type=int,
        default=0,
    )
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
    return parser


def parse_worker_launch(
    arguments: Sequence[str] | None = None,
) -> WorkerLaunchConfig:
    parser = create_worker_cli_parser()
    namespace = parser.parse_args(arguments)
    try:
        return WorkerLaunchConfig.from_namespace(namespace)
    except ValueError as error:
        parser.error(str(error))
