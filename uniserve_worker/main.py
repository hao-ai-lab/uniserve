"""Python worker process entrypoint.

Spawned by the Rust host. Opens the iceoryx2 request-response service, builds a
runner-backed model adapter, and hands off to :class:`~uniserve_worker.server.app.WorkerRuntime`.
The model and KV cache stay resident for the process lifetime; only control-plane
descriptors and small results cross the IPC boundary.

Transport, dispatch, capability gating, typed errors, and metrics live under
``server/``; model-specific work is resolved from ``models/`` and executed by
:class:`~uniserve_worker.execution.runner.ModelRunner`.
"""
from __future__ import annotations

import argparse
import faulthandler
import logging
import signal
import sys

from .backends.attention import init_attention_backends
from .foundation.errors import invalid_descriptor
from .foundation.runtime_config import set_worker_config, worker_config_from_args
from .foundation.sizing import DEFAULT_BLOCK_SIZE
from .nn.mesh import set_current_mesh
from .server.app import WorkerRuntime
from .server.distributed import build_device_mesh
from .server.driver_factory import build_registered_driver
from .server.runner_driver import RunnerDriver, load_runner_engine
from .server.stub import StubUniModel
from .server.worker_kind import (
    FULL,
    RUNNER_BACKED_KINDS,
    WORKER_KINDS,
)

__all__ = [
    'build_arg_parser',
    'build_driver',
    'main',
    'validate_worker_args',
]

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the worker CLI argument schema (independently testable)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-name", required=True)
    ap.add_argument("--pipeline-depth", type=int, default=2)
    ap.add_argument("--ipc-payload-cap", type=int, required=True)
    ap.add_argument("--ipc-max-inflight", type=int, default=1)
    ap.add_argument("--model", default="")
    ap.add_argument(
        "--worker-kind",
        default=FULL,
        choices=sorted(WORKER_KINDS),
        help=(
            "pipeline stage this worker serves: full (default; whole model, all ops "
            "in one mixed forward) or a peeled stage (encoder/prefill/decode/"
            "sampler/postprocess)"
        ),
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--mesh",
        default="",
        help=(
            "parallelism mesh spec: comma-separated key=value entries. Supported "
            "keys: 'tower=text:<dev>;gen:<dev>' places the image-generation tower "
            "on its own device; 'tower-kv-capacity=<tokens>' sizes the gen-tower KV "
            "snapshot. Empty means a single-device worker (no tower axis). Tensor "
            "parallelism is set via --tp-size/--tp-rank."
        ),
    )
    ap.add_argument(
        "--transfer-backend",
        default="inproc",
        help=(
            "data-plane Tier-2 backend for this worker's tensor handoffs: inproc "
            "(default) / shm / cuda_ipc / mooncake"
        ),
    )
    ap.add_argument(
        "--defer-sampling",
        action="store_true",
        default=False,
        help=(
            "Deferred sampling: publish decode logits to the data plane and return "
            "handles instead of sampling inline; a separate Sampler worker samples. "
            "Set by the host when the topology peels a sampler stage."
        ),
    )
    ap.add_argument("--attention-backend", default="auto")
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    ap.add_argument("--kv-token-capacity", type=int, default=None)
    ap.add_argument("--kv-cache-dtype", default=None)
    ap.add_argument("--kv-memory-fraction", type=float, default=0.70)
    ap.add_argument("--model-dtype", default="bfloat16")
    ap.add_argument("--transformers-trust-remote-code", action="store_true", default=False)
    ap.add_argument("--transformers-attn-implementation", default="uniserve")
    ap.add_argument("--disable-model-arch", action="append", default=[])
    ap.add_argument("--strict-model-imports", action="store_true", default=False)
    ap.add_argument("--tp-rank", type=int, default=0, help="tensor-parallel rank")
    ap.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="tensor-parallel world size for ranked workers and model collectives",
    )
    ap.add_argument("--tp-backend", default=None)
    ap.add_argument("--tp-init-method", default=None)
    ap.add_argument("--mooncake-device", default="")
    ap.add_argument("--mooncake-protocol", default="rdma")
    ap.add_argument("--torch-compile", action="store_true", default=False)
    ap.add_argument("--torch-compile-backend", default="inductor")
    ap.add_argument("--torch-compile-mode", default=None)
    ap.add_argument("--torch-compile-fullgraph", action="store_true", default=False)
    ap.add_argument("--torch-compile-dynamic", default=None)
    ap.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cuda-graph-warmup", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cuda-graph-warmup-batches", default=None)
    ap.add_argument("--prefill-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--prefill-cuda-graph-warmup", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--prefill-cuda-graph-warmup-tokens", default=None)
    ap.add_argument("--mixed-text-max-tokens", type=int, default=8192)
    ap.add_argument("--varlen-prefill", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--forward-max-memory-bound-tokens", type=int, default=281)
    ap.add_argument("--green-contexts", action="store_true", default=False)
    ap.add_argument("--logits-processor-chunk-size", type=int, default=0)
    ap.add_argument("--flashinfer-workspace-size", type=int, default=512 * 1024 * 1024)
    ap.add_argument("--flashinfer-use-tensor-core", default=None)
    ap.add_argument("--flashinfer-decode-backend", default="fa2")
    ap.add_argument("--flashinfer-prefill-backend", default="auto")
    ap.add_argument("--flashinfer-decode-split-tile-size", type=int, default=None)
    ap.add_argument("--flashinfer-prefill-split-tile-size", type=int, default=None)
    ap.add_argument("--flashinfer-disable-split-kv", action="store_true", default=False)
    ap.add_argument("--flashinfer-fast-decode-plan", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--allow-stub", action="store_true", default=False)
    return ap


def _parse_tower_devices(value: str, device: str) -> list[str] | None:
    """Parse a ``tower=text:<dev>;gen:<dev>`` value into a per-modality device list.

    TEXT/understanding defaults to the main ``--device``; GEN is the gen tower's
    device. ``None`` (no gen device, or gen == text) means no tower axis, so the
    worker runs its single-device path unchanged.
    """
    import torch  # local import: keep the entrypoint import-light

    modalities: dict[str, str] = {}
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, dev = part.partition(":")
        modalities[name.strip().lower()] = dev.strip()
    gen = modalities.get("gen")
    if not gen:
        return None
    text = modalities.get("text") or device
    text_dev = torch.device(text)
    if text_dev.type == "cuda" and text_dev.index is None:
        text = "cuda:0"
    if str(torch.device(text)) == str(torch.device(gen)):
        return None
    return [text, gen]


def _parse_mesh(spec: str, *, device: str) -> tuple[list[str] | None, int | None]:
    """Parse a ``--mesh`` spec into ``(tower_devices, tower_kv_capacity)``.

    Comma-separated ``key=value`` entries. Supported: ``tower=text:<dev>;gen:<dev>``
    (in-process tower device list) and ``tower-kv-capacity=<tokens>`` (gen-tower KV
    snapshot capacity). An empty spec yields ``(None, None)`` — a single-device
    worker with no tower axis.
    """
    spec = (spec or "").strip()
    if not spec:
        return None, None
    tower_devices: list[str] | None = None
    tower_kv_capacity: int | None = None
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        key, _, val = entry.partition("=")
        key = key.strip().lower()
        val = val.strip()
        if key == "tower":
            tower_devices = _parse_tower_devices(val, device)
        elif key in ("tower-kv-capacity", "tower_kv_capacity"):
            tower_kv_capacity = int(val)
    return tower_devices, tower_kv_capacity


def build_driver(args: argparse.Namespace):
    """Initialize tensor-parallel rank state and build the runner driver.

    Returns the stub echo driver (``--no-model``) or a real model-backed driver.
    Separated from :func:`main` so driver wiring is testable without the IPC/serve
    shell.
    """
    # Composition root: register every available attention backend explicitly so
    # the registered set is visible at one place and optional backends that fail to
    # import are handled uniformly (try/except ImportError inside). Idempotent, so
    # it composes with the module's import-time registration.
    registered = init_attention_backends()
    logger.debug("attention backends registered: %s", registered)
    tower_devices, tower_kv_capacity = _parse_mesh(args.mesh, device=args.device)
    mesh = build_device_mesh(
        tp_rank=args.tp_rank,
        tp_size=args.tp_size,
        device=args.device,
        tower_devices=tower_devices,
        tower_primary=0,
        tp_backend=args.tp_backend,
        tp_init_method=args.tp_init_method,
    )
    set_current_mesh(mesh)
    if args.no_model:
        return _build_stub_driver(args)
    kind = args.worker_kind
    if kind in RUNNER_BACKED_KINDS:
        return _build_runner_backed_driver(args, tower_kv_capacity=tower_kv_capacity)
    driver = build_registered_driver(kind, args)
    if driver is not None:
        logger.info("started registered worker", extra={"worker_kind": kind, "model_path": args.model})
        return driver
    raise invalid_descriptor(f"unsupported worker kind {kind!r}")


def _build_stub_driver(args: argparse.Namespace) -> RunnerDriver:
    logger.warning(
        "STUB MODEL ENABLED: serving the GPU-free StubUniModel echo engine, "
        "not a real model; outputs are synthetic. Unset --no-model for "
        "production serving.",
        extra={"service": args.service_name},
    )
    return RunnerDriver(StubUniModel(), block_size=args.block_size)


def _build_runner_backed_driver(
    args: argparse.Namespace,
    *,
    tower_kv_capacity: int | None,
) -> RunnerDriver:
    # full / prefill / decode all load the whole model and drive the shared
    # ModelRunner; only the advertised OpKind subset differs per worker kind.
    logger.info(
        "loading model",
        extra={"model_path": args.model, "device": args.device, "worker_kind": args.worker_kind},
    )
    driver = load_runner_engine(
        args.model,
        device=args.device,
        gen_snapshot_kv_capacity=tower_kv_capacity,
        attention_backend=args.attention_backend,
        kv_token_capacity=args.kv_token_capacity,
        block_size=args.block_size,
        defer_sampling=args.defer_sampling,
        transfer_backend=args.transfer_backend,
        worker_kind=args.worker_kind,
    )
    logger.info("model loaded; entering serve loop")
    return driver


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_fault_dump_handlers()
    args = build_arg_parser().parse_args()
    validate_worker_args(args)
    set_worker_config(worker_config_from_args(args))

    # Deferred import: opening IPC pulls native transport dependencies only for
    # the actual worker entrypoint, not for CLI/unit-test imports.
    from .server.ipc import Server

    server = Server(
        args.service_name,
        max_payload=args.ipc_payload_cap,
        max_inflight=args.ipc_max_inflight,
    )
    logger.info(
        "ipc service opened",
        extra={
            "service": args.service_name,
            "payload_cap": args.ipc_payload_cap,
            "max_inflight": args.ipc_max_inflight,
        },
    )

    try:
        driver = build_driver(args)
        WorkerRuntime(
            driver,
            server,
            worker_kind=args.worker_kind,
            pipeline_depth=args.pipeline_depth,
        ).serve()
    except KeyboardInterrupt:
        logger.info("worker interrupted; shutting down")
    finally:
        logger.info("worker shut down")


def validate_worker_args(args: argparse.Namespace) -> None:
    parser = build_arg_parser()

    def fail(message: str) -> None:
        parser.error(message)

    if args.block_size <= 0:
        fail("--block-size must be positive")
    if args.pipeline_depth <= 0:
        fail("--pipeline-depth must be positive")
    if args.ipc_payload_cap <= 0:
        fail("--ipc-payload-cap must be positive")
    if args.ipc_max_inflight <= 0:
        fail("--ipc-max-inflight must be positive")
    if args.kv_token_capacity is not None and args.kv_token_capacity <= 0:
        fail("--kv-token-capacity must be positive when provided")
    _, tower_kv_capacity = _parse_mesh(args.mesh, device=args.device)
    if tower_kv_capacity is not None and tower_kv_capacity <= 0:
        fail("--mesh tower-kv-capacity must be positive when provided")
    if args.tp_size <= 0:
        fail("--tp-size must be positive")
    if args.tp_rank < 0 or args.tp_rank >= args.tp_size:
        fail("--tp-rank must satisfy 0 <= rank < tp-size")
    if not args.no_model and not args.model:
        fail("--model is required unless --no-model is set")
    if args.no_model and not args.allow_stub:
        fail(
            "--no-model loads the GPU-free StubUniModel echo engine and must not be "
            "used for production serving; pass --allow-stub to opt in explicitly"
        )
    try:
        worker_config_from_args(args)
    except ValueError as exc:
        fail(str(exc))


def _install_fault_dump_handlers() -> None:
    for sig_name in ("SIGQUIT", "SIGUSR1"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            faulthandler.register(sig, file=sys.stderr, all_threads=True)
        except Exception:
            logger.debug("could not register faulthandler signal", extra={"signal": sig_name})


if __name__ == "__main__":
    main()
