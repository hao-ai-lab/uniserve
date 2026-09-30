"""CLI boundary for the Python worker process.

The Rust launchers (the engine's worker process launcher and the
``uniserve-host`` binary) start each rank as
``python -m uniserve_worker.main`` with the arguments
`uniserve_worker.bootstrap.cli.parse_worker_args` reads; the
``uniserve-worker`` console script runs the same ``main``. This module owns
only process-level concerns: CUDA loading policy, logging setup,
signal-triggered traceback dumps, and the exit status;
`uniserve_worker.bootstrap.launch.run_worker` builds and serves the rank.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys

from uniserve_worker.bootstrap.cli import parse_worker_args

logger = logging.getLogger(__name__)


def main() -> None:
    """Configure CUDA loading and diagnostics, then run the worker.

    The worker runs until shutdown or interruption.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_fault_dump_handlers()
    process_args = parse_worker_args()
    if str(process_args.execution.device).startswith("cuda"):
        # First-use module loading can synchronize a CUDA context while a
        # peer waits in an asynchronous collective. Preload both code and
        # data before importing the execution stack and initializing CUDA.
        os.environ["CUDA_MODULE_LOADING"] = "EAGER"
        os.environ["CUDA_MODULE_DATA_LOADING"] = "EAGER"

    from uniserve_worker.bootstrap.launch import run_worker

    # Both exception paths end the process with ``os._exit``, which skips
    # interpreter shutdown and the ``finally`` clause below; that clause logs
    # only when ``run_worker`` returns.
    try:
        run_worker(process_args)
    except KeyboardInterrupt:
        logger.info("worker interrupted; shutting down")
        # 130 is the shell convention for termination by SIGINT (128 + 2).
        os._exit(130)
    except BaseException:
        # Failed CUDA accesses and transfer threads retain backing until the
        # rank exits. Python's executor/finalizer shutdown would try to drain
        # them and can wait forever for the failed peer. ``os._exit`` does not
        # flush stdio buffers.
        logger.exception("worker failed")
        sys.stderr.flush()
        os._exit(1)
    finally:
        logger.info("worker shut down")


def _install_fault_dump_handlers() -> None:
    """Register signal-triggered Python traceback dumps.

    SIGQUIT and SIGUSR1 write every thread's traceback to stderr and the
    process keeps running, which lets an operator inspect a wedged rank. A
    signal the platform lacks, or one ``faulthandler`` cannot register, is
    skipped. The worker does not enable dumps on fatal signals
    (``faulthandler.enable``).
    """
    for sig_name in ("SIGQUIT", "SIGUSR1"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            faulthandler.register(sig, file=sys.stderr, all_threads=True)
        except Exception:
            logger.debug(
                "could not register faulthandler signal",
                extra={"signal": sig_name},
            )


if __name__ == "__main__":
    main()
