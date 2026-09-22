"""CLI boundary for the Python worker process."""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys

from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.launch import run_worker

logger = logging.getLogger(__name__)


def main() -> None:
    """Configure diagnostics and run the worker.

    The worker runs until shutdown or interruption.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_fault_dump_handlers()
    process_args = parse_worker_args()

    try:
        run_worker(process_args)
    except KeyboardInterrupt:
        logger.info("worker interrupted; shutting down")
        os._exit(130)
    except BaseException:
        # Failed CUDA accesses and transfer threads retain backing until the
        # rank exits. Python's executor/finalizer shutdown would try to drain
        # them and can wait forever for the failed peer.
        logger.exception("worker failed")
        sys.stderr.flush()
        os._exit(1)
    finally:
        logger.info("worker shut down")


def _install_fault_dump_handlers() -> None:
    """Enable Python fault dumps.

    Also register user-triggered traceback signals.
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
