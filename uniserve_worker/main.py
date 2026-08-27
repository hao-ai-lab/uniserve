"""CLI boundary for the Python worker process."""

from __future__ import annotations

import faulthandler
import logging
import signal
import sys

from .bootstrap.cli import parse_worker_args
from .bootstrap.launch import run_worker

logger = logging.getLogger(__name__)


def main() -> None:
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
    finally:
        logger.info("worker shut down")


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
