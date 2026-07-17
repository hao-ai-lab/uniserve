"""CLI boundary for the Python worker process."""

from __future__ import annotations

import faulthandler
import logging
import signal
import sys

from .bootstrap.assembly import run_worker
from .bootstrap.cli import parse_worker_launch

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_fault_dump_handlers()
    launch_config = parse_worker_launch()

    try:
        run_worker(launch_config)
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
