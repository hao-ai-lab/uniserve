"""Compatibility entrypoint for profile-driven serving verification."""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMAND_ALIASES = {
    "bench": "perf",
    "generate": "verify",
}

WORKLOAD_ALIASES = {
    "sensenova-travel-interleave-4x": "gate/sensenova/default-travel",
}


def normalize_args(argv: Sequence[str]) -> list[str]:
    normalized = [COMMAND_ALIASES.get(arg, arg) for arg in argv]
    return [WORKLOAD_ALIASES.get(arg, arg) for arg in normalized]


def main(argv: Sequence[str] | None = None) -> None:
    args = normalize_args(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, "-m", "uniserve_eval", *args]
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
