#!/usr/bin/env python3
"""Deprecated shim: this driver moved to scripts/e2e.py (same CLI, same config).

Kept so existing invocations and docs keep working; new workloads/suites land
in e2e.py + scripts/verify_config.json.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from e2e import main  # noqa: E402

if __name__ == "__main__":
    main()
