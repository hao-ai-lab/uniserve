#!/usr/bin/env python3
"""Smoke-test the standalone ``uniserve-kernel`` provider package."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "uniserve_kernel"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="uniserve-kernel-smoke-") as tmp:
        tmp_dir = Path(tmp)
        venv_dir = tmp_dir / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = _venv_python(venv_dir)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "--no-deps", "-e", str(PROVIDER)],
            cwd=tmp_dir,
            env=env,
        )
        probe = """
import json
from pathlib import Path
import uniserve_kernel
from uniserve_kernel import mm_attn_varlen
print(json.dumps({
    "package_file": str(Path(uniserve_kernel.__file__).resolve()),
    "module_file": str(Path(mm_attn_varlen.__file__).resolve()),
    "available": bool(mm_attn_varlen.available()),
    "import_error": None if mm_attn_varlen.import_error() is None else str(mm_attn_varlen.import_error()),
}))
"""
        raw = subprocess.check_output([str(python), "-c", probe], cwd=tmp_dir, env=env, text=True)
        report = json.loads(raw)

    package_file = Path(report["package_file"])
    module_file = Path(report["module_file"])
    root_bridge = ROOT / "uniserve_kernel" / "__init__.py"
    root_bridge_module = ROOT / "uniserve_kernel" / "mm_attn_varlen.py"
    provider_python = (PROVIDER / "python" / "uniserve_kernel").resolve()
    if root_bridge.exists() or root_bridge_module.exists():
        raise SystemExit(
            "provider pack must not expose a checkout-root compatibility bridge: "
            f"{root_bridge}, {root_bridge_module}"
        )
    if not package_file.is_relative_to(provider_python):
        raise SystemExit(f"provider import did not resolve through packaged source tree: {report}")
    if not module_file.is_relative_to(provider_python):
        raise SystemExit(f"provider module did not resolve through packaged source tree: {report}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
