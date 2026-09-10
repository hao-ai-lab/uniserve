"""Profiler artifacts belong to the evaluator across executable working directories."""

import subprocess
from pathlib import Path

from uniserve_eval.config import ServerLaunch
from uniserve_eval.nsys import NsysCapture


def test_capture_writes_to_requested_directory_from_another_checkout(tmp_path, monkeypatch):
    executable = tmp_path / "nsys"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "target = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--output='))\n"
        "Path(target).with_suffix('.nsys-rep').write_bytes(b'captured')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name: str(executable))
    monkeypatch.chdir(tmp_path)
    checkout = tmp_path / "server"
    checkout.mkdir()
    capture = NsysCapture("request", Path("results"))
    launch = capture.wrap_launch(ServerLaunch(("server",), checkout, {}))

    subprocess.run(launch.command, cwd=launch.working_directory, check=True)

    assert (tmp_path / "results" / "trace.nsys-rep").read_bytes() == b"captured"
