"""Nsight Systems lifecycle for one managed evaluator point."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import ServerLaunch
from .nsys_timeline import transform_timeline


class NsysCapture:
    """Delay collection through warmup, then finalize one managed-server report."""

    def __init__(self, point_name: str, output_dir: Path, *, trace_cuda: bool = True) -> None:
        executable = shutil.which("nsys")
        if executable is None:
            raise RuntimeError("Nsight Systems is required for --nsys")
        self.executable = executable
        self.point_name = point_name
        self.output_dir = Path(output_dir)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", point_name).strip("_") or "point"
        self.session = f"uniserve_eval_{os.getpid()}_{slug}"
        self.report_prefix = self.output_dir / "trace"
        self.log_path = self.output_dir / "nsys.log"
        self.trace_cuda = trace_cuda
        self._started = False
        self._stopped = False

    def wrap_launch(self, launch: ServerLaunch) -> ServerLaunch:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"nsys result directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace = "cuda,nvtx,osrt,cublas,cudnn" if self.trace_cuda else "nvtx,osrt"
        cuda_options = (
            (
                "--cuda-graph-trace=node",
                "--cuda-event-trace=false",
            )
            if self.trace_cuda
            else ()
        )
        command = (
            self.executable,
            "profile",
            "--start-later=true",
            f"--session-new={self.session}",
            "--force-overwrite=true",
            f"--output={self.report_prefix}",
            f"--trace={trace}",
            *cuda_options,
            "--sample=process-tree",
            "--cpuctxsw=process-tree",
            "--wait=all",
            "--show-output=true",
            *launch.command,
        )
        environment = dict(launch.environment)
        environment["UNISERVE_NVTX"] = "1"
        return ServerLaunch(command, launch.working_directory, environment)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("nsys measurement window was started more than once")
        self._run("start", f"--session={self.session}")
        self._started = True

    def stop(self) -> None:
        if not self._started or self._stopped:
            raise RuntimeError("nsys measurement window is not active")
        self._run("stop", f"--session={self.session}")
        self._stopped = True

    def finalize(self) -> None:
        if not self._started:
            return
        if not self._stopped:
            raise RuntimeError("nsys measurement window did not close")
        reports = sorted(self.output_dir.glob("*.nsys-rep"))
        if len(reports) != 1:
            raise RuntimeError(
                f"expected one finalized nsys report in {self.output_dir}, found {len(reports)}"
            )
        report = reports[0]
        exported = self.output_dir / "trace.sqlite"
        self._run(
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            f"--output={exported}",
            str(report),
        )
        timeline = self.output_dir / "timeline.sqlite"
        transform_timeline(
            exported,
            timeline,
            point_name=self.point_name,
            require_cuda=self.trace_cuda,
        )
        manifest = {
            "point": self.point_name,
            "session": self.session,
            "report": report.name,
            "sqlite": exported.name,
            "timeline": timeline.name,
        }
        (self.output_dir / "capture.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def describe(self) -> dict[str, object]:
        return {
            "session": self.session,
            "output_directory": str(self.output_dir),
            "trace": (
                ["cuda", "nvtx", "osrt", "cublas", "cudnn"] if self.trace_cuda else ["nvtx", "osrt"]
            ),
            "cuda_graph_trace": "node" if self.trace_cuda else None,
            "measurement_window": "post-warmup load",
        }

    def _run(self, *arguments: str) -> None:
        result = subprocess.run(
            [self.executable, *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {self.executable} {' '.join(arguments)}\n")
            handle.write(result.stdout)
            if result.stdout and not result.stdout.endswith("\n"):
                handle.write("\n")
        if result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-12:])
            raise RuntimeError(
                f"nsys {' '.join(arguments[:1])} failed with code {result.returncode}:\n{tail}"
            )
