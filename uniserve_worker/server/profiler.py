"""Worker-side per-step profiling capture controller.

The profiler is intentionally controlled by environment variables so production
hot paths stay unchanged unless a profiling run asks for traces. The hot-path
span primitive (``profile_range``) lives in ``foundation/profiling.py``; this
module owns the stateful per-execute-step capture/export orchestration.
"""
from __future__ import annotations

import inspect
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..foundation.env import flag_from_value, int_from_value
from ..foundation.profiling import _parse_activities, profile_range

torch: Any | None
try:  # torch is an optional import for CPU-only control-plane tests.
    import torch as _torch_module
except Exception:  # pragma: no cover - exercised only in torch-free envs.
    torch = None
else:  # pragma: no cover
    torch = _torch_module

__all__ = ["WorkerProfiler", "WorkerProfileConfig"]

logger = logging.getLogger(__name__)

_PROFILE_DIR_ENV = "UNISERVE_PROFILE_DIR"
_TORCH_PROFILE_DIR_ENV = "UNISERVE_TORCH_PROFILER_DIR"
_PROFILE_ACTIVITIES_ENV = "UNISERVE_PROFILE_ACTIVITIES"
_PROFILE_START_STEP_ENV = "UNISERVE_PROFILE_START_STEP"
_PROFILE_STEPS_ENV = "UNISERVE_PROFILE_STEPS"
_PROFILE_PREFIX_ENV = "UNISERVE_PROFILE_PREFIX"
_PROFILE_WITH_STACK_ENV = "UNISERVE_PROFILE_WITH_STACK"
_PROFILE_RECORD_SHAPES_ENV = "UNISERVE_PROFILE_RECORD_SHAPES"
_CUDA_PROFILER_ENV = "UNISERVE_CUDA_PROFILER"


@dataclass(frozen=True)
class WorkerProfileConfig:
    output_dir: Path
    prefix: str
    activities: tuple[str, ...]
    start_step: int
    num_steps: int
    with_stack: bool
    record_shapes: bool
    cuda_profiler: bool


class WorkerProfiler:
    """Small execution-step profiler for worker processes."""

    def __init__(self, config: WorkerProfileConfig | None) -> None:
        self.config = config
        self._seen_steps = 0
        self._profiled_steps = 0
        self._start_step = 0
        self._active = False
        self._finished = False
        self._torch_profiler: Any | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "WorkerProfiler":
        env = os.environ if env is None else env
        output_dir = env.get(_TORCH_PROFILE_DIR_ENV) or env.get(_PROFILE_DIR_ENV)
        if not output_dir:
            return cls(None)
        activities = _parse_activities(env.get(_PROFILE_ACTIVITIES_ENV, "CPU,GPU"))
        cuda_profiler = "CUDA_PROFILER" in activities or flag_from_value(env.get(_CUDA_PROFILER_ENV))
        activities = tuple(activity for activity in activities if activity != "CUDA_PROFILER")
        config = WorkerProfileConfig(
            output_dir=Path(output_dir),
            prefix=env.get(_PROFILE_PREFIX_ENV, "uniserve-worker") or "uniserve-worker",
            activities=activities,
            start_step=max(1, int_from_value(env.get(_PROFILE_START_STEP_ENV), default=1)),
            num_steps=max(1, int_from_value(env.get(_PROFILE_STEPS_ENV), default=1)),
            with_stack=flag_from_value(env.get(_PROFILE_WITH_STACK_ENV)),
            record_shapes=flag_from_value(env.get(_PROFILE_RECORD_SHAPES_ENV)),
            cuda_profiler=cuda_profiler,
        )
        return cls(config)

    @property
    def enabled(self) -> bool:
        return self.config is not None

    @contextmanager
    def step(self, debug_name: str) -> Iterator[None]:
        if self.config is None:
            with profile_range(debug_name):
                yield
            return

        self._seen_steps += 1
        started_for_step = False
        if (
            not self._active
            and not self._finished
            and self._seen_steps >= self.config.start_step
        ):
            started_for_step = self._start(self._seen_steps)

        was_active = self._active
        try:
            with profile_range(debug_name):
                yield
        finally:
            if was_active:
                self._profiled_steps += 1
                if self._profiled_steps >= self.config.num_steps:
                    self._stop()
            elif started_for_step and self._active:
                self._stop()

    def close(self) -> None:
        if self._active:
            self._stop()

    def _start(self, step_id: int) -> bool:
        assert self.config is not None
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._start_step = int(step_id)
        self._profiled_steps = 0
        self._active = True
        try:
            torch_activities = _torch_profiler_activities(self.config.activities)
            if torch_activities and torch is not None:
                kwargs = {
                    "activities": torch_activities,
                    "with_stack": self.config.with_stack,
                    "record_shapes": self.config.record_shapes,
                }
                if _accepts_torch_profiler_arg("acc_events"):
                    kwargs["acc_events"] = True
                profiler = torch.profiler.profile(**kwargs)
                profiler.start()
                self._torch_profiler = profiler
            if self.config.cuda_profiler:
                _cuda_profiler_start()
        except Exception:
            logger.exception("failed to start UniServe worker profiler; disabling this capture")
            self._torch_profiler = None
            self._active = False
            self._finished = True
            return False
        logger.info(
            "UniServe worker profiler started at execute step %s for %s step(s); output_dir=%s",
            step_id,
            self.config.num_steps,
            self.config.output_dir,
        )
        return True

    def _stop(self) -> None:
        assert self.config is not None
        end_step = self._start_step + max(0, self._profiled_steps - 1)
        trace_base = _trace_base(self.config, self._start_step, end_step)
        try:
            if self.config.cuda_profiler:
                _cuda_profiler_stop()
            if self._torch_profiler is not None:
                self._torch_profiler.stop()
                trace_path = self.config.output_dir / f"{trace_base}.trace.json.gz"
                self._torch_profiler.export_chrome_trace(str(trace_path))
                summary_path = self.config.output_dir / f"{trace_base}.summary.txt"
                summary_path.write_text(
                    _profiler_table(
                        self._torch_profiler,
                        prefer_cuda="GPU" in self.config.activities,
                    ),
                    encoding="utf-8",
                )
                logger.info("UniServe worker profiler trace written to %s", trace_path)
        except Exception:
            logger.exception("failed to stop/export UniServe worker profiler")
        finally:
            self._torch_profiler = None
            self._active = False
            self._finished = True


def _trace_base(config: WorkerProfileConfig, start_step: int, end_step: int) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    rank_part = f"-rank{rank}" if rank is not None else ""
    return f"{config.prefix}-pid{os.getpid()}{rank_part}-steps{start_step}-{end_step}-{stamp}"


def _torch_profiler_activities(activities: tuple[str, ...]):
    if torch is None:
        return []
    out = []
    if "CPU" in activities:
        out.append(torch.profiler.ProfilerActivity.CPU)
    if "GPU" in activities:
        if torch.cuda.is_available():
            out.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            logger.warning("UNISERVE_PROFILE_ACTIVITIES requested GPU but CUDA is unavailable")
    return out


def _accepts_torch_profiler_arg(name: str) -> bool:
    if torch is None:
        return False
    try:
        return name in inspect.signature(torch.profiler.profile).parameters
    except (TypeError, ValueError):
        return False


def _cuda_profiler_start() -> None:
    if torch is None or not torch.cuda.is_available():
        logger.warning("CUDA profiler requested but CUDA is unavailable")
        return
    torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop() -> None:
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.cudart().cudaProfilerStop()


def _profiler_table(profiler, *, prefer_cuda: bool) -> str:
    sort_keys = (
        ("cuda_time_total", "self_cuda_time_total", "cpu_time_total", "self_cpu_time_total")
        if prefer_cuda
        else ("cpu_time_total", "self_cpu_time_total", "cuda_time_total", "self_cuda_time_total")
    )
    for sort_by in sort_keys:
        try:
            return profiler.key_averages().table(sort_by=sort_by, row_limit=120)
        except Exception:
            continue
    return profiler.key_averages().table(row_limit=120)
