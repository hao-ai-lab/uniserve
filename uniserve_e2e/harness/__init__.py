"""HTTP-native UniServe serving performance benchmark harness.

Performance-only, model-agnostic. Four workloads over four API endpoints:

* ``text``       -> ``/v1/chat/completions``   (ShareGPT)  -- Family A metrics
* ``interleave`` -> ``/generate``              (UEval)     -- Family A metrics
* ``t2i``        -> ``/v1/images/generations`` (MJHQ-30K)  -- Family B metrics
* ``i2i``        -> ``/generate``              (PIE-Bench) -- Family B metrics

Family A = streaming token metrics (TTFT/TPOT/ITL/E2E + throughput), matching
``refs/sglang`` for LLM serving. Family B = per-image latency + image throughput.
"""

from .runner import BenchmarkRunner, RunResult
from .spec import BenchmarkSpec, TaskName

__all__ = ["BenchmarkRunner", "BenchmarkSpec", "RunResult", "TaskName"]
