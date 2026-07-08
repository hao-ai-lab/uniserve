"""HTTP-native UniServe serving performance benchmark harness.

Performance-only, model-agnostic. Five tasks, each measured over one of the
wire shapes declared in ``spec.TASK_WIRES`` (endpoint derived per wire):

* ``text``       (ShareGPT)   -- Family A metrics; OpenAI chat SSE
* ``interleave`` (UEval)      -- Family A metrics; native SSE or OpenAI chat SSE
* ``i2t``        (image dirs) -- Family A metrics; native SSE, OpenAI chat SSE,
  or one non-streamed chat JSON (diffusion-pipeline backends)
* ``t2i``        (MJHQ-30K)   -- Family B metrics; images-generations JSON or
  image-only chat JSON
* ``i2i``        (PIE-Bench)  -- Family B metrics; native SSE

Family A = streaming token metrics (TTFT/TPOT/ITL/E2E + throughput), matching
``refs/sglang`` for LLM serving. Family B = per-image latency + image throughput.
"""

from .runner import BenchmarkRunner, RunResult
from .spec import BenchmarkSpec, TaskName

__all__ = ["BenchmarkRunner", "BenchmarkSpec", "RunResult", "TaskName"]
