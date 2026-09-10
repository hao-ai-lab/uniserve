#!/usr/bin/env bash
# Profile a complete worker process with Python's standard call profiler.
set -euo pipefail

profile_dir=${UNISERVE_DIAGNOSTIC_DIR:?Set UNISERVE_DIAGNOSTIC_DIR to the output directory}
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p -- "$profile_dir"
exec "$project_dir/.venv/bin/python" -m cProfile -o "$profile_dir/worker-$$.pstats" "$@"
