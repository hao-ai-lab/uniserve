# Shortcuts for local checks and tests. The README's `uv sync` builds the
# environment; serving is `uniserve serve <model> --served-model-name <name>`.

python := env("PYTHON", ".venv/bin/python")
cargo := env("CARGO", "cargo")

# Format check (Rust)
fmt:
    {{cargo}} fmt --all -- --check

# Clippy lint (Rust)
clippy:
    {{cargo}} clippy --workspace --all-targets -- -D warnings

# Run Rust tests
test-rust:
    {{cargo}} test --workspace

# Build benchmarks without running
bench-build:
    {{cargo}} bench --workspace --no-run

# Lint: fmt + clippy + ruff + mypy
lint: fmt clippy
    {{python}} -m ruff check uniserve uniserve_models uniserve_worker uniserve_eval uniserve_kernels/src tests
    {{python}} -m ruff format --check uniserve uniserve_models uniserve_worker uniserve_eval uniserve_kernels/src tests
    {{python}} -m mypy

# Fast Python tests (CPU only)
test-python-fast:
    {{python}} -m pytest -m "(unit or architecture) and not gpu"

# Integration Python tests (CPU only)
test-python-integration:
    {{python}} -m pytest -m "integration and not gpu"

# Unit and integration Python tests that require a CUDA device
test-python-cuda:
    {{python}} -m pytest -m "(unit or integration) and gpu"

# Build the debug binary if missing
[private]
build-debug:
    {{cargo}} build -p uniserve

# End-to-end Python tests (no GPU)
test-python-e2e: build-debug
    {{python}} -m pytest -m "e2e and not gpu"

# End-to-end Python tests that need GPUs and the model checkpoints they name
test-python-gpu: build-debug
    UNISERVE_RUN_GPU_E2E=1 {{python}} -m pytest -m "e2e and gpu"

# Run all checks
test-all: fmt clippy test-rust bench-build test-python-fast test-python-integration test-python-e2e

# Resolve an evaluation suite without starting a server
eval-plan config="uniserve_eval/fast_h3.toml" suite="gb200-4-bf16":
    {{python}} -m uniserve_eval --config {{config}} plan {{suite}}
