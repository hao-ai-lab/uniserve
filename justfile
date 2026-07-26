# Shortcuts for local checks and tests. Build is `pip install` / `cargo`;
# serving is `uniserve serve <model>`.

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
    {{python}} -m ruff check uniserve_worker uniserve_eval tests/python
    {{python}} -m mypy uniserve_worker uniserve_eval

# Fast Python tests (unit / contract / architecture)
test-python-fast:
    {{python}} -m pytest -m "unit or contract or architecture"

# Integration Python tests
test-python-integration:
    {{python}} -m pytest -m integration

# Build the debug binary if missing
[private]
build-debug:
    {{cargo}} build -p uniserve-cli

# End-to-end Python tests (no GPU)
test-python-e2e: build-debug
    {{python}} -m pytest -m "e2e and not gpu"

# End-to-end Python tests (GPU, sensenova)
test-python-gpu: build-debug
    UNISERVE_RUN_GPU_E2E=1 {{python}} -m pytest -m "e2e and gpu and sensenova"

# Run all checks
test-all: fmt clippy test-rust bench-build test-python-fast test-python-integration test-python-e2e

# Quick benchmark smoke test
bench-smoke: build-debug
    {{python}} -m uniserve_eval.harness.cli \
        --base-url http://127.0.0.1:18080 \
        --model sim-model \
        --task interleave \
        --dataset trace \
        --dataset-path uniserve_eval/data/smoke_trace.jsonl \
        --output-dir results/benchmarks/smoke \
        --smoke
