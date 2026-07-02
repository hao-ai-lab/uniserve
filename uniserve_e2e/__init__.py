"""End-to-end verification, benchmarking, and comparison for UniServe serving.

One package owns the whole stack: profile-driven server lifecycle
(``backends``), native-SSE correctness gates (``verify``), perf points with
metric bounds (``perf``), cross-run comparisons (``compare``), and the
HTTP-native measurement harness (``harness``). The CLI entry point is
``uniserve-e2e`` (or ``python -m uniserve_e2e``); profiles live in
``profiles.json`` next to this file.
"""
