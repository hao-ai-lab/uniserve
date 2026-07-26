"""Serving verification and benchmarking for UniServe.

One package owns the whole stack: profile-driven server lifecycle
(``backends``), native-SSE correctness gates (``verify``), and the HTTP-native
measurement harness (``harness``) that produces canonical benchmark point
artifacts and cross-runtime comparisons. The CLI entry point is
``uniserve-eval`` (or ``python -m uniserve_eval``); profiles live in
``profiles.json`` next to this file.
"""
