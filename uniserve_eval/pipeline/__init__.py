"""Exposes benchmark execution and result reporting."""

from .report import build_summary, render_markdown
from .run import run_point

__all__ = [
    "build_summary",
    "render_markdown",
    "run_point",
]
