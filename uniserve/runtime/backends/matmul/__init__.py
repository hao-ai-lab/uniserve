"""Local matrix-multiplication providers.

:func:`resolve` selects the ``torch``, ``cublas`` or ``flashinfer`` provider;
each provider's ``prepare`` returns an independent :class:`Operator`.
"""

from .base import Backend, MergedOperator, Operator, resolve

__all__ = ["Backend", "MergedOperator", "Operator", "resolve"]
