"""Concrete asynchronous transfer owners.

Transport backends publish immutable tensors as locators and serve bounded
asynchronous reads; transfer tickets track each read from stream readiness
through physical retirement of the storage it touched.
"""
