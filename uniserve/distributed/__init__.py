"""Borrowed tensor communication and mathematical distribution."""

from .distribution import Distribution
from .mesh import Communicator, DeviceMesh

__all__ = [
    "Distribution",
    "Communicator",
    "DeviceMesh",
    "parallelize_",
    "partition_experts",
    "communication_axes",
    "communicators",
]


def __getattr__(name):
    if name in {
        "parallelize_",
        "partition_experts",
        "communication_axes",
        "communicators",
    }:
        from . import parallelize

        return getattr(parallelize, name)
    raise AttributeError(name)
