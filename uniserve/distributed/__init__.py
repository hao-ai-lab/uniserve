"""Borrowed tensor communication and mathematical distribution."""

from .distribution import Distribution
from .mesh import Communicator, DeviceMesh

__all__ = [
    "Distribution",
    "Communicator",
    "DeviceMesh",
    "parallelize_",
]


def __getattr__(name):
    if name == "parallelize_":
        from .parallelize import parallelize_

        return parallelize_
    raise AttributeError(name)
