"""Borrowed tensor communication and caller-owned process-group resources."""

from .mesh import Communicator, DeviceMesh
from .parallel import ParallelConfig, SequenceParallel

__all__ = ["Communicator", "DeviceMesh", "ParallelConfig", "SequenceParallel"]
