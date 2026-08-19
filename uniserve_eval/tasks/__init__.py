from . import _registry
from .base import BenchmarkTask
from .i2i import I2ITask
from .i2t import I2TTask
from .interleave import InterleaveTask
from .t2i import T2ITask
from .text import TextTask

_ = _registry

__all__ = [
    "BenchmarkTask",
    "I2ITask",
    "I2TTask",
    "InterleaveTask",
    "T2ITask",
    "TextTask",
]
