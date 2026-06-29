from .i2i import I2ITask
from .interleave import InterleaveTask
from .t2i import T2ITask
from .text import TextTask

TASKS = {
    "text": TextTask,
    "t2i": T2ITask,
    "i2i": I2ITask,
    "interleave": InterleaveTask,
}

__all__ = ["TASKS", "I2ITask", "InterleaveTask", "T2ITask", "TextTask"]
