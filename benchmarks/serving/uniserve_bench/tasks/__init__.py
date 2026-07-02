from .i2i import I2ITask
from .i2t import I2TTask
from .interleave import InterleaveTask
from .t2i import T2ITask
from .text import TextTask

TASKS = {
    "text": TextTask,
    "t2i": T2ITask,
    "i2i": I2ITask,
    "i2t": I2TTask,
    "interleave": InterleaveTask,
}

__all__ = ["TASKS", "I2ITask", "I2TTask", "InterleaveTask", "T2ITask", "TextTask"]
