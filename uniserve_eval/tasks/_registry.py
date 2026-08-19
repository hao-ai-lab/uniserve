from ..registry import TaskSpec, register_task
from ..types import CHAT_COMPLETIONS, IMAGES_GENERATIONS, TaskName
from .i2i import I2ITask
from .i2t import I2TTask
from .interleave import InterleaveTask
from .t2i import T2ITask
from .text import TextTask

register_task(
    TaskSpec(
        name=TaskName.TEXT,
        allowed_endpoints=(CHAT_COMPLETIONS,),
        default_endpoint=CHAT_COMPLETIONS,
        default_stream=True,
        accepts_image=False,
        accepts_question=False,
        requires_image_count=False,
        forbids_image_count=True,
        factory=TextTask,
    )
)
register_task(
    TaskSpec(
        name=TaskName.T2I,
        allowed_endpoints=(CHAT_COMPLETIONS, IMAGES_GENERATIONS),
        default_endpoint=CHAT_COMPLETIONS,
        default_stream=False,
        accepts_image=True,
        accepts_question=False,
        requires_image_count=True,
        forbids_image_count=False,
        factory=T2ITask,
    )
)
register_task(
    TaskSpec(
        name=TaskName.I2I,
        allowed_endpoints=(CHAT_COMPLETIONS,),
        default_endpoint=CHAT_COMPLETIONS,
        default_stream=False,
        accepts_image=True,
        accepts_question=False,
        requires_image_count=False,
        forbids_image_count=False,
        factory=I2ITask,
    )
)
register_task(
    TaskSpec(
        name=TaskName.I2T,
        allowed_endpoints=(CHAT_COMPLETIONS,),
        default_endpoint=CHAT_COMPLETIONS,
        default_stream=True,
        accepts_image=False,
        accepts_question=True,
        requires_image_count=False,
        forbids_image_count=True,
        factory=I2TTask,
    )
)
register_task(
    TaskSpec(
        name=TaskName.INTERLEAVE,
        allowed_endpoints=(CHAT_COMPLETIONS,),
        default_endpoint=CHAT_COMPLETIONS,
        default_stream=True,
        accepts_image=True,
        accepts_question=False,
        requires_image_count=False,
        forbids_image_count=True,
        factory=InterleaveTask,
    )
)
