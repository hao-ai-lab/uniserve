"""Decoder numerical calls and public output layouts."""

from typing import cast

from uniserve.model import AudioDecoder, ImageDecoder, VideoPostprocessor
from uniserve.tensors import OutputLayout

from .model_runner import ModelRunner
from .output import ExecutionOutput


class DecoderRunner(ModelRunner):
    """Decode prepared latent tensors with the capability's workspace views."""

    def resources(self):
        """Context views supplied to direct numerical decoder calls."""
        # A video decoder's windows arrive unpacked, so its calls borrow no
        # context views.
        if isinstance(self.model, VideoPostprocessor):
            return {
                "constants": self.execution.context.constants,
                "workspace": self.execution.context.workspace,
            }
        if isinstance(self.model, AudioDecoder):
            return {"workspace": self.execution.context.workspace}
        return {}

    def batch_forward(self, batch, *, padded=False):
        decoder = cast(ImageDecoder, self.model)
        values = decoder.decode(batch.inputs.latents, sizes=batch.inputs.sizes)
        return ExecutionOutput(
            values,
            layouts=tuple(
                OutputLayout(
                    tuple(value.shape),
                    value.dtype,
                    tuple(slice(0, extent) for extent in value.shape),
                    value_range=decoder.decoder.value_range,
                )
                for value in values
            ),
        )
