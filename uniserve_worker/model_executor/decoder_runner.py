"""Decoder numerical calls and public output layouts."""

from uniserve.model import AudioDecoder, VideoPostprocessor
from uniserve.tensors import OutputLayout

from .model_runner import ModelRunner
from .output import ExecutionOutput


class DecoderRunner(ModelRunner):
    """Decode prepared latent tensors with the capability's workspace views."""

    def resources(self):
        """Context views ``execute_model`` passes to direct decoder calls."""
        # A video decoder's windows arrive unpacked, so its calls borrow no
        # context views.
        if isinstance(self.model, VideoPostprocessor):
            return {
                "constants": self.context.constants,
                "workspace": self.context.workspace,
            }
        if isinstance(self.model, AudioDecoder):
            return {"workspace": self.context.workspace}
        return {}

    def batch_forward(self, batch, *, padded=False):
        values = self.model.decode(
            batch.inputs.latents, sizes=batch.inputs.sizes
        )
        return ExecutionOutput(
            values,
            layouts=tuple(
                OutputLayout(
                    tuple(value.shape),
                    value.dtype,
                    tuple(slice(0, extent) for extent in value.shape),
                    value_range=self.model.decoder.value_range,
                )
                for value in values
            ),
        )
