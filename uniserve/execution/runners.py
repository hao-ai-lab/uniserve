"""Reusable execution of model capabilities over bound numerical resources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Generic, TypeVar

import torch
from torch import nn

from uniserve.diffusion import DenoisingStep, Schedule
from uniserve.media import image
from uniserve.model import (
    AudioDecoder,
    CausalLM,
    Denoiser,
    DenoiserInput,
    Encoder,
    ImageDecoder,
    TextInput,
    TextSize,
    VideoDecoder,
    VideoPostprocessor,
)
from uniserve.model.logits import Logits
from uniserve.nn.vae import PatchAutoencoder
from uniserve.runtime import ExecutionContext
from uniserve.tensors import TensorOutput

ModelT = TypeVar("ModelT", bound=nn.Module)
InputT = TypeVar("InputT")
DenoiserInputT = TypeVar("DenoiserInputT", bound=DenoiserInput)
SizeT = TypeVar("SizeT")
ResultT = TypeVar("ResultT")


class ModelRunner(Generic[ModelT, SizeT]):
    """Run one numerical module through its already-bound execution context.

    The model and context are borrowed. The runner connects the caller's CUDA
    stream to the context stream without exposing stream policy to the model.
    """

    def __init__(
        self, model: ModelT, *, context: ExecutionContext[SizeT]
    ) -> None:
        if context.module is not model:
            raise ValueError(
                "execution context must be bound to the runner's model"
            )
        self.model = model
        self.context = context
        self.closed = False

    def warmup(self, size: SizeT) -> None:
        """Prepare numerical operators and workspace.

        For the requested maximum size.
        """
        self._ensure_open()
        self.context.prepare(size)

    def _run(self, call: Callable[..., ResultT], *args, **kwargs) -> ResultT:
        """Run on the bound stream.

        Connect the result to the caller's stream.
        """
        self._ensure_open()
        stream = self.context.stream
        caller = None
        if stream is not None:
            with torch.cuda.device(stream.device):
                caller = torch.cuda.current_stream(stream.device)
                if caller != stream:
                    stream.wait_stream(caller)

        with self.context.activate():
            result = call(*args, **kwargs)

        if caller is not None and caller != stream:
            caller.wait_stream(stream)
        return result

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("model runner is closed")

    def close(self) -> None:
        """Reject future calls after callers have finished borrowed outputs."""
        if self.closed:
            return
        self.closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"Model runner cleanup also failed: {error!r}")


class TextRunner(ModelRunner[CausalLM, TextSize]):
    """Run a causal model while binding its numerical attention input."""

    def __init__(
        self, model: CausalLM, *, context: ExecutionContext[TextSize]
    ) -> None:
        super().__init__(model, context=context)
        self.inputs = None

    def forward(self, inputs: TextInput) -> torch.Tensor:
        def call() -> torch.Tensor:
            self.context.bind_attention(inputs.attention)
            return self.model(inputs)

        return self._run(call)

    def compute_logits(
        self, hidden: torch.Tensor, *, token_indices: torch.Tensor
    ) -> Logits | None:
        return self._run(
            self.model.compute_logits, hidden, token_indices=token_indices
        )


class EncoderRunner(
    ModelRunner[Encoder[InputT], SizeT], Generic[InputT, SizeT]
):
    """Run a homogeneous encoder through prepared numerical resources."""

    def encode(
        self, inputs: InputT, *, size: SizeT
    ) -> tuple[torch.Tensor, ...] | None:
        self.warmup(size)
        return self._run(self.model.encode, inputs)


class LatentRunner(ModelRunner[PatchAutoencoder, object]):
    """Encode raster pixels into canonical latent patches."""

    def encode(
        self,
        pixels: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self._run(self.model.encode, pixels, generator=generator)


class DenoisingRunner(
    ModelRunner[Denoiser[DenoiserInputT, SizeT], SizeT],
    Generic[DenoiserInputT, SizeT],
):
    """Prepare latent state and run one denoising step.

    Without request policy.
    """

    def __init__(
        self,
        model: Denoiser[DenoiserInputT, SizeT],
        *,
        context: ExecutionContext[SizeT],
    ) -> None:
        super().__init__(model, context=context)
        self.inputs: DenoiserInputT | None = None

    def prepare_latents(
        self,
        sizes: tuple[SizeT, ...],
        *,
        noise: Mapping[str, torch.Tensor],
        state: Mapping[str, torch.Tensor],
    ) -> None:
        self._run(
            self.model.prepare_latents,
            sizes,
            noise=noise,
            state=state,
            constants=self.context.constants,
            workspace=self.context.workspace,
        )

    def step(
        self,
        inputs: DenoiserInputT,
        schedules: Mapping[str, Schedule],
        *,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, tuple[torch.Tensor, ...]]:
        self.inputs = inputs
        call = DenoisingStep(
            self.model,
            inputs,
            schedules,
            state,
            self.context.constants,
            self.context.workspace,
        )
        return self._run(call)


class ImageRunner(ModelRunner[ImageDecoder, image.Config]):
    """Decode canonical image latents through prepared numerical resources."""

    def decode(
        self,
        latents: tuple[torch.Tensor, ...],
        *,
        sizes: tuple[image.Config, ...],
    ) -> tuple[torch.Tensor, ...]:
        return self._run(self.model.decode, latents, sizes=sizes)


class AudioRunner(ModelRunner[AudioDecoder, int]):
    """Decode audio latents using the context's prepared workspace."""

    def decode(
        self,
        latents: tuple[torch.Tensor, ...],
        *,
        num_samples: tuple[int, ...],
    ) -> tuple[torch.Tensor, ...]:
        return self._run(
            self.model.decode,
            latents,
            num_samples=num_samples,
            workspace=self.context.workspace,
        )


class VideoRunner(ModelRunner[VideoDecoder, int]):
    """Decode video latent windows using prepared constants and workspace."""

    def decode(
        self,
        latents: tuple[torch.Tensor, ...],
        *,
        frames: tuple[slice, ...],
        num_frames: tuple[int, ...],
    ) -> tuple[TensorOutput | None, ...]:
        return self._run(
            self.model.decode,
            latents,
            frames=frames,
            num_frames=num_frames,
            constants=self.context.constants,
            workspace=self.context.workspace,
        )


class VideoProcessor(ModelRunner[VideoPostprocessor, int]):
    """Postprocess decoded video windows using caller-owned temporal state."""

    def forward(
        self,
        segments: tuple[TensorOutput, ...],
        *,
        frames: tuple[slice, ...],
        num_frames: tuple[int, ...],
        state: Mapping[str, torch.Tensor],
    ) -> tuple[TensorOutput, ...]:
        return self._run(
            self.model,
            segments,
            frames=frames,
            num_frames=num_frames,
            state=state,
            constants=self.context.constants,
            workspace=self.context.workspace,
        )
