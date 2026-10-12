"""Reusable execution of model capabilities over bound numerical resources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Generic, TypeVar

import torch
from torch import nn

from uniserve.diffusion import DenoisingStep, Schedule
from uniserve.model import Denoiser, DenoiserInput
from uniserve.runtime import ExecutionContext

ModelT = TypeVar("ModelT", bound=nn.Module)
DenoiserInputT = TypeVar("DenoiserInputT", bound=DenoiserInput)
SizeT = TypeVar("SizeT")
ResultT = TypeVar("ResultT")


class ModelRunner(Generic[ModelT, SizeT]):
    """Run one numerical module through its already-bound execution context.

    The model and context are borrowed; the context's owner decides their
    lifetime, and calls fail once that context is closed. The runner connects
    the caller's CUDA stream to the context stream without exposing stream
    policy to the model.
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

    def warmup(self, size: SizeT) -> None:
        """Prepare numerical operators and workspace for a maximum size.

        Preparation replaces the context's previous capacity. Every later call
        must fit within ``size``; preparing again is needed only to change it.
        """
        self.context.prepare(size)

    def _run(self, call: Callable[..., ResultT], *args, **kwargs) -> ResultT:
        """Run on the bound stream.

        Connect the result to the caller's stream.
        """
        owner = self.context.stream
        caller = None
        if owner is not None:
            caller = torch.cuda.current_stream(owner.device)
            owner.wait(caller)

        with self.context.activate():
            result = call(*args, **kwargs)

        if owner is not None and caller is not None and caller != owner.stream:
            caller.wait_stream(owner.stream)
        return result


class DenoisingRunner(
    ModelRunner[Denoiser[DenoiserInputT, SizeT], SizeT],
    Generic[DenoiserInputT, SizeT],
):
    """Prepare a request's state and run its denoising steps.

    The context is prepared for a layout, any size that holds the request
    (``Denoiser.holds``; ``Denoiser.layout_size`` is the smallest), and steps
    evaluate that layout. ``prepare_latents`` fills the samples from native
    draws and ``prepare_state`` the request's remaining state fields from its
    exact size in that layout; the caller stages both into the state views it
    passes to ``step``. The runner holds no request policy.
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

    def prepare_state(
        self,
        sizes: tuple[SizeT, ...],
        *,
        layouts: tuple[SizeT, ...],
        out: Mapping[str, torch.Tensor],
    ) -> None:
        """Fill host views of the state fields other than the samples.

        Each size is evaluated in the aligned layout, which must hold it
        (``Denoiser.holds``).
        """
        self.model.prepare_state(sizes, layouts=layouts, out=out)

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
