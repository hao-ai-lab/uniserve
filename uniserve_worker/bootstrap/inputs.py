"""Bind the worker's input staging to a model's declared capabilities.

The worker finds a model's capabilities (``ImageDenoiser``, ``VideoDenoiser``,
``VideoPostprocessor``, ...) by type within its ordinary module tree rather
than by concrete model identity, and builds the matching request-input
builder: ``ImageBuilder`` for image denoising and ``MediaBuilder`` for video.
"""

from torch import nn

from uniserve.model import ImageDenoiser, VideoDenoiser, VideoPostprocessor
from uniserve_worker.bootstrap.components import describe_components
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.model_executor.diffusion_inputs import ImageBuilder
from uniserve_worker.model_executor.media_inputs import MediaBuilder


def capability(model: nn.Module, kind: type[nn.Module]):
    """Find one capability within ordinary module composition.

    ``model.modules()`` yields a module shared under several paths once, so
    such a module counts as one capability.

    Returns:
        The single submodule (or ``model`` itself) that is an instance of
        ``kind``, or ``None`` when there is none.

    Raises:
        ValueError: More than one distinct submodule is an instance of
            ``kind``.
    """
    values = tuple(
        module for module in model.modules() if isinstance(module, kind)
    )
    if len(values) > 1:
        raise ValueError(
            f"worker requires an unambiguous {kind.__name__} capability"
        )
    return None if not values else values[0]


def image_builder(model: nn.Module):
    """Instantiate the image-denoising input builder or ``None`` without one."""
    denoiser = capability(model, ImageDenoiser)
    return None if denoiser is None else ImageBuilder(denoiser)


def video_denoiser(model: nn.Module, config: WorkerConfig):
    """Find the video denoiser the deployment places.

    A checkpoint may hold several video denoisers, one per task family; a
    deployment places exactly one of them, and every worker of it serves
    that one. With a single candidate it is the denoiser; with several, the
    deployment's components (``config.deployment_components``) name the
    placed one through their entry points.

    Returns:
        The placed denoiser, or ``None`` for a model without one.

    Raises:
        ValueError: The deployment places several of the model's video
            denoisers, or none of several.
    """
    candidates = tuple(
        dict.fromkeys(
            module
            for module in model.modules()
            if isinstance(module, VideoDenoiser)
        )
    )
    if len(candidates) <= 1:
        return candidates[0] if candidates else None

    declared = describe_components(model)
    placed = tuple(
        dict.fromkeys(
            call.module
            for name in config.deployment_components
            for call in declared.get(name, ())
            if any(call.module is candidate for candidate in candidates)
        )
    )
    if len(placed) != 1:
        names = sorted(
            name
            for name, calls in declared.items()
            if any(
                call.module is candidate
                for call in calls
                for candidate in candidates
            )
        )
        raise ValueError(
            "a deployment places exactly one of the model's video "
            f"denoisers {names}; it places {len(placed)}"
        )
    return placed[0]


# Video tasks whose request calls the worker executes: generation
# conditioned on the prompt's text encoding alone.
EXECUTED_VIDEO_TASKS = frozenset({"t2va"})


def executed_video_tasks(denoiser: VideoDenoiser) -> tuple[str, ...]:
    """The denoiser's tasks the worker executes, in the denoiser's order."""
    return tuple(
        task for task in denoiser.tasks if task in EXECUTED_VIDEO_TASKS
    )


def condition_capacity(denoiser: VideoDenoiser, config: WorkerConfig) -> int:
    """Packed condition rows a worker provisions for one request.

    Only a conditioned task carries conditions, so a deployment whose
    denoiser executes none provisions none, whatever ``max_condition_rows``
    grants.
    """
    if not set(executed_video_tasks(denoiser)) - {"t2va"}:
        return 0
    return config.max_condition_rows


def media_builder(model: nn.Module, config: WorkerConfig):
    """Instantiate the video input builder within the worker's frame budget.

    The frame budget is ``max_video_seconds`` at the post-processor's output
    frame rate, rounded half to even to whole frames, and admitted frame
    counts start at ``min_video_seconds`` converted the same way; the text
    budget is ``max_sequence_tokens``, divided into
    ``video_text_capacities``.

    Returns:
        The builder, or ``None`` for a model without a ``VideoDenoiser``.

    Raises:
        ValueError: The model has a ``VideoDenoiser`` but no
            ``VideoPostprocessor`` to supply its frame rate, or either
            capability is ambiguous.
    """
    denoiser = video_denoiser(model, config)
    if denoiser is None:
        return None

    output = capability(model, VideoPostprocessor)
    if output is None:
        raise ValueError(
            "media input construction requires its output sampling clock"
        )

    # The server counts a duration's frames rounded half to even at the
    # output clock, so the capacity provisions exactly the frame count its
    # longest admitted request resolves to.
    return MediaBuilder(
        denoiser,
        max_frames=round(config.max_video_seconds * output.frame_rate),
        max_text_tokens=config.max_sequence_tokens,
        min_frames=1
        if config.min_video_seconds is None
        else round(config.min_video_seconds * output.frame_rate),
        text_capacities=config.video_text_capacities,
        condition_rows=condition_capacity(denoiser, config),
    )
