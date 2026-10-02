"""Generate one H3 video in eager library calls.

``generate`` runs a request through the same numerical modules serving uses:
the text encoder, the placed denoiser's conditioner and every solver step,
then the VAE decoders and the RGB post-processor. Every rank that holds a
bound component calls it with the same arguments (SPMD); a denoiser sharded
by sequence parallelism gathers its final samples before decoding. Serving
composes the same calls through its worker instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from uniserve.diffusion import normal_noise
from uniserve.execution import DenoisingRunner, EncoderRunner
from uniserve.media import image, video
from uniserve.model import LatentInput, TextSize
from uniserve.runtime import ExecutionContext, TensorBuffers
from uniserve.tensors import TensorOutput

from .inputs import DenoiserInput
from .model import Model


@dataclass(frozen=True, slots=True)
class Generation:
    """One generated video with its stereo track.

    Attributes:
        video_latent: [rows, 96] FP32 final video latent, packed order.
        audio_latent: [2 * frames, 32] FP32 final audio latent,
            channel-major.
        frames: [num_frames, height, width, 3] uint8 RGB frames.
        audio: [samples, 2] int16 PCM at the model's sample rate.
    """

    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    frames: torch.Tensor
    audio: torch.Tensor


def _gather(sample: torch.Tensor, rows: int, group) -> torch.Tensor:
    """Concatenate every sequence rank's canonical rows in rank order.

    Each rank holds one contiguous range of the ``rows``-row canonical
    sample and the ranges follow the ranks, so gathering shards padded to the
    whole sample and dropping each rank's padding restores the sample.
    """
    if group.size == 1:
        return sample
    width = rows
    padded = sample.new_zeros((width, sample.shape[1]))
    padded[: sample.shape[0]].copy_(sample)
    counts = torch.tensor([sample.shape[0]], device=sample.device)
    every = group.all_gather(counts)
    shards = group.all_gather(padded).view(group.size, width, -1)
    return torch.cat(
        [shards[rank, : int(every[rank])] for rank in range(group.size)]
    )


@torch.inference_mode()
def generate(
    model: Model,
    *,
    prompt_token_ids: tuple[int, ...],
    num_frames: int,
    canvas: image.Config,
    seed: int,
    component: str = "denoiser",
) -> Generation:
    """Generate a text-conditioned video and audio track eagerly.

    Args:
        model: A loaded H3 model holding the text encoder, ``component``'s
            denoiser, both decoders and the post-processor.
        prompt_token_ids: The presented prompt's token ids.
        num_frames: Output frames at 24 fps, of the form ``17 * n + 5``.
        canvas: The output raster, one the denoiser generates.
        seed: The request seed of the native CPU noise draw.
        component: The denoising component to run.

    Raises:
        ValueError: The denoiser rejects the size (a canvas it does not
            generate, a frame count it does not produce).
    """
    denoiser = getattr(model, component)
    device = next(denoiser.parameters()).device
    tokens = len(prompt_token_ids)

    with ExecutionContext(model.text_encoder) as context:
        runner = EncoderRunner(model.text_encoder, context=context)
        runner.warmup(TextSize(tokens, 1))
        features = runner.encode(
            (torch.tensor(prompt_token_ids, device=device),)
        )[0]

    size = denoiser.make_size(num_frames, tokens, canvas=canvas)
    layout = denoiser.layout_size(size)
    conditioning = torch.zeros(
        denoiser.text_condition_rows(layout),
        denoiser.text_condition_width,
        dtype=torch.bfloat16,
        device=device,
    )
    with ExecutionContext(denoiser.conditioner) as context:
        runner = EncoderRunner(denoiser.conditioner, context=context)
        runner.warmup(TextSize(tokens, 1))
        conditioning[:tokens].copy_(runner.encode((features,))[0])

    requirements = denoiser.state_buffers(layout)
    noise = {
        name: torch.empty(
            (1, *denoiser.noise_shape(name, layout)), dtype=torch.float32
        )
        for name in denoiser.modalities
    }
    normal_noise((seed,), out=tuple(noise.values()))
    with (
        ExecutionContext(denoiser) as context,
        TensorBuffers.allocate(requirements, device="cpu") as host,
        TensorBuffers.allocate(requirements, device=device) as backing,
    ):
        runner = DenoisingRunner(denoiser, context=context)
        runner.warmup(layout)
        request, staged = backing.view(requirements), host.view(requirements)
        runner.prepare_latents(
            (layout,),
            noise=noise,
            state={
                name: staged[name].unsqueeze(0) for name in denoiser.modalities
            },
        )
        runner.prepare_state(
            (size,),
            layouts=(layout,),
            out={
                name: value
                for name, value in staged.items()
                if name not in denoiser.modalities
            },
        )
        for name, value in request.items():
            value.copy_(staged[name])
        schedules = denoiser.make_schedules(
            denoiser.num_steps, shift=None, device=device
        )
        samples = {name: request[name] for name in denoiser.modalities}
        for step in range(denoiser.num_steps):
            inputs = DenoiserInput(
                {
                    name: (LatentInput(value, schedules[name].timesteps[step]),)
                    for name, value in samples.items()
                },
                (layout,),
                schedules["video"].step(step),
                (conditioning,),
            )
            runner.step(inputs, schedules, state=request)
        group = denoiser._sequence_group()
        latents = {
            name: _gather(
                samples[name],
                denoiser.latent_shape(name, layout)[0],
                group,
            ).clone()
            for name in denoiser.modalities
        }

    output = video.Config(num_frames, canvas)
    decoder, postprocessor = model.video_decoder, model.video_postprocessor
    frames = []
    with (
        ExecutionContext(decoder) as decoding,
        ExecutionContext(postprocessor) as processing,
    ):
        decoding.prepare(output)
        processing.prepare(output)
        overlap = {
            name: torch.zeros(config.shape, dtype=config.dtype, device=device)
            for name, config in postprocessor.state_buffers(output).items()
        }
        for window in decoder.frame_slices(num_frames):
            with decoding.activate():
                segment = decoder.decode(
                    (latents["video"],),
                    frames=(window,),
                    sizes=(output,),
                    constants=decoding.constants,
                    workspace=decoding.workspace,
                )
            with processing.activate():
                rgb = postprocessor(
                    tuple(
                        value
                        for value in segment
                        if isinstance(value, TensorOutput)
                    ),
                    frames=(window,),
                    sizes=(output,),
                    state=overlap,
                    constants=processing.constants,
                    workspace=processing.workspace,
                )
            frames.extend(value.tensor.clone() for value in rgb)

    audio_decoder = model.audio_decoder
    samples_count = audio_decoder.track_samples(
        num_frames, postprocessor.frame_rate
    )
    frames_count = audio_decoder.latent_frames(samples_count)
    with ExecutionContext(audio_decoder) as context:
        context.prepare(frames_count)
        with context.activate():
            audio = audio_decoder.decode(
                (latents["audio"],),
                frames=(slice(0, frames_count),),
                num_samples=(samples_count,),
                workspace=context.workspace,
            )[0]
    return Generation(
        video_latent=latents["video"],
        audio_latent=latents["audio"],
        frames=torch.cat(frames),
        audio=audio,
    )
