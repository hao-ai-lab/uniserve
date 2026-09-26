"""The portable canvas sampler step follows the reference denoising loop.

The reference is the pinned Transformers ``DiffusionGemmaGenerationMixin``:
its temperature processor, entropy-bound acceptance, stable-and-confident
stop, self-conditioning embedding and canvas finalization. Its multinomial
and re-noise draws come from torch's generator; the step draws them from
the Philox contract of ``uniserve.diffusion.tokens`` instead, so the test
checks those draws against that contract.
"""

import pytest
import torch
from transformers.models.diffusion_gemma import (
    generation_diffusion_gemma as reference,
)

from uniserve.diffusion import canvas, tokens

pytestmark = pytest.mark.unit

ROWS, LENGTH, VOCAB, HIDDEN = 3, 16, 64, 8
EOS, PAD = (1, 6, 5), 0


def _sampling(steps, stability=1):
    return canvas.CanvasSampling(
        steps=steps,
        entropy_bound=0.5,
        t_min=0.4,
        t_max=0.8,
        confidence=0.2,
        stability=stability,
        eos_ids=EOS,
        pad_id=PAD,
    )


def _state(seed, block, step, stability=1):
    state = canvas.CanvasState.empty(
        ROWS, LENGTH, HIDDEN, stability=stability, dtype=torch.bfloat16
    )
    state.seed.copy_(torch.tensor(seed))
    state.block.copy_(torch.tensor(block))
    state.step.copy_(torch.tensor(step))
    state.canvas.fill_(7)
    state.history.fill_(3)
    state.self_conditioning.fill_(1.0)
    return state


def _buffers():
    return (
        canvas.CanvasScores.empty(ROWS, LENGTH),
        canvas.CanvasDecision.empty(ROWS, LENGTH),
        canvas.CanvasWorkspace.empty(ROWS, LENGTH, VOCAB, HIDDEN),
    )


def test_start_draws_the_initial_canvas_of_rows_at_step_zero():
    state = _state(seed=[5, 5, 9], block=[0, 2, 1], step=[0, 4, 0])
    canvas.start_canvas(state, vocab_size=VOCAB)

    bits = tokens.canvas_bits(
        state.seed,
        state.block,
        torch.zeros(ROWS, dtype=torch.int64),
        tokens.NoiseStream.INITIAL,
        positions=LENGTH,
        words=1,
    )
    initial = tokens.random_tokens(bits[..., 0], VOCAB)
    for row, starting in enumerate((True, False, True)):
        if starting:
            assert torch.equal(state.canvas[row], initial[row])
            assert (state.history[row] == -1).all()
            assert (
                state.self_conditioning.view(ROWS, LENGTH, HIDDEN)[row] == 0
            ).all()
        else:
            assert (state.canvas[row] == 7).all()
            assert (state.history[row] == 3).all()
            assert (
                state.self_conditioning.view(ROWS, LENGTH, HIDDEN)[row] == 1
            ).all()


def test_steps_follow_the_reference_denoising_loop():
    generator = torch.Generator().manual_seed(21)
    steps = 3
    sampling = _sampling(steps)
    embedding = torch.randn(VOCAB, HIDDEN, generator=generator).bfloat16()
    scale = HIDDEN**0.5
    state = _state(seed=[3, 3, 11], block=[0, 1, 0], step=[0, 0, 0])
    canvas.start_canvas(state, vocab_size=VOCAB)
    scores, decision, workspace = _buffers()

    processor = reference.LinearTemperatureScheduleLogitsProcessor(
        t_min=0.4, t_max=0.8, max_denoising_steps=steps
    )
    sampler = reference.EntropyBoundSampler(
        reference.EntropyBoundSamplerConfig(entropy_bound=0.5),
        canvas_length=LENGTH,
        vocab_size=VOCAB,
        max_denoising_steps=steps,
    )
    criterion = reference.StableAndConfidentStoppingCriteria(
        stability_threshold=1, confidence_threshold=0.2
    )
    # Sharpness grows along the canvas and over the steps, so rows cross the
    # entropy budget and, later, the confidence threshold.
    base = torch.randn(ROWS, LENGTH, VOCAB, generator=generator)
    sharpness = torch.linspace(0.5, 30, LENGTH)[None, :, None]
    for step in range(steps):
        logits = torch.tanh(base * sharpness * (1 + 2 * step) / 30) * 30
        if step == steps - 1:
            logits[0, 4, EOS[1]] = 40.0  # row 0 ends with an EOS mid-canvas
        state.step.fill_(step)
        canvas.denoise_canvas(
            logits,
            embedding,
            scale,
            state,
            sampling,
            scores=scores,
            decision=decision,
            workspace=workspace,
        )

        remaining = torch.tensor(steps - step, dtype=torch.int32)
        processed = processor(None, logits, cur_step=remaining)
        argmax = processed.argmax(dim=-1)
        assert torch.equal(scores.argmax, argmax)

        # Accepted positions take the Gumbel sample; the others re-noise.
        sampler.accept_canvas(
            torch.zeros_like(argmax),
            torch.ones_like(argmax),
            processed,
            remaining,
        )
        accepted = sampler.accepted_token_mask
        bits = tokens.canvas_bits(
            state.seed,
            state.block,
            state.step,
            tokens.NoiseStream.SAMPLE,
            positions=LENGTH,
            words=VOCAB // 4,
        )
        sample = tokens.gumbel_sample(processed, tokens.uniform(bits))
        noise = tokens.canvas_bits(
            state.seed,
            state.block,
            state.step,
            tokens.NoiseStream.RENOISE,
            positions=LENGTH,
            words=1,
        )[..., 0]
        expected = torch.where(
            accepted, sample, tokens.random_tokens(noise, VOCAB)
        )
        assert torch.equal(scores.sample, sample)
        assert torch.equal(state.canvas, expected)

        stop = criterion(argmax, processed)
        assert torch.equal(decision.finished[:, 0], stop | (step == steps - 1))

        # The reference decoder's embedding of the carried logits (its HF
        # parity is tested with the primitive).
        soft = tokens.self_conditioning_embedding(
            processed.reshape(-1, VOCAB), embedding, torch.tensor(scale)
        )
        assert torch.equal(state.self_conditioning, soft)

    finished = torch.zeros(ROWS, dtype=torch.bool)
    committed, ended = reference.DiffusionGemmaGenerationMixin._finalize_canvas(
        argmax.clone(),
        finished,
        reference.DiffusionGemmaGenerationConfig(
            eos_token_id=list(EOS), pad_token_id=PAD
        ),
        reference.StoppingCriteriaList(
            [reference.EosTokenCriteria(eos_token_id=list(EOS))]
        ),
        canvas_length=LENGTH,
        eos_tensor=torch.tensor(EOS),
    )
    assert torch.equal(decision.tokens, committed)
    assert torch.equal(decision.finished[:, 1], ended)
    assert decision.finished[0, 1]


def test_rows_without_stability_history_stop_on_confidence_alone():
    sampling = _sampling(steps=48, stability=0)
    state = _state(seed=[1, 2, 3], block=[0, 0, 0], step=[5, 5, 5], stability=0)
    scores, decision, _ = _buffers()
    scores.entropy.copy_(
        torch.tensor([0.01, 0.3, 0.19])
        .repeat_interleave(LENGTH)
        .view(ROWS, LENGTH)
    )
    scores.argmax.fill_(9)
    scores.sample.fill_(9)
    canvas.advance_canvas(
        scores, state, sampling, decision=decision, vocab_size=VOCAB
    )
    assert decision.finished[:, 0].tolist() == [True, False, True]
