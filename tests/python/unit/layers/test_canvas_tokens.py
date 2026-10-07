"""Canvas denoising primitives follow the reference block-diffusion sampler.

The reference is the pinned Transformers DiffusionGemma generation code; the
random bits follow the published Philox4x32-10 definition.
"""

import pytest
import torch
from transformers.models.diffusion_gemma import (
    generation_diffusion_gemma as reference,
)

from uniserve.diffusion import tokens

pytestmark = pytest.mark.unit


def _words(*values):
    return torch.tensor(values, dtype=torch.int64)


def test_philox_matches_the_published_known_answers():
    # Random123 kat_vectors, philox4x32 with 10 rounds.
    cases = (
        (
            (0, 0, 0, 0),
            (0, 0),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        ),
        (
            (0xFFFFFFFF,) * 4,
            (0xFFFFFFFF,) * 2,
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    )
    for counter, key, expected in cases:
        assert tokens.philox(_words(*key), _words(*counter)).tolist() == list(
            expected
        )


def test_canvas_draws_depend_only_on_their_request_coordinates():
    seed = _words(7, 7, 8)
    block = _words(0, 0, 0)
    step = _words(3, 3, 3)
    bits = tokens.canvas_bits(
        seed, block, step, tokens.NoiseStream.SAMPLE, positions=5, words=2
    )
    assert bits.shape == (3, 5, 8)
    # Identical coordinates draw identical bits whatever the batch holds.
    assert torch.equal(bits[0], bits[1])
    assert not torch.equal(bits[0], bits[2])
    alone = tokens.canvas_bits(
        seed[:1],
        block[:1],
        step[:1],
        tokens.NoiseStream.SAMPLE,
        positions=5,
        words=2,
    )
    assert torch.equal(alone[0], bits[0])
    other = tokens.canvas_bits(
        seed[:1],
        block[:1],
        step[:1],
        tokens.NoiseStream.RENOISE,
        positions=5,
        words=2,
    )
    assert not torch.equal(other[0], bits[0])

    extremes = _words(0, 0xFFFFFFFF)
    values = tokens.uniform(extremes)
    assert (values > 0).all() and (values < 1).all()
    assert tokens.random_tokens(extremes, 262144).tolist() == [0, 262143]


def test_temperature_matches_the_linear_schedule_processor():
    processor = reference.LinearTemperatureScheduleLogitsProcessor(
        t_min=0.4, t_max=0.8, max_denoising_steps=48
    )
    logits = torch.randn(2, 3, 11, generator=torch.Generator().manual_seed(3))
    for remaining in (48, 31, 1):
        step = torch.tensor(remaining, dtype=torch.int32)
        temperature = tokens.step_temperature(
            step, t_min=0.4, t_max=0.8, steps=48
        )
        torch.testing.assert_close(
            logits / temperature,
            processor(None, logits, cur_step=step),
            rtol=0,
            atol=0,
        )


def test_entropy_and_acceptance_match_the_entropy_bound_sampler():
    generator = torch.Generator().manual_seed(5)
    # Varied sharpness spreads per-position entropies across the budget.
    logits = (
        torch.randn(3, 16, 50, generator=generator)
        * torch.linspace(0.2, 12, 16)[None, :, None]
    )
    entropy = tokens.token_entropy(logits)
    torch.testing.assert_close(
        entropy,
        torch.distributions.Categorical(logits=logits).entropy(),
        rtol=0,
        atol=0,
    )

    for bound in (0.1, 1.0, 5.0):
        sampler = reference.EntropyBoundSampler(
            reference.EntropyBoundSamplerConfig(entropy_bound=bound),
            canvas_length=16,
            vocab_size=50,
            max_denoising_steps=48,
        )
        current = torch.zeros(3, 16, dtype=torch.int64)
        proposed = torch.ones(3, 16, dtype=torch.int64)
        expected = sampler.accept_canvas(current, proposed, logits, 1)
        accepted = tokens.accept_by_entropy(entropy, bound)
        assert torch.equal(torch.where(accepted, proposed, current), expected)
        # The lowest-entropy position of every row is always accepted.
        assert accepted.gather(-1, entropy.argmin(-1, keepdim=True)).all()


def test_gumbel_draws_follow_the_softmax_distribution():
    probabilities = torch.tensor([0.5, 0.3, 0.15, 0.05])
    draws = 40000
    bits = tokens.canvas_bits(
        _words(11),
        _words(0),
        _words(0),
        tokens.NoiseStream.SAMPLE,
        positions=draws,
        words=1,
    )
    logits = probabilities.log().expand(1, draws, 4)
    samples = tokens.gumbel_sample(logits, tokens.uniform(bits))
    counts = torch.bincount(samples.flatten(), minlength=4).double()
    # Each count is binomial; five standard deviations bound a fixed draw.
    deviation = (draws * probabilities * (1 - probabilities)).double().sqrt()
    assert ((counts - draws * probabilities).abs() <= 5 * deviation).all()


def test_renoise_keeps_accepted_tokens_only():
    canvas = torch.tensor([[5, 6, 7, 8]])
    accepted = torch.tensor([[True, False, True, False]])
    noise = torch.tensor([[1, 2, 3, 4]])
    assert tokens.renoise(canvas, accepted, noise).tolist() == [[5, 2, 7, 4]]


def test_stopping_matches_the_stable_and_confident_criterion():
    generator = torch.Generator().manual_seed(9)
    criterion = reference.StableAndConfidentStoppingCriteria(
        stability_threshold=2, confidence_threshold=0.5
    )
    history = torch.full((2, 2, 6), -1, dtype=torch.int64)
    fixed = torch.randint(0, 20, (2, 6), generator=generator)
    for step in range(5):
        argmax = fixed.clone()
        if step < 2:
            argmax[1, step] += 1  # row 1 changes during the first two steps
        # Row 0 is confident; row 1 becomes confident from step 3.
        sharpness = torch.tensor([30.0, 30.0 if step >= 3 else 0.1])
        logits = (
            torch.randn(2, 6, 20, generator=generator)
            * sharpness[:, None, None]
        )
        expected = criterion(argmax, logits)
        stop, history = tokens.stable_and_confident(
            history, argmax, tokens.token_entropy(logits), confidence=0.5
        )
        assert torch.equal(stop, expected)

    without_history = torch.empty((0, 1, 3), dtype=torch.int64)
    stop, kept = tokens.stable_and_confident(
        without_history,
        torch.zeros(1, 3, dtype=torch.int64),
        torch.zeros(1, 3),
        confidence=0.5,
    )
    assert stop.tolist() == [True] and kept.shape == (0, 1, 3)


def test_eos_truncation_matches_canvas_finalization():
    canvas = torch.tensor(
        [[9, 106, 7, 1, 5], [9, 8, 7, 6, 5], [50, 3, 3, 3, 3]]
    )
    eos = torch.tensor([1, 106, 50])
    config = reference.DiffusionGemmaGenerationConfig(
        eos_token_id=[1, 106, 50], pad_token_id=0
    )
    stopping = reference.StoppingCriteriaList(
        [reference.EosTokenCriteria(eos_token_id=[1, 106, 50])]
    )
    expected, finished = (
        reference.DiffusionGemmaGenerationMixin._finalize_canvas(
            canvas.clone(),
            torch.zeros(3, dtype=torch.bool),
            config,
            stopping,
            canvas_length=5,
            eos_tensor=eos,
        )
    )
    actual, stopped = tokens.truncate_after_eos(canvas, eos, pad_id=0)
    assert torch.equal(actual, expected)
    assert torch.equal(stopped, finished)


def test_candidate_logprobs_normalize_over_the_full_vocabulary():
    logits = torch.tensor([[2.0, 0.0, -1.0, 3.0]])
    candidates = torch.tensor([[3, 0]])
    expected = torch.log_softmax(logits, dim=-1)[:, [3, 0]]
    torch.testing.assert_close(
        tokens.candidate_logprobs(logits.bfloat16().float(), candidates),
        expected,
    )


def test_self_conditioning_embedding_matches_the_reference_decoder():
    generator = torch.Generator().manual_seed(13)
    embedding = torch.randn(40, 8, generator=generator).bfloat16()
    logits = torch.randn(6, 40, generator=generator) * 4
    scale = torch.tensor(8**0.5)
    # The reference decoder casts the processed logits to the embedding
    # dtype, softmaxes in FP32 and scales by its embedding scale buffer.
    carried = logits.to(torch.bfloat16)
    expected = torch.matmul(
        carried.softmax(dim=-1, dtype=torch.float32).to(torch.bfloat16),
        embedding,
    ) * scale.to(torch.bfloat16)
    torch.testing.assert_close(
        tokens.self_conditioning_embedding(logits, embedding, scale),
        expected,
        rtol=0,
        atol=0,
    )
