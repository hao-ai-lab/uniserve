"""The SM100 canvas sampler kernels follow the portable sampler step.

Comparison rules and FP32 bounds were registered before any comparison ran
(``artifacts/diffusion_gemma/stage4/sampler/tolerances.json``, SHA256
adda64ef4f063ab1389019a02a612e171a1ec4e65786bd7d3784ef1ef474f19a):

- R1: Philox draws (initial and re-noise tokens) and block starts match
  bitwise.
- R2: the argmax matches exactly.
- R3: the Gumbel sample matches unless the two winners' CPU scores lie
  within the logarithms' FP32 rounding of each other.
- R4: entropies agree within the sum of both implementations' derived FP32
  error bounds.
- R5: the row decisions of the kernel equal the portable decisions on the
  same scores, except a stop whose mean entropy lies within the summation
  rounding of the confidence threshold.
- R7: self-conditioning embeddings agree within the BF16 weight rounding,
  the dot-product accumulation bound and the output roundings.
- R8: CUDA graph replay equals eager execution, also after the device
  inputs change.
- R9: a row's results do not depend on the rows beside it.
"""

import dataclasses
import math

import pytest
import torch
from uniserve_kernels.diffusion import canvas as kernels

from uniserve.diffusion import canvas, tokens

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 0),
        reason="the canvas sampler kernels require an SM100 GPU",
    ),
]

DEVICE = torch.device("cuda:0")
EOS, PAD, HIDDEN = (1, 106, 50), 0, 64
U = 2.0**-24
N = 256


def _sampling(steps=48, stability=1):
    return canvas.CanvasSampling(
        steps=steps,
        entropy_bound=0.1,
        t_min=0.4,
        t_max=0.8,
        confidence=0.005,
        stability=stability,
        eos_ids=EOS,
        pad_id=PAD,
    )


def _logits(generator, rows, length, vocab):
    """Soft-capped logits whose per-position sharpness spans the decisions.

    Some positions tie their two largest raw logits, some favour an
    end-of-sequence token.
    """
    base = torch.randn(rows, length, vocab, generator=generator)
    sharpness = torch.logspace(-1, 1.5, length)[None, :, None]
    logits = torch.tanh(base * sharpness * 3 / 30) * 30
    top = torch.randint(0, vocab, (rows, length), generator=generator)
    boost = torch.linspace(0, 45, rows * length).view(rows, length)
    logits.scatter_(-1, top[..., None], (logits.amax(-1) + boost)[..., None])
    logits[:, 1, 7] = logits[:, 1].amax(-1)  # a tie with an earlier index
    logits[0, 3, EOS[1] % vocab] = 60.0
    return logits.clamp(-30, 30)


def _state(rows, length, seeds, blocks, steps, stability=1, device=DEVICE):
    state = canvas.CanvasState.empty(
        rows, length, HIDDEN, stability=stability, device=device
    )
    state.seed.copy_(torch.tensor(seeds))
    state.block.copy_(torch.tensor(blocks))
    state.step.copy_(torch.tensor(steps))
    state.canvas.fill_(0)
    state.history.fill_(-1)
    state.self_conditioning.fill_(0)
    return state


def _clone(state, device):
    return canvas.CanvasState(
        *(
            getattr(state, name).to(device, copy=True)
            for name in state.__slots__
        )
    )


def _buffers(rows, length, vocab, device):
    return (
        canvas.CanvasScores.empty(rows, length, device=device),
        canvas.CanvasDecision.empty(rows, length, device=device),
        canvas.CanvasWorkspace.empty(
            rows, length, vocab, HIDDEN, device=device
        ),
    )


def _processed(logits, state, sampling):
    remaining = sampling.steps - state.step
    temperature = tokens.step_temperature(
        remaining,
        t_min=sampling.t_min,
        t_max=sampling.t_max,
        steps=sampling.steps,
    )
    return logits / temperature[:, None, None]


def _entropy_bound(q, entropy_cpu):
    """Registered R4 bound tau_cuda + tau_cpu per position (FP64)."""
    q = q.double()
    top = q.amax(-1, keepdim=True)
    lse = top[..., 0] + torch.log(torch.exp(q - top).sum(-1))
    log_p = q - lse[..., None]
    p = log_p.exp()
    h = -(p * log_p).sum(-1)
    e2 = (p * (q - top) ** 2).sum(-1)
    a = (p * q.abs() * (log_p + h[..., None]).abs()).sum(-1)
    cuda = 2 * (U * ((4 + N) + (16 + 2 * N) * h + 6 * e2) + 3 * U * a)
    cpu = 2 * U * ((2 + N) + lse.abs() + (9 + 2 * N) * h + 3 * h**2 + 3 * e2)
    return cuda + cpu


def _check_scores(logits, state, sampling, cuda_scores, cpu_scores):
    """Rules R2-R4 on one step's scores; returns the R3 near-tie count."""
    q = _processed(logits, state, sampling)
    assert torch.equal(cuda_scores.argmax.cpu(), cpu_scores.argmax)  # R2

    bound = _entropy_bound(q, cpu_scores.entropy)
    difference = (
        cuda_scores.entropy.cpu().double() - cpu_scores.entropy.double()
    ).abs()
    assert (difference <= bound).all(), difference.max()  # R4

    kernel = cuda_scores.sample.cpu()
    reference = cpu_scores.sample
    ties = 0
    if not torch.equal(kernel, reference):  # R3
        bits = tokens.canvas_bits(
            state.seed,
            state.block,
            state.step,
            tokens.NoiseStream.SAMPLE,
            positions=q.shape[1],
            words=q.shape[2] // 4,
        )
        inner = torch.log(-torch.log(tokens.uniform(bits)))
        scores = q - inner
        eps = (
            2.0**-23 * (1 + inner.double().abs())
            + 2.0**-24 * scores.double().abs()
        )
        for row, position in (kernel != reference).nonzero().tolist():
            j, k = reference[row, position], kernel[row, position]
            gap = (
                scores[row, position, j].double()
                - scores[row, position, k].double()
            )
            assert gap <= 2 * (eps[row, position, j] + eps[row, position, k])
            ties += 1
    return ties


def _check_decisions(scores, before, after, decision, sampling, vocab):
    """Rule R5: the kernel's decisions equal the portable ones on its scores."""
    rows, length = before.canvas.shape
    portable = _clone(before, "cpu")
    cpu_scores = canvas.CanvasScores(
        scores.entropy.cpu(), scores.argmax.cpu(), scores.sample.cpu()
    )
    cpu_decision = canvas.CanvasDecision.empty(rows, length)
    canvas.advance_canvas(
        cpu_scores, portable, sampling, decision=cpu_decision, vocab_size=vocab
    )
    assert torch.equal(after.canvas.cpu(), portable.canvas)
    assert torch.equal(after.history.cpu(), portable.history)
    assert torch.equal(decision.tokens.cpu(), cpu_decision.tokens)
    assert torch.equal(
        decision.finished[:, 1].cpu(), cpu_decision.finished[:, 1]
    )
    differs = decision.finished[:, 0].cpu() != cpu_decision.finished[:, 0]
    threshold = float(torch.tensor(sampling.confidence, dtype=torch.float32))
    for row in differs.nonzero().flatten().tolist():
        mean = cpu_scores.entropy[row].double().mean().item()
        stable = bool(
            (before.history[row].cpu() == cpu_scores.argmax[row]).all()
        )
        assert stable and abs(mean - threshold) <= U * 256 * mean + 2.0**-149


def _check_embedding(q, embedding, scale, cuda_state, cpu_state):
    """Rule R7 on the self-conditioning embedding."""
    vocab = q.shape[-1]
    p = torch.softmax(q.to(torch.bfloat16).double(), dim=-1).reshape(-1, vocab)
    rounded = float(torch.tensor(scale, dtype=torch.bfloat16))
    spread = rounded * (p @ embedding[:vocab].double().abs())
    reference = cpu_state.self_conditioning.double()
    difference = (cuda_state.self_conditioning.cpu().double() - reference).abs()
    bound = (
        (2.0**-8 + 2.0**-13 + 2 * vocab * U) * spread
        + 2.0**-7 * reference.abs()
        + 2.0**-120
    )
    assert (difference <= bound).all()


def _step_both(logits, embedding, scale, state, sampling):
    """One denoise step on CUDA and on the CPU from the same inputs."""
    rows, length, vocab = logits.shape
    cuda_state = _clone(state, DEVICE)
    cpu_state = _clone(state, "cpu")
    cuda = _buffers(rows, length, vocab, DEVICE)
    cpu = _buffers(rows, length, vocab, "cpu")
    for target, buffers, device in (
        (cuda_state, cuda, DEVICE),
        (cpu_state, cpu, "cpu"),
    ):
        scores, decision, workspace = buffers
        canvas.denoise_canvas(
            logits.to(device),
            embedding.to(device),
            scale,
            target,
            sampling,
            scores=scores,
            decision=decision,
            workspace=workspace,
        )
    torch.cuda.synchronize()
    return cuda_state, cuda, cpu_state, cpu


def test_steps_match_the_portable_sampler():
    generator = torch.Generator().manual_seed(3)
    rows, length, vocab = 4, 64, 8192
    sampling = _sampling(steps=48)
    embedding = torch.randn(vocab, HIDDEN, generator=generator).bfloat16()
    scale = HIDDEN**0.5
    state = _state(
        rows, length, [7, 7, 2**40 + 3, 12], [0, 3, 1, 9], [0, 17, 47, 5]
    )
    logits = _logits(generator, rows, length, vocab)
    # Row 1 repeats its previous argmax canvas, so it can stop when confident.
    q = _processed(logits, _clone(state, "cpu"), sampling)
    state.history[1].copy_(q[1].argmax(-1))

    cuda_state, cuda, cpu_state, cpu = _step_both(
        logits, embedding, scale, state, sampling
    )
    _check_scores(logits, _clone(state, "cpu"), sampling, cuda[0], cpu[0])
    # R5 also fixes the re-noised positions to the RENOISE stream (R1).
    _check_decisions(cuda[0], state, cuda_state, cuda[1], sampling, vocab)
    _check_embedding(q, embedding, scale, cuda_state, cpu_state)
    assert cuda[1].finished[2, 0]  # the last step ends the block


def test_full_vocabulary_scores_match_the_portable_sampler():
    generator = torch.Generator().manual_seed(8)
    rows, length, vocab = 1, 32, 262144
    sampling = _sampling(steps=48)
    embedding = (
        torch.randn(vocab, HIDDEN, generator=generator) * 0.05
    ).bfloat16()
    state = _state(rows, length, [123], [2], [1])
    logits = _logits(generator, rows, length, vocab)

    cuda_state, cuda, cpu_state, cpu = _step_both(
        logits, embedding, 2816**0.5, state, sampling
    )
    _check_scores(logits, _clone(state, "cpu"), sampling, cuda[0], cpu[0])
    _check_decisions(cuda[0], state, cuda_state, cuda[1], sampling, vocab)
    q = _processed(logits, _clone(state, "cpu"), sampling)
    _check_embedding(q, embedding, 2816**0.5, cuda_state, cpu_state)


def test_block_start_matches_the_portable_draws():
    rows, length, vocab = 3, 256, 262144
    state = _state(rows, length, [5, 5, 99], [0, 4, 1], [0, 3, 0])
    state.canvas.fill_(11)
    state.history.fill_(8)
    state.self_conditioning.fill_(2.0)
    cuda_state, cpu_state = _clone(state, DEVICE), _clone(state, "cpu")
    canvas.start_canvas(cuda_state, vocab_size=vocab)
    canvas.start_canvas(cpu_state, vocab_size=vocab)
    for name in ("canvas", "history", "self_conditioning"):
        assert torch.equal(
            getattr(cuda_state, name).cpu(), getattr(cpu_state, name)
        )


def test_graph_replay_follows_changed_device_inputs():
    generator = torch.Generator().manual_seed(11)
    rows, length, vocab = 2, 64, 8192
    sampling = _sampling(steps=48)
    embedding = (
        torch.randn(vocab, HIDDEN, generator=generator).bfloat16().to(DEVICE)
    )
    inputs = [
        (
            _logits(generator, rows, length, vocab).to(DEVICE),
            _state(rows, length, seeds, [0, 1], steps),
        )
        for seeds, steps in (([4, 5], [0, 6]), ([9, 1], [3, 0]))
    ]

    logits = torch.empty(rows, length, vocab, device=DEVICE)
    state = _state(rows, length, [0, 0], [0, 0], [0, 0])
    scores, decision, workspace = _buffers(rows, length, vocab, DEVICE)

    def step():
        canvas.start_canvas(state, vocab_size=vocab)
        canvas.denoise_canvas(
            logits,
            embedding,
            53.0,
            state,
            sampling,
            scores=scores,
            decision=decision,
            workspace=workspace,
        )

    def load(values, initial):
        logits.copy_(values)
        for name in initial.__slots__:
            getattr(state, name).copy_(getattr(initial, name))

    def outputs():
        return [
            t.clone()
            for t in (
                scores.entropy,
                scores.argmax,
                scores.sample,
                decision.tokens,
                decision.finished,
                state.canvas,
                state.history,
                state.self_conditioning,
            )
        ]

    eager = []
    for values, initial in inputs:
        load(values, initial)
        step()
        eager.append(outputs())

    graph = torch.cuda.CUDAGraph()
    load(*inputs[0])
    with torch.cuda.graph(graph):
        step()
    for (values, initial), expected in zip(inputs, eager, strict=True):
        load(values, initial)
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(outputs(), expected, strict=True):
            assert torch.equal(actual, reference)


def test_row_results_do_not_depend_on_the_batch():
    generator = torch.Generator().manual_seed(17)
    length, vocab = 64, 8192
    sampling = _sampling(steps=48)
    embedding = (
        torch.randn(vocab, HIDDEN, generator=generator).bfloat16().to(DEVICE)
    )
    logits = _logits(generator, 3, length, vocab).to(DEVICE)
    batch = _state(3, length, [1, 2, 3], [0, 5, 2], [4, 9, 0])

    def run(values, state):
        rows = values.shape[0]
        scores, decision, workspace = _buffers(rows, length, vocab, DEVICE)
        canvas.start_canvas(state, vocab_size=vocab)
        canvas.denoise_canvas(
            values,
            embedding,
            53.0,
            state,
            sampling,
            scores=scores,
            decision=decision,
            workspace=workspace,
        )
        return scores, decision, state

    together = run(logits, _clone(batch, DEVICE))
    alone_state = canvas.CanvasState(
        *(
            getattr(batch, name)[1:2].clone()
            if name != "self_conditioning"
            else batch.self_conditioning[length : 2 * length].clone()
            for name in batch.__slots__
        )
    )
    alone = run(logits[1:2].contiguous(), alone_state)
    for field in ("entropy", "argmax", "sample"):
        assert torch.equal(
            getattr(together[0], field)[1:2], getattr(alone[0], field)
        )
    assert torch.equal(together[1].tokens[1:2], alone[1].tokens)
    assert torch.equal(together[1].finished[1:2], alone[1].finished)
    assert torch.equal(together[2].canvas[1:2], alone[2].canvas)
    assert torch.equal(together[2].history[1:2], alone[2].history)

    q = _processed(logits[1:2].cpu(), _clone(alone_state, "cpu"), sampling)
    p = torch.softmax(q.to(torch.bfloat16).double(), dim=-1).reshape(-1, vocab)
    spread = 53.0 * (p @ embedding.cpu().double().abs())
    reference = alone[2].self_conditioning.cpu().double()
    difference = (
        together[2].self_conditioning[length : 2 * length].cpu().double()
        - reference
    ).abs()
    assert (
        difference
        <= (2.0**-8 + 2.0**-13 + 2 * vocab * U) * spread
        + 2.0**-7 * reference.abs()
        + 2.0**-120
    ).all()
    assert math.isfinite(float(reference.abs().max()))


@pytest.mark.parametrize(("confidence", "stops"), [(0.0, False), (100.0, True)])
def test_rows_without_history_stop_on_the_confidence_threshold(
    confidence, stops
):
    # Without stability history every row is stable, so the confidence
    # threshold alone decides: 100 nats exceeds every mean entropy, and no
    # mean entropy lies below zero. The positions are near one-hot, with a
    # runner-up 6 to 16 raw units below the winner, before or after it.
    generator = torch.Generator().manual_seed(29)
    rows, length, vocab = 3, 64, 262144
    sampling = dataclasses.replace(
        _sampling(), stability=0, confidence=confidence
    )
    embedding = (
        torch.randn(vocab, HIDDEN, generator=generator).bfloat16().to(DEVICE)
    )
    logits = torch.full((rows, length, vocab), -30.0)
    first = torch.randint(
        0, vocab - 4096, (rows, length, 1), generator=generator
    )
    gap = 6 + 10 * torch.rand(rows, length, 1, generator=generator)
    winner_first = torch.rand(rows, length, 1, generator=generator) < 0.5
    logits.scatter_(-1, first, torch.where(winner_first, 20.0, 20.0 - gap))
    logits.scatter_(
        -1, first + 4096, torch.where(winner_first, 20.0 - gap, 20.0)
    )
    state = _state(rows, length, [4, 5, 6], [0, 0, 0], [5, 30, 47], stability=0)
    scores, decision, workspace = _buffers(rows, length, vocab, DEVICE)

    canvas.denoise_canvas(
        logits.to(DEVICE),
        embedding,
        53.0,
        state,
        sampling,
        scores=scores,
        decision=decision,
        workspace=workspace,
    )
    assert (scores.entropy >= 0).all()
    assert decision.finished[:, 0].tolist() == [stops, stops, True]


def test_product_reports_the_algorithm_it_measured():
    # A shape no other test multiplies, so its first product happens here.
    generator = torch.Generator().manual_seed(41)
    positions, vocab, hidden = 96, 4096, 40
    weights = torch.rand(positions, vocab, generator=generator).bfloat16()
    table = torch.randn(vocab, hidden, generator=generator).bfloat16()
    output = torch.empty(positions, hidden, device=DEVICE)
    scratch = torch.empty(1 << 22, dtype=torch.uint8, device=DEVICE)
    operands = (weights.to(DEVICE), table.to(DEVICE), output, scratch)

    assert kernels.product_algorithm(*operands) is None
    kernels.product(*operands)
    chosen = kernels.product_algorithm(*operands)
    assert chosen is not None and chosen["cublaslt_version"] > 0

    expected = weights.double() @ table.double()
    spread = weights.double().abs() @ table.double().abs()
    difference = (output.cpu().double() - expected).abs()
    assert (difference <= 2 * vocab * U * spread).all()


@pytest.mark.parametrize(("rows", "hidden"), [(1, 64), (4, 2816)])
def test_workspace_size_matches_its_allocation(rows, hidden):
    workspace = canvas.CanvasWorkspace.empty(
        rows, 256, 4096, hidden, device=DEVICE
    )
    allocated = sum(
        getattr(workspace, name).nbytes for name in workspace.__slots__
    )
    assert allocated == canvas.CanvasWorkspace.nbytes(rows, 256, 4096, hidden)
