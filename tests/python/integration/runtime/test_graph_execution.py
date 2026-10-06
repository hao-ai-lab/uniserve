"""Prepared input lifetime and numerical denoising.

Both are exercised across graph residency changes.
"""

from contextlib import nullcontext

import pytest
import torch

from tests.python.fixtures.diffusion import LinearDenoiser, Size
from tests.python.fixtures.encoding import Model
from uniserve.model import DenoiserInput, EntryPoint, LatentInput
from uniserve.runtime import CUDAStream, partition_streams
from uniserve_worker.config.execution import WorkerConfig
from uniserve_worker.execution.model_executor import ModelExecutor
from uniserve_worker.model_executor.component_binding import Call
from uniserve_worker.model_executor.diffusion_runner import DiffusionRunner
from uniserve_worker.model_executor.graph_storage import GraphStorage
from uniserve_worker.storage.latent_pool import LatentPool

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("failure", [None, "compute", "copy"])
def test_initial_inputs_are_ready_for_consumption_after_preparation(
    failure,
):
    device = torch.device("cuda", 0)
    model = Model().to(device)
    runner = ModelExecutor(model, WorkerConfig(device=str(device)))
    source = torch.empty(
        4 * 1024 * 1024, dtype=torch.float32, pin_memory=True
    ).fill_(7)
    destination = torch.empty_like(source, device=device)
    try:
        for value in (7, 13):
            source.fill_(value)
            copies = ((destination, source),)
            if failure == "copy":
                # Reject a later pair after the valid copy was submitted.
                copies += ((destination[:1], source),)
            error = (
                pytest.raises(
                    ValueError,
                    match="compute rejected"
                    if failure == "compute"
                    else "match destination shape and dtype",
                )
                if failure is not None
                else nullcontext()
            )
            with error:
                with runner.preparing_inputs(copies):
                    if failure == "compute":
                        raise ValueError("compute rejected")
            # This GPU consumer uses the calling stream. Copy completion must
            # precede it even when independent preparation computation failed.
            actual = (destination + 1).cpu()
            torch.testing.assert_close(
                actual, torch.full_like(source, value + 1), rtol=0, atol=0
            )
    finally:
        runner.close()


def _pool(slots, *, device):
    """A consumer-owned latent pool in which slot ``s`` owns page ``s``."""
    return LatentPool(
        request_pool_size=slots,
        num_pages=slots + 1,
        page_units=64,
        latent_width=1,
        dtype=torch.float32,
        device=device,
        with_workspace=False,
    )


def _runner(model, size, *, device, stream, pool, devices=(), slots=2):
    """Prepare the linear denoiser's diffusion runner for one size."""
    runner = DiffusionRunner.for_layouts(
        "denoiser",
        Call("denoiser", model, EntryPoint("forward")),
        size,
        device=device,
        stream=stream,
        storage=GraphStorage(),
        devices=devices,
        bank={},
        slots=slots,
        pool=pool,
        pages=1,
    )
    runner.prepare(size, pages=1)
    return runner


def _committed(pool, bank, slot, size):
    """A slot's samples in one bank of its page."""
    return pool.bank_view(bank, (slot,)).view(-1)[: size.width]


def _inputs(schedules, sample, size, steps):
    return tuple(
        DenoiserInput(
            {
                "image": (
                    LatentInput(sample, schedules["image"].timesteps[step]),
                )
            },
            (size,),
            schedules["image"].step(step),
        )
        for step in range(steps)
    )


def _bind(runner, schedules, size, steps, slot):
    """Bind a slot's ladder over the runner's samples."""
    sample = runner.samples.view(-1)[: size.width]
    return runner.bind(
        size,
        _inputs(schedules, sample, size, steps),
        schedules,
        state={},
        slot=slot,
        pages=(slot,),
    )


def _advanced(size, steps, *, device):
    """The linear denoiser's samples after ``steps`` unit-shift steps."""
    value = torch.full((size.width,), 7.0, device=device)
    for _ in range(steps):
        value.add_(0.5 * (value * 0.25 + size.offset))
    return value


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("execution", ["cuda:0", "cuda:1", "green"])
@torch.inference_mode()
def test_denoising_steps_read_the_committed_bank_and_write_the_other(
    graphs, execution
):
    """Each step advances a slot's committed samples into the other bank.

    Runners are prepared again for every size and slot. A step reads the
    bank holding the committed samples, leaves it unchanged and writes the
    successor to the other bank of the slot's page, so the next step reads
    that bank.
    """
    device = torch.device("cuda:0")
    branch_device = "cuda:0" if execution == "green" else execution
    model = LinearDenoiser().to(device)
    model.projection.to(branch_device)
    schedules = model.make_schedules(2, shift=1.0, device=device)
    partition = (
        partition_streams(device, (64,))[0] if execution == "green" else None
    )
    stream = (
        partition
        if partition is not None
        else CUDAStream.external(torch.cuda.Stream(device=device))
    )
    devices = (
        (device,)
        + ((torch.device(branch_device),) if branch_device != "cuda:0" else ())
        if graphs
        else ()
    )
    pool = _pool(2, device=device)
    try:
        for index, (slot, width, key) in enumerate(
            ((1, 32, 2), (1, 64, 3), (2, 32, 2), (1, 32, 2))
        ):
            size = Size(width, float(key + index))
            _committed(pool, 1, slot, size).fill_(7.0)
            reference = _committed(pool, 1, slot, size).clone()
            runner = _runner(
                model,
                size,
                device=device,
                stream=stream,
                pool=pool,
                devices=devices,
            )
            try:
                sample = runner.samples.view(-1)[:width]
                sample.copy_(reference)
                ladder = _bind(runner, schedules, size, 2, slot)
                runner.warmup(ladder)
                torch.testing.assert_close(sample, reference, rtol=0, atol=0)
                if graphs:
                    runner.capture(ladder)
                bank = 1
                for step in (0, 1):
                    committed = _committed(pool, bank, slot, size).clone()
                    reference.add_(0.5 * (reference * 0.25 + size.offset))
                    actual, _ = runner.step(ladder, step, bank)
                    torch.testing.assert_close(
                        actual["image"][0], reference, rtol=1e-6, atol=1e-6
                    )
                    torch.testing.assert_close(
                        _committed(pool, bank, slot, size),
                        committed,
                        rtol=0,
                        atol=0,
                    )
                    bank = 1 - bank
                    torch.testing.assert_close(
                        _committed(pool, bank, slot, size),
                        actual["image"][0],
                        rtol=0,
                        atol=0,
                    )
            finally:
                torch.cuda.current_stream(device).synchronize()
                stream.synchronize()
                runner.close()
    finally:
        pool.close()
        stream.close()


@torch.inference_mode()
def test_captured_ladders_replay_on_every_slot_with_eager_values():
    """Startup capture leaves one ladder resident that every slot replays.

    One captured graph reads whichever pages the device rows name. The
    values on every slot must match the same ladder evaluated eagerly.
    """
    device = torch.device("cuda:0")
    model = LinearDenoiser().to(device)
    steps = 2
    schedules = model.make_schedules(steps, shift=1.0, device=device)
    size = Size(32, 2.0)
    slots = (1, 2)

    def ladder(runner, pool):
        values = {}
        for slot in slots:
            _committed(pool, 1, slot, size).fill_(7.0)
            bound = _bind(runner, schedules, size, steps, slot)
            bank = 1
            for step in range(steps):
                result, _ = runner.step(bound, step, bank)
                bank = 1 - bank
            values[slot] = result["image"][0].clone()
        return values

    pool = _pool(2, device=device)
    eager = _runner(model, size, device=device, stream=None, pool=pool)
    try:
        expected = ladder(eager, pool)
    finally:
        torch.cuda.current_stream(device).synchronize()
        eager.close()

    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    runner = _runner(
        model,
        size,
        device=device,
        stream=stream,
        pool=pool,
        devices=(device,),
    )
    try:
        _committed(pool, 1, 1, size).fill_(7.0)
        bound = _bind(runner, schedules, size, steps, 1)
        runner.warmup(bound)
        runner.capture(bound)
        # Capture must leave the slot's committed samples where it found them.
        torch.testing.assert_close(
            _committed(pool, 1, 1, size),
            torch.full((size.width,), 7.0, device=device),
            rtol=0,
            atol=0,
        )

        actual = ladder(runner, pool)
        for slot in slots:
            torch.testing.assert_close(
                actual[slot], expected[slot], rtol=1e-6, atol=1e-6
            )
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()
        pool.close()
        stream.close()


@torch.inference_mode()
def test_prepared_layout_preserves_step_values_when_captured():
    """Preparing and later capturing a layout preserve its solver values."""
    device = torch.device("cuda:0")
    model = LinearDenoiser().to(device)
    steps = 2
    schedules = model.make_schedules(steps, shift=1.0, device=device)
    size = Size(32, 2.0)
    stream = CUDAStream.external(torch.cuda.Stream(device=device))
    pool = _pool(1, device=device)
    runner = _runner(
        model,
        size,
        device=device,
        stream=stream,
        pool=pool,
        devices=(device,),
        slots=1,
    )
    try:
        bound = _bind(runner, schedules, size, steps, 1)
        for capture in (False, True):
            _committed(pool, 1, 1, size).fill_(7.0)
            if capture:
                runner.warmup(bound)
                runner.capture(bound)

            bank = 1
            for step in range(steps):
                actual, _ = runner.step(bound, step, bank)
                torch.testing.assert_close(
                    actual["image"][0],
                    _advanced(size, step + 1, device=device),
                    rtol=1e-6,
                    atol=1e-6,
                )
                bank = 1 - bank
    finally:
        torch.cuda.current_stream(device).synchronize()
        runner.close()
        pool.close()
        stream.close()
