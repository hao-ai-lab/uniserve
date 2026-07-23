from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from uniserve_worker.contracts.model_spec import FlowSpec
from uniserve_worker.execution.flow import (
    GenState,
    GuidePlan,
    PagedGenFlowExecution,
    PreparedFlowStep,
)
from uniserve_worker.models.bagel import _BagelGraph
from uniserve_worker.nn.diffusion import FlowMatchSchedule, ScheduleDirection, euler_step
from uniserve_worker.nn.diffusion.noise import init_latent
from uniserve_worker.runtime.request_state import RequestState, flow_noise_seed
from uniserve_worker.runtime.residency import LatentStore

pytestmark = pytest.mark.unit


def _gen_flow_driver(device: str = "cpu") -> PagedGenFlowExecution:
    return PagedGenFlowExecution(
        SimpleNamespace(device=device),
        flow=FlowSpec(
            latent_downsample=16,
            prediction="velocity",
            schedule_direction="descending",
            schedule_shift_domain="time",
            cfg_recipe="image_over_text",
        ),
    )


def test_latent_noise_can_sample_cpu_float32_before_output_cast():
    shape = (6, 8)
    seed = 37
    expected = torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16)

    actual = init_latent(
        shape,
        rng=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.bfloat16,
        source_device="cpu",
        source_dtype=torch.float32,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_latent_noise_can_transfer_cpu_float32_source_exactly():
    shape = (6, 8)
    seed = 37
    expected = torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float32,
    ).to(device="cuda")

    actual = init_latent(
        shape,
        rng=torch.Generator(device="cpu").manual_seed(seed),
        device="cuda",
        dtype=torch.float32,
        source_device="cpu",
        source_dtype=torch.float32,
    )

    assert actual.device.type == "cuda"
    assert torch.equal(actual, expected)


def test_latent_noise_default_samples_directly_in_output_format():
    shape = (2, 3, 4)
    seed = 19
    expected = torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float64,
    )

    actual = init_latent(
        shape,
        rng=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float64,
    )

    assert torch.equal(actual, expected)


def test_gen_flow_initial_noise_keeps_exact_cpu_float32_values():
    driver = _gen_flow_driver("cpu")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        expected = torch.randn((6, 64), device="cpu", dtype=torch.float32)

    actual = driver._init_generation_noise((6, 64), seed=0)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_flow_noise_seed_is_a_pure_function_of_session_seed_and_op_id():
    # Deterministic in its coordinates.
    assert flow_noise_seed(11, 5) == flow_noise_seed(11, 5)
    # A different op id (a different image/generation of the same session) and a
    # different session seed each decorrelate the stream.
    assert flow_noise_seed(11, 5) != flow_noise_seed(11, 6)
    assert flow_noise_seed(11, 5) != flow_noise_seed(12, 5)
    # Neighboring op ids are decorrelated by the finalizer, and every seed is a
    # 64-bit unsigned integer torch.Generator.manual_seed accepts.
    assert flow_noise_seed(11, 5) != flow_noise_seed(11, 4)
    for coord in ((0, 0), (11, 5), (2**63, 2**40 + 7)):
        assert 0 <= flow_noise_seed(*coord) <= 0xFFFFFFFFFFFFFFFF


def test_gen_flow_noise_is_counter_derived_and_invariant_to_draw_history():
    driver = _gen_flow_driver("cpu")
    session_seed = 4242
    op_a, op_b = 0x100, 0x200

    first_a = driver._init_generation_noise((6, 64), seed=flow_noise_seed(session_seed, op_a))
    # Sampling an unrelated operation in between must not perturb op_a: the
    # generator is reseeded per operation, so nothing carries between draws.
    noise_b = driver._init_generation_noise((6, 64), seed=flow_noise_seed(session_seed, op_b))
    second_a = driver._init_generation_noise((6, 64), seed=flow_noise_seed(session_seed, op_a))

    assert torch.equal(first_a, second_a)  # retry / batch-position invariance
    assert not torch.equal(first_a, noise_b)  # distinct op ids -> distinct noise


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
        ),
    ],
)
def test_gen_flow_update_keeps_fp32_state_and_matches_reference_arithmetic(device):
    driver = _gen_flow_driver(device)
    latent = driver._init_generation_noise((2, 3), seed=7)
    velocity = torch.tensor(
        [[0.125, -0.25, 0.5], [-0.75, 1.0, -1.25]],
        dtype=torch.bfloat16,
        device=device,
    )
    t = torch.tensor(0.8, dtype=torch.float32, device=device)
    t_next = torch.tensor(0.7, dtype=torch.float32, device=device)
    expected = latent - velocity * (t - t_next)
    store = LatentStore()
    generation_state = GenState(
        latent_pool=store,
        latent_handle=1,
        vae_pos_ids=torch.zeros(latent.shape[0], dtype=torch.long),
        num_vae=int(latent.shape[0]),
        H=32,
        W=48,
        schedule=FlowMatchSchedule(
            num_steps=1, shift=1.0, direction=ScheduleDirection.DESCENDING
        ),
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_renorm_type="global",
        cfg_renorm_min=0.0,
        cfg_interval=(0.4, 1.0),
        cond_pos=0,
    )
    generation_state.x_t = latent
    committed = latent.clone()
    step = PreparedFlowStep(
        req_id=1,
        state=RequestState(),
        op={},
        latent=generation_state.x_t,
        t=t,
        t_next=t_next,
        step_index=0,
        total_steps=1,
        guide=GuidePlan.resolve(
            recipe="image_over_text",
            text_scale=1.0,
            img_scale=1.0,
            interval=(0.4, 1.0),
            renorm="global",
            renorm_min=0.0,
            t=t,
        ),
        extra={"gs": generation_state},
    )

    updated = euler_step(generation_state.x_t, velocity, t, t_next)
    driver.apply_flow_update(step, updated)

    assert generation_state.x_t.dtype == torch.float32
    assert torch.equal(generation_state.x_t, expected)
    # The accepted update replaces the system latent buffer with the euler
    # output itself — no copy on the success path — while the prior committed
    # tensor object stays intact for transactional rollback.
    assert generation_state.x_t is updated
    assert store.get(1) is updated
    assert updated is not latent
    assert torch.equal(latent, committed)


def test_euler_update_preserves_existing_bfloat16_latent_behavior():
    latent = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
    velocity = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)

    updated = euler_step(
        latent,
        velocity,
        torch.tensor(0.8),
        torch.tensor(0.7),
    )

    assert updated.dtype == torch.bfloat16
    assert torch.equal(updated, latent + torch.tensor(-0.1, dtype=torch.bfloat16) * velocity)


class _LanguageModelStub(nn.Module):
    @staticmethod
    def embed_tokens(ids):
        return torch.zeros((ids.shape[0], 4), dtype=torch.bfloat16, device=ids.device)


class _ProjectionCapture(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dtype = None

    def forward(self, latent):
        self.input_dtype = latent.dtype
        return torch.zeros((latent.shape[0], 4), dtype=latent.dtype, device=latent.device)


class _EmbeddingStub(nn.Module):
    @staticmethod
    def forward(values):
        rows = values.shape[0]
        return torch.zeros((rows, 4), dtype=torch.bfloat16, device=values.device)


class _VaeCapture(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.input_dtype = None

    def decode(self, latent):
        self.input_dtype = latent.dtype
        return torch.zeros((1, 3, 4, 6), dtype=latent.dtype, device=latent.device)


def _bare_bagel_graph():
    graph = _BagelGraph.__new__(_BagelGraph)
    nn.Module.__init__(graph)
    graph.cfg = SimpleNamespace(
        llm=SimpleNamespace(hidden_size=4),
        start_of_image_id=1,
        end_of_image_id=2,
        latent_downsample=16,
        latent_patch_size=2,
        latent_channel=16,
    )
    graph.lm = _LanguageModelStub()
    graph.lm_head = nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)
    return graph


def test_bagel_denoiser_casts_fp32_state_only_at_model_compute_boundary():
    graph = _bare_bagel_graph()
    projection = _ProjectionCapture()
    graph.vae2llm = projection
    graph.time_embedder = _EmbeddingStub()
    graph.latent_pos_embed = _EmbeddingStub()
    latent = torch.randn((6, 64), dtype=torch.float32)
    before = latent.clone()

    embeds = graph.gen_segment_embeds(6, torch.arange(6), latent, 0.8)

    assert projection.input_dtype == torch.bfloat16
    assert embeds.dtype == torch.bfloat16
    assert latent.dtype == torch.float32
    assert torch.equal(latent, before)


def test_bagel_decode_casts_fp32_state_at_vae_boundary():
    graph = _bare_bagel_graph()
    vae = _VaeCapture()
    graph.vae = vae
    latent = torch.randn((6, 64), dtype=torch.float32)
    before = latent.clone()

    image = graph.vae_decode(latent, height=32, width=48)

    assert image.size == (6, 4)
    assert vae.input_dtype == torch.bfloat16
    assert latent.dtype == torch.float32
    assert torch.equal(latent, before)
