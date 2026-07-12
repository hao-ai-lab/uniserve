from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from uniserve_worker.execution.denoise_driver import TextImageDenoiseStep
from uniserve_worker.execution.text_image_denoise_session import TextImageDenoiseSession
from uniserve_worker.models.bagel import (
    BagelConfig,
    BagelForUnifiedGeneration,
    _BagelGraph,
)
from uniserve_worker.nn.diffusion import euler_step
from uniserve_worker.nn.diffusion.noise import init_latent
from uniserve_worker.runtime.request_state import RequestState

pytestmark = pytest.mark.unit


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


class _InitialNoiseGraph:
    cfg = SimpleNamespace(patch_latent_dim=64, timestep_shift=3.0)

    @staticmethod
    def latent_hw(height, width):
        return height // 16, width // 16

    @staticmethod
    def latent_position_ids(height, width):
        return torch.arange((height // 16) * (width // 16))

    @staticmethod
    def new_cache():
        return object()


def test_bagel_generation_state_keeps_exact_cpu_float32_initial_noise(monkeypatch):
    owner = BagelForUnifiedGeneration(config=BagelConfig(), device="cpu")
    graph = _InitialNoiseGraph()
    monkeypatch.setattr(owner, "_ensure_loaded", lambda: SimpleNamespace(model=graph))
    request_id = 5
    owner.states[request_id] = RequestState(seed=999)
    owner.generation_session.begin_request(
        request_id,
        image={
            "width": 48,
            "height": 32,
            "steps": 50,
            "cfg_text_scale": 1.0,
            "cfg_img_scale": 1.0,
            "cfg_interval": [0.4, 1.0],
            "cfg_renorm_type": "global",
            "cfg_renorm_min": 0.0,
            "timestep_shift": 3.0,
            "seed": 0,
        },
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        expected = torch.randn((6, 64), device="cpu", dtype=torch.float32)

    owner._init_gen({"req_id": request_id, "cond_pos": 0})

    state = owner._gen_state(request_id)
    assert state is not None
    assert state.x_t.dtype == torch.float32
    assert torch.equal(state.x_t, expected)


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
def test_bagel_euler_update_keeps_fp32_state_and_matches_reference_arithmetic(device):
    owner = BagelForUnifiedGeneration(config=BagelConfig(), device=device)
    latent = owner._init_generation_noise((2, 3), seed=7)
    velocity = torch.tensor(
        [[0.125, -0.25, 0.5], [-0.75, 1.0, -1.25]],
        dtype=torch.bfloat16,
        device=device,
    )
    t = torch.tensor(0.8, dtype=torch.float32, device=device)
    t_next = torch.tensor(0.7, dtype=torch.float32, device=device)
    expected = latent - velocity * (t - t_next)
    generation_state = SimpleNamespace(x_t=latent)
    step = TextImageDenoiseStep(
        req_id=1,
        state=RequestState(),
        op={},
        latent=latent,
        t=t,
        t_next=t_next,
        step_index=0,
        total_steps=1,
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_type="global",
        cfg_renorm_min=0.0,
        extra={"gs": generation_state},
    )

    session = TextImageDenoiseSession(
        owner,
        step,
        combine_velocity=lambda _step, values: values["cond"],
        accept_update=lambda model, current_step, updated: model.apply_denoise_update(
            current_step, updated
        ),
    )
    session.apply_update({"cond": velocity})

    assert generation_state.x_t.dtype == torch.float32
    assert torch.equal(generation_state.x_t, expected)


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
