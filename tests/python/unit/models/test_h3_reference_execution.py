"""Request-to-transformer input contract with real modules and synthetic weights."""

import pytest
import torch
from torch import nn

from tests.python.unit.models.test_h3_image_conditioning import encoder  # noqa: F401
from uniserve_worker.execution.batch import (
    DecodedReference,
    DiffusionRequestParams,
    DType,
    MediaGeometry,
    PointRange,
    ProductKind,
    ProductRef,
    RequestKey,
    ShapeBound,
    StaticDim,
    StorageClass,
)
from uniserve_worker.execution.bounded_storage import BoundedTensorStorage
from uniserve_worker.models.minimax_h3.layout import H3Layout
from uniserve_worker.models.minimax_h3.model import MiniMaxH3Model
from uniserve_worker.models.minimax_h3.packing import audio_latent_frames
from uniserve_worker.models.minimax_h3.transformer import MiniMaxH3Transformer
from uniserve_worker.models.minimax_h3.video_vae import H3ImagePosterior, MiniMaxH3VideoVAE
from uniserve_worker.models.minimax_h3.weights import H3Components
from uniserve_worker.nn.diffusion.modulation import ModulationPlan
from uniserve_worker.nn.mesh import Communicator, DeviceMesh, EntryBindings
from uniserve_worker.nn.parallel import EntryConfig, ParallelConfig
from uniserve_worker.runtime.distributed import DistributedEnvironment

pytestmark = pytest.mark.unit


@torch.inference_mode()
def test_reference_request_reaches_transformer_without_colocated_decoder(encoder):  # noqa: F811
    pixels = ProductRef(
        RequestKey(1, 1, 1),
        1,
        65535,
        1,
        ProductKind.TENSOR,
        StorageClass.HOST_STAGING,
        DType.U8,
        ShapeBound(tuple(StaticDim(n) for n in (1, 64, 64, 3))),
        PointRange(),
    )
    media = DiffusionRequestParams(
        (42, 87, 99),
        123,
        MediaGeometry(39, 2, 3, 1),
        (DecodedReference("image", "reference", "reference", False, pixels, None, 0, 1),),
    )
    config = ParallelConfig()
    mesh = DeviceMesh((0,), 0, config)
    bindings = EntryBindings(
        {"denoiser": EntryConfig((0,), config)}, {"denoiser": mesh}, Communicator()
    )
    transformer = MiniMaxH3Transformer(
        mesh,
        parameter_device="meta",
        attention_linear_precision="bf16",
        mlp_linear_precision="bf16",
        attention="dense",
    )
    # Denoising weights are not consumed by preparation. Keep the full real
    # transformer on meta; only its real rotary module executes in this test.
    transformer.modulation_plan = ModulationPlan(
        torch.empty(1, 50, 3, 6 * 5376, device="meta"),
        torch.empty(1, 3, 2 * 5376, device="meta"),
    )
    posterior = H3ImagePosterior(parameter_device="cpu")
    for parameter in posterior.parameters():
        parameter.zero_()
    image_vae = MiniMaxH3VideoVAE(posterior, linear_precision="fp32")
    conditioner = nn.Sequential(nn.Linear(16, 5376, dtype=torch.bfloat16))
    capacity = H3Layout.build(
        bindings,
        frames=39,
        text_rows=1024,
        audio_frames=audio_latent_frames(39),
        height=480,
        width=832,
        attention="dense",
        reference_shape=(64, 64),
        presentation_tags=torch.ones(1024, dtype=torch.long),
    )
    model = MiniMaxH3Model(
        bindings,
        H3Components(transformer, conditioner, None, None, None, image_vae),
        capacity,
        denoise_steps=1,
        presentation_processor=encoder.processor,
    )
    geometry = model.media_geometry(media)
    environment = DistributedEnvironment(0, 1, torch.device("cpu"), "gloo")
    scratch = BoundedTensorStorage.allocate(model.scratch_schema, "cpu", environment=environment)
    execution = model.build_execution(geometry, scratch, None)
    storage = BoundedTensorStorage(
        {
            name: torch.zeros(spec.shape, dtype=spec.dtype)
            for name, spec in model.resource_geometry.request_tensors.items()
        }
    )
    slot = model.request_tensors(storage, geometry, execution)
    image = torch.arange(64 * 64 * 3).remainder(256).to(torch.uint8).reshape(1, 64, 64, 3)
    states, tags = encoder.numerical_entry(torch.tensor([media.prompt_token_ids]), image)
    refined = conditioner(states)
    for destination, source in model.initialize_tensors(slot, media.seed):
        destination.copy_(source)
    video_before, audio_before = slot.video_rows.clone(), slot.audio_rows.clone()
    model.prepare_tensors(
        slot,
        execution,
        refined,
        tags.numel(),
        presentation_tags=tags,
        reference_image=image[0],
    )
    # FastVideo's image presentation is six label tokens, 64 merged patches,
    # two vision boundaries, and three prompt tokens; VAE patches are separate.
    assert tags.tolist() == [1] * 6 + [0] * 66 + [1] * 3
    assert torch.equal(slot.text_condition[:, :75], refined)
    assert not slot.text_condition[:, 75:].count_nonzero()
    packed = execution.layout.packed
    assert packed.text_indices.numel() == 128
    assert packed.reference_indices.tolist() == list(range(128, 132))
    assert packed.audio_indices[0] == 192
    assert slot.reference_rows.shape == (4, 96)
    assert torch.isfinite(slot.reference_rows).all()
    assert slot.reference_rows.count_nonzero()
    # Zero posterior weights imply N(0, 1), sampled with the checkpoint's
    # independent seed 42 and FP16 round-trip before channel normalization.
    noise = (
        torch.randn((1, 24, 1, 4, 4), generator=torch.Generator().manual_seed(42)).half().float()
    )
    first_patch = (noise[0, 0, 0, :2, :2].flatten() - 0.858090341091156) / 1.2223774194717407
    assert torch.equal(slot.reference_rows[0, :4], first_patch)
    assert torch.equal(slot.video_rows.view(torch.uint8), video_before.view(torch.uint8))
    assert torch.equal(slot.audio_rows.view(torch.uint8), audio_before.view(torch.uint8))
    # Fixed image time is 75; target audio/video begin at 76, not page 192.
    positions = execution.scratch.rotary_positions
    assert torch.equal(positions[packed.reference_indices, 0], torch.full((4,), 75.0))
    assert positions[packed.audio_indices[0], 0] == 76
    assert positions[packed.video_untile_indices[0], 0] == 76
    cosine, sine = transformer.rope(positions)
    assert torch.equal(slot.rotary_cosine, cosine)
    assert torch.equal(slot.rotary_sine, sine)
    with pytest.raises(ValueError, match="presentation tags"):
        model.prepare_tensors(
            slot,
            execution,
            refined,
            len(media.prompt_token_ids),
            presentation_tags=tags,
            reference_image=image[0],
        )

    plain_media = DiffusionRequestParams(media.prompt_token_ids, media.seed, media.geometry)
    plain_geometry = model.media_geometry(plain_media)
    plain_execution = model.build_execution(plain_geometry, scratch, None)
    plain_slot = model.request_tensors(storage, plain_geometry, plain_execution)
    plain_conditioning = conditioner(
        encoder.numerical_entry(torch.tensor([media.prompt_token_ids]))
    )
    for destination, source in model.initialize_tensors(plain_slot, media.seed):
        destination.copy_(source)
    model.prepare_tensors(plain_slot, plain_execution, plain_conditioning, 3)
    assert torch.equal(
        plain_slot.text_condition[:, :3].view(torch.uint8), plain_conditioning.view(torch.uint8)
    )
    assert torch.equal(plain_slot.video_rows.view(torch.uint8), video_before.view(torch.uint8))
    assert torch.equal(plain_slot.audio_rows.view(torch.uint8), audio_before.view(torch.uint8))
    environment.close()
