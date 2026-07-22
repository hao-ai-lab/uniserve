"""Product transaction conformance for executor-driven image commits.

A failed commit must leave every touched request's previously committed
products visible in the ``ProductStore`` and its generation scratch state
restorable, no matter whether the failure lands before or after the neural
decode or before or after the KV writeback. Retrying the same operation must
then produce the committed product exactly once.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from uniserve_worker.contracts import UniModel
from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.contracts.resource_plan import ResourcePlan
from uniserve_worker.execution import ExecutorConfig, ModelExecutor
from uniserve_worker.runtime.image_utils import pil_image_to_png_b64
from uniserve_worker.runtime.residency import KvCacheSpec, ResidencyManager
from uniserve_worker.runtime.resources import ResourceRuntime

pytestmark = pytest.mark.integration


class GenerationCommitCPUModel(UniModel):
    """Commit-capable CPU model exposing only the neural commit entries.

    ``vae_decode`` derives a deterministic image from the latent value and
    ``commit_generated_kv`` records the KV writeback; per-request failure sets
    inject faults at the four product boundaries the spec names: before and
    after the neural decode, and before and after the KV writeback.
    """

    resource_classes = ("kv_block",)
    resource_plan = ResourcePlan(kv_block="per_block")
    supported_ops = ("prefill_und", "decode_und", "commit_gen")
    adapter_mode = "none"
    device = "cpu"
    num_layers = 1
    num_blocks = 8
    block_size = 16
    eos_id = 2
    img_start_id = 3

    def __init__(self, residency: ResidencyManager) -> None:
        self.residency = residency
        self.segment_executor = SimpleNamespace(release_staging=lambda cache: None)
        self.fail_before_decode: set[int] = set()
        self.fail_after_decode: set[int] = set()
        self.fail_before_writeback: set[int] = set()
        self.fail_after_writeback: set[int] = set()
        self.writebacks: list[tuple[int, tuple[int, ...]]] = []

    def vae_decode(self, latent: Any, *, height=None, width=None) -> Image.Image:
        req_id = int(latent.reshape(-1)[0].item())
        if req_id in self.fail_before_decode:
            raise RuntimeError("injected failure before image decode")
        image = Image.new("RGB", (int(width), int(height)), (req_id % 256, 0, 0))
        if req_id in self.fail_after_decode:
            raise RuntimeError("injected failure after image decode")
        return image

    def commit_generated_kv(self, req_id: int, gen_state: Any, block_ids: Any) -> int:
        if int(req_id) in self.fail_before_writeback:
            raise RuntimeError("injected failure before image writeback")
        self.writebacks.append((int(req_id), tuple(int(b) for b in block_ids)))
        if int(req_id) in self.fail_after_writeback:
            raise RuntimeError("injected failure after image writeback")
        return int(gen_state.num_vae) + 2


def _fresh() -> tuple[GenerationCommitCPUModel, ModelExecutor]:
    ledger = ResourceRuntime(("kv_block",), totals={"kv_block": 8})
    residency = ResidencyManager.build(
        KvCacheSpec(num_layers=1, num_kv_heads=1, head_dim=4, dtype=torch.float32),
        num_blocks=8,
        block_size=16,
        device="cpu",
        ledger=ledger,
    )
    model = GenerationCommitCPUModel(residency)
    executor = ModelExecutor(
        model,
        config=ExecutorConfig(simulation=True),
        resource_runtime=ledger,
        residency=residency,
    )
    assert executor.product_store is not None and executor.latent_store is not None
    return model, executor


def _gen_state(req_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        x_t=torch.tensor([float(req_id)]),
        H=8,
        W=8,
        num_vae=2,
        vae_pos_ids=None,
        cond_pos=3,
    )


def _commit_batch(step_id: int, req_ids: list[int], **kwargs: Any) -> dict[str, Any]:
    return seal_batch(
        step_id,
        [{"req_id": req_id, "kind": "commit_gen"} for req_id in req_ids],
        **kwargs,
    )


def _expected_png_b64(req_id: int) -> str:
    return pil_image_to_png_b64(Image.new("RGB", (8, 8), (req_id % 256, 0, 0)))


def test_commit_persists_frame_reclaims_state_and_reports_writeback():
    model, executor = _fresh()
    executor.latent_store.set_state(5, _gen_state(5))

    result = executor.execute(
        _commit_batch(1, [5], new_reqs=[{"req_id": 5, "block_ids": [0]}])
    )

    seq = result["per_seq"][0]
    assert seq["image_png_b64"] == _expected_png_b64(5)
    assert seq["image_hw"] == [8, 8]
    assert seq["num_tokens"] == 4
    assert executor.product_store.frames(5) != ()
    assert executor.latent_store.state(5) is None
    assert model.writebacks == [(5, (0,))]


def test_retain_images_off_skips_the_kv_writeback_but_still_materializes():
    model, executor = _fresh()
    executor.latent_store.set_state(6, _gen_state(6))

    result = executor.execute(
        _commit_batch(
            1,
            [6],
            new_reqs=[{"req_id": 6, "block_ids": [], "image": {"retain_images": False}}],
        )
    )

    seq = result["per_seq"][0]
    assert seq["image_png_b64"] == _expected_png_b64(6)
    assert seq["num_tokens"] == 0
    assert model.writebacks == []
    assert executor.product_store.frames(6) != ()
    assert executor.latent_store.state(6) is None


@pytest.mark.parametrize(
    "failure_site",
    ["fail_before_decode", "fail_after_decode", "fail_before_writeback", "fail_after_writeback"],
)
def test_failed_commit_keeps_previous_product_and_restores_generation_scratch(failure_site):
    model, executor = _fresh()

    # First image commits cleanly and becomes the previous committed product.
    executor.latent_store.set_state(5, _gen_state(5))
    executor.execute(_commit_batch(1, [5], new_reqs=[{"req_id": 5, "block_ids": [0]}]))
    committed_frames = executor.product_store.frames(5)
    assert len(committed_frames) == 1
    writebacks_before = list(model.writebacks)

    # Second image fails at the injected product boundary.
    seeded = _gen_state(5)
    executor.latent_store.set_state(5, seeded)
    getattr(model, failure_site).add(5)
    with pytest.raises(RuntimeError, match="injected failure"):
        executor.execute(_commit_batch(2, [5], base_version=1))
    getattr(model, failure_site).clear()

    # The previous committed product stays visible and the in-flight image's
    # scratch state is restored, so the same operation can be retried.
    assert executor.product_store.frames(5) == committed_frames
    assert executor.latent_store.state(5) is seeded
    assert executor.sessions.get(5).version == 1

    retry = executor.execute(_commit_batch(3, [5], base_version=1))
    assert retry["per_seq"][0]["image_png_b64"] == _expected_png_b64(5)
    frames = executor.product_store.frames(5)
    assert len(frames) == 2 and frames[0] == committed_frames[0]
    assert executor.latent_store.state(5) is None
    expected_retry_writebacks = writebacks_before + [(5, (0,))]
    if failure_site == "fail_after_writeback":
        # The injected fault lands after the writeback forward ran; the retry
        # then repeats it, and the step transaction guarantees the failed
        # attempt left no committed effects behind.
        expected_retry_writebacks = writebacks_before + [(5, (0,)), (5, (0,))]
    assert model.writebacks == expected_retry_writebacks


def test_failure_in_sibling_commit_rolls_back_the_whole_steps_products():
    model, executor = _fresh()
    executor.latent_store.set_state(5, _gen_state(5))
    executor.latent_store.set_state(6, _gen_state(6))

    # Request 5 materializes fully before request 6 fails; the step
    # transaction must discard request 5's provisionally appended frame.
    model.fail_before_decode.add(6)
    with pytest.raises(RuntimeError, match="injected failure before image decode"):
        executor.execute(
            _commit_batch(
                1,
                [5, 6],
                new_reqs=[
                    {"req_id": 5, "block_ids": [0]},
                    {"req_id": 6, "block_ids": [1]},
                ],
            )
        )
    model.fail_before_decode.clear()

    assert executor.product_store.frames(5) == ()
    assert executor.product_store.frames(6) == ()
    assert executor.latent_store.state(5) is not None
    assert executor.latent_store.state(6) is not None

    result = executor.execute(
        _commit_batch(
            2,
            [5, 6],
            new_reqs=[
                {"req_id": 5, "block_ids": [0]},
                {"req_id": 6, "block_ids": [1]},
            ],
        )
    )
    assert [seq["image_png_b64"] for seq in result["per_seq"]] == [
        _expected_png_b64(5),
        _expected_png_b64(6),
    ]
    assert len(executor.product_store.frames(5)) == 1
    assert len(executor.product_store.frames(6)) == 1
    assert executor.latent_store.state(5) is None
    assert executor.latent_store.state(6) is None
