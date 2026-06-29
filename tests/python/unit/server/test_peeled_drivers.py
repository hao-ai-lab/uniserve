"""Peeled-stage worker drivers (docs/staged-workers.md §6.1/§6.4/§6.5).

Each peeled stage is a plain ``WorkerDriver`` (caps / execute / drop_request)
that shares the same worker shell as the ``full`` worker and differs only in its
declared OpKind subset + device profile. These tests pin that contract.
"""
from __future__ import annotations

import base64

import pytest
import torch

from uniserve_worker.contracts.caps import validate_caps
from uniserve_worker.contracts.model_protocols import ModelHooks
from uniserve_worker.contracts.outputs import EncodeOutput
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.tensor_store import TensorStore
from uniserve_worker.server.app import WorkerRuntime
from uniserve_worker.server.encode_only_driver import EncodeOnlyDriver
from uniserve_worker.server.postprocess_driver import PostProcessDriver
from uniserve_worker.server.sampler_driver import SamplerDriver
from uniserve_worker.server.worker_kind import (
    FULL,
    SAMPLER,
    SUPPORTED_OPS,
    restrict_supported_ops,
)

pytestmark = pytest.mark.unit


class _FakeServer:
    """Minimal server stand-in: WorkerRuntime.handle() never touches it."""

    def respond(self, _resp):  # pragma: no cover - unused in handle()
        raise AssertionError("respond should not be called in these tests")


def test_tensor_store_publish_fetch_release_roundtrip():
    store = TensorStore(id_base=1000)
    t = torch.arange(4)
    handle = store.publish(t, "logits")
    assert handle > 1000
    assert torch.equal(store.fetch(handle), t)
    assert store.kind_of(handle) == "logits"
    store.release(handle)
    with pytest.raises(WorkerError):
        store.fetch(handle)


def test_sampler_driver_greedy_samples_argmax_from_handle():
    store = TensorStore()
    driver = SamplerDriver(tensor_store=store)
    validate_caps(driver.caps(), owner="SamplerDriver")
    assert driver.caps().supported_ops == ("sample",)

    logits = torch.full((128,), -10.0)
    logits[42] = 5.0
    handle = store.publish(logits, "logits")
    out = driver.execute(
        {
            "step_id": 1,
            "new_reqs": [{"req_id": 7, "sampling": {"temperature": 0.0, "n_logprobs": 0}}],
            "ops": [{"req_id": 7, "kind": "sample", "logits_handle": handle}],
        }
    )
    assert out["per_seq"][0]["sampled_token_id"] == 42

    driver.free_logits([handle])
    assert handle not in store
    driver.drop_request(7)


def test_postprocess_driver_counts_frames_and_finalizes():
    driver = PostProcessDriver()
    validate_caps(driver.caps(), owner="PostProcessDriver")
    assert driver.caps().supported_ops == ("encode_frame",)
    frame = base64.b64encode(b"\x00\x01\x02").decode()
    out = driver.execute(
        {
            "step_id": 2,
            "new_reqs": [],
            "ops": [
                {"req_id": 5, "kind": "encode_frame", "image_b64": frame},
                {"req_id": 5, "kind": "encode_frame", "image_b64": frame},
            ],
        }
    )
    assert [s["num_tokens"] for s in out["per_seq"]] == [1, 2]
    status = driver.finalize_video(5)
    assert status["frames"] == 2


class _FakeVisionModel(ModelHooks):
    num_layers = 2
    max_latent_size = 0
    latent_downsample = 1
    encoder_cache_budget = 4

    def encode_image(self, pixels, grid, op=None):
        return EncodeOutput(req_id=int(op["req_id"]), encoder_handle=999, num_tokens=16)

    def free_encoder(self, handles):
        pass


def test_encode_only_driver_dispatches_vision_encode():
    driver = EncodeOnlyDriver(_FakeVisionModel())
    validate_caps(driver.caps(), owner="EncodeOnlyDriver")
    assert driver.caps().supported_ops == ("vit_encode",)
    assert "free_encoder" in driver.caps().supported_controls
    out = driver.execute(
        {
            "step_id": 3,
            "new_reqs": [{"req_id": 1}],
            "ops": [{"req_id": 1, "kind": "vit_encode", "mm_hash": 123}],
        }
    )
    assert out["per_seq"][0]["encoder_handle"] == 999
    assert out["per_seq"][0]["num_tokens"] == 16


def test_worker_kind_subsets_match_op_vocabulary():
    # full's subset is exactly the model-op vocabulary (every model op is in it).
    assert restrict_supported_ops(FULL, ["prefill_und", "decode_und"]) == [
        "prefill_und",
        "decode_und",
    ]
    assert SUPPORTED_OPS[SAMPLER] == frozenset({"sample"})


def test_worker_runtime_rejects_op_outside_peeled_subset():
    store = TensorStore()
    runtime = WorkerRuntime(
        SamplerDriver(tensor_store=store), _FakeServer(), worker_kind=SAMPLER
    )
    # A sampler worker advertises only "sample".
    assert runtime.caps["supported_ops"] == ["sample"]
    # An op outside the subset is rejected (routing/causality guard, §9.2).
    resp = runtime.handle(
        {
            "kind": "execute",
            "batch": {
                "step_id": 1,
                "new_reqs": [],
                "ops": [{"req_id": 1, "kind": "decode_und", "token_ids": [1]}],
            },
        }
    )
    assert resp["kind"] == "error"
    assert "outside its OpKind subset" in (resp.get("message") or "")


def test_worker_runtime_full_kind_is_unrestricted():
    # `full` advertises the model's ops verbatim and applies no gate.
    from uniserve_worker.server.stub import StubEngine

    runtime = WorkerRuntime(StubEngine(), _FakeServer(), worker_kind=FULL)
    stub_ops = set(StubEngine.supported_ops)
    assert set(runtime.caps["supported_ops"]) == stub_ops
