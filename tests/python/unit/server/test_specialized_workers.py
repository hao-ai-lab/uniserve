"""Peeled-stage workers (docs/staged-workers.md §6.1/§6.4/§6.5).

Each peeled stage is a plain ``Worker`` (caps / execute / drop_request)
that shares the same worker shell as the ``full`` worker and differs only in its
declared OpKind subset + device profile. These tests pin that contract.
"""

from __future__ import annotations

import base64

import pytest
import torch

from uniserve_worker.contracts import UniModel
from uniserve_worker.contracts.batches import seal_batch
from uniserve_worker.contracts.caps import validate_caps
from uniserve_worker.contracts.outputs import EncodeOutput
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.runtime.tensor_store import TensorStore
from uniserve_worker.server.app import WorkerServer
from uniserve_worker.server.stub import StubWorker
from uniserve_worker.server.worker_kind import WorkerKind
from uniserve_worker.worker.encoder import EncoderWorker
from uniserve_worker.worker.frame_accumulator import (
    FrameAccumulatorWorker,
)
from uniserve_worker.worker.sampler import SamplerWorker

pytestmark = pytest.mark.unit


class _FakeServer:
    """Minimal endpoint stand-in: WorkerServer.handle() never touches it."""

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


def test_sampler_worker_greedy_samples_argmax_from_handle():
    store = TensorStore()
    worker = SamplerWorker(tensor_store=store)
    validate_caps(worker.caps(), owner="SamplerWorker")
    assert worker.caps().supported_ops == ("sample",)

    logits = torch.full((128,), -10.0)
    logits[42] = 5.0
    handle = store.publish(logits, "logits")
    out = worker.execute(
        seal_batch(
            1,
            [{"req_id": 7, "kind": "sample", "logits_handle": handle}],
            new_reqs=[{"req_id": 7, "sampling": {"temperature": 0.0, "n_logprobs": 0}}],
        )
    )
    assert out["per_seq"][0]["sampled_token_id"] == 42

    worker.free_logits([handle])
    assert handle not in store
    worker.drop_request(7)


def test_frame_accumulator_worker_counts_and_releases_frames():
    worker = FrameAccumulatorWorker()
    validate_caps(worker.caps(), owner="FrameAccumulatorWorker")
    assert worker.caps().supported_ops == ("encode_frame",)
    frame = base64.b64encode(b"\x00\x01\x02").decode()
    out = worker.execute(
        seal_batch(
            2,
            [
                {"req_id": 5, "kind": "encode_frame", "image_b64": frame},
                {"req_id": 6, "kind": "encode_frame", "image_b64": frame},
            ],
            new_reqs=[{"req_id": 5}, {"req_id": 6}],
        )
    )
    assert [s["num_tokens"] for s in out["per_seq"]] == [1, 1]
    worker.execute(
        seal_batch(
            3,
            [{"req_id": 5, "kind": "encode_frame", "image_b64": frame}],
            base_version=1,
        )
    )
    status = worker.release_request_frames(5)
    assert status["frames"] == 2


class _FakeVisionModel(UniModel):
    supported_ops = ("vit_encode",)
    num_layers = 2
    max_latent_size = 0
    latent_downsample = 1
    encoder_cache_budget = 4

    def encode_image(self, pixels, grid=None, *, ctx):
        return EncodeOutput(req_id=ctx.req_id, encoder_handle=999, num_tokens=16)

    def free_encoder(self, handles):
        pass


def test_encoder_worker_dispatches_vision_encode():
    worker = EncoderWorker(_FakeVisionModel())
    validate_caps(worker.caps(), owner="EncoderWorker")
    assert worker.caps().supported_ops == ("vit_encode",)
    assert "free_encoder" in worker.caps().supported_controls
    out = worker.execute(
        seal_batch(
            3,
            [{"req_id": 1, "kind": "vit_encode", "image_in": 123}],
            new_reqs=[{"req_id": 1}],
        )
    )
    assert out["per_seq"][0]["encoder_handle"] == 999
    assert out["per_seq"][0]["num_tokens"] == 16


def test_worker_kind_subsets_match_op_vocabulary():
    assert {"prefill_und", "decode_und"} <= WorkerKind.FULL.supported_ops
    assert WorkerKind.SAMPLER.supported_ops == frozenset({"sample"})


def test_worker_server_rejects_op_outside_worker_contract():
    store = TensorStore()
    runtime = WorkerServer(SamplerWorker(tensor_store=store), _FakeServer())
    # A sampler worker advertises only "sample".
    assert runtime.caps["supported_ops"] == ["sample"]
    # An op outside the subset is rejected (routing/causality guard, §9.2).
    resp = runtime.handle(
        {
            "kind": "execute",
            "batch": seal_batch(
                1,
                [{"req_id": 1, "kind": "decode_und", "token_ids": [1]}],
                new_reqs=[{"req_id": 1}],
            ),
        }
    )
    assert resp["kind"] == "error"
    assert "outside its capability set" in (resp.get("message") or "")


def test_worker_server_uses_exact_worker_contract():
    runtime = WorkerServer(StubWorker(), _FakeServer())
    stub_ops = set(StubWorker.supported_ops)
    assert set(runtime.caps["supported_ops"]) == stub_ops
