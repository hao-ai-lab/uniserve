from __future__ import annotations

import pytest
import torch

from uniserve_worker.server.distributed import build_device_mesh

pytestmark = pytest.mark.unit


def test_tp_cuda_mesh_rejects_unavailable_rank_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(Exception, match=r"cuda device cuda:1 is outside the 1 visible CUDA device"):
        build_device_mesh(tp_rank=1, tp_size=2, device="cuda", tp_init_method="tcp://127.0.0.1:1")


def test_tower_cuda_mesh_rejects_unavailable_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(Exception, match=r"cuda device cuda:1 is outside the 1 visible CUDA device"):
        build_device_mesh(tp_rank=0, tp_size=1, device="cpu", tower_devices=["cuda:0", "cuda:1"])
