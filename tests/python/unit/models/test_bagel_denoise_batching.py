from types import SimpleNamespace

import pytest
import torch

from uniserve_worker.execution.runner import PreparedFlowStep
from uniserve_worker.models.bagel import BagelConfig, BagelForUnifiedGeneration
from uniserve_worker.nn.diffusion.cfg import Branch
from uniserve_worker.runtime.paged_denoise import PagedDenoiseBranchSet

pytestmark = pytest.mark.unit


class _Cache:
    def __init__(self, pool):
        self.pool = pool
        self.layers = [object()]


class _LanguageModel:
    def __init__(self):
        self.calls = []

    def forward_paged_gen_batch(self, inputs, positions, is_gen, cache):
        self.calls.append((inputs.clone(), positions.clone(), is_gen.clone(), cache))
        return inputs


class _Graph:
    def __init__(self):
        self.lm = _LanguageModel()

    def gen_segment_embeds(self, num_vae, vae_pos_ids, latent, timestep):
        del vae_pos_ids
        marker = latent.new_full((1, latent.shape[-1]), float(timestep))
        return torch.cat((marker, latent + float(timestep), marker), dim=0)

    @staticmethod
    def gen_segment_is_gen(num_vae):
        return torch.tensor([False, *([True] * int(num_vae)), False])

    @staticmethod
    def gen_segment_graph_layout(batch_size, num_vae):
        total = int(num_vae) + 2
        return (
            torch.tensor([False, *([True] * int(num_vae)), False]),
            torch.tensor(
                [
                    offset
                    for row in range(int(batch_size))
                    for offset in (row * total, (row + 1) * total - 1)
                ]
            ),
        )

    @staticmethod
    def llm2vae(hidden):
        return hidden


def _step(req_id, timestep, latent, branches):
    return PreparedFlowStep(
        req_id=req_id,
        state=None,
        op={},
        latent=latent,
        t=torch.tensor(timestep),
        t_next=torch.tensor(timestep - 0.1),
        step_index=0,
        total_steps=10,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_type="global",
        cfg_renorm_min=0.0,
        extra={
            "gs": SimpleNamespace(
                num_vae=2,
                vae_pos_ids=torch.zeros(2),
                paged_branches=branches,
                graph_image=SimpleNamespace(token_h=1, token_w=4, height=32, width=32),
            )
        },
    )


def test_bagel_coalesces_compatible_requests_into_one_denoise_forward(monkeypatch):
    owner = BagelForUnifiedGeneration(config=BagelConfig(), device="cpu")
    graph = _Graph()
    monkeypatch.setattr(owner, "_ensure_loaded", lambda: SimpleNamespace(model=graph))
    pool = object()
    branch_names = (Branch.COND, Branch.TEXT_UNCOND)

    def paged(position):
        return PagedDenoiseBranchSet(
            caches={name.value: _Cache(pool) for name in branch_names},
            positions={name.value: position + offset for offset, name in enumerate(branch_names)},
        )

    first_latent = torch.arange(8, dtype=torch.float32).view(2, 4)
    second_latent = first_latent + 10
    steps = [
        _step(1, 0.8, first_latent, paged(20)),
        _step(2, 0.6, second_latent, paged(40)),
    ]
    captured = []

    def graph_forward(rows):
        captured.append(rows)
        return torch.stack(
            [row.step.latent + float(row.step.t) for row in rows],
            dim=0,
        )

    monkeypatch.setattr(owner.flow_graph_execution, "maybe_run_graph", graph_forward)

    outputs = owner.predict_flow_velocity_batch(steps, [branch_names, branch_names])

    assert len(captured) == 1
    assert [row.step.req_id for row in captured[0]] == [1, 1, 2, 2]
    assert [row.branch for row in captured[0]] == ["cond", "text_uncond"] * 2
    assert list(outputs[0]) == list(branch_names)
    assert list(outputs[1]) == list(branch_names)
    torch.testing.assert_close(outputs[0]["cond"], first_latent + 0.8)
    torch.testing.assert_close(outputs[1]["cond"], second_latent + 0.6)
