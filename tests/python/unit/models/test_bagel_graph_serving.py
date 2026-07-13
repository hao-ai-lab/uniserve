from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from uniserve_worker.execution.forward.graph.programs import PackedVisibleGraphProgram
from uniserve_worker.foundation.errors import WorkerError
from uniserve_worker.models import bagel as bagel_model
from uniserve_worker.models.bagel import BagelForUnifiedGeneration, LLMConfig
from uniserve_worker.nn.decoder import Modality, MoTModel

pytestmark = pytest.mark.unit


def test_bagel_binds_packed_visible_mixed_graph_program():
    owner = BagelForUnifiedGeneration(device="cpu")
    program = PackedVisibleGraphProgram(owner=owner, request_states=object())
    plan = SimpleNamespace(
        shape=SimpleNamespace(text_row_count=1, denoise_row_count=1, commit_row_count=0),
        ops=[
            {"req_id": 1, "kind": "decode_und"},
            {"req_id": 2, "kind": "denoise_gen"},
        ],
    )

    eligibility = program.can_run(SimpleNamespace(), plan)

    assert eligibility.eligible is True


def test_bagel_denoise_rejects_eager_execution(monkeypatch):
    owner = BagelForUnifiedGeneration(device="cpu")
    monkeypatch.setattr(owner, "_ensure_loaded", lambda: object())

    with pytest.raises(WorkerError, match="requires CUDA graphs"):
        owner.predict_text_image_velocity_batch([], [], graph_mode="eager")


def test_bagel_mot_packed_visible_routes_marker_tokens_through_text_expert():
    calls = []

    class PackedLayer(nn.Module):
        def forward_packed_visible(
            self,
            layer_idx,
            hidden_states,
            *,
            text_mask,
            gen_mask,
            cos,
            sin,
            forward_stream,
            kv_view,
            any_text,
            any_gen,
        ):
            calls.append(
                {
                    "layer_idx": layer_idx,
                    "text_mask": text_mask.clone(),
                    "gen_mask": gen_mask.clone(),
                    "cos": cos,
                    "sin": sin,
                    "forward_stream": forward_stream,
                    "kv_view": kv_view,
                    "any_text": any_text,
                    "any_gen": any_gen,
                }
            )
            return hidden_states + 1

    class Rotary(nn.Module):
        @staticmethod
        def cos_sin_1d(positions):
            return positions.float().unsqueeze(-1), positions.float().unsqueeze(-1) + 10

    model = MoTModel(
        LLMConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            vocab_size=16,
        )
    )
    model.layers = nn.ModuleList([PackedLayer(), PackedLayer()])
    model.rotary = Rotary()
    model.norm = nn.Identity()
    model.norm_moe_gen = nn.Identity()
    model.final_norm = {
        Modality.TEXT: model.norm,
        Modality.GEN: model.norm_moe_gen,
    }
    hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    # Two ordinary text tokens followed by one BAGEL denoise segment:
    # marker, two latent tokens, marker.
    is_gen = torch.tensor([False, False, False, True, True, False])
    indexes = torch.tensor(
        [
            [0, 1, 7, 7, 7, 7],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    stream = SimpleNamespace(
        segments=(
            SimpleNamespace(modality="und", q_len=2),
            SimpleNamespace(modality="gen", q_len=4),
        )
    )
    kv_view = object()

    output = model.forward_packed_visible(
        hidden,
        image_gen_indicators=is_gen,
        indexes=indexes,
        forward_stream=stream,
        kv_view=kv_view,
    )

    assert len(calls) == 2
    assert all(call["any_text"] is True and call["any_gen"] is True for call in calls)
    assert all(call["text_mask"].tolist() == [True, True, True, False, False, True] for call in calls)
    assert all(call["gen_mask"].tolist() == is_gen.tolist() for call in calls)
    torch.testing.assert_close(output, hidden + 2)


def test_bagel_graph_only_text_hook_preserves_position_and_cache_mirrors():
    calls = []
    synced = []
    past = SimpleNamespace(length=23)
    state = SimpleNamespace(cond=SimpleNamespace(past=past, t_index=-1))
    logits = [torch.tensor([[1.0, 2.0]])]

    class GraphDriver:
        def try_run_text_graph_logits_batch(self, ops):
            calls.append([dict(op) for op in ops])
            return logits

    class Owner:
        _prepare_text_logits_batch = BagelForUnifiedGeneration._prepare_text_logits_batch
        _sync_text_cache_lengths = BagelForUnifiedGeneration._sync_text_cache_lengths

        def _ensure_loaded(self):
            return None

        def interleaved_image_state(self, req_id):
            assert int(req_id) == 7
            return state

        def _text_driver(self):
            return GraphDriver()

        def _set_length(self, req_id, length):
            synced.append((int(req_id), int(length)))

    op = {
        "req_id": 7,
        "kind": "prefill_und",
        "token_ids": [3, 4],
        "pos_range": [11, 13],
    }
    owner = Owner()

    result = BagelForUnifiedGeneration.try_run_text_graph_logits_batch(owner, [op])

    assert result is logits
    assert calls == [[op]]
    assert state.cond.t_index == 10
    assert synced == [(7, 23)]


def test_bagel_denoise_require_mode_uses_shared_graph_rows(monkeypatch):
    cache_a = SimpleNamespace(pool=object())
    cache_b = SimpleNamespace(pool=cache_a.pool)

    class Branches:
        caches = {"cond": cache_a, "text_uncond": cache_b}

        def has_all(self, names):
            return all(name in self.caches for name in names)

        def positions_tensor(self, names, *, device, width):
            del device
            return torch.stack(
                [torch.full((width,), index + 5, dtype=torch.long) for index, _ in enumerate(names)]
            )

    graph_image = SimpleNamespace(token_h=1, token_w=4, height=32, width=32)
    gs = SimpleNamespace(
        paged_branches=Branches(),
        graph_image=graph_image,
        num_vae=2,
        vae_pos_ids=torch.zeros(2, dtype=torch.long),
    )
    step = SimpleNamespace(
        extra={"gs": gs},
        latent=torch.ones((1, 2, 1)),
        t=torch.tensor(0.5),
    )

    class Model:
        def gen_segment_embeds(self, num_vae, vae_pos_ids, latent, timestep):
            assert num_vae == 2
            assert vae_pos_ids is gs.vae_pos_ids
            assert latent is step.latent
            assert timestep == pytest.approx(0.5)
            return torch.arange(12, dtype=torch.float32).reshape(4, 3)

        @staticmethod
        def gen_segment_graph_layout(batch_size, num_vae):
            assert (batch_size, num_vae) == (2, 2)
            return torch.tensor([False, True, True, False]), torch.tensor([0, 3, 4, 7])

    class Owner:
        device = "cpu"
        _predict_text_image_velocity_graph = (
            BagelForUnifiedGeneration._predict_text_image_velocity_graph
        )

        def _ensure_loaded(self):
            return SimpleNamespace(model=Model())

    captured = []

    def graph_forward(owner, rows):
        captured.append((owner, rows))
        return torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    monkeypatch.setattr(bagel_model, "maybe_run_denoise_step_graph", graph_forward)
    owner = Owner()

    result = BagelForUnifiedGeneration.predict_text_image_velocity_batch(
        owner,
        [step],
        [("cond", "text_uncond")],
        graph_mode="require",
    )

    assert len(captured) == 1
    rows = captured[0][1]
    assert [row.branch for row in rows] == ["cond", "text_uncond"]
    assert [row.cache for row in rows] == [cache_a, cache_b]
    assert step.extra["img"] is graph_image
    assert tuple(step.extra["image_embeds"].shape) == (1, 4, 3)
    torch.testing.assert_close(result[0]["cond"], torch.tensor([[1.0], [2.0]]))
    torch.testing.assert_close(result[0]["text_uncond"], torch.tensor([[3.0], [4.0]]))


def test_bagel_denoise_graph_forward_maps_static_positions_and_latent_rows():
    calls = []

    class Language:
        def forward_paged_gen_batch(self, embeds, positions, is_gen, cache, *, text_idx=None):
            calls.append((embeds, positions, is_gen, cache, text_idx))
            return torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    class Model:
        lm = Language()

        @staticmethod
        def gen_segment_graph_layout(batch_size, num_vae):
            assert (batch_size, num_vae) == (2, 2)
            return torch.tensor([False, True, True, False]), torch.tensor([0, 3, 4, 7])

        @staticmethod
        def llm2vae(hidden):
            return hidden[..., :1]

    class Owner:
        def _ensure_loaded(self):
            return SimpleNamespace(model=Model())

    embeds = torch.zeros((2, 4, 3))
    token_by_row_positions = torch.tensor([[5, 8], [5, 8], [5, 8], [5, 8]])
    cache = object()
    velocity = BagelForUnifiedGeneration.interleaved_image_predict_velocity(
        Owner(),
        embeds,
        token_by_row_positions,
        None,
        cache,
        torch.tensor(0.5),
        torch.zeros((2, 2, 1)),
        image_token_num=4,
        image_size=(32, 32),
    )

    assert calls[0][0] is embeds
    torch.testing.assert_close(calls[0][1], token_by_row_positions.transpose(0, 1))
    assert calls[0][3] is cache
    torch.testing.assert_close(calls[0][4], torch.tensor([0, 3, 4, 7]))
    assert tuple(velocity.shape) == (2, 2, 1)


def test_bagel_interleaved_text_decode_uses_row_batched_paged_forward():
    calls = []

    class Language:
        def forward_paged_text_batch(self, embeds, positions, past):
            calls.append((embeds, positions, past))
            return embeds

    class Model:
        lm = Language()

        @staticmethod
        def embed_tokens(input_ids):
            return input_ids.to(torch.bfloat16).unsqueeze(-1).expand(*input_ids.shape, 3)

        @staticmethod
        def logits(hidden):
            return hidden[:, :1].expand(int(hidden.shape[0]), 5)

    class Owner:
        interleaved_text_forward = BagelForUnifiedGeneration.interleaved_text_forward

        def _ensure_loaded(self):
            return SimpleNamespace(model=Model())

    past = object()
    output = Owner().interleaved_text_forward(
        input_ids=torch.tensor([[11], [12]]),
        indexes=torch.tensor([[5, 7]]),
        past_key_values=past,
    )

    assert len(calls) == 1
    embeds, positions, seen_past = calls[0]
    assert tuple(embeds.shape) == (2, 1, 3)
    torch.testing.assert_close(positions, torch.tensor([5, 7]))
    assert seen_past is past
    assert tuple(output.logits.shape) == (2, 1, 5)
