"""Batch submission preserves parameters and rejects unsafe KV assignments."""

import pickle
from types import MappingProxyType

import pytest

from uniserve_worker.errors import WorkerError
from uniserve_worker.protocol.batch import (
    Batch,
    BlockTable,
    CacheUnitAllocation,
    CanvasSampling,
    GenerationParams,
    NewRequest,
    Start,
)
from uniserve_worker.protocol.call import (
    Bounds,
    Call,
    CallCoordinates,
    ForwardMode,
)
from uniserve_worker.protocol.identity import CallId, RequestKey

pytestmark = pytest.mark.unit


def _batch_fields():
    request = RequestKey(1, 2, 3)
    canvas = CanvasSampling(16, 4, 0.5, 0.4, 0.8, 0.2, 1)
    return {
        "batch_id": 7,
        "calls": (
            Call(
                request,
                CallId(7, 0),
                CallCoordinates(),
                ForwardMode.PREFILL,
                Bounds(max_tokens=2),
                input_token_ids=(3, 7),
            ),
        ),
        "commands": (
            Start(NewRequest(request, 1, GenerationParams(canvas=canvas))),
        ),
        "block_tables": (BlockTable(1, 0, 0, (1, 2), 32),),
        "new_cache_units": (CacheUnitAllocation(1, 0, (1,)),),
        "forward_call_indices": (0,),
        "request_pool_indices": (1,),
        "seq_lens": (2,),
        "query_lens": (2,),
        "write_kv": (True,),
    }


def test_batch_serialization_and_replacement_preserve_submission():
    fields = _batch_fields()
    batch = Batch(**fields)
    assert batch.admissions[0] == fields["commands"][0].request
    for restored in (
        Batch.from_mapping(batch.to_mapping()),
        pickle.loads(pickle.dumps(batch)),
    ):
        assert restored == batch
        assert restored.calls[0].input_token_ids == (3, 7)
        assert restored.admissions == (batch.commands[0].request,)
        assert restored.block_tables == batch.block_tables
        assert restored.new_cache_units == batch.new_cache_units
        assert restored.request_pool_indices == (1,)
        assert restored.write_kv == (True,)

    changed = batch.replace(new_cache_units=(CacheUnitAllocation(1, 0, (2,)),))
    assert changed.new_cache_units[0].unit_ids == (2,)
    assert batch.new_cache_units[0].unit_ids == (1,)
    assert changed.calls == batch.calls


@pytest.mark.parametrize("source", ["constructor", "mapping", "replacement"])
def test_batch_rejects_cache_units_outside_assigned_table(source):
    fields = _batch_fields()
    batch = Batch(**fields)
    unassigned = CacheUnitAllocation(1, 0, (3,))

    with pytest.raises(WorkerError, match="outside its block table"):
        if source == "constructor":
            Batch(**(fields | {"new_cache_units": (unassigned,)}))
        elif source == "mapping":
            data = batch.to_mapping()
            data["new_cache_units"] = [unassigned.to_mapping()]
            Batch.from_mapping(data)
        else:
            batch.replace(new_cache_units=(unassigned,))


def test_batch_rejects_canvas_without_denoising_steps():
    data = Batch(**_batch_fields()).to_mapping()
    data["commands"][0]["value"]["request"]["ar"]["canvas"]["max_steps"] = 0

    with pytest.raises(WorkerError, match="canvas sampling max_steps"):
        Batch.from_mapping(data)


@pytest.mark.parametrize("mapping", (dict, MappingProxyType))
def test_sampling_mapping_rejects_boolean_token_limits(mapping):
    with pytest.raises(WorkerError):
        GenerationParams.from_mapping({"sampling": mapping({"top_k": True})})
