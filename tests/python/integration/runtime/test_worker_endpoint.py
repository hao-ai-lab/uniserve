"""Endpoint ownership across real native IPC and worker launch failures."""

import json
import socket
from uuid import uuid4

import pytest

from tests.python.fixtures.launch import worker_args
from uniserve_worker.bootstrap.launch import WorkerIpcEndpoint, run_worker
from uniserve_worker.protocol.call import ForwardMode
from uniserve_worker.protocol.identity import CallId

pytestmark = pytest.mark.integration


def _model_config(directory, **overrides):
    """Launch one rank against a model path that cannot load."""
    directory.mkdir(parents=True, exist_ok=True)
    return worker_args(
        directory,
        ipc_payload_cap=65536,
        max_batch_tokens=256,
        device="cpu",
        model=str(directory),
        **overrides,
    )


def _registration_listener():
    """Stand in for the head, which binds a rank's channel from its report."""
    listener = socket.create_server(("127.0.0.1", 0))
    return listener, "{}:{}".format(*listener.getsockname())


def _reported_endpoint(listener):
    """Return the endpoint the rank named for itself."""
    connection, _ = listener.accept()
    with connection:
        connection.settimeout(60)
        return json.loads(connection.makefile("r").readline())["endpoint"]


def test_native_endpoint_close_releases_service_and_rejects_io():
    service = f"worker-{uuid4().hex}"
    endpoint = WorkerIpcEndpoint(service, max_payload=65536)
    assert not endpoint.closed
    assert endpoint.try_recv() is None
    endpoint.close()
    endpoint.close()
    assert endpoint.closed
    with pytest.raises(RuntimeError, match="closed"):
        endpoint.try_recv()
    with pytest.raises(RuntimeError, match="closed"):
        endpoint.wake()
    with pytest.raises(RuntimeError, match="closed"):
        with endpoint:
            pytest.fail("a closed endpoint cannot enter a service scope")
    with WorkerIpcEndpoint(service, max_payload=65536) as replacement:
        assert replacement.try_recv() is None


@pytest.mark.parametrize("failing_scope", (False, True))
def test_endpoint_scope_closes_and_preserves_body_error(failing_scope):
    service = f"worker-{uuid4().hex}"
    endpoint = WorkerIpcEndpoint(service, max_payload=65536)
    failure = RuntimeError("service startup failed")
    try:
        with endpoint as bound:
            assert bound is endpoint
            assert bound.try_recv() is None
            if failing_scope:
                raise failure
    except RuntimeError as error:
        assert error is failure
    else:
        assert not failing_scope, (
            "endpoint scope suppressed the startup exception"
        )
    assert endpoint.closed
    with WorkerIpcEndpoint(service, max_payload=65536) as replacement:
        assert replacement.try_recv() is None


def test_model_loading_failure_releases_the_reported_endpoint(tmp_path):
    # A rank reports the endpoint it created before it loads anything, so the
    # endpoint has to survive the report and be released by the failure.
    listener, address = _registration_listener()
    config = _model_config(tmp_path / "launch", registration_address=address)
    with listener:
        with pytest.raises(FileNotFoundError, match="modular_model_index.json"):
            run_worker(config)
        service = _reported_endpoint(listener)
    with WorkerIpcEndpoint(service, max_payload=65536) as endpoint:
        assert endpoint.try_recv() is None


def _stub_config(directory, *, rank=0, world_size=1, init_method=None):
    """Launch one weightless rank of a tensor-parallel stub component."""
    directory.mkdir(parents=True, exist_ok=True)
    return worker_args(
        directory,
        ipc_payload_cap=65536,
        max_batch_tokens=256,
        max_batch_calls=2,
        device="cpu",
        no_model=True,
        allow_stub=True,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        kv_token_capacity=4096,
        components={
            "model": {
                "ranks": list(range(world_size)),
                "parallel_config": {"tensor_parallel_size": world_size},
            }
        },
        distributed_init_method=init_method,
    )


@pytest.mark.parametrize("failure", (None, "model_loading", "process_world"))
def test_worker_preserves_a_caller_owned_process_group(tmp_path, failure):
    import torch
    import torch.distributed as dist

    from uniserve_worker.errors import WorkerError, WorkerErrorCode
    from uniserve_worker.worker import Worker

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/world", rank=0, world_size=1
    )
    try:
        if failure == "model_loading":
            with pytest.raises(
                FileNotFoundError, match="modular_model_index.json"
            ):
                Worker.from_config(_model_config(tmp_path / "borrowed"))
        elif failure == "process_world":
            with pytest.raises(WorkerError, match="rank/world_size") as raised:
                Worker.from_config(
                    _stub_config(tmp_path / "borrowed-world", world_size=2)
                )
            assert raised.value.code is WorkerErrorCode.UNSUPPORTED_SETUP
            assert not raised.value.fatal
        else:
            with Worker.from_config(_stub_config(tmp_path / "borrowed-world")):
                pass
        value = torch.tensor([7])
        dist.all_reduce(value)
        assert value.item() == 7
    finally:
        dist.destroy_process_group()


def _owned_world(rank, directory, failure):
    from dataclasses import replace
    from pathlib import Path

    import torch
    import torch.distributed as dist

    from uniserve_worker.config.deployment import ModelLaunchConfig
    from uniserve_worker.errors import WorkerError
    from uniserve_worker.worker import Worker

    config = _stub_config(
        Path(directory) / f"owned-world-{rank}",
        rank=rank,
        world_size=2,
        init_method=f"file://{directory}/world-{failure}",
    )
    if failure == "model_loading":
        config = replace(
            config,
            use_stub_model=False,
            model=ModelLaunchConfig(directory, {}),
        )
        with pytest.raises(FileNotFoundError, match="modular_model_index.json"):
            Worker.from_config(config)
    elif failure == "execution_setup":
        config = replace(
            config,
            data_plane=replace(config.data_plane, publication_backends=()),
        )
        with pytest.raises(
            WorkerError, match="publication backends must be unique"
        ):
            Worker.from_config(config)
    else:
        with Worker.from_config(config):
            value = torch.tensor([rank + 1])
            dist.all_reduce(value)
            assert value.item() == 3
    if failure == "execution_setup":
        import os

        # Failed construction aborts this rank without synchronizing peers;
        # the process must exit with its still-owned world retained.
        assert dist.is_initialized()
        os._exit(0)
    assert not dist.is_initialized()


@pytest.mark.parametrize("failure", ("model_loading", "execution_setup", None))
def test_worker_releases_process_groups_created_by_its_factory(
    tmp_path, failure
):
    import torch.multiprocessing as mp

    mp.spawn(_owned_world, args=(str(tmp_path), failure), nprocs=2, join=True)


@pytest.mark.gpu
@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_partial_cuda_binding_failure_preserves_error_and_allows_reconstruction(
    monkeypatch, cleanup_fails
):
    import torch
    from cuda.bindings import driver

    from tests.python.fixtures.depth_one import (
        ar_params,
        execution_batch,
        finalized_report,
        root_parent,
        token_call,
    )
    from tests.python.fixtures.execution_worker import execution_worker
    from uniserve_worker.config.execution import LaneConfig, WorkerConfig
    from uniserve_worker.protocol.call import CALL_KINDS, CallStatus

    policy = WorkerConfig(
        prefill_cuda_graph=False,
        graph_policy="off",
        lanes=(
            LaneConfig("decode", 64, (ForwardMode.DECODE, ForwardMode.VERIFY)),
            LaneConfig(
                "compute",
                88,
                tuple(
                    kind
                    for kind in CALL_KINDS
                    if kind not in {ForwardMode.DECODE, ForwardMode.VERIFY}
                ),
            ),
        ),
    )
    create_stream = driver.cuGreenCtxStreamCreate
    destroy_context = driver.cuGreenCtxDestroy
    failure = torch.cuda.OutOfMemoryError("CUDA stream allocation failed")
    allocations = 0

    def unavailable_stream(*args):
        nonlocal allocations
        allocations += 1
        if allocations == 2:
            raise failure
        return create_stream(*args)

    def failed_cleanup(*args):
        result = destroy_context(*args)
        if result[0] == driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA context teardown reported failure")
        return result

    # Inject at the external driver boundary after a physical binding exists.
    # No Worker, runner, Lane, or storage collaborator is replaced.
    with monkeypatch.context() as patch:
        patch.setattr(driver, "cuGreenCtxStreamCreate", unavailable_stream)
        if cleanup_fails:
            patch.setattr(driver, "cuGreenCtxDestroy", failed_cleanup)
        with pytest.raises(torch.cuda.OutOfMemoryError) as raised:
            execution_worker(device="cuda:0", execution=policy)
        assert raised.value is failure
        if cleanup_fails:
            assert any(
                "CUDA context teardown reported failure" in note
                for note in failure.__notes__
            )

    with execution_worker(device="cuda:0", execution=policy) as worker:
        admission = ar_params(1, block_ids=(0,))
        call = token_call(
            admission.request_key,
            call_id=CallId(1, 0),
            predecessor=root_parent(admission),
            mode=ForwardMode.PREFILL,
            tokens=(3, 4),
        )
        result = finalized_report(
            worker,
            worker.submit(
                execution_batch(
                    batch_id=1,
                    admissions=(admission,),
                    calls=(call,),
                )
            ),
        )
        assert result.completions[0].status is CallStatus.OK
        assert result.completions[0].committed_tokens
