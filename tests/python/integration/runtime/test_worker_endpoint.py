"""Endpoint ownership across real native IPC and worker launch failures."""

import json
import socket
from uuid import uuid4

import pytest

from tests.python.fixtures.launch import worker_args
from uniserve_worker.bootstrap.launch import WorkerIpcEndpoint, run_worker

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


@pytest.mark.parametrize("transport", ("iceoryx2", "tcp"))
def test_model_loading_failure_releases_the_reported_endpoint(
    tmp_path, transport
):
    # A rank reports the endpoint it created before it loads anything, so the
    # endpoint has to survive the report and be released by the failure.
    listener, address = _registration_listener()
    config = _model_config(
        tmp_path / "launch",
        registration_address=address,
        channel_transport=transport,
    )
    with listener:
        with pytest.raises(FileNotFoundError, match="modular_model_index.json"):
            run_worker(config)
        service = _reported_endpoint(listener)
    if transport == "tcp":
        host, port = service.rsplit(":", 1)
        assert host == "127.0.0.1"
        with socket.create_server((host, int(port))):
            pass
    else:
        with WorkerIpcEndpoint(service, max_payload=65536) as endpoint:
            assert endpoint.try_recv() is None


def _stub_config(
    directory, *, rank=0, world_size=1, rendezvous=None, listen_fd=None
):
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
        rendezvous_address=rendezvous,
        rendezvous_listen_fd=listen_fd,
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


def _owned_world(rank, directory, failure, store):
    from dataclasses import replace
    from pathlib import Path

    import torch
    import torch.distributed as dist

    from uniserve_worker.config.deployment import ModelLaunchConfig
    from uniserve_worker.errors import WorkerError
    from uniserve_worker.worker import Worker

    # Every rank inherits the store socket the parent bound; the first rank
    # serves the world's store on it, as it does under the engine.
    address = "{}:{}".format(*store.getsockname())
    if rank == 0:
        listen_fd = store.detach()
    else:
        listen_fd = None
        store.close()
    config = _stub_config(
        Path(directory) / f"owned-world-{rank}",
        rank=rank,
        world_size=2,
        rendezvous=address,
        listen_fd=listen_fd,
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
            data_plane=replace(config.data_plane, export_backends=()),
        )
        with pytest.raises(WorkerError, match="export backends must be unique"):
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

    # Spawning passes the listening socket to each rank at launch, so the
    # store's port stays bound from here until the ranks exit.
    with socket.create_server(("127.0.0.1", 0)) as store:
        mp.spawn(
            _owned_world,
            args=(str(tmp_path), failure, store),
            nprocs=2,
            join=True,
        )
