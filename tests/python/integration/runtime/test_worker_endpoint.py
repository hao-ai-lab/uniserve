"""Endpoint ownership across real native IPC and worker launch failures."""

import json
from uuid import uuid4

import pytest

from uniserve_worker.bootstrap.cli import parse_worker_args
from uniserve_worker.bootstrap.ipc import WorkerIpcEndpoint
from uniserve_worker.bootstrap.launch import run_worker

pytestmark = pytest.mark.integration


def _model_config(service, directory):
    return parse_worker_args(
        [
            "--service-name",
            service,
            "--ipc-payload-cap",
            "65536",
            "--max-batch-tokens",
            "256",
            "--device",
            "cpu",
            "--model",
            str(directory),
        ]
    )


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
        assert not failing_scope, "endpoint scope suppressed the startup exception"
    assert endpoint.closed
    with WorkerIpcEndpoint(service, max_payload=65536) as replacement:
        assert replacement.try_recv() is None


def test_model_loading_failure_releases_the_launch_endpoint(tmp_path):
    service = f"worker-{uuid4().hex}"
    config = _model_config(service, tmp_path)
    with pytest.raises(FileNotFoundError, match="config.json"):
        run_worker(config)
    with WorkerIpcEndpoint(service, max_payload=65536) as endpoint:
        assert endpoint.try_recv() is None


def test_endpoint_binding_failure_prevents_model_loading(tmp_path):
    service = f"worker-{uuid4().hex}"
    config = _model_config(service, tmp_path)
    # The model directory is invalid, but an occupied endpoint prevents launch
    # from reaching checkpoint I/O in the first place.
    with WorkerIpcEndpoint(service, max_payload=65536) as endpoint:
        with pytest.raises(RuntimeError, match="failed to bind IPC service"):
            run_worker(config)
        assert endpoint.try_recv() is None


def _stub_config(service, *, rank=0, world_size=1, init_method=None):
    args = [
        "--service-name",
        service,
        "--ipc-payload-cap",
        "65536",
        "--max-batch-tokens",
        "256",
        "--max-batch-operations",
        "2",
        "--device",
        "cpu",
        "--no-model",
        "--allow-stub",
        "--rank",
        str(rank),
        "--world-size",
        str(world_size),
        "--kv-token-capacity",
        "4096",
    ]
    args.extend(
        [
            "--entries",
            json.dumps(
                {
                    "model": {
                        "ranks": list(range(world_size)),
                        "parallel_config": {"tensor_parallel_size": world_size},
                    }
                }
            ),
        ]
    )
    if init_method is not None:
        args.extend(["--distributed-init-method", init_method])
    return parse_worker_args(args)


@pytest.mark.parametrize("loading_fails", (False, True))
def test_worker_preserves_a_caller_owned_process_group(tmp_path, loading_fails):
    import torch
    import torch.distributed as dist

    from uniserve_worker.worker import Worker

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/world", rank=0, world_size=1)
    try:
        if loading_fails:
            with pytest.raises(FileNotFoundError, match="config.json"):
                Worker.from_config(_model_config("borrowed-world", tmp_path))
        else:
            with Worker.from_config(_stub_config("borrowed-world")):
                pass
        value = torch.tensor([7])
        dist.all_reduce(value)
        assert value.item() == 7
    finally:
        dist.destroy_process_group()


def _owned_world(rank, directory):
    from dataclasses import replace

    import torch
    import torch.distributed as dist

    from uniserve_worker.bootstrap.config import ModelLaunchConfig
    from uniserve_worker.foundation.errors import WorkerError
    from uniserve_worker.worker import Worker

    # Failure at either construction stage and a normal close must leave the
    # process free to create another world.
    for failure in ("model_loading", "execution_setup", None):
        config = _stub_config(
            "owned-world",
            rank=rank,
            world_size=2,
            init_method=f"file://{directory}/world-{failure}",
        )
        if failure == "model_loading":
            config = replace(
                config,
                use_stub_model=False,
                model=ModelLaunchConfig(directory, {}, 16, 1.0),
            )
            with pytest.raises(FileNotFoundError, match="config.json"):
                Worker.from_config(config)
        elif failure == "execution_setup":
            config = replace(config, data_plane=replace(config.data_plane, publication_backends=()))
            with pytest.raises(WorkerError, match="publication backends must be unique"):
                Worker.from_config(config)
        else:
            with Worker.from_config(config):
                value = torch.tensor([rank + 1])
                dist.all_reduce(value)
                assert value.item() == 3
        assert not dist.is_initialized()


def test_worker_releases_process_groups_created_by_its_factory(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_owned_world, args=(str(tmp_path),), nprocs=2, join=True)


@pytest.mark.gpu
@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_partial_cuda_binding_failure_preserves_error_and_allows_reconstruction(
    monkeypatch, cleanup_fails
):
    import torch
    from cuda.bindings import driver

    from tests.python.fixtures.depth_one import (
        ar_params,
        execution_run,
        finalized_report,
        root_parent,
        token_operation,
    )
    from tests.python.fixtures.execution_worker import execution_worker
    from uniserve_worker.config import LaneConfig, WorkerConfig
    from uniserve_worker.execution.batch import Domain, OpStatus, TokenMode

    policy = WorkerConfig(
        prefill_cuda_graph=False,
        graph_policy="off",
        lanes=(
            LaneConfig("decode", 64, (Domain.DECODE,)),
            LaneConfig("compute", 88, (Domain.PREFILL, Domain.FLOW)),
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
                "CUDA context teardown reported failure" in note for note in failure.__notes__
            )

    with execution_worker(device="cuda:0", execution=policy) as worker:
        admission = ar_params(1, block_ids=(0,))
        operation, payload = token_operation(
            admission.request_key,
            op_id=1,
            parent=root_parent(admission),
            mode=TokenMode.EXTEND,
            tokens=(3, 4),
        )
        result = finalized_report(
            worker.execute(
                execution_run(
                    run_id=1,
                    admissions=(admission,),
                    operations=(operation,),
                    input_products=(payload,),
                )
            )
        )
        assert result.completions[0].status is OpStatus.OK
        assert result.completions[0].committed_tokens
