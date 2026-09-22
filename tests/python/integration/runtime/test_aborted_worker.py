"""A failed process preserves live device storage without awaiting its work."""

import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_aborted_worker_returns_before_device_work_and_preserves_its_storage():
    script = textwrap.dedent("""
        import os
        import traceback
        import torch
        from tests.python.fixtures.encoding import Model
        from tests.python.fixtures.execution_worker import execution_worker
        from uniserve_worker.config.execution import WorkerConfig

        try:
            worker = execution_worker(
                Model().to("cuda:0"), device="cuda:0",
                execution=WorkerConfig(
                    graph_policy="full", model_dtype="float32",
                ),
            )
            runner = worker.runner
            tokens = torch.tensor([3, 8, 1], device="cuda:0")
            runner.run_encoder("text", tokens)
            stream = runner.module_stream("text_encoder", method="encode")
            torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                torch.cuda._sleep(5_000_000_000)
            output = runner.run_encoder("text", tokens).values[0]
            finished = torch.cuda.Event()
            finished.record(stream)
            assert not finished.query()
            worker.close(aborted=True)
            assert not finished.query(), "aborted close waited for device work"
            finished.synchronize()
            expected = tokens.cpu()[:, None] * 4 + torch.arange(4)
            torch.testing.assert_close(output.cpu(), expected.float())
            try:
                worker.advance()
            except RuntimeError as error:
                assert "closed" in str(error)
            else:
                raise AssertionError("failed worker accepted new execution")
        except BaseException:
            traceback.print_exc()
            os._exit(1)
        os._exit(0)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
