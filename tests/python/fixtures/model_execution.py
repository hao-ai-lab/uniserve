"""Canonical model and deployment for execution-boundary tests."""

from uniserve_worker.models.stub import StubModel, stub_deployment

TEST_MODEL = StubModel()
TEST_DEPLOYMENT = stub_deployment(64, max_batch_tokens=8192)

__all__ = ["TEST_DEPLOYMENT", "TEST_MODEL"]
