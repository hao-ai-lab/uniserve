"""Canonical model and deployment for execution-boundary tests."""

from uniserve_worker.server.stub import StubModel, stub_deployment

TEST_MODEL = StubModel()
TEST_DEPLOYMENT = stub_deployment(64)

__all__ = ["TEST_DEPLOYMENT", "TEST_MODEL"]
