from __future__ import annotations

import argparse
import json

import pytest

import uniserve_eval.backends
import uniserve_eval.verify
from uniserve_eval import cli

pytestmark = pytest.mark.unit


def test_run_suite_can_manage_verify_workload_servers(tmp_path, monkeypatch):
    config_path = tmp_path / "profiles.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "off-server": {
                        "model": "m",
                        "served_model_name": "m",
                        "host": "127.0.0.1",
                        "port": 18082,
                    },
                    "on-server": {
                        "model": "m",
                        "served_model_name": "m",
                        "host": "127.0.0.1",
                        "port": 18082,
                    },
                },
                "workloads": {
                    "off": {
                        "type": "verify",
                        "server": "off-server",
                        "manage_server": True,
                        "payload": {},
                    },
                    "on": {
                        "type": "verify",
                        "server": "on-server",
                        "manage_server": True,
                        "payload": {},
                    },
                },
                "suites": {"fusion": ["off", "on"]},
            }
        ),
        encoding="utf-8",
    )
    events: list[tuple[str, str]] = []

    def fake_clean(args):
        events.append(("clean", args.server))

    def fake_launch(args):
        events.append(("launch", args.server))

    def fake_verify(args):
        events.append(("verify", args.workload))

    monkeypatch.setattr(uniserve_eval.backends, "clean", fake_clean)
    monkeypatch.setattr(uniserve_eval.backends, "launch", fake_launch)
    monkeypatch.setattr(uniserve_eval.verify, "verify", fake_verify)

    cli.run_suite(
        argparse.Namespace(
            config=config_path,
            suite="fusion",
            server=None,
            manage_servers=False,
            launch_timeout_s=1.0,
            clean_grace_s=0.0,
        )
    )

    assert events == [
        ("clean", "off-server"),
        ("launch", "off-server"),
        ("verify", "off"),
        ("clean", "off-server"),
        ("clean", "on-server"),
        ("launch", "on-server"),
        ("verify", "on"),
        ("clean", "on-server"),
    ]
