from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "Cargo.toml").exists())
CRATES = ROOT / "crates"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def manifest(path: Path) -> dict:
    return tomllib.loads(read(path))


def rust_files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.rs"))


def package_deps(crate: Path) -> set[str]:
    data = manifest(crate / "Cargo.toml")
    deps = set(data.get("dependencies", {}))
    deps.update(data.get("dev-dependencies", {}))
    deps.update(data.get("build-dependencies", {}))
    return deps


def test_target_serving_crates_exist_and_are_workspace_dependencies():
    workspace = manifest(ROOT / "Cargo.toml")
    deps = set(workspace["workspace"]["dependencies"])
    expected = {
        "uniserve-serving": CRATES / "frontend" / "serving",
        "uniserve-model-profile": CRATES / "frontend" / "model-profile",
        "uniserve-engine-gateway": CRATES / "frontend" / "engine-gateway",
        "uniserve-protocol-adapters": CRATES / "frontend" / "protocol-adapters",
    }
    assert set(expected).issubset(deps)
    for name, path in expected.items():
        assert path.joinpath("Cargo.toml").is_file(), name


def test_server_state_owns_serving_runtime():
    app = CRATES / "server" / "app"
    assert "uniserve-serving" in package_deps(app)
    state = read(app / "src" / "state.rs")
    assert "runtime: ServingRuntime" in state
    assert "pub fn runtime(&self) -> &ServingRuntime" in state


def test_openai_completions_uses_semantic_runtime_path():
    route = read(CRATES / "server" / "http" / "src" / "routes" / "openai" / "completions.rs")
    assert "ServeRequest::from_text_request" in route
    assert ".runtime()" in route
    assert ".serve(serve_request)" in route
    assert ".chat().text().generate" not in route
    assert ".chat().text().generate_raw" not in route


def test_openai_chat_uses_semantic_runtime_path_for_chat_submission():
    route = read(CRATES / "server" / "http" / "src" / "routes" / "openai" / "chat_completions.rs")
    assert "ServeRequest::from_chat_request" in route
    assert ".runtime()" in route
    assert ".serve(serve_request)" in route
    assert ".chat(prepared.chat_request)" not in route
    assert ".chat().chat" not in route


def test_raw_generate_uses_semantic_runtime_path():
    route = read(CRATES / "server" / "http" / "src" / "routes" / "inference" / "generate.rs")
    assert "ServeRequest::from_text_request" in route
    assert ".runtime()" in route
    assert ".serve(serve_request)" in route
    assert ".chat().text().generate" not in route
    assert ".chat().text().generate_raw" not in route


def test_native_generation_submission_is_owned_by_serving_runtime():
    runtime = read(CRATES / "frontend" / "serving" / "src" / "lib.rs")
    native_route = read(CRATES / "server" / "http" / "src" / "routes" / "native.rs")
    chat_route = read(CRATES / "server" / "http" / "src" / "routes" / "openai" / "chat_completions.rs")
    assert "pub async fn serve_native" in runtime
    assert "ServeEvent::ImageDone" in runtime
    assert ".runtime()" in native_route
    assert ".serve_native(" in native_route
    assert ".runtime()" in chat_route
    assert ".serve_native(" in chat_route
    assert ".uniserve_engine_client().generate_native" not in native_route
    assert ".uniserve_engine_client().generate_native" not in chat_route


def test_public_multimodal_generation_surface_is_chat_completions():
    routes = read(CRATES / "server" / "http" / "src" / "routes" / "mod.rs")
    spec = read(ROOT / "uniserve_eval" / "harness" / "spec.py")
    profiles = json.loads(read(ROOT / "uniserve_eval" / "profiles.json"))

    assert '.route("/generate"' not in routes
    assert '"/generate"' not in spec
    assert '"wire": "native"' not in json.dumps(profiles)


def test_grpc_generate_uses_semantic_runtime_path():
    service = read(CRATES / "server" / "grpc" / "src" / "lib.rs")
    assert "ServeRequest::from_text_request" in service
    assert ".runtime()" in service
    assert ".serve(serve_request)" in service
    assert ".chat().text().generate" not in service
    assert ".chat().text().generate_raw" not in service


def test_serving_public_surface_has_no_protocol_dtos():
    serving = "\n".join(read(path) for path in rust_files(CRATES / "frontend" / "serving" / "src"))
    forbidden = [
        "uniserve_openai_types",
        "uniserve_grpc_proto",
        "CompletionRequest",
        "ChatCompletionRequest",
        "GenerateRequest as Pb",
    ]
    offenders = [token for token in forbidden if token in serving]
    assert offenders == []


def test_protocol_adapters_do_not_import_scheduler_or_server_internals():
    adapters = "\n".join(read(path) for path in rust_files(CRATES / "frontend" / "protocol-adapters" / "src"))
    assert "uniserve_scheduler" not in adapters
    assert "uniserve_server_app" not in adapters
    assert "uniserve_server_http" not in adapters


def test_scheduler_does_not_import_public_protocol_or_route_dtos():
    scheduler = "\n".join(read(path) for path in rust_files(CRATES / "engine" / "scheduler" / "src"))
    forbidden = [
        "uniserve_openai_types",
        "uniserve_openai_api",
        "uniserve_native_api",
        "uniserve_grpc_proto",
        "CompletionRequest",
        "ChatCompletionRequest",
        "NativeGenerateBody",
    ]
    offenders = [token for token in forbidden if token in scheduler]
    assert offenders == []


def test_core_crate_has_no_frontend_or_transport_dependencies():
    deps = package_deps(CRATES / "foundation" / "core")
    forbidden = {
        "uniserve-tokenizer",
        "uniserve-serving",
        "uniserve-protocol-adapters",
        "uniserve-engine-client",
        "uniserve-engine-gateway",
        "axum",
        "tonic",
        "prometheus-client",
    }
    assert deps.isdisjoint(forbidden)


def test_no_opaque_extension_map_in_semantic_request_surface():
    serving = read(CRATES / "frontend" / "serving" / "src" / "lib.rs")
    serve_request_block = re.search(r"pub struct ServeRequest \{(?P<body>.*?)\n\}", serving, re.S)
    assert serve_request_block is not None
    assert "HashMap<String, serde_json::Value>" not in serve_request_block.group("body")
    assert "uniserve_xargs" not in serve_request_block.group("body")
