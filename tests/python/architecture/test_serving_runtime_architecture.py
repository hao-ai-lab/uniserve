from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from uniserve_eval.harness.metrics.common import RequestRecord
from uniserve_eval.harness.report import benchmark_contract, build_summary
from uniserve_eval.harness.runner import reference_request_summary
from uniserve_eval.harness.spec import BenchmarkSpec, TaskName
from uniserve_eval.harness.tasks.t2i import T2ITask

pytestmark = pytest.mark.architecture

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "Cargo.toml").exists())
CRATES = ROOT / "crates"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def manifest(path: Path) -> dict:
    return tomllib.loads(read(path))


def rust_files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.rs"))


def normal_deps(crate: Path) -> set[str]:
    data = manifest(crate / "Cargo.toml")
    return set(data.get("dependencies", {})) | set(data.get("build-dependencies", {}))


def source_tree(path: Path) -> str:
    return "\n".join(read(file) for file in rust_files(path))


def production_source(path: Path) -> str:
    return "\n".join(re.split(r"#\[cfg\(test\)\]\s*mod tests", read(file), maxsplit=1)[0] for file in rust_files(path))


def struct_body(source: str, name: str) -> str:
    match = re.search(rf"pub(?:\(crate\))? struct {name} \{{(?P<body>.*?)\n\}}", source, re.S)
    assert match is not None, name
    return match.group("body")


def enum_body(source: str, name: str) -> str:
    match = re.search(rf"pub(?:\(crate\))? enum {name} \{{(?P<body>.*?)\n\}}", source, re.S)
    assert match is not None, name
    return match.group("body")


def test_canonical_crates_exist_and_shallow_crates_are_absent() -> None:
    workspace = manifest(ROOT / "Cargo.toml")
    deps = set(workspace["workspace"]["dependencies"])
    expected = {
        "uniserve-serving": CRATES / "frontend" / "serving",
        "uniserve-model-profile": CRATES / "frontend" / "model-profile",
        "uniserve-engine-gateway": CRATES / "frontend" / "engine-gateway",
        "uniserve-protocol-adapters": CRATES / "frontend" / "protocol-adapters",
        "uniserve-server": CRATES / "server" / "app",
    }
    assert set(expected).issubset(deps)
    assert all(path.joinpath("Cargo.toml").is_file() for path in expected.values())
    forbidden_shallow_crates = {
        "uniserve-chat",
        "uniserve-chat-output",
        "uniserve-chat-protocol",
        "uniserve-chat-template",
        "uniserve-engine-client",
        "uniserve-llm",
        "uniserve-model-assets",
        "uniserve-native-api",
        "uniserve-openai-api",
        "uniserve-reasoning-parser",
        "uniserve-server-app",
        "uniserve-server-grpc",
        "uniserve-server-http",
        "uniserve-text",
        "uniserve-tokenizer",
        "uniserve-tool-parser",
    }
    assert deps.isdisjoint(forbidden_shallow_crates)


def test_production_crate_count_is_within_target_range() -> None:
    manifests = list(CRATES.rglob("Cargo.toml"))
    production = [path for path in manifests if "support" not in path.parts and "bin" not in path.parts]
    assert 20 <= len(production) <= 25


def test_production_package_graph_has_no_upward_edges() -> None:
    core = normal_deps(CRATES / "foundation" / "core")
    scheduler = normal_deps(CRATES / "engine" / "scheduler")
    serving = normal_deps(CRATES / "frontend" / "serving")
    profile = normal_deps(CRATES / "frontend" / "model-profile")
    adapters = normal_deps(CRATES / "frontend" / "protocol-adapters")
    assert core.isdisjoint({"axum", "tonic", "prometheus-client", "uniserve-engine-gateway", "uniserve-model-profile", "uniserve-protocol-adapters", "uniserve-serving"})
    assert scheduler.isdisjoint({"axum", "tonic", "uniserve-grpc-proto", "uniserve-openai-types", "uniserve-protocol-adapters", "uniserve-serving"})
    assert serving.isdisjoint({"axum", "tonic", "uniserve-engine-wire", "uniserve-grpc-proto", "uniserve-openai-types", "uniserve-protocol-adapters", "uniserve-server"})
    assert profile.isdisjoint({"axum", "tonic", "uniserve-engine-gateway", "uniserve-grpc-proto", "uniserve-openai-types", "uniserve-protocol-adapters", "uniserve-serving"})
    assert adapters.isdisjoint({"uniserve-scheduler", "uniserve-server"})


def test_every_production_route_enters_serving_runtime() -> None:
    server = CRATES / "server" / "app" / "src"
    state = read(server / "state.rs")
    assert "runtime: ServingRuntime" in state
    assert "pub fn runtime(&self) -> &ServingRuntime" in state
    route_paths = [
        server / "http" / "routes" / "openai" / "completions.rs",
        server / "http" / "routes" / "openai" / "chat_completions.rs",
        server / "http" / "routes" / "openai" / "images.rs",
        server / "http" / "routes" / "inference" / "generate.rs",
        server / "http" / "routes" / "inference" / "native.rs",
        server / "grpc" / "mod.rs",
    ]
    for path in route_paths:
        route = read(path)
        assert ".runtime()" in route, path
        assert ".serve(" in route, path
        assert ".chat().text().generate" not in route, path
        assert "generate_native" not in route, path


def test_serving_runtime_owns_all_execution_branches_and_controls() -> None:
    runtime = read(CRATES / "frontend" / "serving" / "src" / "lib.rs")
    state = read(CRATES / "server" / "app" / "src" / "state.rs")
    body = struct_body(runtime, "ServingRuntime")
    state_body = struct_body(state, "AppState")
    assert "gateway: EngineGateway" in body
    assert "engine_control" not in body
    assert "engine_control: EngineAppControl" in state_body
    assert "PlannedRequest::Text" in runtime
    assert "PlannedRequest::Chat" in runtime
    assert "PlannedRequest::DialectGeneration" in runtime
    assert "pub async fn serve(" in runtime
    assert "pub async fn execute(" in runtime
    assert "pub async fn cancel(" in runtime
    assert "pub async fn abort(" in runtime
    assert "pub fn request_stats(" in runtime
    assert "pub async fn drain_request(" in runtime
    assert "pub async fn drain(" in runtime
    assert "pub async fn shutdown(" in runtime
    assert "pub async fn serve_native" not in runtime


def test_protocol_and_engine_dtos_do_not_cross_canonical_surfaces() -> None:
    serving = source_tree(CRATES / "frontend" / "serving" / "src")
    adapters = source_tree(CRATES / "frontend" / "protocol-adapters" / "src")
    profile = source_tree(CRATES / "frontend" / "model-profile" / "src")
    scheduler = production_source(CRATES / "engine" / "scheduler" / "src")
    assert all(token not in serving for token in ["uniserve_engine_wire", "uniserve_grpc_proto", "uniserve_openai_types", "ChatCompletionRequest", "CompletionRequest"])
    assert all(token not in adapters for token in ["uniserve_scheduler", "uniserve_server"])
    assert all(token not in profile for token in ["axum::", "tonic::", "ChatCompletionRequest", "CompletionRequest", "NativeGenerateBody"])
    assert all(token not in scheduler for token in ["uniserve_openai_types", "uniserve_grpc_proto", "ChatCompletionRequest", "CompletionRequest", "NativeGenerateBody"])


def test_scheduler_plans_from_lowered_descriptors_without_public_labels() -> None:
    scheduler = production_source(CRATES / "engine" / "scheduler" / "src")
    forbidden = ["GenerationConstraint", "UndOnly", "GenOnly", "sensenova", "thinkmorph", "bagel", "image_start_ids", "start_of_image", "render_prompt", "chat_template"]
    assert [token for token in forbidden if token in scheduler] == []
    assert "transition: PlannedTransition" in scheduler
    assert "transition.validate_result(&sr)" in scheduler
    assert "cursor.apply_transition(&transition, &sr)" in scheduler


def test_generation_values_are_pure_and_submission_is_separate() -> None:
    core = read(CRATES / "foundation" / "core" / "src" / "generation.rs")
    engine_api = read(CRATES / "protocol" / "engine-api" / "src" / "lib.rs")
    request = struct_body(core, "GenerationRequest")
    assert all(token not in request for token in ["Sender", "Receiver", "Stream", "EngineHandle", "WorkerHandle", "Tokenizer"])
    assert "pub struct GenerationSubmission" in engine_api
    assert "pub request: GenerationRequest" in engine_api
    assert "pub event_tx: EventTx" in engine_api


def test_generation_transport_uses_only_canonical_request_shapes() -> None:
    translate = production_source(CRATES / "protocol" / "engine-wire" / "src")
    wire = read(CRATES / "protocol" / "engine-wire" / "src" / "lib.rs")
    gateway = production_source(CRATES / "frontend" / "engine-gateway" / "src")
    gateway_canonical = read(CRATES / "frontend" / "engine-gateway" / "src" / "generation" / "canonical.rs")
    serving = production_source(CRATES / "frontend" / "serving" / "src")
    native_schema = read(CRATES / "frontend" / "protocol-adapters" / "src" / "native" / "schema.rs")
    engine_request = struct_body(wire, "EngineCoreRequest")
    serve_event = enum_body(serving, "ServeEvent")
    assert "pub generation: GenerationRequest" in engine_request
    assert all(field not in engine_request for field in ["prompt_token_ids", "sampling_params", "prompt_embeds", "extra_args"])
    assert "req.generation.clone()" in translate
    assert "NativeRequestExt" not in translate
    assert "GenerationConstraint::UndOnly" not in translate
    assert "generation_request_to_wire(submission" in gateway_canonical
    assert "pub request: GenerationRequest" in gateway_canonical
    assert "mm_features" not in gateway_canonical
    assert all(token not in gateway for token in ["GenerateRequest", "GenerateOutput", "struct Llm"])
    assert all(token not in serving for token in ["EngineCoreRequest", "EngineCoreSamplingParams", "StructuredOutputsParams", "LoraRequest"])
    assert "TokenLogprobs" not in serve_event
    assert "MmFeatureSpec" not in translate
    assert "mm_features" not in serving
    assert "input_image_b64" not in production_source(CRATES / "frontend" / "protocol-adapters" / "src" / "native")
    assert "alias =" not in native_schema


def test_every_production_crate_has_a_survival_reason() -> None:
    topology = read(ROOT / "specs" / "serving_runtime_crate_audit.md")
    production_manifests = [path for path in CRATES.rglob("Cargo.toml") if "support" not in path.parts and "bin" not in path.parts]
    names = {manifest(path)["package"]["name"] for path in production_manifests}
    production_topology = topology.split("## Binaries", maxsplit=1)[0]
    documented = set(re.findall(r"^\| `([^`]+)` \|", production_topology, re.M))
    assert names == documented


def test_cursor_is_the_single_lifecycle_owner() -> None:
    generation = read(CRATES / "engine" / "scheduler" / "src" / "generation" / "mod.rs")
    scheduler = read(CRATES / "engine" / "scheduler" / "src" / "scheduler.rs")
    cursor = struct_body(generation, "GenerationCursor")
    state = struct_body(scheduler, "ReqState")
    assert "cursor: GenerationCursor" in state
    for field in ["lifecycle", "ingest", "und", "image_gen", "feedback", "resources", "replay"]:
        assert re.search(rf"\b{field}\s*:", cursor)
    for field in ["phase", "pos", "kvlen", "prompt_cursor", "image_id", "steps_done", "feedback_locator", "replayability"]:
        assert not re.search(rf"\b{field}\s*:", cursor)
        assert not re.search(rf"\b{field}\s*:", state)


def test_opaque_behavior_maps_do_not_cross_runtime_or_scheduler_boundaries() -> None:
    serving = read(CRATES / "frontend" / "serving" / "src" / "lib.rs")
    core = read(CRATES / "foundation" / "core" / "src" / "generation.rs")
    request = struct_body(serving, "ServeRequest")
    generation = struct_body(core, "GenerationRequest")
    forbidden = ["HashMap<String, serde_json::Value>", "extra_args", "uniserve_xargs"]
    assert all(token not in request for token in forbidden)
    assert all(token not in generation for token in forbidden)


def test_benchmark_support_uses_canonical_runtime_dependencies() -> None:
    benchmark_deps = normal_deps(CRATES / "support" / "benchmarks")
    example_deps = normal_deps(CRATES / "support" / "examples")
    assert "uniserve-serving" in benchmark_deps
    assert {"uniserve-serving", "uniserve-protocol-adapters"}.issubset(example_deps)
    forbidden = {
        "uniserve-chat",
        "uniserve-engine-client",
        "uniserve-native-api",
        "uniserve-openai-api",
        "uniserve-text",
    }
    assert benchmark_deps.isdisjoint(forbidden)
    assert example_deps.isdisjoint(forbidden)


def test_benchmark_profiles_produce_schema_valid_canonical_artifacts() -> None:
    profiles = json.loads(read(ROOT / "uniserve_eval" / "profiles.json"))
    points = profiles["benchmarks"]["main"]["points"].values()
    required = {"runtime_profile_id", "output_constraint", "preprocessing", "plan_evidence_policy"}
    assert all(required.issubset(point["harness"]) for point in points)
    assert all(
        point["harness"]["plan_evidence_policy"]
        in {"runtime_inspection", "reference_protocol"}
        for point in points
    )
    assert '"wire": "native"' not in json.dumps(profiles)

    spec = BenchmarkSpec(
        task=TaskName.T2I,
        model="reference-model",
        num_prompts=1,
        output_constraint="gen_only",
        plan_evidence_policy="reference_protocol",
        runtime_profile_id="reference-profile",
        width=64,
        height=64,
        steps=2,
        max_images=1,
    )
    request = T2ITask(spec).build_request({"prompt": "not retained"})
    evidence = {"source": "reference_protocol", "request": reference_request_summary(request)}
    contract = benchmark_contract(spec, [{"id": "reference-row"}])
    records = [
        RequestRecord(
            request_id="reference-row",
            task="t2i",
            success=True,
            latency=1.0,
            images=1,
            classifier="ok",
        )
    ]
    valid = build_summary(
        spec,
        "http://reference.invalid",
        records,
        dur_s=1.0,
        plan_evidence=evidence,
        contract=contract,
    )
    schema = json.loads(
        read(ROOT / "uniserve_eval" / "harness" / "schemas" / "summary.schema.json")
    )
    Draft202012Validator(schema).validate(valid)
    assert valid["artifact"]["valid"] is True
    evidence["request"]["image"]["steps"] = 3
    mismatched = build_summary(
        spec,
        "http://reference.invalid",
        records,
        dur_s=1.0,
        plan_evidence=evidence,
        contract=contract,
    )
    assert mismatched["artifact"]["valid"] is False
