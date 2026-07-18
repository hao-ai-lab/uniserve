"""Import-layering guardrails for the worker module-boundary redesign.

The headline contract (docs/uniserve_worker-module-boundary-audit.md): the
*package* import graph must be a DAG and every cross-package edge must point at a
strictly-lower (or equal, if acyclic) layer. ``if TYPE_CHECKING:`` imports are
ignored throughout — they are never executed and so cannot create a runtime
import cycle or layer violation.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "uniserve_worker").is_dir()
)
WORKER = ROOT / "uniserve_worker"
PKG = "uniserve_worker"
PYTHON_TESTS = ROOT / "tests" / "python"

# Layer numbers: a package may import only strictly-lower-numbered layers;
# same-layer edges are permitted as long as the graph stays acyclic.
LAYER = {
    "foundation": 0,
    "contracts": 1,
    "backends": 2,
    "nn": 2,
    "ops": 2,
    "runtime": 3,
    # ``spec`` is the speculative-sampling policy: it imports only foundation
    # (L0) + nn.sampler (L2), so it sits at L3 and the system-side spec-verify
    # orchestration in ``execution`` (L4) can use it without an upward edge.
    "spec": 3,
    "execution": 4,
    "loader": 5,
    "processors": 5,
    "models": 6,
    "worker": 7,
    "server": 8,
    "bootstrap": 9,
    "main": 10,
    "<root>": 0,  # empty package __init__ + native _uniserve_ipc ext (leaves)
}

# Lower layers that must stay model-neutral (no concrete-model imports / names).
MODEL_NEUTRAL = (
    "foundation",
    "contracts",
    "backends",
    "nn",
    "runtime",
    "execution",
    "worker",
)
FORBIDDEN_MODEL_NAMES = re.compile(r"\b(?:SenseNova|BAGEL|Bagel|sensenova|bagel)\b")
FORBIDDEN_MODEL_CLASS_NAMES = re.compile(r".*Adapter$")
LEGACY_WORKER_TYPES = re.compile(
    r"\b(?:WorkerRuntime|WorkerDriver|BaseWorkerDriver|RunnerDriver|"
    r"EncodeOnlyDriver|SamplerDriver|PostProcessDriver|StubEngine)\b"
)


def _py_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(WORKER.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _rel(path: Path) -> str:
    return path.relative_to(WORKER).as_posix()


def _first_level_pkg(module: str) -> str:
    parts = module.split(".")
    if len(parts) <= 1:
        return "<root>"
    if len(parts) == 2:
        return parts[1] if parts[1] in LAYER else "<root>"
    return parts[1]


def _type_checking_imports(tree: ast.AST) -> set[ast.AST]:
    tc: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            )
            if is_tc:
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        tc.add(sub)
    return tc


def _all_modules() -> dict[str, Path]:
    files = [p for p in WORKER.rglob("*.py") if "__pycache__" not in p.parts]
    mods = {_module_name(p): p for p in files}
    mods[f"{PKG}._uniserve_ipc"] = WORKER / "_uniserve_ipc"  # native ext leaf
    return mods


def _defining_module(target: str, modules: dict[str, Path]) -> str | None:
    parts = target.split(".")
    while parts:
        cand = ".".join(parts)
        if cand in modules:
            return cand
        parts = parts[:-1]
    return None


def _runtime_import_edges() -> set[tuple[str, str]]:
    """Module->module runtime import edges (TYPE_CHECKING imports excluded)."""
    modules = _all_modules()
    edges: set[tuple[str, str]] = set()
    for module, path in modules.items():
        if not path.suffix == ".py":
            continue
        is_init = path.name == "__init__.py"
        pkg_parts = module.split(".") if is_init else module.split(".")[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        tc = _type_checking_imports(tree)
        for node in ast.walk(tree):
            if node in tc:
                continue
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [
                    a.name for a in node.names if a.name == PKG or a.name.startswith(PKG + ".")
                ]
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                    target_mod = ".".join(base + ([node.module] if node.module else []))
                else:
                    target_mod = node.module or ""
                if not (target_mod == PKG or target_mod.startswith(PKG + ".")):
                    continue
                targets.append(target_mod)
                targets.extend(f"{target_mod}.{a.name}" for a in node.names)
            for t in targets:
                dst = _defining_module(t, modules)
                if dst is not None and dst != module:
                    edges.add((module, dst))
    return edges


def _package_edges() -> dict[str, set[str]]:
    pkg_edges: dict[str, set[str]] = {}
    for src, dst in _runtime_import_edges():
        ps, pd = _first_level_pkg(src), _first_level_pkg(dst)
        if ps != pd:
            pkg_edges.setdefault(ps, set()).add(pd)
    return pkg_edges


def _sccs(graph: dict[str, set[str]]) -> list[list[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: dict[str, bool] = {}
    stack: list[str] = []
    out: list[list[str]] = []
    counter = [0]
    nodes = set(graph) | {d for ds in graph.values() for d in ds}

    def strong(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on[v] = True
        for w in graph.get(v, ()):  # noqa: SIM118
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif on.get(w):
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on[w] = False
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for v in nodes:
        if v not in index:
            strong(v)
    return out


def test_package_import_graph_is_a_dag():
    """The whole-codebase headline contract: no package-level import cycles."""
    multi = [c for c in _sccs(_package_edges()) if len(c) > 1]
    assert multi == [], f"package import cycles (SCCs): {[sorted(c) for c in multi]}"


def test_worker_assembly_legacy_driver_surface_is_absent():
    legacy_modules = (
        "server/base_driver.py",
        "server/driver_factory.py",
        "server/encode_only_driver.py",
        "server/postprocess_driver.py",
        "server/runner_driver.py",
        "server/sampler_driver.py",
    )
    assert [path for path in legacy_modules if (WORKER / path).exists()] == []

    offenders = [
        _rel(path)
        for path in _py_files(WORKER)
        if LEGACY_WORKER_TYPES.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_worker_server_does_not_reinterpret_deployment_roles():
    source = (WORKER / "server" / "app.py").read_text(encoding="utf-8")
    assert "WorkerKind" not in source
    assert "worker_kind" not in source


def test_no_upward_layer_edges():
    """Every cross-package edge points at a layer no higher than its source."""
    offenders = []
    for src, dsts in sorted(_package_edges().items()):
        for dst in sorted(dsts):
            ls, ld = LAYER.get(src), LAYER.get(dst)
            assert ls is not None, f"unmapped package {src!r}"
            assert ld is not None, f"unmapped package {dst!r}"
            if ls < ld:
                offenders.append(f"L{ls} {src} -> L{ld} {dst}")
    assert offenders == [], f"upward layer edges: {offenders}"


def test_model_neutral_layers_do_not_import_models():
    offenders: list[str] = []
    for src, dst in sorted(_runtime_import_edges()):
        if _first_level_pkg(src) in MODEL_NEUTRAL and _first_level_pkg(dst) == "models":
            offenders.append(f"{src} imports {dst}")
    assert offenders == []


def test_model_neutral_layers_do_not_name_specific_models():
    roots = [WORKER / pkg for pkg in MODEL_NEUTRAL]
    offenders = [
        _rel(path)
        for path in _py_files(*roots)
        if FORBIDDEN_MODEL_NAMES.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_production_code_does_not_import_quarantined_reference_models():
    offenders: list[str] = []
    for path in _py_files(WORKER):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == f"{PKG}.reference_models" or alias.name.startswith(
                        f"{PKG}.reference_models."
                    ):
                        offenders.append(f"{_rel(path)} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == f"{PKG}.reference_models" or module.startswith(
                    f"{PKG}.reference_models."
                ):
                    offenders.append(f"{_rel(path)} imports {module}")
                if node.level >= 2 and module.startswith("reference_models"):
                    offenders.append(f"{_rel(path)} imports relative {'.' * node.level}{module}")
    assert offenders == []


def test_models_tree_has_no_runtime_packages_or_adapter_classes():
    models = WORKER / "models"
    runtime_dirs = [
        path.relative_to(models).as_posix()
        for path in models.rglob("*")
        if path.is_dir() and path.name.endswith("_runtime")
    ]
    adapter_classes: list[str] = []
    for path in _py_files(models):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and FORBIDDEN_MODEL_CLASS_NAMES.fullmatch(node.name):
                adapter_classes.append(f"{_rel(path)}::{node.name}")
    assert runtime_dirs == []
    assert adapter_classes == []


def _family_entry_files(models):
    return _py_files(models)


def test_models_tree_matches_canonical_file_set():
    models = WORKER / "models"
    top_level = {p.name for p in models.glob("*.py")}
    assert top_level == {
        "__init__.py",
        "registry.py",
        "qwen3.py",
        "bagel.py",
        # Family cache registrations for the unified KV runtime
        # (specs/unified_kv_attention_runtime.md); family-naming data stays in
        # the models layer, never in model-neutral contracts.
        "cache_registrations.py",
    }
    # SenseNova-U1 is its own family package; operation lifecycle is system-owned.
    sensenova = {p.name for p in (models / "sensenova").glob("*.py")}
    assert sensenova == {"__init__.py", "model.py", "config.py"}


def test_openai_chat_protocol_stays_out_of_worker_runtime_layers():
    """Worker execution/model code consumes engine-native contracts, not HTTP DTOs."""

    roots = [
        WORKER / "backends",
        WORKER / "contracts",
        WORKER / "execution",
        WORKER / "models",
        WORKER / "nn",
        WORKER / "ops",
        WORKER / "runtime",
        WORKER / "spec",
    ]
    forbidden = ("ChatCompletion", "image_config", "image_url", "delta.images", "OpenAI", "openai")
    offenders: list[str] = []
    for path in _py_files(*roots):
        source = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in source:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_operation_execution_has_single_system_owner():
    owned_by_module = {
        WORKER / "execution" / "sequence.py": {
            "SequenceCache",
            "SequenceAdapter",
            "SequenceExecutor",
        },
        WORKER / "execution" / "flow.py": {
            "FlowState",
            "FlowRow",
            "ProgramState",
            "FlowAdapter",
            "FlowGraphExecution",
            "FlowExecution",
        },
        WORKER / "execution" / "segment.py": {
            "SegmentGraphRunner",
            "SegmentAdapter",
            "SegmentRuntime",
            "SegmentPlan",
            "SegmentExecutor",
        },
        WORKER / "execution" / "products.py": {
            "ImageMaterializeAdapter",
            "ImageMaterializer",
            "ImageEncodeAdapter",
            "ImageEncoder",
            "ProductTransferSession",
        },
    }
    for module, owned in owned_by_module.items():
        assert module.is_file(), f"{_rel(module)} must exist"
        defined = {
            node.name
            for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
            if isinstance(node, ast.ClassDef)
        }
        assert owned <= defined, f"{_rel(module)} must define {owned}, has {defined}"

    all_owned = set().union(*owned_by_module.values())
    offenders: list[str] = []
    for path in _py_files(WORKER / "models"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in all_owned:
                offenders.append(f"{_rel(path)}::{node.name}")
    assert offenders == [], f"models must not redefine operation execution: {offenders}"


def test_family_models_compose_system_execution_objects():
    forbidden_bases = {
        "FlowExecution",
        "FlowGraphExecution",
        "SegmentRuntime",
        "SegmentExecutor",
        "SequenceExecutor",
        "ImageEncoder",
        "ImageMaterializer",
        "ProductTransferSession",
    }
    offenders: list[str] = []
    for path in _py_files(WORKER / "models"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
            inherited = sorted(bases & forbidden_bases)
            if inherited:
                offenders.append(f"{_rel(path)}::{node.name} inherits {inherited}")
    assert offenders == []


def test_operation_execution_uses_family_surfaces_not_model_backbones():
    offenders: list[str] = []
    for name in ("sequence.py", "flow.py", "segment.py", "products.py"):
        path = WORKER / "execution" / name
        source = path.read_text(encoding="utf-8")
        for needle in ("owner.model.language_model", "owner.model._build_t2i"):
            if needle in source:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_image_materializer_uses_adapter_not_model_backbone():
    path = WORKER / "execution" / "products.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    materializer = ""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ImageMaterializer":
            materializer = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    assert materializer, "ImageMaterializer must be defined"
    assert "self.owner.model." not in materializer


def test_flow_execution_uses_adapter_for_family_primitives():
    source = (WORKER / "execution" / "flow.py").read_text(encoding="utf-8")
    assert "self.model." not in source


def test_sensenova_does_not_define_a_second_native_qwen3_backbone_namespace():
    source = (WORKER / "models" / "sensenova" / "model.py").read_text(encoding="utf-8")
    assert "_NativeQwen3" not in source


def test_segment_execution_uses_owner_adapter_not_model_backbone():
    source = (WORKER / "execution" / "segment.py").read_text(encoding="utf-8")
    assert "owner.model." not in source


# Residency and graph classes are constructed by the system layer. Model files
# may reference their types in annotations.
FORBIDDEN_MODEL_CONSTRUCTIONS = frozenset(
    {
        "PagedKVPool",
        "BatchedPagedTextCache",
        "Executor",
        # ``torch.cuda.CUDAGraph()`` denotes graph lifecycle ownership.
        "CUDAGraph",
    }
)

# Class-name shapes that denote CUDA-graph lifecycle ownership. A model may still
# *reference* a system graph type in an annotation, but defining a class whose
# name matches these means the model owns capture/replay state — which belongs to
# ``execution``. Encoding the ownership shape (not a fixed name list) is what would
# have caught ``_SenseNovaTextDecodeGraph{Runner,State,Past,RequestCache}``.
MODEL_GRAPH_OWNER_CLASS_NAME = re.compile(
    r"(?:CUDAGraph|CudaGraph|DecodeGraphRunner|PrefillGraphRunner|GraphRunner|GraphState|GraphPast|GraphRequestCache)"
)


def test_models_tree_owns_no_pools_or_graphs():
    """Models declare residency geometry; the system constructs + owns the pools.

    The worker runtime (``ResidencyManager`` / graph ``Executor``) is the single
    owner of the KV pools and CUDA graphs. A model may name these types in an
    annotation but must not instantiate one — that ownership moved to the system.
    """

    offenders: list[str] = []
    for path in _family_entry_files(WORKER / "models"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name in FORBIDDEN_MODEL_CONSTRUCTIONS:
                offenders.append(f"{_rel(path)}:{node.lineno} constructs {name}")
    assert offenders == [], f"family entry files must not construct pools or graphs: {offenders}"


def test_models_do_not_query_cuda_memory_directly():
    """CUDA free-memory policy belongs to the system sizing layer, not models."""

    offenders: list[str] = []
    for path in _py_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        for needle in ("mem_get_info",):
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_models_do_not_define_local_block_free_list_allocators():
    offenders: list[str] = []
    forbidden = (
        "BlockFreeList",
        "_alloc_from_free_list",
        "_release_to_free_list",
        "bisect.bisect_left",
    )
    for path in _py_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_models_do_not_define_local_text_image_param_parsers():
    offenders: list[str] = []
    forbidden = (
        "class _ImageParams",
        "def _image_param",
        "def _required_image_param",
        "def _require_image_param",
    )
    for path in _py_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_models_do_not_define_local_paged_cache_copy_helpers():
    offenders: list[str] = []
    forbidden = ("def _copy_cache_prefix", "def _copy_cache_span", "def _append_packed_chunk")
    for path in _family_entry_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_models_do_not_define_packed_forward_sampling_or_relay_helpers():
    offenders: list[str] = []
    forbidden = (
        "def _sample_text_logits",
        "def _store_forward_sampled_token_relay",
        "def _forward_text_input_ids",
    )
    for path in _family_entry_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_models_tree_defines_no_cuda_graph_runner_classes():
    """Models must not *define* a CUDA-graph runner/state class.

    The construction ban keyed on a fixed name list and so missed the model-local
    ``_SenseNovaTextDecodeGraphRunner`` and its graph state/past/request-cache
    helpers. Encode the ownership invariant by shape: no model file may define a
    class whose name denotes CUDA-graph lifecycle ownership. Graph capture/replay
    is a system (``execution``) responsibility.
    """

    offenders: list[str] = []
    for path in _family_entry_files(WORKER / "models"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and MODEL_GRAPH_OWNER_CLASS_NAME.search(node.name):
                offenders.append(f"{_rel(path)}::{node.name}")
    assert offenders == [], (
        f"family entry files must not define CUDA-graph runner/state classes: {offenders}"
    )


def test_python_tests_do_not_use_deleted_model_modules_as_live_oracles():
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in _py_files(PYTHON_TESTS)
        if path.name != "test_layering_contract.py"
        and re.search(
            r"reference_models|bagel_runtime|sensenova_runtime", path.read_text(encoding="utf-8")
        )
    ]
    assert offenders == []


def test_backend_provider_pack_imports_do_not_mutate_sys_path_or_use_env_paths():
    """Kernel providers are importable packages, never runtime path injections."""

    offenders: list[str] = []
    for root in (WORKER, ROOT / "uniserve_kernel", ROOT / "scripts", ROOT / "uniserve_eval"):
        if not root.exists():
            continue
        for path in _py_files(root):
            text = path.read_text(encoding="utf-8")
            if "sys.path" in text:
                offenders.append(
                    f"{path.relative_to(ROOT).as_posix()}: sys.path mutation/reference"
                )
            if re.search(r"os\.environ(?:\.get)?\([^)]*FA4[^)]*PATH", text):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: legacy FA4 env path")
    assert offenders == []


def test_fa4_backend_uses_uniserve_kernel_provider_surface():
    source = (WORKER / "backends" / "attention" / "fa4_cute.py").read_text(encoding="utf-8")
    assert "from uniserve_kernel import mm_attn_varlen" in source
    assert "flash_attn.cute.interface" not in source
    assert "hybrid_mask_attn.mask" not in source


def test_uniserve_kernel_is_standalone_provider_pack():
    pack = ROOT / "uniserve_kernel"
    root_pyproject = ROOT / "pyproject.toml"
    pyproject = pack / "pyproject.toml"
    public_surface = pack / "python" / "uniserve_kernel" / "mm_attn_varlen.py"
    runtime_loader = pack / "python" / "uniserve_kernel" / "_fa4_runtime.py"
    prefix_bounds = pack / "python" / "uniserve_kernel" / "_prefix_bounds.py"
    visible_mask = pack / "python" / "uniserve_kernel" / "_visible_end_mask.py"
    vendored_flash_attn = pack / "python" / "uniserve_kernel" / "_cute" / "flash_attn"
    vendored_hybrid_mask = pack / "python" / "uniserve_kernel" / "_cute" / "hybrid_mask_attn"
    root_bridge = pack / "__init__.py"
    root_bridge_module = pack / "mm_attn_varlen.py"

    root_config = tomllib.loads(root_pyproject.read_text(encoding="utf-8"))
    root_includes = (
        root_config.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
        .get("include", [])
    )
    assert all(not str(pattern).startswith("uniserve_kernel") for pattern in root_includes)
    assert not root_bridge.exists()
    assert not root_bridge_module.exists()
    assert pyproject.exists()
    provider_config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert provider_config["project"]["name"] == "uniserve-kernel"
    assert any("nvidia-cutlass-dsl" in dep for dep in provider_config["project"]["dependencies"])
    assert public_surface.exists()
    assert runtime_loader.exists()
    assert prefix_bounds.exists()
    assert visible_mask.exists()
    assert not vendored_flash_attn.exists()
    assert not vendored_hybrid_mask.exists()
    public = public_surface.read_text(encoding="utf-8")
    assert "from ._prefix_bounds import compute_prefix_bounds" in public
    assert "from ._fa4_runtime import flash_attn_fwd" in public
    assert "flash_attn.cute.interface" not in public
    assert "hybrid_mask_attn.mask" not in public
    runtime_source = runtime_loader.read_text(encoding="utf-8")
    assert "package_path.insert" not in runtime_source
    assert "_append_package_path(package_path, overlay_path, prepend=True)" in runtime_source


def test_fa4_runtime_path_helper_accepts_namespace_package_paths():
    runtime_loader = ROOT / "uniserve_kernel" / "python" / "uniserve_kernel" / "_fa4_runtime.py"
    tree = ast.parse(runtime_loader.read_text(encoding="utf-8"))
    helper_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_append_package_path"
    ]
    assert len(helper_defs) == 1
    module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(runtime_loader), "exec"), namespace)
    append_path = namespace["_append_package_path"]

    class AppendOnlyPath:
        def __init__(self) -> None:
            self.items: list[str] = []

        def __contains__(self, item: object) -> bool:
            return item in self.items

        def append(self, item: str) -> None:
            self.items.append(item)

    namespace_path = AppendOnlyPath()
    append_path(namespace_path, "/provider", prepend=True)
    append_path(namespace_path, "/provider", prepend=True)
    assert namespace_path.items == ["/provider"]

    regular_path: list[str] = []
    append_path(regular_path, "/overlay", prepend=True)
    append_path(regular_path, "/provider")
    assert regular_path == ["/overlay", "/provider"]


def test_models_do_not_import_attention_backends_or_vendor_kernels():
    """Models call the ops facade instead of backend registries or vendor symbols."""

    forbidden = (
        "uniserve_worker.backends.attention",
        "...backends.attention",
        "..backends.attention",
        "torch.ops.sgl_kernel",
        "sgl_kernel",
        "flashinfer",
        "flash_attn",
        "hybrid_mask_attn",
    )
    offenders: list[str] = []
    for path in _py_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_shared_layers_do_not_import_sgl_kernel_pack_directly():
    """Optional SGL fused-op kernels live behind the ops provider pack."""

    offenders: list[str] = []
    allowed = {WORKER / "ops" / "providers.py", WORKER / "backends" / "attention" / "sgl_kernel.py"}
    for path in _py_files(WORKER / "nn"):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if (
            "from sgl_kernel" in text
            or "import sgl_kernel" in text
            or "torch.ops.sgl_kernel" in text
        ):
            offenders.append(_rel(path))
    assert offenders == []


def test_shared_layers_do_not_use_legacy_kernel_router():
    assert not (WORKER / "nn" / "_kernel_router.py").exists()
    offenders: list[str] = []
    for path in _py_files(WORKER / "nn"):
        text = path.read_text(encoding="utf-8")
        for needle in ("_kernel_router", "KernelRouter", "FusedKernel"):
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_mixed_forward_side_tables_use_forward_names():
    assert (WORKER / "runtime" / "forward_stream.py").exists()
    assert not (WORKER / "execution" / "forward_stream.py").exists()
    assert not (WORKER / "execution" / "fused_stream.py").exists()
    offenders: list[str] = []
    for path in _py_files(WORKER / "execution") + _py_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        for needle in (
            "execution.fused_stream",
            ".fused_stream",
            "FusedStream",
            "FusedPagedKV",
            "fused_stream",
        ):
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_sensenova_qk_norm_rope_has_no_model_local_legacy_path():
    model_source = (WORKER / "models" / "sensenova" / "model.py").read_text(encoding="utf-8")
    provider_source = (WORKER / "ops" / "providers.py").read_text(encoding="utf-8")
    config_source = (ROOT / "uniserve_eval" / "profiles.json").read_text(encoding="utf-8")
    spec_source = (ROOT / "specs" / "backend_ops.md").read_text(encoding="utf-8")

    assert "_qk_norm_rope_3d_legacy" not in model_source
    assert "SENSENOVA_FUSED" not in model_source
    assert 'env_override="UNISERVE_QK_NORM_ROPE_PROVIDER"' in provider_source
    assert not (ROOT / "benchmarks" / "serving" / "sensenova_fusion_matrix.py").exists()
    assert not (ROOT / "benchmarks" / "kernels" / "sensenova_projection_microbench.py").exists()
    for source in (config_source, spec_source):
        assert "FUSED_QK_NORM_ROPE" not in source
        assert "SENSENOVA_FUSED" not in source
    for needle in ("_fused_qkv_proj", "_fused_gate_up", "self.q_proj", "self.gate_proj"):
        assert needle not in model_source


def test_radix_attention_invokes_backends_only_through_ops_attention():
    """RadixAttention plans residency but does not call backend kernels directly."""

    source = (WORKER / "nn" / "attention.py").read_text(encoding="utf-8")
    forbidden = (
        "select_attention_backend_name",
        "get_attention_backend",
        "resolve_varlen_backend",
        "varlen_backend_usable",
        "_fallback_paged_backend",
        "_select_paged_backend",
        "_select_varlen_backend",
        "_select_plain_backend",
        "_paged_backend_usable",
        "_paged_kernel_supports",
        ".forward_paged(",
        ".forward_varlen(",
    )
    offenders = [needle for needle in forbidden if needle in source]
    assert offenders == []
    assert "ops.attention(" in source


def test_text_backend_gate_uses_ops_attention_dispatcher():
    source = (WORKER / "backends" / "attention" / "text_dispatch.py").read_text(encoding="utf-8")
    forbidden = ("select_attention_backend_name", "get_attention_backend(")
    offenders = [needle for needle in forbidden if needle in source]
    assert offenders == []
    assert "ops.attention_dispatcher()" in source


def test_vision_attention_invokes_varlen_only_through_ops_attention():
    source = (WORKER / "nn" / "vision" / "encoder.py").read_text(encoding="utf-8")
    forbidden = ("resolve_varlen_backend", ".forward_varlen(")
    offenders = [needle for needle in forbidden if needle in source]
    assert offenders == []
    assert "ops.attention(" in source


def test_models_do_not_inline_png_or_base64_wire_encoding():
    """PNG/base64 output encoding is a wire-format policy owned by runtime/image_utils."""

    offenders: list[str] = []
    forbidden = ("BytesIO", "b64encode", 'format="PNG"')
    for path in _py_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_torch_is_compiling_has_a_single_owner():
    """torch.compile introspection lives in foundation.torch_compat only."""

    offenders: list[str] = []
    owner = WORKER / "foundation" / "torch_compat.py"
    for path in _py_files(WORKER):
        if path == owner:
            continue
        text = path.read_text(encoding="utf-8")
        if "def torch_is_compiling" in text or "def _torch_is_compiling" in text:
            offenders.append(_rel(path))
    assert offenders == []


def test_execution_graph_packages_match_completion_criterion():
    """Graph responsibilities live in one namespace with explicit physical seams."""

    execution = WORKER / "execution"
    assert {p.name for p in execution.glob("*.py")} == {
        "__init__.py",
        "engine.py",
        "flow.py",
        "products.py",
        "segment.py",
        "sequence.py",
    }
    assert {p.name for p in execution.iterdir() if p.is_dir() and p.name != "__pycache__"} == {
        "graph"
    }

    graph_core = execution / "graph"
    assert {p.name for p in graph_core.glob("*.py")} == {
        "__init__.py",
        "bucket.py",
        "capture.py",
        "dispatch.py",
        "executor.py",
        "path.py",
        "span.py",
        "step.py",
    }
    assert [p for p in graph_core.iterdir() if p.is_dir() and p.name != "__pycache__"] == []


def test_graph_core_identity_excludes_dynamic_query_and_composition_data():
    source = "\n".join(
        (WORKER / "execution" / "graph" / name).read_text(encoding="utf-8")
        for name in ("bucket.py", "capture.py", "dispatch.py")
    )
    forbidden = (
        "ForwardMode",
        "op_modes",
        "segment_geometry",
        "unit_query",
        "ragged_query",
        "modality",
    )
    offenders = [needle for needle in forbidden if needle in source]
    assert offenders == []


def test_piecewise_compile_helper_has_a_single_owner():
    """Config-gated piecewise ``torch.compile`` application is registry glue."""

    offenders: list[str] = []
    owner = WORKER / "models" / "registry.py"
    for path in _py_files(WORKER):
        if path == owner:
            continue
        text = path.read_text(encoding="utf-8")
        if "def _maybe_compile_piecewise" in text:
            offenders.append(_rel(path))
    assert offenders == []


def test_host_staging_helpers_have_a_single_owner():
    """Pinned-host staging mechanics live in runtime.host_staging only."""

    offenders: list[str] = []
    owner = WORKER / "runtime" / "host_staging.py"
    forbidden = (
        "def _canonical_device",
        "def canonical_device",
        "def _fill_cpu_long",
        "def _fill_cpu_int",
    )
    for path in _py_files(WORKER):
        if path == owner:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_tp_head_sharding_rule_has_a_single_owner():
    """The attention head/KV tp-sharding rule lives in nn.linear only.

    Models and other layers must consume ``local_attention_head_count`` /
    ``local_kv_head_count`` rather than re-deriving per-rank head counts from
    ``tp_size`` arithmetic, so pool geometry and sharded projections can never
    drift apart.
    """

    owner = WORKER / "nn" / "linear.py"
    offenders: list[str] = []
    forbidden = (
        "def _local_kv_head_count",
        "def _local_num_kv_heads",
        "def local_kv_head_count",
        "def local_attention_head_count",
    )
    for path in _py_files(WORKER):
        if path == owner:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []
