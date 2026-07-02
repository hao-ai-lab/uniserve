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
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "uniserve_worker").is_dir()
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
    "server": 7,
    "main": 8,
    "<root>": 0,  # empty package __init__ + native _uniserve_ipc ext (leaves)
}

# Lower layers that must stay model-neutral (no concrete-model imports / names).
MODEL_NEUTRAL = ("foundation", "contracts", "backends", "nn", "runtime", "execution")
FORBIDDEN_MODEL_NAMES = re.compile(r"\b(?:SenseNova|BAGEL|Bagel|sensenova|bagel)\b")
FORBIDDEN_MODEL_CLASS_NAMES = re.compile(r".*Adapter$")


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
                targets = [a.name for a in node.names if a.name == PKG or a.name.startswith(PKG + ".")]
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


def test_models_tree_matches_target_file_set():
    models = WORKER / "models"
    top_level = {p.name for p in models.glob("*.py")}
    assert top_level == {
        "__init__.py",
        "registry.py",
        "transformers_fallback.py",
        "qwen3.py",
        "bagel.py",
    }
    # SenseNova-U1 is its own package (model + config + interleaved_image moved
    # out of the shared contract layer).
    sensenova = {p.name for p in (models / "sensenova").glob("*.py")}
    assert sensenova == {"__init__.py", "model.py", "config.py", "interleaved_image.py"}


def test_interleaved_text_stepper_is_system_owned():
    """The interleaved text-decode orchestration is a system component.

    The text metadata-building + forward orchestration (``InterleavedTextCacheDriver``
    / ``TextCache`` / ``InterleavedModelOwner``) lives under ``execution`` (the system),
    not ``models`` -- a model only provides the duck-typed compute primitives. A model
    file may import these names (re-export), but must not *define* them.
    """

    stepper = WORKER / "execution" / "interleaved_text_stepper.py"
    assert stepper.is_file(), "execution/interleaved_text_stepper.py must exist (system InterleavedStepper)"
    owned = {"InterleavedTextCacheDriver", "TextCache", "InterleavedModelOwner"}
    defined_in_stepper = {
        node.name
        for node in ast.walk(ast.parse(stepper.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    assert owned <= defined_in_stepper, f"system stepper must define {owned}, has {defined_in_stepper}"

    offenders: list[str] = []
    for path in _py_files(WORKER / "models"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in owned:
                offenders.append(f"{_rel(path)}::{node.name}")
    assert offenders == [], f"models/ must not define system stepper classes: {offenders}"


def test_interleaved_text_execution_uses_owner_adapter_not_model_backbone():
    offenders: list[str] = []
    for rel in ("execution/interleaved_text_stepper.py", "execution/interleaved_text_graph_runner.py"):
        source = (WORKER / rel).read_text(encoding="utf-8")
        for needle in ("owner.model.language_model", "owner.model._build_t2i"):
            if needle in source:
                offenders.append(f"{rel} contains {needle}")
    assert offenders == []


def test_generated_image_commit_driver_uses_owner_adapter_not_model_backbone():
    source = (WORKER / "models" / "sensenova" / "interleaved_image.py").read_text(
        encoding="utf-8"
    )
    commit_driver_source = source[source.index("class GeneratedImageCommitDriver") :]
    assert "self.owner.model." not in commit_driver_source


def test_interleaved_image_mixin_uses_owner_adapter_for_t2i_model_primitives():
    source = (WORKER / "models" / "sensenova" / "interleaved_image.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("self.model.",)
    offenders = [needle for needle in forbidden if needle in source]
    assert offenders == []


def test_sensenova_does_not_define_a_second_native_qwen3_backbone_namespace():
    source = (WORKER / "models" / "sensenova" / "model.py").read_text(encoding="utf-8")
    assert "_NativeQwen3" not in source


def test_packed_mixed_forward_uses_owner_adapter_not_model_backbone():
    source = (WORKER / "execution" / "packed_mixed_forward.py").read_text(encoding="utf-8")
    assert "owner.model." not in source


# Residency/graph classes the system owns: a model file may *reference* them in
# type annotations but must never *construct* one (system-managed-worker-redesign
# §6/§17 Phase 6 layering goal — "no pool/graph ownership left under models/").
FORBIDDEN_MODEL_CONSTRUCTIONS = frozenset(
    {
        "PagedKVPool",
        "DecodeCudaGraphRunner",
        "PrefillCudaGraphRunner",
        "TextGraphRunner",
        "InterleavedTextDecodeGraphRunner",
        # ``torch.cuda.CUDAGraph()`` — no model-local CUDA graph lifecycle. The
        # old set omitted this, so the model-local ``_SenseNovaTextDecodeGraphRunner``
        # (which captured its own ``torch.cuda.CUDAGraph``) slipped past the guard.
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

    The worker runtime (``ResidencyManager`` / ``TextGraphRunner``) is the single
    owner of the KV pools and CUDA graphs. A model may name these types in an
    annotation but must not instantiate one — that ownership moved to the system.
    """

    offenders: list[str] = []
    for path in _py_files(WORKER / "models"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if name in FORBIDDEN_MODEL_CONSTRUCTIONS:
                offenders.append(f"{_rel(path)}:{node.lineno} constructs {name}")
    assert offenders == [], f"models/ must not construct system-owned pools/graphs: {offenders}"


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
    forbidden = ("_alloc_from_free_list", "_release_to_free_list", "bisect.bisect_left")
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
    for path in _py_files(WORKER / "models"):
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
    for path in _py_files(WORKER / "models"):
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
    for path in _py_files(WORKER / "models"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and MODEL_GRAPH_OWNER_CLASS_NAME.search(node.name):
                offenders.append(f"{_rel(path)}::{node.name}")
    assert offenders == [], f"models/ must not define CUDA-graph runner/state classes: {offenders}"


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
    for root in (WORKER, ROOT / "uniserve_kernel", ROOT / "scripts", ROOT / "benchmarks"):
        if not root.exists():
            continue
        for path in _py_files(root):
            text = path.read_text(encoding="utf-8")
            if "sys.path" in text:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: sys.path mutation/reference")
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
        if "from sgl_kernel" in text or "import sgl_kernel" in text or "torch.ops.sgl_kernel" in text:
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
    assert (WORKER / "execution" / "forward_stream.py").exists()
    assert not (WORKER / "execution" / "fused_stream.py").exists()
    offenders: list[str] = []
    for path in _py_files(WORKER / "execution") + _py_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        for needle in ("execution.fused_stream", ".fused_stream", "FusedStream", "FusedPagedKV", "fused_stream"):
            if needle in text:
                offenders.append(f"{_rel(path)} contains {needle}")
    assert offenders == []


def test_sensenova_qk_norm_rope_has_no_model_local_legacy_path():
    model_source = (WORKER / "models" / "sensenova" / "model.py").read_text(encoding="utf-8")
    provider_source = (WORKER / "ops" / "providers.py").read_text(encoding="utf-8")
    config_source = (ROOT / "scripts" / "verify_config.json").read_text(encoding="utf-8")
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


def test_models_do_not_redefine_piecewise_compile_helper():
    """Config-gated piecewise torch.compile application is UniModelBase glue."""

    offenders: list[str] = []
    for path in _py_files(WORKER / "models"):
        text = path.read_text(encoding="utf-8")
        if "def _maybe_compile_piecewise" in text:
            offenders.append(_rel(path))
    assert offenders == []


def test_host_staging_helpers_have_a_single_owner():
    """Pinned-host staging mechanics live in runtime.host_staging only."""

    offenders: list[str] = []
    owner = WORKER / "runtime" / "host_staging.py"
    forbidden = ("def _canonical_device", "def canonical_device", "def _fill_cpu_long", "def _fill_cpu_int")
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
