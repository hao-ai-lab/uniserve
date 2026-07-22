"""Build-time import-graph conformance for the uniserve_worker package.

Checks the dependency direction declared in specs/model-execution.md sections 22 and 25.7.
This is an architectural dependency check on the package import graph, not a behavioral
test: it walks import statements of uniserve_worker modules and validates edge direction.
Imports guarded by TYPE_CHECKING count as dependency edges.
"""

import ast
import functools
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

PACKAGE = "uniserve_worker"
PACKAGE_ROOT = Path(__file__).resolve().parents[3] / PACKAGE

SYSTEM_LAYER_SEGMENTS = ("bootstrap", "execution", "runtime", "server", "worker")
COMPOSITION_ROOT_SEGMENTS = ("bootstrap", "server", "worker")
GENERIC_PACKAGE_NAMES = ("base", "common", "core", "interfaces", "utils")

# Import edges from models/ into system-layer modules, one (importer, imported) pair per
# line, sorted. This list may only shrink: an entry is deleted when the importer no longer
# imports the target, and no new entry may be added. All listed edges are runtime imports;
# a TYPE_CHECKING-only edge would be marked as such here rather than listed silently.
MODELS_SYSTEM_LAYER_EDGES = (
    ("uniserve_worker.models.bagel", "uniserve_worker.execution.flow"),
    ("uniserve_worker.models.bagel", "uniserve_worker.execution.segment"),
    ("uniserve_worker.models.bagel", "uniserve_worker.execution.sequence"),
    ("uniserve_worker.models.bagel", "uniserve_worker.runtime.kv_pool"),
    ("uniserve_worker.models.bagel", "uniserve_worker.runtime.paged_text_cache"),
    ("uniserve_worker.models.bagel", "uniserve_worker.runtime.request_state"),
    ("uniserve_worker.models.bagel", "uniserve_worker.runtime.residency"),
    ("uniserve_worker.models.qwen3", "uniserve_worker.runtime.compile"),
    ("uniserve_worker.models.qwen3", "uniserve_worker.runtime.residency"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.execution.flow"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.execution.products"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.execution.segment"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.execution.sequence"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.runtime.compile"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.runtime.forward_stream"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.runtime.kv_pool"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.runtime.masks"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.runtime.residency"),
    ("uniserve_worker.models.sensenova.model", "uniserve_worker.runtime.tower_handoff"),
)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(importer: str, is_package: bool, level: int, module: str | None) -> str | None:
    parts = importer.split(".")
    if not is_package:
        parts = parts[:-1]
    for _ in range(level - 1):
        if not parts:
            return None
        parts.pop()
    if not parts:
        return None
    return ".".join(parts + module.split(".")) if module else ".".join(parts)


@functools.cache
def _import_edges() -> frozenset[tuple[str, str]]:
    """Intra-package import edges as (importer module, imported module) pairs.

    ``from x import y`` records ``x.y`` when ``y`` is itself a module of the package and
    ``x`` otherwise; relative imports are resolved to absolute module names.
    """
    sources = {
        _module_name(path): path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    edges: set[tuple[str, str]] = set()
    for module, path in sources.items():
        is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == PACKAGE or alias.name.startswith(f"{PACKAGE}."):
                        edges.add((module, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = _resolve_relative(module, is_package, node.level, node.module)
                else:
                    base = node.module
                if base is None or (base != PACKAGE and not base.startswith(f"{PACKAGE}.")):
                    continue
                for alias in node.names:
                    qualified = f"{base}.{alias.name}"
                    edges.add((module, qualified if qualified in sources else base))
    return frozenset(edges)


def _segment(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def test_execution_imports_no_concrete_models():
    violations = sorted(
        edge
        for edge in _import_edges()
        if _segment(edge[0]) == "execution" and _segment(edge[1]) == "models"
    )
    assert not violations, (
        f"execution/ must not import concrete model modules: {violations}"
    )


def test_concrete_models_imported_only_by_composition_roots():
    allowed_segments = ("models",) + COMPOSITION_ROOT_SEGMENTS
    violations = sorted(
        edge
        for edge in _import_edges()
        if _segment(edge[1]) == "models" and _segment(edge[0]) not in allowed_segments
    )
    assert not violations, (
        f"only worker composition roots {COMPOSITION_ROOT_SEGMENTS} may import "
        f"concrete model modules: {violations}"
    )


def test_models_system_layer_edges_match_declared_list():
    assert list(MODELS_SYSTEM_LAYER_EDGES) == sorted(set(MODELS_SYSTEM_LAYER_EDGES)), (
        "MODELS_SYSTEM_LAYER_EDGES must be sorted and duplicate-free"
    )
    observed = {
        edge
        for edge in _import_edges()
        if _segment(edge[0]) == "models" and _segment(edge[1]) in SYSTEM_LAYER_SEGMENTS
    }
    declared = set(MODELS_SYSTEM_LAYER_EDGES)
    undeclared = sorted(observed - declared)
    assert not undeclared, (
        f"models/ gained system-layer imports; models/ may depend on PyTorch, nn/, the "
        f"forward contracts, and spec declarations only: {undeclared}"
    )
    stale = sorted(declared - observed)
    assert not stale, (
        f"declared edges are no longer present; delete them from "
        f"MODELS_SYSTEM_LAYER_EDGES: {stale}"
    )


def test_no_generic_dumping_ground_packages():
    package_dirs = [
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_dir() and path.name in GENERIC_PACKAGE_NAMES
    ]
    top_level_modules = [
        PACKAGE_ROOT / f"{name}.py"
        for name in GENERIC_PACKAGE_NAMES
        if (PACKAGE_ROOT / f"{name}.py").is_file()
    ]
    offenders = sorted(str(path.relative_to(PACKAGE_ROOT.parent)) for path in package_dirs + top_level_modules)
    assert not offenders, (
        f"{PACKAGE} must not contain generic {GENERIC_PACKAGE_NAMES} packages: {offenders}"
    )
