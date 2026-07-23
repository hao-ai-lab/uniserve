"""Loader for the optional FlashAttention CUTE visible-end runtime."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path


def _ordered_package_paths(package_path: object, overlay_path: str, provider_path: str) -> list[str]:
    try:
        existing = [str(path) for path in package_path]
    except TypeError as exc:
        raise ImportError("flash_attn package path cannot resolve flash_attn.cute") from exc
    ordered = [overlay_path]
    ordered.extend(path for path in existing if path != overlay_path)
    if provider_path not in ordered:
        ordered.append(provider_path)
    return ordered


def _install_flash_attn_4_cute_path() -> None:
    """Resolve ``flash-attn-4``'s CUTE provider package.

    Some deployment images also install FA2 as a regular ``flash_attn`` package
    in system site-packages. ``flash-attn-4`` contributes ``flash_attn/cute`` as
    a provider package, but Python will not discover that subpackage once FA2's
    regular package wins top-level import resolution. Extend the package's own
    search path to the provider distribution instead of mutating process import
    paths.
    """

    try:
        dist = metadata.distribution("flash-attn-4")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "FlashAttention CUTE runtime is unavailable. Install "
            "flash-attn-4[cu13]."
        ) from exc

    provider_root: Path | None = None
    for entry in dist.files or ():
        if entry.as_posix() == "flash_attn/cute/interface.py":
            provider_root = Path(dist.locate_file(entry)).parent.parent
            break
    if provider_root is None:  # pragma: no cover
        raise ImportError(
            "FlashAttention CUTE runtime is unavailable. The installed "
            "flash-attn-4 package does not contain flash_attn.cute.interface."
        )

    import flash_attn as flash_attn_pkg

    package_path = getattr(flash_attn_pkg, "__path__", None)
    if package_path is None:  # pragma: no cover
        raise ImportError("flash_attn is not a package and cannot expose flash_attn.cute")
    overlay_root = Path(__file__).resolve().parent / "_fa4_overlay" / "flash_attn"
    overlay_path = str(overlay_root)
    if not overlay_root.exists():  # pragma: no cover
        raise ImportError("UniServe FlashAttention CUTE overlay is unavailable")
    provider_path = str(provider_root)
    flash_attn_pkg.__path__ = _ordered_package_paths(package_path, overlay_path, provider_path)


try:  # pragma: no cover - optional runtime dependency.
    _install_flash_attn_4_cute_path()
    from flash_attn.cute.interface import _flash_attn_fwd as flash_attn_fwd  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "FlashAttention CUTE runtime is unavailable. Install flash-attn-4[cu13] "
        "with its CUTE runtime dependencies."
    ) from exc

from ._visible_end_mask import hybrid_multimodal_mask  # noqa: E402

__all__ = ["flash_attn_fwd", "hybrid_multimodal_mask"]
