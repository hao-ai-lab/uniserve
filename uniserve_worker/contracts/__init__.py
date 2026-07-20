"""Model-neutral data contracts and enums."""

from importlib import import_module

__all__ = [
    "BatchPolicy",
    "FlowContext",
    "ForwardStats",
    "ModelLoadScope",
    "UniModel",
]

_EXPORTS = {
    "BatchPolicy": ("forward_batch", "BatchPolicy"),
    "FlowContext": ("model_protocols", "FlowContext"),
    "ForwardStats": ("forward_context", "ForwardStats"),
    "ModelLoadScope": ("model_family", "ModelLoadScope"),
    "UniModel": ("model_protocols", "UniModel"),
}


def __getattr__(name: str):
    """Load consolidated public contracts without coupling torch-free modules to torch."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
