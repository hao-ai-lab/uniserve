"""Model path and config loading helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["read_config", "resolve_model_path"]


def resolve_model_path(model_path: str) -> str:
    """Return a local path, downloading an HF repo id to a snapshot if needed."""
    if Path(model_path).exists():
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


def read_config(model_path: str) -> dict[str, Any]:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
