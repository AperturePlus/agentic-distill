"""Utility helpers for dynamic imports and batching."""

from __future__ import annotations

import importlib
from typing import Any


def import_from_path(path: str) -> Any:
    """Import an attribute given a string path 'module:attr'."""

    if ":" not in path:
        raise ValueError(f"Invalid import path '{path}', expected format 'module:attr'.")
    module_path, attr = path.split(":", 1)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Module '{module_path}' has no attribute '{attr}'.") from exc

