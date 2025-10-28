"""Utility helpers for dynamic imports and dictionary operations."""

from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any, Dict


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


def deep_merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep merge of two dictionaries without mutating the inputs."""

    result: Dict[str, Any] = deepcopy(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result

