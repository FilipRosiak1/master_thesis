from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def maybe_resolve_path(path: str | None, root_dir: str | None = None) -> str | None:
    if path is None:
        return None
    as_path = Path(path)
    if as_path.is_absolute():
        return str(as_path)
    if root_dir is None:
        return str(as_path)
    return str((Path(root_dir) / as_path).resolve())
