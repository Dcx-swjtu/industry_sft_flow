"""Configuration loading helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        value = os.getenv(name)
        if value is None:
            return default if default is not None else ""
        return value

    return _ENV_RE.sub(replace, text)


def _expand_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_expand_value(child) for child in value]
    if isinstance(value, str):
        return _expand_env(value)
    return value


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    config = _expand_value(config)
    config["_meta"] = {"config_path": str(path), "config_dir": str(path.parent)}
    return config

