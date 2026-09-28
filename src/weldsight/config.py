"""YAML config loading with `_base_` inheritance and dotted CLI overrides.

A config file may contain `_base_: path/or/list` (relative to the file). Bases are
merged first (left to right), then the file itself on top. Overrides use the form
`train.lr=3e-4` and are parsed with YAML so `true`, `null`, lists etc. work.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, load_config(path.parent / b))
    merged = deep_merge(merged, cfg)
    for ov in overrides or []:
        apply_override(merged, ov)
    return merged


def apply_override(cfg: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must look like key.sub=value, got {override!r}")
    key, raw = override.split("=", 1)
    value = yaml.safe_load(raw)
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
