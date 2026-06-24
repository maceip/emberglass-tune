"""YAML preset loader for unified train pipelines."""
from __future__ import annotations

from pathlib import Path

import yaml

PRESET_DIR = Path(__file__).resolve().parent.parent / "configs" / "presets"


def load_preset(name: str) -> dict:
    path = PRESET_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Unknown preset {name!r}. Available: {', '.join(available) or '(none)'}"
        )
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Preset {path} must be a YAML mapping")
    return data
