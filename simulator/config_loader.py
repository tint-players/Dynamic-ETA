"""
Loads corridor/train/simulation definitions from YAML into validated
Pydantic models. Also accepts plain dicts, so this same function works
whether the config came from a file, a JSON body in a future API route,
or a hardcoded dict in a test.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import SimulationConfig


def load_simulation_config(path: str | Path) -> SimulationConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return SimulationConfig(**raw)


def simulation_config_from_dict(raw: dict) -> SimulationConfig:
    return SimulationConfig(**raw)
