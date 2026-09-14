from __future__ import annotations

from pathlib import Path
import yaml

from .models import SimulationConfig
from .track_block_validation import validate_track_block_config


def load_simulation_config(path: str | Path) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return validate_track_block_config(SimulationConfig(**raw))


def simulation_config_from_dict(raw: dict) -> SimulationConfig:
    return validate_track_block_config(SimulationConfig(**raw))
