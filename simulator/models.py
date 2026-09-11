"""
Core data models for the railway telemetry simulator.

These define the *shape* of everything the simulator works with:
tracks, trains, weather/signal state, and the telemetry frames it emits.

Built on Pydantic so that:
  - track/train definitions can be loaded from YAML/JSON/dict interchangeably
  - bad configs fail fast with a clear validation error, instead of silently
    producing garbage physics later
  - new fields can be added without breaking existing configs (just give
    new fields sensible defaults)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Enums — the fixed vocabularies used across the simulator
# --------------------------------------------------------------------------

class SignalAspect(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    DOUBLE_YELLOW = "DOUBLE_YELLOW"
    RED = "RED"


class WeatherCondition(str, Enum):
    CLEAR = "CLEAR"
    RAIN = "RAIN"
    HEAVY_FOG = "HEAVY_FOG"


class AnomalyType(str, Enum):
    SIGNAL_ASPECT_CHANGE = "SIGNAL_ASPECT_CHANGE"
    WEATHER_MODIFIER = "WEATHER_MODIFIER"
    TEMPORARY_SPEED_RESTRICTION = "TEMPORARY_SPEED_RESTRICTION"
    BLOCK_MAINTENANCE_HOLD = "BLOCK_MAINTENANCE_HOLD"


# --------------------------------------------------------------------------
# Track / corridor definitions (the "map")
# --------------------------------------------------------------------------

class TrackBlock(BaseModel):
    """A single signalling block along the corridor."""
    block_id: str
    length_m: float = Field(gt=0)
    max_speed_kmh: float = Field(gt=0)

    # mutable runtime state (anomalies act on these)
    signal_aspect: SignalAspect = SignalAspect.GREEN
    weather: WeatherCondition = WeatherCondition.CLEAR
    temporary_speed_restriction_kmh: Optional[float] = None
    maintenance_hold: bool = False


class Corridor(BaseModel):
    """An ordered sequence of blocks forming a rail corridor."""
    corridor_id: str
    name: str
    blocks: list[TrackBlock]

    def block_index(self, block_id: str) -> int:
        for i, b in enumerate(self.blocks):
            if b.block_id == block_id:
                return i
        raise KeyError(f"Unknown block_id: {block_id}")

    def lookahead(self, current_index: int, horizon: int = 4) -> list[TrackBlock]:
        """Blocks 0..+horizon ahead of current_index (Block 0 = current)."""
        return self.blocks[current_index: current_index + horizon + 1]


# --------------------------------------------------------------------------
# Train definition
# --------------------------------------------------------------------------

class TrainConfig(BaseModel):
    train_id: str
    name: str = ""
    start_block_id: str
    initial_speed_kmh: float = 0.0

    # physics constants — overridable per train (e.g. freight vs express)
    service_decel_ms2: float = 0.7
    emergency_decel_ms2: float = 1.0
    accel_ms2: float = 0.5


# --------------------------------------------------------------------------
# Simulation-level config
# --------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    corridor: Corridor
    trains: list[TrainConfig]
    tick_seconds: float = 1.0
    max_ticks: int = 3600
    caution_speed_kmh: float = 30.0
    lookahead_blocks: int = 4
    random_seed: Optional[int] = None


# --------------------------------------------------------------------------
# Output: telemetry frame emitted every tick, per train
# --------------------------------------------------------------------------

class TelemetryFrame(BaseModel):
    """One tick of state for one train. This is the atomic unit of output,
    whether it ends up in a Parquet file (batch/training) or a WebSocket
    message (live demo)."""

    scenario_id: str
    train_id: str
    tick: int
    sim_time_s: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    block_id: str
    position_in_block_m: float
    speed_kmh: float
    acceleration_ms2: float

    effective_speed_ceiling_kmh: float
    signal_aspect: SignalAspect
    weather: WeatherCondition

    ebd_distance_m: float
    remaining_block_distance_m: float

    anomaly_active: bool = False
    guardrail_triggered: bool = False
    guardrail_event: Optional[str] = None
