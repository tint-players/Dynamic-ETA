from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class SignalAspect(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class SignalType(str, Enum):
    STANDARD = "STANDARD"
    ROUTE_INDICATOR = "ROUTE_INDICATOR"


class WeatherCondition(str, Enum):
    CLEAR = "CLEAR"
    RAIN = "RAIN"
    HEAVY_RAIN = "HEAVY_RAIN"
    FOG = "FOG"
    HEAVY_FOG = "HEAVY_FOG"


class CrossingState(str, Enum):
    OPEN_FOR_TRAIN = "OPEN_FOR_TRAIN"
    CLOSED_FOR_TRAIN = "CLOSED_FOR_TRAIN"


class CurveDirection(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class TrainDirection(str, Enum):
    FORWARD = "FORWARD"
    REVERSE = "REVERSE"


class TrackBlock(BaseModel):
    block_id: str
    length_m: float = Field(gt=0)
    speed_limit_kmh: float = Field(gt=0)
    gradient_percent: float = 0.0
    curve_radius_m: Optional[float] = Field(default=None, gt=0)
    curve_speed_limit_kmh: Optional[float] = Field(default=None, gt=0)
    curve_direction: Optional[CurveDirection] = None

    @model_validator(mode="after")
    def validate_curve_geometry(self):
        if self.curve_direction is not None and self.curve_radius_m is None:
            raise ValueError("curve_direction requires curve_radius_m")
        return self


class Route(BaseModel):
    route_id: str
    route_name: str
    track_ids: list[str] = Field(default_factory=lambda: ["TRACK-1"], min_length=1)
    blocks: list[TrackBlock] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_route_ids(self):
        ids = [b.block_id for b in self.blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("block_id values must be unique")
        if len(self.track_ids) != len(set(self.track_ids)):
            raise ValueError("route.track_ids values must be unique")
        return self

    def block_index(self, block_id: str) -> int:
        for i, block in enumerate(self.blocks):
            if block.block_id == block_id:
                return i
        raise KeyError(f"Unknown block_id: {block_id}")

    def block_start_distance_m(self, block_id: str) -> float:
        idx = self.block_index(block_id)
        return sum(b.length_m for b in self.blocks[:idx])

    @property
    def total_length_m(self) -> float:
        return sum(b.length_m for b in self.blocks)

    def locate(self, route_position_m: float) -> tuple[int, TrackBlock, float]:
        pos = min(max(route_position_m, 0.0), self.total_length_m)
        cumulative = 0.0
        for i, block in enumerate(self.blocks):
            end = cumulative + block.length_m
            if pos < end or i == len(self.blocks) - 1:
                return i, block, min(block.length_m, max(0.0, pos - cumulative))
            cumulative = end
        raise RuntimeError("Unable to locate route position")


class Signal(BaseModel):
    signal_id: str
    protected_block_id: str
    track_id: str = "TRACK-1"
    direction: TrainDirection = TrainDirection.FORWARD
    signal_type: SignalType = SignalType.STANDARD
    position_in_block_m: Optional[float] = Field(default=None, ge=0)
    crossover_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_route_indicator(self):
        if self.signal_type == SignalType.ROUTE_INDICATOR:
            if self.position_in_block_m is None:
                raise ValueError("route-indicator signal requires position_in_block_m")
            if self.crossover_id is None:
                raise ValueError("route-indicator signal requires crossover_id")
        return self


class InitialTrainState(BaseModel):
    start_block_id: str
    position_in_block_m: float = Field(default=0.0, ge=0)
    initial_speed_kmh: float = Field(default=0.0, ge=0)


class TrainConfig(BaseModel):
    train_id: str
    train_name: str = ""
    track_id: str = "TRACK-1"
    max_speed_kmh: float = Field(gt=0)
    length_m: float = Field(gt=0)
    accel_ms2: float = Field(gt=0)
    service_decel_ms2: float = Field(gt=0)
    emergency_decel_ms2: float = Field(gt=0)
    dual_cab: bool = False
    initial_state: InitialTrainState


class JourneyEndpoint(BaseModel):
    block_id: str
    position_in_block_m: float = Field(ge=0)


class Journey(BaseModel):
    source: JourneyEndpoint
    destination: JourneyEndpoint


class StationPlatform(BaseModel):
    platform_id: str
    track_id: str
    block_id: str
    position_in_block_m: float = Field(ge=0)
    length_m: float = Field(default=350.0, gt=0)


class Station(BaseModel):
    station_id: str
    station_name: str
    platforms: list[StationPlatform] = Field(min_length=1)


class StationStop(BaseModel):
    station_id: str
    dwell_time_s: float = Field(default=30.0, ge=0)


class Crossover(BaseModel):
    crossover_id: str
    block_id: str
    start_position_m: float = Field(ge=0)
    end_position_m: float = Field(gt=0)
    from_track_id: str
    to_track_id: str

    @model_validator(mode="after")
    def validate_crossover(self):
        if self.end_position_m <= self.start_position_m:
            raise ValueError("crossover end_position_m must exceed start_position_m")
        if self.from_track_id == self.to_track_id:
            raise ValueError("crossover must connect two different tracks")
        return self


class TrackChangePlan(BaseModel):
    crossover_id: str
    reverse_after_change: bool = False
    turnaround_signal_id: Optional[str] = None


class TrainRun(BaseModel):
    train: TrainConfig
    journey: Journey
    departure_time_s: float = Field(default=0.0, ge=0)
    station_stops: list[StationStop] = Field(default_factory=list)
    track_changes: list[TrackChangePlan] = Field(default_factory=list)


class WeatherTimelineEntry(BaseModel):
    start_time_s: float = Field(ge=0)
    condition: WeatherCondition
    visibility_m: float = Field(gt=0)


class BlockWeatherSchedule(BaseModel):
    block_id: str
    timeline: list[WeatherTimelineEntry] = Field(min_length=1)


class SignalTimelineEntry(BaseModel):
    start_time_s: float = Field(ge=0)
    aspect: SignalAspect


class SignalStateSchedule(BaseModel):
    signal_id: str
    timeline: list[SignalTimelineEntry] = Field(min_length=1)


class TemporarySpeedRestriction(BaseModel):
    restriction_id: str
    block_id: str
    start_position_m: float = Field(ge=0)
    end_position_m: float = Field(gt=0)
    speed_limit_kmh: float = Field(gt=0)
    start_time_s: float = Field(default=0.0, ge=0)
    end_time_s: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_position_m <= self.start_position_m:
            raise ValueError("restriction end_position_m must exceed start_position_m")
        if self.end_time_s is not None and self.end_time_s <= self.start_time_s:
            raise ValueError("restriction end_time_s must exceed start_time_s")
        return self


class MaintenanceRestriction(TemporarySpeedRestriction):
    pass


class CrossingTimelineEntry(BaseModel):
    start_time_s: float = Field(ge=0)
    state: CrossingState


class Crossing(BaseModel):
    crossing_id: str
    block_id: str
    position_in_block_m: float = Field(ge=0)
    timeline: list[CrossingTimelineEntry] = Field(min_length=1)


class EnvironmentConfig(BaseModel):
    weather: list[BlockWeatherSchedule] = Field(default_factory=list)
    signal_states: list[SignalStateSchedule] = Field(default_factory=list)
    temporary_speed_restrictions: list[TemporarySpeedRestriction] = Field(default_factory=list)
    maintenance_restrictions: list[MaintenanceRestriction] = Field(default_factory=list)
    crossings: list[Crossing] = Field(default_factory=list)


class SimulationSettings(BaseModel):
    tick_seconds: float = Field(default=1.0, gt=0)
    max_simulation_time_s: float = Field(default=7200.0, gt=0)
    random_seed: Optional[int] = None
    braking_safety_margin_m: float = Field(default=15.0, ge=0)
    speed_tolerance_kmh: float = Field(default=0.5, ge=0)
    train_separation_m: float = Field(default=120.0, ge=0)


class SimulationConfig(BaseModel):
    route: Route
    signals: list[Signal]
    train: TrainConfig
    journey: Journey
    primary_station_stops: list[StationStop] = Field(default_factory=list)
    primary_track_changes: list[TrackChangePlan] = Field(default_factory=list)
    additional_train_runs: list[TrainRun] = Field(default_factory=list)
    stations: list[Station] = Field(default_factory=list)
    crossovers: list[Crossover] = Field(default_factory=list)
    dynamic_signalling: bool = False
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)

    def _endpoint_m(self, endpoint: JourneyEndpoint) -> float:
        return self.route.block_start_distance_m(endpoint.block_id) + endpoint.position_in_block_m

    def _validate_train(self, train: TrainConfig, label: str, block_ids: set[str], track_ids: set[str]) -> None:
        if train.track_id not in track_ids:
            raise ValueError(f"{label}.track_id references unknown track {train.track_id}")
        if train.initial_state.start_block_id not in block_ids:
            raise ValueError(f"{label}.initial_state.start_block_id is unknown")
        start_block = self.route.blocks[self.route.block_index(train.initial_state.start_block_id)]
        if train.initial_state.position_in_block_m > start_block.length_m:
            raise ValueError(f"{label} initial position exceeds start block length")
        if train.initial_state.initial_speed_kmh > train.max_speed_kmh:
            raise ValueError(f"{label}.initial_state.initial_speed_kmh cannot exceed train max_speed_kmh")

    def _validate_journey(self, journey: Journey, label: str, block_ids: set[str]) -> tuple[float, float]:
        for endpoint_name, endpoint in (("source", journey.source), ("destination", journey.destination)):
            if endpoint.block_id not in block_ids:
                raise ValueError(f"{label}.{endpoint_name}.block_id is unknown")
            block = self.route.blocks[self.route.block_index(endpoint.block_id)]
            if endpoint.position_in_block_m > block.length_m:
                raise ValueError(f"{label}.{endpoint_name}.position_in_block_m exceeds block length")
        source_m = self._endpoint_m(journey.source)
        destination_m = self._endpoint_m(journey.destination)
        if abs(destination_m - source_m) <= 1e-6:
            raise ValueError(f"{label} source and destination must differ")
        return source_m, destination_m

    @model_validator(mode="after")
    def validate_references(self):
        block_ids = {b.block_id for b in self.route.blocks}
        track_ids = set(self.route.track_ids)
        signal_ids = [s.signal_id for s in self.signals]
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("signal_id values must be unique")
        signal_set = set(signal_ids)

        self._validate_train(self.train, "train", block_ids, track_ids)
        primary_source, primary_destination = self._validate_journey(self.journey, "journey", block_ids)
        if primary_destination <= primary_source:
            raise ValueError("primary journey destination must be after source for legacy single-train compatibility")

        for signal in self.signals:
            if signal.protected_block_id not in block_ids:
                raise ValueError(f"Signal {signal.signal_id} references unknown block {signal.protected_block_id}")
            if signal.track_id not in track_ids:
                raise ValueError(f"Signal {signal.signal_id} references unknown track {signal.track_id}")
            if signal.position_in_block_m is not None:
                block = self.route.blocks[self.route.block_index(signal.protected_block_id)]
                if signal.position_in_block_m > block.length_m:
                    raise ValueError(f"Signal {signal.signal_id} position exceeds block length")

        station_ids = [station.station_id for station in self.stations]
        if len(station_ids) != len(set(station_ids)):
            raise ValueError("station_id values must be unique")
        platform_ids: set[str] = set()
        for station in self.stations:
            for platform in station.platforms:
                if platform.platform_id in platform_ids:
                    raise ValueError(f"Duplicate platform_id {platform.platform_id}")
                platform_ids.add(platform.platform_id)
                if platform.track_id not in track_ids:
                    raise ValueError(f"Platform {platform.platform_id} references unknown track")
                if platform.block_id not in block_ids:
                    raise ValueError(f"Platform {platform.platform_id} references unknown block")
                block = self.route.blocks[self.route.block_index(platform.block_id)]
                if platform.position_in_block_m > block.length_m:
                    raise ValueError(f"Platform {platform.platform_id} exceeds block length")

        crossover_ids = [item.crossover_id for item in self.crossovers]
        if len(crossover_ids) != len(set(crossover_ids)):
            raise ValueError("crossover_id values must be unique")
        for crossover in self.crossovers:
            if crossover.block_id not in block_ids:
                raise ValueError(f"Crossover {crossover.crossover_id} references unknown block")
            if crossover.from_track_id not in track_ids or crossover.to_track_id not in track_ids:
                raise ValueError(f"Crossover {crossover.crossover_id} references unknown track")
            block = self.route.blocks[self.route.block_index(crossover.block_id)]
            if crossover.end_position_m > block.length_m:
                raise ValueError(f"Crossover {crossover.crossover_id} exceeds block length")
        crossover_set = set(crossover_ids)

        for signal in self.signals:
            if signal.crossover_id is not None and signal.crossover_id not in crossover_set:
                raise ValueError(f"Signal {signal.signal_id} references unknown crossover {signal.crossover_id}")

        station_set = set(station_ids)
        for stop in self.primary_station_stops:
            if stop.station_id not in station_set:
                raise ValueError(f"Primary stop references unknown station {stop.station_id}")
        for change in self.primary_track_changes:
            if change.crossover_id not in crossover_set:
                raise ValueError(f"Primary track change references unknown crossover {change.crossover_id}")
            if change.turnaround_signal_id is not None and change.turnaround_signal_id not in signal_set:
                raise ValueError(f"Primary track change references unknown turnaround signal {change.turnaround_signal_id}")

        train_ids = [self.train.train_id]
        for i, run in enumerate(self.additional_train_runs):
            self._validate_train(run.train, f"additional_train_runs[{i}].train", block_ids, track_ids)
            self._validate_journey(run.journey, f"additional_train_runs[{i}].journey", block_ids)
            train_ids.append(run.train.train_id)
            for stop in run.station_stops:
                if stop.station_id not in station_set:
                    raise ValueError(f"Train {run.train.train_id} stop references unknown station {stop.station_id}")
            for change in run.track_changes:
                if change.crossover_id not in crossover_set:
                    raise ValueError(f"Train {run.train.train_id} references unknown crossover {change.crossover_id}")
                if change.turnaround_signal_id is not None and change.turnaround_signal_id not in signal_set:
                    raise ValueError(f"Train {run.train.train_id} references unknown turnaround signal {change.turnaround_signal_id}")
        if len(train_ids) != len(set(train_ids)):
            raise ValueError("train_id values must be unique across all train runs")

        for schedule in self.environment.signal_states:
            if schedule.signal_id not in signal_set:
                raise ValueError(f"Signal schedule references unknown signal {schedule.signal_id}")
            self._validate_timeline(schedule.timeline, f"signal {schedule.signal_id}")
        for schedule in self.environment.weather:
            if schedule.block_id not in block_ids:
                raise ValueError(f"Weather schedule references unknown block {schedule.block_id}")
            self._validate_timeline(schedule.timeline, f"weather {schedule.block_id}")
        for restriction in [*self.environment.temporary_speed_restrictions, *self.environment.maintenance_restrictions]:
            if restriction.block_id not in block_ids:
                raise ValueError(f"Restriction references unknown block {restriction.block_id}")
            block = self.route.blocks[self.route.block_index(restriction.block_id)]
            if restriction.end_position_m > block.length_m:
                raise ValueError(f"Restriction {restriction.restriction_id} exceeds block length")
        for crossing in self.environment.crossings:
            if crossing.block_id not in block_ids:
                raise ValueError(f"Crossing {crossing.crossing_id} references unknown block")
            block = self.route.blocks[self.route.block_index(crossing.block_id)]
            if crossing.position_in_block_m > block.length_m:
                raise ValueError(f"Crossing {crossing.crossing_id} exceeds block length")
            self._validate_timeline(crossing.timeline, f"crossing {crossing.crossing_id}")
        return self

    @staticmethod
    def _validate_timeline(timeline, label: str) -> None:
        starts = [entry.start_time_s for entry in timeline]
        if starts != sorted(starts):
            raise ValueError(f"{label} timeline must be sorted by start_time_s")
        if starts and starts[0] != 0:
            raise ValueError(f"{label} timeline must start at time 0")


class TelemetryFrame(BaseModel):
    scenario_id: str
    train_id: str
    tick: int
    sim_time_s: float

    track_id: str = "TRACK-1"
    direction: TrainDirection = TrainDirection.FORWARD
    active: bool = True
    completed: bool = False
    current_station_id: Optional[str] = None

    current_block_id: str
    position_in_block_m: float
    route_position_m: float
    speed_kmh: float
    acceleration_ms2: float
    control_action: str
    control_reason: str

    current_block_speed_limit_kmh: float
    current_gradient_percent: float
    current_curve_speed_limit_kmh: Optional[float] = None
    effective_speed_ceiling_kmh: float

    weather: WeatherCondition
    visibility_m: float

    distance_to_destination_m: float
    route_progress: float

    next_signal_aspect: Optional[SignalAspect] = None
    distance_to_next_signal_m: Optional[float] = None
    second_signal_aspect: Optional[SignalAspect] = None
    distance_to_second_signal_m: Optional[float] = None

    next_speed_limit_kmh: Optional[float] = None
    distance_to_next_speed_change_m: Optional[float] = None
    next_curve_limit_kmh: Optional[float] = None
    distance_to_next_curve_m: Optional[float] = None

    next_tsr_limit_kmh: Optional[float] = None
    distance_to_next_tsr_m: Optional[float] = None
    next_crossing_state: Optional[CrossingState] = None
    distance_to_next_crossing_m: Optional[float] = None

    actual_remaining_time_s: Optional[float] = None
    actual_arrival_simulation_s: Optional[float] = None
    total_journey_time_s: Optional[float] = None
