from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .models import (
    CrossingState,
    MaintenanceType,
    SignalAspect,
    SimulationConfig,
    TelemetryFrame,
    TrainConfig,
    TrainDirection,
    WeatherCondition,
)
from .track_blocks import enumerate_track_blocks, track_block_id


SEQUENCE_WINDOW_STEPS = 60
SPEED_SCALE_KMH = 200.0
ACCEL_SCALE_MS2 = 2.0
EMERGENCY_DECEL_SCALE_MS2 = 3.0
LENGTH_SCALE_M = 1000.0
GRADIENT_SCALE_PERCENT = 5.0
CURVE_RADIUS_SCALE_M = 5000.0
VISIBILITY_SCALE_M = 10000.0


WEATHER_ORDER = (
    WeatherCondition.CLEAR,
    WeatherCondition.RAIN,
    WeatherCondition.HEAVY_RAIN,
    WeatherCondition.FOG,
    WeatherCondition.HEAVY_FOG,
)
SIGNAL_ORDER = (SignalAspect.GREEN, SignalAspect.YELLOW, SignalAspect.RED)


SEQUENCE_FEATURE_NAMES = (
    "history_present",
    "speed_norm",
    "acceleration_norm",
    "position_in_block_ratio",
    "route_progress",
    "distance_to_destination_ratio",
    "current_block_speed_limit_norm",
    "effective_speed_ceiling_norm",
    "gradient_norm",
    "current_curve_limit_norm",
    "current_curve_limit_present",
    "weather_clear",
    "weather_rain",
    "weather_heavy_rain",
    "weather_fog",
    "weather_heavy_fog",
    "visibility_norm",
    "next_signal_green",
    "next_signal_yellow",
    "next_signal_red",
    "next_signal_present",
    "distance_to_next_signal_ratio",
    "second_signal_green",
    "second_signal_yellow",
    "second_signal_red",
    "second_signal_present",
    "distance_to_second_signal_ratio",
    "next_speed_limit_norm",
    "next_speed_limit_present",
    "distance_to_next_speed_change_ratio",
    "next_curve_limit_norm",
    "next_curve_limit_present",
    "distance_to_next_curve_ratio",
    "next_tsr_limit_norm",
    "next_tsr_limit_present",
    "distance_to_next_tsr_ratio",
    "next_crossing_open",
    "next_crossing_closed",
    "next_crossing_present",
    "distance_to_next_crossing_ratio",
    "action_initial",
    "action_wait",
    "action_stop",
    "action_brake",
    "action_accelerate",
    "action_maintain",
    "action_other",
    "reason_destination",
    "reason_scheduled_departure",
    "reason_station",
    "reason_signal",
    "reason_train_ahead",
    "reason_crossing",
    "reason_crossover",
    "reason_turnaround",
    "reason_speed_control",
    "reason_other",
)


GRAPH_FEATURE_NAMES = (
    "length_ratio",
    "speed_limit_norm",
    "gradient_norm",
    "curve_radius_norm",
    "curve_radius_present",
    "curve_speed_limit_norm",
    "curve_speed_limit_present",
    "track_index_norm",
    "station_present",
    "signal_present",
    "crossing_present",
    "crossover_from",
    "crossover_to",
    "occupied",
    "train_count_ratio",
    "mean_speed_norm",
    "min_speed_norm",
    "stopped_train_count_ratio",
    "weather_clear",
    "weather_rain",
    "weather_heavy_rain",
    "weather_fog",
    "weather_heavy_fog",
    "visibility_norm",
    "tsr_active",
    "tsr_limit_norm",
    "tsr_limit_present",
    "maintenance_active",
    "maintenance_full_closure",
    "maintenance_limit_norm",
    "maintenance_limit_present",
    "platform_occupied",
)


CONTEXT_FEATURE_NAMES = (
    "train_max_speed_norm",
    "train_length_norm",
    "train_accel_norm",
    "train_service_decel_norm",
    "train_emergency_decel_norm",
    "direction_forward",
    "direction_reverse",
    "active",
    "at_station",
    "track_index_norm",
)


@dataclass(frozen=True, slots=True)
class MLFeatureBatch:
    """Leakage-safe model inputs for one target train at one simulation second.

    This object intentionally contains no training label. The same builder is
    used for live inference and offline dataset preparation; post-run outcome
    fields on TelemetryFrame are therefore ignored by construction.
    """

    sequence: tuple[tuple[float, ...], ...]
    graph: tuple[tuple[float, ...], ...]
    context: tuple[float, ...]
    node_ids: tuple[str, ...]
    current_node_index: int

    @property
    def sequence_shape(self) -> tuple[int, int]:
        return len(self.sequence), len(SEQUENCE_FEATURE_NAMES)

    @property
    def graph_shape(self) -> tuple[int, int]:
        return len(self.graph), len(GRAPH_FEATURE_NAMES)


class MLFeatureBuilder:
    """Shared offline/live feature builder for ETA model inputs.

    Inputs are simulator TelemetryFrame observations plus SimulationConfig.
    There is no pandas, torch, or frontend dependency here so the exact same
    feature logic can be called during Parquet preprocessing and live inference.
    """

    def __init__(self, config: SimulationConfig, *, sequence_window_steps: int = SEQUENCE_WINDOW_STEPS):
        if sequence_window_steps <= 0:
            raise ValueError("sequence_window_steps must be positive")
        self.config = config
        self.sequence_window_steps = sequence_window_steps
        self._sections = enumerate_track_blocks(config.route)
        self.node_ids = tuple(section.track_block_id for section in self._sections)
        self._node_index = {node_id: i for i, node_id in enumerate(self.node_ids)}
        self._train_configs = self._build_train_config_index(config)

    @staticmethod
    def _build_train_config_index(config: SimulationConfig) -> dict[str, TrainConfig]:
        index = {config.train.train_id: config.train}
        index.update({run.train.train_id: run.train for run in config.additional_train_runs})
        return index

    @staticmethod
    def _clip(value: float, low: float = -2.0, high: float = 2.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _one_hot(value, order: Sequence) -> tuple[float, ...]:
        return tuple(1.0 if value == item else 0.0 for item in order)

    @staticmethod
    def _optional_scaled(value: float | None, scale: float) -> tuple[float, float]:
        if value is None:
            return 0.0, 0.0
        return value / scale, 1.0

    @staticmethod
    def _active_timeline_entry(timeline, sim_time_s: float):
        if not timeline:
            return None
        active = timeline[0]
        for entry in timeline:
            if entry.start_time_s <= sim_time_s:
                active = entry
            else:
                break
        return active

    @staticmethod
    def _restriction_active(restriction, sim_time_s: float) -> bool:
        return sim_time_s >= restriction.start_time_s and (
            restriction.end_time_s is None or sim_time_s < restriction.end_time_s
        )

    def _weather_at(self, block_id: str, sim_time_s: float) -> tuple[WeatherCondition, float]:
        schedule = next((item for item in self.config.environment.weather if item.block_id == block_id), None)
        if schedule is None:
            return WeatherCondition.CLEAR, VISIBILITY_SCALE_M
        entry = self._active_timeline_entry(schedule.timeline, sim_time_s)
        if entry is None:
            return WeatherCondition.CLEAR, VISIBILITY_SCALE_M
        return entry.condition, entry.visibility_m

    def _sequence_row(self, frame: TelemetryFrame) -> tuple[float, ...]:
        route_length = max(1.0, self.config.route.total_length_m)
        block = self.config.route.blocks[self.config.route.block_index(frame.current_block_id)]
        block_length = max(1.0, block.length_m)

        curve_value, curve_present = self._optional_scaled(
            frame.current_curve_speed_limit_kmh,
            SPEED_SCALE_KMH,
        )
        next_speed_value, next_speed_present = self._optional_scaled(frame.next_speed_limit_kmh, SPEED_SCALE_KMH)
        next_curve_value, next_curve_present = self._optional_scaled(frame.next_curve_limit_kmh, SPEED_SCALE_KMH)
        next_tsr_value, next_tsr_present = self._optional_scaled(frame.next_tsr_limit_kmh, SPEED_SCALE_KMH)

        next_signal = self._one_hot(frame.next_signal_aspect, SIGNAL_ORDER)
        second_signal = self._one_hot(frame.second_signal_aspect, SIGNAL_ORDER)
        weather = self._one_hot(frame.weather, WEATHER_ORDER)

        crossing_open = 1.0 if frame.next_crossing_state == CrossingState.OPEN_FOR_TRAIN else 0.0
        crossing_closed = 1.0 if frame.next_crossing_state == CrossingState.CLOSED_FOR_TRAIN else 0.0
        crossing_present = 1.0 if frame.next_crossing_state is not None else 0.0

        action = (frame.control_action or "").upper()
        action_order = ("INITIAL", "WAIT", "STOP", "BRAKE", "ACCELERATE", "MAINTAIN")
        action_flags = tuple(1.0 if action == item else 0.0 for item in action_order)
        action_other = 1.0 if action and action not in action_order else 0.0

        reason = (frame.control_reason or "").upper()
        reason_flags = (
            1.0 if reason.startswith("DESTINATION") else 0.0,
            1.0 if reason.startswith("SCHEDULED_DEPARTURE") else 0.0,
            1.0 if reason.startswith("STATION") else 0.0,
            1.0 if "SIGNAL" in reason else 0.0,
            1.0 if reason.startswith("TRAIN_AHEAD") else 0.0,
            1.0 if reason.startswith("CROSSING") else 0.0,
            1.0 if reason.startswith("CROSSOVER") else 0.0,
            1.0 if reason.startswith("TURNAROUND") else 0.0,
            1.0 if reason.startswith("CURRENT_SPEED_CEILING") or reason.startswith("UPCOMING_BLOCK") else 0.0,
        )
        reason_other = 1.0 if reason and not any(reason_flags) else 0.0

        values = (
            1.0,
            frame.speed_kmh / SPEED_SCALE_KMH,
            self._clip(frame.acceleration_ms2 / ACCEL_SCALE_MS2),
            frame.position_in_block_m / block_length,
            frame.route_progress,
            frame.distance_to_destination_m / route_length,
            frame.current_block_speed_limit_kmh / SPEED_SCALE_KMH,
            frame.effective_speed_ceiling_kmh / SPEED_SCALE_KMH,
            self._clip(frame.current_gradient_percent / GRADIENT_SCALE_PERCENT),
            curve_value,
            curve_present,
            *weather,
            frame.visibility_m / VISIBILITY_SCALE_M,
            *next_signal,
            1.0 if frame.next_signal_aspect is not None else 0.0,
            (frame.distance_to_next_signal_m or 0.0) / route_length,
            *second_signal,
            1.0 if frame.second_signal_aspect is not None else 0.0,
            (frame.distance_to_second_signal_m or 0.0) / route_length,
            next_speed_value,
            next_speed_present,
            (frame.distance_to_next_speed_change_m or 0.0) / route_length,
            next_curve_value,
            next_curve_present,
            (frame.distance_to_next_curve_m or 0.0) / route_length,
            next_tsr_value,
            next_tsr_present,
            (frame.distance_to_next_tsr_m or 0.0) / route_length,
            crossing_open,
            crossing_closed,
            crossing_present,
            (frame.distance_to_next_crossing_m or 0.0) / route_length,
            *action_flags,
            action_other,
            *reason_flags,
            reason_other,
        )
        if len(values) != len(SEQUENCE_FEATURE_NAMES):
            raise RuntimeError("Sequence feature schema mismatch")
        return tuple(float(value) for value in values)

    def _sequence(self, history: Iterable[TelemetryFrame], target_train_id: str, current_time_s: float) -> tuple[tuple[float, ...], ...]:
        rows = sorted(
            (
                frame for frame in history
                if frame.train_id == target_train_id and frame.sim_time_s <= current_time_s + 1e-9
            ),
            key=lambda frame: (frame.sim_time_s, frame.tick),
        )[-self.sequence_window_steps:]

        padding = self.sequence_window_steps - len(rows)
        zero_row = tuple(0.0 for _ in SEQUENCE_FEATURE_NAMES)
        return tuple([zero_row] * padding + [self._sequence_row(frame) for frame in rows])

    def _occupancy_tracks(self, frame: TelemetryFrame) -> set[str]:
        tracks = {frame.track_id}
        for crossover in self.config.crossovers:
            if frame.track_id not in {crossover.from_track_id, crossover.to_track_id}:
                continue
            start = self.config.route.block_start_distance_m(crossover.block_id) + crossover.start_position_m
            end = self.config.route.block_start_distance_m(crossover.block_id) + crossover.end_position_m
            low, high = sorted((start, end))
            if low <= frame.route_position_m <= high:
                tracks.update({crossover.from_track_id, crossover.to_track_id})
        return tracks

    def _occupied_nodes(self, current_frames: Sequence[TelemetryFrame]) -> dict[str, list[TelemetryFrame]]:
        occupied: dict[str, list[TelemetryFrame]] = {node_id: [] for node_id in self.node_ids}
        for frame in current_frames:
            if not frame.active or frame.completed:
                continue
            train = self._train_configs.get(frame.train_id)
            if train is None:
                continue
            if frame.direction == TrainDirection.FORWARD:
                body_low = frame.route_position_m - train.length_m
                body_high = frame.route_position_m
            else:
                body_low = frame.route_position_m
                body_high = frame.route_position_m + train.length_m
            body_low, body_high = sorted((body_low, body_high))

            for track_id in self._occupancy_tracks(frame):
                for section in self._sections:
                    if section.track_id != track_id:
                        continue
                    if body_high >= section.route_start_m - 1e-6 and body_low <= section.route_end_m + 1e-6:
                        occupied[section.track_block_id].append(frame)
        return occupied

    def _graph(self, current_frames: Sequence[TelemetryFrame], sim_time_s: float) -> tuple[tuple[float, ...], ...]:
        route_length = max(1.0, self.config.route.total_length_m)
        train_total = max(1, len(self._train_configs))
        occupied = self._occupied_nodes(current_frames)

        station_nodes = {
            track_block_id(platform.track_id, platform.block_id)
            for station in self.config.stations
            for platform in station.platforms
        }
        signal_nodes = {
            track_block_id(signal.track_id, signal.protected_block_id)
            for signal in self.config.signals
        }
        crossing_blocks = {crossing.block_id for crossing in self.config.environment.crossings}
        crossover_from = {
            track_block_id(crossover.from_track_id, crossover.block_id)
            for crossover in self.config.crossovers
        }
        crossover_to = {
            track_block_id(crossover.to_track_id, crossover.block_id)
            for crossover in self.config.crossovers
        }

        occupied_platform_nodes: set[str] = set()
        for frame in current_frames:
            if not frame.current_station_id:
                continue
            station = next((item for item in self.config.stations if item.station_id == frame.current_station_id), None)
            if station is None:
                continue
            for platform in station.platforms:
                if platform.track_id == frame.track_id and platform.block_id == frame.current_block_id:
                    occupied_platform_nodes.add(track_block_id(platform.track_id, platform.block_id))

        rows: list[tuple[float, ...]] = []
        track_denominator = max(1, len(self.config.route.track_ids) - 1)
        for section in self._sections:
            node_id = section.track_block_id
            block = section.geometry
            node_frames = occupied[node_id]
            speeds = [frame.speed_kmh for frame in node_frames]
            weather, visibility = self._weather_at(section.block_id, sim_time_s)
            weather_flags = self._one_hot(weather, WEATHER_ORDER)

            tsrs = [
                restriction
                for restriction in self.config.environment.temporary_speed_restrictions
                if restriction.block_id == section.block_id
                and restriction.track_id in {None, section.track_id}
                and self._restriction_active(restriction, sim_time_s)
            ]
            maint = [
                restriction
                for restriction in self.config.environment.maintenance_restrictions
                if restriction.block_id == section.block_id
                and restriction.track_id in {None, section.track_id}
                and self._restriction_active(restriction, sim_time_s)
            ]
            tsr_limit = min((item.speed_limit_kmh for item in tsrs), default=None)
            maintenance_limit = min((item.speed_limit_kmh for item in maint), default=None)
            tsr_limit_value, tsr_limit_present = self._optional_scaled(tsr_limit, SPEED_SCALE_KMH)
            maint_limit_value, maint_limit_present = self._optional_scaled(maintenance_limit, SPEED_SCALE_KMH)

            curve_radius_value, curve_radius_present = self._optional_scaled(block.curve_radius_m, CURVE_RADIUS_SCALE_M)
            curve_speed_value, curve_speed_present = self._optional_scaled(block.curve_speed_limit_kmh, SPEED_SCALE_KMH)

            row = (
                block.length_m / route_length,
                block.speed_limit_kmh / SPEED_SCALE_KMH,
                self._clip(block.gradient_percent / GRADIENT_SCALE_PERCENT),
                curve_radius_value,
                curve_radius_present,
                curve_speed_value,
                curve_speed_present,
                section.track_index / track_denominator,
                1.0 if node_id in station_nodes else 0.0,
                1.0 if node_id in signal_nodes else 0.0,
                1.0 if section.block_id in crossing_blocks else 0.0,
                1.0 if node_id in crossover_from else 0.0,
                1.0 if node_id in crossover_to else 0.0,
                1.0 if node_frames else 0.0,
                len({frame.train_id for frame in node_frames}) / train_total,
                (sum(speeds) / len(speeds) / SPEED_SCALE_KMH) if speeds else 0.0,
                (min(speeds) / SPEED_SCALE_KMH) if speeds else 0.0,
                len({frame.train_id for frame in node_frames if frame.speed_kmh <= 1e-6}) / train_total,
                *weather_flags,
                visibility / VISIBILITY_SCALE_M,
                1.0 if tsrs else 0.0,
                tsr_limit_value,
                tsr_limit_present,
                1.0 if maint else 0.0,
                1.0 if any(item.maintenance_type == MaintenanceType.FULL_CLOSURE for item in maint) else 0.0,
                maint_limit_value,
                maint_limit_present,
                1.0 if node_id in occupied_platform_nodes else 0.0,
            )
            if len(row) != len(GRAPH_FEATURE_NAMES):
                raise RuntimeError("Graph feature schema mismatch")
            rows.append(tuple(float(value) for value in row))
        return tuple(rows)

    def _context(self, frame: TelemetryFrame) -> tuple[float, ...]:
        train = self._train_configs.get(frame.train_id)
        if train is None:
            raise KeyError(f"Unknown train_id for ML context: {frame.train_id}")
        try:
            track_index = self.config.route.track_ids.index(frame.track_id)
        except ValueError as exc:
            raise KeyError(f"Unknown track_id for ML context: {frame.track_id}") from exc
        track_denominator = max(1, len(self.config.route.track_ids) - 1)
        values = (
            train.max_speed_kmh / SPEED_SCALE_KMH,
            train.length_m / LENGTH_SCALE_M,
            train.accel_ms2 / ACCEL_SCALE_MS2,
            train.service_decel_ms2 / ACCEL_SCALE_MS2,
            train.emergency_decel_ms2 / EMERGENCY_DECEL_SCALE_MS2,
            1.0 if frame.direction == TrainDirection.FORWARD else 0.0,
            1.0 if frame.direction == TrainDirection.REVERSE else 0.0,
            1.0 if frame.active else 0.0,
            1.0 if frame.current_station_id is not None else 0.0,
            track_index / track_denominator,
        )
        if len(values) != len(CONTEXT_FEATURE_NAMES):
            raise RuntimeError("Context feature schema mismatch")
        return tuple(float(value) for value in values)

    def build(
        self,
        *,
        history: Iterable[TelemetryFrame],
        current_frames: Iterable[TelemetryFrame],
        target_train_id: str,
    ) -> MLFeatureBatch:
        current = list(current_frames)
        target = next((frame for frame in current if frame.train_id == target_train_id), None)
        if target is None:
            raise ValueError(f"No current frame for target train {target_train_id}")
        if any(abs(frame.sim_time_s - target.sim_time_s) > 1e-6 for frame in current):
            raise ValueError("current_frames must all describe the same simulation second")

        node_id = track_block_id(target.track_id, target.current_block_id)
        if node_id not in self._node_index:
            raise ValueError(f"Target train is on unknown operational section {node_id}")

        return MLFeatureBatch(
            sequence=self._sequence(history, target_train_id, target.sim_time_s),
            graph=self._graph(current, target.sim_time_s),
            context=self._context(target),
            node_ids=self.node_ids,
            current_node_index=self._node_index[node_id],
        )

    def build_from_records(
        self,
        *,
        history_records: Iterable[Mapping[str, object]],
        current_records: Iterable[Mapping[str, object]],
        target_train_id: str,
    ) -> MLFeatureBatch:
        """Offline adapter for Parquet/dataframe records using the same core path."""
        history = [TelemetryFrame.model_validate(dict(record)) for record in history_records]
        current = [TelemetryFrame.model_validate(dict(record)) for record in current_records]
        return self.build(history=history, current_frames=current, target_train_id=target_train_id)
