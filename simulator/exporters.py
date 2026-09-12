from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import SimulationConfig, TelemetryFrame


class BatchExporter:
    def __init__(self):
        self._rows: list[dict] = []

    def add(self, frames: list[TelemetryFrame]) -> None:
        self._rows.extend(frame.model_dump(mode="json") for frame in frames)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def to_parquet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

    def clear(self) -> None:
        self._rows = []


class ParquetTelemetryExporter:
    """Build the ML-facing one-second telemetry table.

    CSV export intentionally remains the raw TelemetryFrame dump.  Parquet rows
    are enriched with the static train/block prerequisites and current manual
    constraint context while preserving one row per train per simulator tick.
    Absolute schedule timestamps are left null until a timetable supplies them.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._blocks = {block.block_id: block for block in config.route.blocks}
        self._stations = {station.station_id: station for station in config.stations}
        self._train_context = self._build_train_context()

    def _build_train_context(self) -> dict[str, dict]:
        contexts = {
            self.config.train.train_id: {
                "train": self.config.train,
                "journey": self.config.journey,
                "stops": self.config.primary_station_stops,
            }
        }
        for run in self.config.additional_train_runs:
            contexts[run.train.train_id] = {
                "train": run.train,
                "journey": run.journey,
                "stops": run.station_stops,
            }
        return contexts

    @staticmethod
    def _restriction_is_active(restriction, frame: TelemetryFrame) -> bool:
        if restriction.block_id != frame.current_block_id:
            return False
        if not (restriction.start_position_m <= frame.position_in_block_m <= restriction.end_position_m):
            return False
        if frame.sim_time_s < restriction.start_time_s:
            return False
        return restriction.end_time_s is None or frame.sim_time_s < restriction.end_time_s

    def _active_limit(self, restrictions: Iterable, frame: TelemetryFrame) -> float | None:
        limits = [
            restriction.speed_limit_kmh
            for restriction in restrictions
            if self._restriction_is_active(restriction, frame)
        ]
        return min(limits) if limits else None

    def _station_context(self, frames: list[TelemetryFrame]) -> dict[tuple[str, int, float], tuple[str | None, float | None]]:
        result: dict[tuple[str, int, float], tuple[str | None, float | None]] = {}
        served: dict[str, set[str]] = {train_id: set() for train_id in self._train_context}
        previous_station: dict[str, str | None] = {train_id: None for train_id in self._train_context}

        for frame in frames:
            context = self._train_context[frame.train_id]
            stops = context["stops"]
            last_station = previous_station[frame.train_id]
            if last_station is not None and frame.current_station_id != last_station:
                served[frame.train_id].add(last_station)

            station_id = frame.current_station_id
            if station_id is None:
                station_id = next(
                    (stop.station_id for stop in stops if stop.station_id not in served[frame.train_id]),
                    None,
                )

            halt_time = next(
                (stop.dwell_time_s for stop in stops if stop.station_id == station_id),
                None,
            )
            result[(frame.train_id, frame.tick, frame.sim_time_s)] = (station_id, halt_time)
            previous_station[frame.train_id] = frame.current_station_id

        return result

    @staticmethod
    def _speed_gradient_30s(current_mps: float, previous_mps: float | None) -> float | None:
        if previous_mps is None:
            return None
        scale = max(abs(current_mps), abs(previous_mps), 1.0)
        return (current_mps - previous_mps) / scale

    def to_dataframe(self, frames: list[TelemetryFrame]) -> pd.DataFrame:
        station_context = self._station_context(frames)
        speed_history: dict[str, dict[float, float]] = {}
        rows: list[dict] = []

        for frame in frames:
            block = self._blocks[frame.current_block_id]
            context = self._train_context[frame.train_id]
            train = context["train"]
            station_id, scheduled_halt_time_s = station_context[(frame.train_id, frame.tick, frame.sim_time_s)]
            station = self._stations.get(station_id) if station_id is not None else None

            train_speed_mps = frame.speed_kmh / 3.6
            history = speed_history.setdefault(frame.train_id, {})
            previous_mps = history.get(round(frame.sim_time_s - 30.0, 6))
            speed_gradient_30s = self._speed_gradient_30s(train_speed_mps, previous_mps)
            history[round(frame.sim_time_s, 6)] = train_speed_mps

            current_tsr_limit = self._active_limit(self.config.environment.temporary_speed_restrictions, frame)
            current_maintenance_limit = self._active_limit(self.config.environment.maintenance_restrictions, frame)

            rows.append({
                "scenario_id": frame.scenario_id,
                "trajectory_id": f"{frame.scenario_id}:{frame.train_id}",
                "train_id": frame.train_id,
                "train_name": train.train_name,
                "tick": frame.tick,
                "sim_time_s": frame.sim_time_s,
                "timestamp": None,
                "train_max_speed_kmh": train.max_speed_kmh,
                "train_length_m": train.length_m,
                "train_accel_ms2": train.accel_ms2,
                "train_service_decel_ms2": train.service_decel_ms2,
                "train_emergency_decel_ms2": train.emergency_decel_ms2,
                "block_id": block.block_id,
                "block_length_m": block.length_m,
                "block_speed_limit_kmh": block.speed_limit_kmh,
                "block_gradient_percent": block.gradient_percent,
                "block_curve_radius_m": block.curve_radius_m,
                "block_curve_speed_limit_kmh": block.curve_speed_limit_kmh,
                "block_curve_direction": block.curve_direction.value if block.curve_direction is not None else None,
                "track_id": frame.track_id,
                "direction": frame.direction.value,
                "position_in_block_m": frame.position_in_block_m,
                "route_position_m": frame.route_position_m,
                "route_progress": frame.route_progress,
                "train_speed_kmh": frame.speed_kmh,
                "train_speed_mps": train_speed_mps,
                "acceleration_ms2": frame.acceleration_ms2,
                "speed_gradient_30s": speed_gradient_30s,
                "control_action": frame.control_action,
                "control_reason": frame.control_reason,
                "effective_speed_ceiling_kmh": frame.effective_speed_ceiling_kmh,
                "weather": frame.weather.value,
                "visibility_m": frame.visibility_m,
                "tsr_active": current_tsr_limit is not None,
                "current_tsr_limit_kmh": current_tsr_limit,
                "next_tsr_limit_kmh": frame.next_tsr_limit_kmh,
                "distance_to_next_tsr_m": frame.distance_to_next_tsr_m,
                "maintenance_active": current_maintenance_limit is not None,
                "maintenance_limit_kmh": current_maintenance_limit,
                "next_signal_aspect": frame.next_signal_aspect.value if frame.next_signal_aspect is not None else None,
                "distance_to_next_signal_m": frame.distance_to_next_signal_m,
                "second_signal_aspect": frame.second_signal_aspect.value if frame.second_signal_aspect is not None else None,
                "distance_to_second_signal_m": frame.distance_to_second_signal_m,
                "next_speed_limit_kmh": frame.next_speed_limit_kmh,
                "distance_to_next_speed_change_m": frame.distance_to_next_speed_change_m,
                "next_curve_limit_kmh": frame.next_curve_limit_kmh,
                "distance_to_next_curve_m": frame.distance_to_next_curve_m,
                "next_crossing_state": frame.next_crossing_state.value if frame.next_crossing_state is not None else None,
                "distance_to_next_crossing_m": frame.distance_to_next_crossing_m,
                "current_station_id": frame.current_station_id,
                "station_id": station_id,
                "station_name": station.station_name if station is not None else None,
                "scheduled_arrival_time": None,
                "scheduled_departure_time": None,
                "scheduled_halt_time_s": scheduled_halt_time_s,
                "expected_arrival_time": None,
                "delta_time_s": None,
                "distance_to_destination_m": frame.distance_to_destination_m,
                "active": frame.active,
                "completed": frame.completed,
                "actual_remaining_time_s": frame.actual_remaining_time_s,
                "actual_arrival_simulation_s": frame.actual_arrival_simulation_s,
                "total_journey_time_s": frame.total_journey_time_s,
            })

        return pd.DataFrame(rows)

    def to_parquet(self, frames: list[TelemetryFrame], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(frames).to_parquet(path, index=False)


class LiveExporter:
    @staticmethod
    def to_json(frame: TelemetryFrame) -> str:
        return frame.model_dump_json()

    @staticmethod
    def to_json_batch(frames: list[TelemetryFrame]) -> str:
        return json.dumps([f.model_dump(mode="json") for f in frames])
