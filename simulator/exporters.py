from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .models import MaintenanceType, SimulationConfig, TelemetryFrame, TrainDirection


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
    """Build the Phase-2 one-second telemetry table.

    The same enriched dataframe can be written to CSV and Parquet. Every raw
    TelemetryFrame field is preserved, then static prerequisites and observable
    variable-feature context are added. No future event duration, future block
    exit, or future arrival information is introduced as a live feature.
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
    def _restriction_time_active(restriction, sim_time_s: float) -> bool:
        if sim_time_s < restriction.start_time_s:
            return False
        return restriction.end_time_s is None or sim_time_s < restriction.end_time_s

    @classmethod
    def _restriction_is_active(cls, restriction, frame: TelemetryFrame) -> bool:
        if restriction.block_id != frame.current_block_id:
            return False
        if not (restriction.start_position_m <= frame.position_in_block_m < restriction.end_position_m):
            return False
        return cls._restriction_time_active(restriction, frame.sim_time_s)

    def _active_limit(self, restrictions: Iterable, frame: TelemetryFrame) -> float | None:
        limits = [
            restriction.speed_limit_kmh
            for restriction in restrictions
            if self._restriction_is_active(restriction, frame)
        ]
        return min(limits) if limits else None

    def _maintenance_context(self, frame: TelemetryFrame) -> tuple[bool, str | None, float | None]:
        closure_prefix = "MAINTENANCE_CLOSURE:"
        if frame.control_reason.startswith(closure_prefix):
            restriction_id = frame.control_reason[len(closure_prefix):]
            restriction = next(
                (
                    item
                    for item in self.config.environment.maintenance_restrictions
                    if item.restriction_id == restriction_id
                    and item.maintenance_type == MaintenanceType.FULL_CLOSURE
                    and self._restriction_time_active(item, frame.sim_time_s)
                ),
                None,
            )
            if restriction is not None:
                return True, MaintenanceType.FULL_CLOSURE.value, None

        active = [
            restriction
            for restriction in self.config.environment.maintenance_restrictions
            if self._restriction_is_active(restriction, frame)
        ]
        if not active:
            return False, None, None

        full_closure = next(
            (restriction for restriction in active if restriction.maintenance_type == MaintenanceType.FULL_CLOSURE),
            None,
        )
        if full_closure is not None:
            return True, MaintenanceType.FULL_CLOSURE.value, None

        limit = min(restriction.speed_limit_kmh for restriction in active)
        return True, MaintenanceType.SPEED_RESTRICTION.value, limit

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

    def _platform_for_station(self, station_id: str | None, track_id: str):
        if station_id is None:
            return None
        station = self._stations.get(station_id)
        if station is None:
            return None
        return next((platform for platform in station.platforms if platform.track_id == track_id), None)

    def _body_bounds(self, frame: TelemetryFrame) -> tuple[float, float]:
        train = self._train_context[frame.train_id]["train"]
        if frame.direction == TrainDirection.FORWARD:
            return frame.route_position_m - train.length_m, frame.route_position_m
        return frame.route_position_m, frame.route_position_m + train.length_m

    def _platform_occupancy(
        self,
        frame: TelemetryFrame,
        station_id: str | None,
        same_time_frames: list[TelemetryFrame],
    ) -> tuple[str | None, bool, str | None]:
        platform = self._platform_for_station(station_id, frame.track_id)
        if platform is None:
            return None, False, None
        center = self.config.route.block_start_distance_m(platform.block_id) + platform.position_in_block_m
        platform_start = center - platform.length_m / 2.0
        platform_end = center + platform.length_m / 2.0
        for other in same_time_frames:
            if other.train_id == frame.train_id or not other.active or other.completed:
                continue
            if other.track_id != frame.track_id:
                continue
            body_start, body_end = self._body_bounds(other)
            if body_end >= platform_start - 1e-6 and body_start <= platform_end + 1e-6:
                return platform.platform_id, True, other.train_id
        return platform.platform_id, False, None

    def _train_ahead_context(
        self,
        frame: TelemetryFrame,
        same_time_frames: list[TelemetryFrame],
    ) -> tuple[bool, float | None, str | None]:
        if not frame.active or frame.completed:
            return False, None, None
        candidates: list[tuple[float, str]] = []
        for other in same_time_frames:
            if other.train_id == frame.train_id or not other.active or other.completed:
                continue
            if other.track_id != frame.track_id or other.direction != frame.direction:
                continue
            other_train = self._train_context[other.train_id]["train"]
            other_rear = other.route_position_m - (1.0 if frame.direction == TrainDirection.FORWARD else -1.0) * other_train.length_m
            sign = 1.0 if frame.direction == TrainDirection.FORWARD else -1.0
            gap = (other_rear - frame.route_position_m) * sign
            if gap >= 0.0:
                candidates.append((gap, other.train_id))
        if not candidates:
            return False, None, None
        candidates.sort(key=lambda item: item[0])
        return True, candidates[0][0], candidates[0][1]

    def to_dataframe(self, frames: list[TelemetryFrame]) -> pd.DataFrame:
        station_context = self._station_context(frames)
        speed_history: dict[str, dict[float, float]] = {}
        occupancy_wait: defaultdict[tuple[str, str], float] = defaultdict(float)
        frames_by_time: defaultdict[tuple[int, float], list[TelemetryFrame]] = defaultdict(list)
        for frame in frames:
            frames_by_time[(frame.tick, frame.sim_time_s)].append(frame)

        rows: list[dict] = []
        tick_seconds = self.config.simulation.tick_seconds

        for frame in frames:
            block = self._blocks[frame.current_block_id]
            context = self._train_context[frame.train_id]
            train = context["train"]
            station_id, scheduled_halt_time_s = station_context[(frame.train_id, frame.tick, frame.sim_time_s)]
            station = self._stations.get(station_id) if station_id is not None else None
            same_time_frames = frames_by_time[(frame.tick, frame.sim_time_s)]

            train_speed_mps = frame.speed_kmh / 3.6
            history = speed_history.setdefault(frame.train_id, {})
            previous_mps = history.get(round(frame.sim_time_s - 30.0, 6))
            speed_gradient_30s = self._speed_gradient_30s(train_speed_mps, previous_mps)
            history[round(frame.sim_time_s, 6)] = train_speed_mps

            current_tsr_limit = self._active_limit(self.config.environment.temporary_speed_restrictions, frame)
            maintenance_active, maintenance_type, current_maintenance_limit = self._maintenance_context(frame)
            platform_id, platform_occupied, occupying_train_id = self._platform_occupancy(
                frame, station_id, same_time_frames
            )
            train_ahead_present, distance_to_train_ahead_m, train_ahead_id = self._train_ahead_context(
                frame, same_time_frames
            )

            if (
                station_id is not None
                and frame.active
                and frame.current_station_id is None
                and platform_occupied
                and frame.speed_kmh <= 1e-6
            ):
                occupancy_wait[(frame.train_id, station_id)] += tick_seconds
            station_occupancy_wait_s = occupancy_wait[(frame.train_id, station_id)] if station_id is not None else 0.0

            row = frame.model_dump(mode="json")
            row.update({
                "trajectory_id": f"{frame.scenario_id}:{frame.train_id}",
                "train_name": train.train_name,
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
                "train_speed_kmh": frame.speed_kmh,
                "train_speed_mps": train_speed_mps,
                "speed_gradient_30s": speed_gradient_30s,
                "tsr_active": current_tsr_limit is not None,
                "current_tsr_limit_kmh": current_tsr_limit,
                "maintenance_active": maintenance_active,
                "maintenance_type": maintenance_type,
                "maintenance_limit_kmh": current_maintenance_limit,
                "station_id": station_id,
                "station_name": station.station_name if station is not None else None,
                "platform_id": platform_id,
                "station_platform_occupied_on_approach": platform_occupied,
                "occupying_train_id": occupying_train_id,
                "station_occupancy_wait_s": station_occupancy_wait_s,
                "train_ahead_present": train_ahead_present,
                "train_ahead_id": train_ahead_id,
                "distance_to_train_ahead_m": distance_to_train_ahead_m,
                "scheduled_arrival_time": None,
                "scheduled_departure_time": None,
                "scheduled_halt_time_s": scheduled_halt_time_s,
                "expected_arrival_time": None,
                "delta_time_s": None,
            })
            rows.append(row)

        return pd.DataFrame(rows)

    def to_parquet(self, frames: list[TelemetryFrame], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(frames).to_parquet(path, index=False)

    def to_csv(self, frames: list[TelemetryFrame], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(frames).to_csv(path, index=False)


class BlockVisitExporter:
    """Post-run one-row-per-train/block-visit outcome table.

    This table is intentionally separate from live tick telemetry because exit
    times and realised traversal durations are future outcomes until the visit is
    complete.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    @staticmethod
    def _count_entries(values: list[str | None], target: str) -> int:
        count = 0
        previous = None
        for value in values:
            if value == target and previous != target:
                count += 1
            previous = value
        return count

    @staticmethod
    def _weighted_duration(segment: pd.DataFrame, mask: pd.Series, exit_time: float) -> float:
        times = segment["sim_time_s"].astype(float).tolist()
        durations: list[float] = []
        for index, value in enumerate(times):
            next_time = times[index + 1] if index + 1 < len(times) else exit_time
            durations.append(max(0.0, next_time - value))
        return sum(duration for duration, enabled in zip(durations, mask.tolist()) if bool(enabled))

    def to_dataframe(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        if telemetry.empty:
            return pd.DataFrame()

        rows: list[dict] = []
        visit_counts: defaultdict[tuple[str, str], int] = defaultdict(int)

        for train_id, train_df in telemetry.groupby("train_id", sort=False):
            train_df = train_df.sort_values(["sim_time_s", "tick"], kind="stable").reset_index(drop=True)
            start_index = 0
            while start_index < len(train_df):
                first = train_df.iloc[start_index]
                block_id = first["block_id"]
                direction = first["direction"]
                end_index = start_index + 1
                while end_index < len(train_df):
                    candidate = train_df.iloc[end_index]
                    if candidate["block_id"] != block_id or candidate["direction"] != direction:
                        break
                    end_index += 1

                segment = train_df.iloc[start_index:end_index].copy()
                entry_time = float(segment.iloc[0]["sim_time_s"])
                if end_index < len(train_df):
                    exit_time = float(train_df.iloc[end_index]["sim_time_s"])
                else:
                    exit_time = float(segment.iloc[-1]["sim_time_s"])
                actual_block_time_s = max(0.0, exit_time - entry_time)

                visit_key = (str(train_id), str(block_id))
                visit_counts[visit_key] += 1
                visit_index = visit_counts[visit_key]

                weather_durations: Counter[str] = Counter()
                times = segment["sim_time_s"].astype(float).tolist()
                weathers = segment["weather"].astype(str).tolist()
                for index, weather in enumerate(weathers):
                    next_time = times[index + 1] if index + 1 < len(times) else exit_time
                    weather_durations[weather] += max(0.0, next_time - times[index])
                dominant_weather = (
                    weather_durations.most_common(1)[0][0]
                    if weather_durations and sum(weather_durations.values()) > 0
                    else str(segment.iloc[0]["weather"])
                )

                speed = segment["train_speed_kmh"].astype(float)
                stopped = speed <= 1e-6
                tsr_mask = segment["tsr_active"].fillna(False).astype(bool)
                maintenance_mask = segment["maintenance_active"].fillna(False).astype(bool)
                platform_wait_mask = (
                    segment["station_platform_occupied_on_approach"].fillna(False).astype(bool)
                    & stopped
                    & segment["current_station_id"].isna()
                )
                signal_hold_mask = segment["control_reason"].astype(str).str.startswith("RED_SIGNAL:") & stopped
                crossing_hold_mask = segment["control_reason"].astype(str).str.startswith("CROSSING:") & stopped
                traffic_hold_mask = segment["control_reason"].astype(str).str.startswith("TRAIN_AHEAD:") & stopped
                crossover_hold_mask = segment["control_reason"].astype(str).str.startswith("CROSSOVER_RESERVED:") & stopped
                maintenance_hold_mask = segment["control_reason"].astype(str).str.startswith("MAINTENANCE_CLOSURE:") & stopped
                dwell_mask = segment["current_station_id"].notna() & stopped

                tsr_limits = pd.to_numeric(segment.loc[tsr_mask, "current_tsr_limit_kmh"], errors="coerce").dropna()
                maintenance_limits = pd.to_numeric(segment.loc[maintenance_mask, "maintenance_limit_kmh"], errors="coerce").dropna()
                maintenance_types = segment.loc[maintenance_mask, "maintenance_type"].dropna().astype(str)
                if (maintenance_types == MaintenanceType.FULL_CLOSURE.value).any():
                    dominant_maintenance_type = MaintenanceType.FULL_CLOSURE.value
                elif not maintenance_types.empty:
                    dominant_maintenance_type = MaintenanceType.SPEED_RESTRICTION.value
                else:
                    dominant_maintenance_type = None
                signal_values = segment["next_signal_aspect"].where(segment["next_signal_aspect"].notna(), None).tolist()

                rows.append({
                    "scenario_id": segment.iloc[0]["scenario_id"],
                    "train_id": train_id,
                    "block_id": block_id,
                    "block_visit_index": visit_index,
                    "direction": direction,
                    "track_id_at_entry": segment.iloc[0]["track_id"],
                    "track_id_at_exit": segment.iloc[-1]["track_id"],
                    "entry_sim_time_s": entry_time,
                    "exit_sim_time_s": exit_time,
                    "actual_block_time_s": actual_block_time_s,
                    "entry_speed_kmh": float(speed.iloc[0]),
                    "exit_speed_kmh": float(speed.iloc[-1]),
                    "average_speed_kmh": float(speed.mean()),
                    "minimum_speed_kmh": float(speed.min()),
                    "stopped_time_s": self._weighted_duration(segment, stopped, exit_time),
                    "block_speed_limit_kmh": segment.iloc[0]["block_speed_limit_kmh"],
                    "train_max_speed_kmh": segment.iloc[0]["train_max_speed_kmh"],
                    "weather_at_entry": segment.iloc[0]["weather"],
                    "weather_at_exit": segment.iloc[-1]["weather"],
                    "dominant_weather": dominant_weather,
                    "minimum_visibility_m": float(segment["visibility_m"].astype(float).min()),
                    "average_visibility_m": float(segment["visibility_m"].astype(float).mean()),
                    "tsr_encountered": bool(tsr_mask.any()),
                    "tsr_min_limit_kmh": float(tsr_limits.min()) if not tsr_limits.empty else None,
                    "tsr_exposure_s": self._weighted_duration(segment, tsr_mask, exit_time),
                    "maintenance_encountered": bool(maintenance_mask.any()),
                    "maintenance_type": dominant_maintenance_type,
                    "maintenance_limit_kmh": float(maintenance_limits.min()) if not maintenance_limits.empty else None,
                    "maintenance_exposure_s": self._weighted_duration(segment, maintenance_mask, exit_time),
                    "maintenance_hold_time_s": self._weighted_duration(segment, maintenance_hold_mask, exit_time),
                    "yellow_signal_count": self._count_entries(signal_values, "YELLOW"),
                    "red_signal_count": self._count_entries(signal_values, "RED"),
                    "signal_hold_time_s": self._weighted_duration(segment, signal_hold_mask, exit_time),
                    "station_in_block": bool(segment["current_station_id"].notna().any()),
                    "station_platform_occupied_on_approach": bool(segment["station_platform_occupied_on_approach"].fillna(False).any()),
                    "station_occupancy_wait_s": self._weighted_duration(segment, platform_wait_mask, exit_time),
                    "station_dwell_s": self._weighted_duration(segment, dwell_mask, exit_time),
                    "crossing_hold_time_s": self._weighted_duration(segment, crossing_hold_mask, exit_time),
                    "traffic_hold_time_s": self._weighted_duration(segment, traffic_hold_mask, exit_time),
                    "crossover_hold_time_s": self._weighted_duration(segment, crossover_hold_mask, exit_time),
                    "train_ahead_present": bool(segment["train_ahead_present"].fillna(False).any()),
                    "minimum_train_ahead_distance_m": pd.to_numeric(
                        segment["distance_to_train_ahead_m"], errors="coerce"
                    ).min(),
                })

                start_index = end_index

        return pd.DataFrame(rows)

    def to_parquet(self, telemetry: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(telemetry).to_parquet(path, index=False)

    def to_csv(self, telemetry: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(telemetry).to_csv(path, index=False)


class LiveExporter:
    @staticmethod
    def to_json(frame: TelemetryFrame) -> str:
        return frame.model_dump_json()

    @staticmethod
    def to_json_batch(frames: list[TelemetryFrame]) -> str:
        return json.dumps([f.model_dump(mode="json") for f in frames])
