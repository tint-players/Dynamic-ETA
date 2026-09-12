from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from simulator.config_loader import load_simulation_config
from simulator.dataset import label_completed_journey, label_completed_multi_train_journey
from simulator.engine import SimulationEngine
from simulator.exporters import BlockVisitExporter, ParquetTelemetryExporter
from simulator.models import (
    BlockWeatherSchedule,
    MaintenanceRestriction,
    SignalAspect,
    SimulationConfig,
    TelemetryFrame,
    TemporarySpeedRestriction,
    WeatherCondition,
    WeatherTimelineEntry,
)
from simulator.network_engine_v4 import NetworkSimulationEngineV4
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "dashboard_runs"


def _display_path(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _new_engine(config: SimulationConfig, scenario_id: str):
    if config.additional_train_runs or config.stations or config.dynamic_signalling:
        return NetworkSimulationEngineV4Restrictive(config, scenario_id=scenario_id)
    return SimulationEngine(config, scenario_id=scenario_id)


@dataclass
class SimulationSession:
    session_id: str
    scenario_id: str
    scenario_name: str
    config: SimulationConfig
    engine: Any
    playback_speed: float = 1.0
    playing: bool = False
    lock: Lock = field(default_factory=Lock)
    frames: list[TelemetryFrame] = field(default_factory=list)
    export_paths: dict[str, str] | None = None
    baseline_config: SimulationConfig = field(init=False, repr=False)
    manual_weather_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.baseline_config = self.config.model_copy(deep=True)
        if not self.frames:
            self.frames = self.snapshots()

    @property
    def is_multi_train(self) -> bool:
        return isinstance(self.engine, NetworkSimulationEngineV4)

    def snapshots(self) -> list[TelemetryFrame]:
        if self.is_multi_train:
            return self.engine.snapshot_all()
        return [self.engine.snapshot()]

    def snapshot(self) -> TelemetryFrame:
        return self.snapshots()[0]

    def signal_states(self) -> dict[str, str]:
        if self.is_multi_train:
            return {key: value.value for key, value in self.engine.signal_states().items()}
        return {}

    def crossing_states(self) -> dict[str, str]:
        if self.is_multi_train:
            return {key: value.value for key, value in self.engine.crossing_states().items()}
        return {}

    def _block(self, block_id: str):
        return next((block for block in self.config.route.blocks if block.block_id == block_id), None)

    def _refresh_weather_cache(self) -> None:
        if hasattr(self.engine, "_weather"):
            self.engine._weather = {item.block_id: item for item in self.config.environment.weather}

    def inject_weather(
        self,
        block_id: str,
        condition: WeatherCondition,
        visibility_m: float,
        duration_s: float | None,
    ) -> str:
        if self._block(block_id) is None:
            raise ValueError(f"Unknown block_id: {block_id}")

        start_time_s = self.engine.sim_time_s
        end_time_s = start_time_s + duration_s if duration_s is not None else None
        schedule = next((item for item in self.config.environment.weather if item.block_id == block_id), None)
        entry = WeatherTimelineEntry(start_time_s=start_time_s, condition=condition, visibility_m=visibility_m)
        if schedule is None:
            schedule = BlockWeatherSchedule(block_id=block_id, timeline=[entry])
            self.config.environment.weather.append(schedule)
        else:
            schedule.timeline.append(entry)

        if end_time_s is not None:
            schedule.timeline.append(WeatherTimelineEntry(
                start_time_s=end_time_s,
                condition=WeatherCondition.CLEAR,
                visibility_m=10000,
            ))

        schedule.timeline.sort(key=lambda item: item.start_time_s)
        self._refresh_weather_cache()
        self.manual_weather_overrides[block_id] = {
            "block_id": block_id,
            "condition": condition.value,
            "visibility_m": visibility_m,
            "start_time_s": start_time_s,
            "end_time_s": end_time_s,
        }
        duration_text = "until reset" if duration_s is None else f"for {duration_s:g}s"
        return f"Weather {condition.value} applied to {block_id} {duration_text}"

    def inject_signal(self, signal_id: str, aspect: SignalAspect, duration_s: float | None) -> str:
        if not isinstance(self.engine, NetworkSimulationEngineV4Restrictive):
            raise ValueError("Manual signal injection requires the restrictive network engine")
        self.engine.set_manual_signal_override(signal_id, aspect, duration_s)
        duration_text = "until reset" if duration_s is None else f"for {duration_s:g}s"
        return f"Signal {signal_id} forced to {aspect.value} {duration_text}"

    def inject_speed_restriction(
        self,
        kind: str,
        block_id: str,
        start_position_m: float,
        end_position_m: float,
        speed_limit_kmh: float,
        duration_s: float | None,
    ) -> str:
        block = self._block(block_id)
        if block is None:
            raise ValueError(f"Unknown block_id: {block_id}")
        if start_position_m < 0 or end_position_m > block.length_m:
            raise ValueError(f"Restriction range must stay within {block_id} (0-{block.length_m:g}m)")

        start_time_s = self.engine.sim_time_s
        end_time_s = start_time_s + duration_s if duration_s is not None else None
        suffix = uuid4().hex[:8].upper()

        if kind == "tsr":
            restriction = TemporarySpeedRestriction(
                restriction_id=f"TSR-MANUAL-{suffix}",
                block_id=block_id,
                start_position_m=start_position_m,
                end_position_m=end_position_m,
                speed_limit_kmh=speed_limit_kmh,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
            )
            self.config.environment.temporary_speed_restrictions.append(restriction)
            label = "TSR"
        elif kind == "maintenance":
            restriction = MaintenanceRestriction(
                restriction_id=f"MAINT-MANUAL-{suffix}",
                block_id=block_id,
                start_position_m=start_position_m,
                end_position_m=end_position_m,
                speed_limit_kmh=speed_limit_kmh,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
            )
            self.config.environment.maintenance_restrictions.append(restriction)
            label = "Maintenance speed limit"
        else:
            raise ValueError(f"Unknown restriction kind: {kind}")

        duration_text = "until reset" if duration_s is None else f"for {duration_s:g}s"
        return (
            f"{label} {restriction.restriction_id} applied to {block_id} "
            f"{start_position_m:g}-{end_position_m:g}m at {speed_limit_kmh:g} km/h {duration_text}"
        )

    def active_constraints(self) -> dict[str, Any]:
        sim_time_s = self.engine.sim_time_s
        expired_weather = [
            block_id
            for block_id, item in self.manual_weather_overrides.items()
            if item["end_time_s"] is not None and sim_time_s >= item["end_time_s"]
        ]
        for block_id in expired_weather:
            self.manual_weather_overrides.pop(block_id, None)

        weather = [dict(item) for item in self.manual_weather_overrides.values()]
        tsr = [
            restriction.model_dump(mode="json")
            for restriction in self.config.environment.temporary_speed_restrictions
            if restriction.restriction_id.startswith("TSR-MANUAL-")
            and restriction.start_time_s <= sim_time_s
            and (restriction.end_time_s is None or sim_time_s < restriction.end_time_s)
        ]
        maintenance = [
            restriction.model_dump(mode="json")
            for restriction in self.config.environment.maintenance_restrictions
            if restriction.restriction_id.startswith("MAINT-MANUAL-")
            and restriction.start_time_s <= sim_time_s
            and (restriction.end_time_s is None or sim_time_s < restriction.end_time_s)
        ]
        signals: list[dict[str, Any]] = []
        if isinstance(self.engine, NetworkSimulationEngineV4Restrictive):
            for signal_id, (aspect, end_time_s) in self.engine.manual_signal_override_details().items():
                signals.append({
                    "signal_id": signal_id,
                    "aspect": aspect.value,
                    "start_time_s": None,
                    "end_time_s": end_time_s,
                })
        return {
            "sim_time_s": sim_time_s,
            "weather": weather,
            "tsr": tsr,
            "maintenance": maintenance,
            "signals": signals,
        }

    def reset_constraints(self) -> None:
        baseline_environment = self.baseline_config.environment.model_copy(deep=True)
        self.config.environment.weather = baseline_environment.weather
        self.config.environment.temporary_speed_restrictions = baseline_environment.temporary_speed_restrictions
        self.config.environment.maintenance_restrictions = baseline_environment.maintenance_restrictions
        self.config.environment.signal_states = baseline_environment.signal_states
        self.manual_weather_overrides.clear()
        self._refresh_weather_cache()
        if isinstance(self.engine, NetworkSimulationEngineV4Restrictive):
            self.engine.clear_manual_signal_overrides()

    def tick(self) -> list[TelemetryFrame]:
        with self.lock:
            frames = self.engine.tick()
            if frames:
                self.frames.extend(frames)
                if self.engine.is_complete and self.export_paths is None:
                    self.export_paths = self._export_completed_journey()
            return frames

    def _export_completed_journey(self) -> dict[str, str]:
        labelled = label_completed_multi_train_journey(self.frames) if self.is_multi_train else label_completed_journey(self.frames)
        telemetry_exporter = ParquetTelemetryExporter(self.config)
        telemetry = telemetry_exporter.to_dataframe(labelled)
        block_exporter = BlockVisitExporter(self.config)
        block_visits = block_exporter.to_dataframe(telemetry)

        stem = f"{self.scenario_id}_{self.session_id[:8]}"
        csv_path = OUTPUT_DIR / f"{stem}.csv"
        parquet_path = OUTPUT_DIR / f"{stem}.parquet"
        block_csv_path = OUTPUT_DIR / f"{stem}_block_visits.csv"
        block_parquet_path = OUTPUT_DIR / f"{stem}_block_visits.parquet"

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry.to_csv(csv_path, index=False)
        telemetry.to_parquet(parquet_path, index=False)
        block_visits.to_csv(block_csv_path, index=False)
        block_visits.to_parquet(block_parquet_path, index=False)

        return {
            "csv": _display_path(csv_path),
            "parquet": _display_path(parquet_path),
            "block_csv": _display_path(block_csv_path),
            "block_parquet": _display_path(block_parquet_path),
        }

    def reset(self) -> list[TelemetryFrame]:
        with self.lock:
            self.config = self.baseline_config.model_copy(deep=True)
            self.engine = _new_engine(self.config, self.scenario_id)
            self.manual_weather_overrides.clear()
            self.playing = False
            self.playback_speed = 1.0
            frames = self.snapshots()
            self.frames = list(frames)
            self.export_paths = None
            return frames


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, SimulationSession] = {}

    @staticmethod
    def list_scenarios() -> list[str]:
        return sorted(path.name for path in EXAMPLES_DIR.glob("*.yaml"))

    @staticmethod
    def load_scenario(scenario_name: str) -> SimulationConfig:
        allowed = {name: EXAMPLES_DIR / name for name in SessionManager.list_scenarios()}
        path = allowed.get(scenario_name)
        if path is None:
            raise KeyError(f"Unknown scenario: {scenario_name}")
        return load_simulation_config(path)

    def create(self, scenario_name: str) -> SimulationSession:
        config = self.load_scenario(scenario_name)
        session_id = uuid4().hex
        scenario_id = f"live-{session_id[:8]}"
        session = SimulationSession(
            session_id=session_id,
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            config=config,
            engine=_new_engine(config, scenario_id),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SimulationSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session: {session_id}") from exc


sessions = SessionManager()
