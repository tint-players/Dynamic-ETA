from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from simulator.config_loader import load_simulation_config
from simulator.dataset import label_completed_journey, label_completed_multi_train_journey
from simulator.engine import SimulationEngine
from simulator.exporters import BatchExporter, ParquetTelemetryExporter
from simulator.models import SimulationConfig, TelemetryFrame
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

    def __post_init__(self) -> None:
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
        csv_exporter = BatchExporter()
        csv_exporter.add(labelled)
        parquet_exporter = ParquetTelemetryExporter(self.config)

        stem = f"{self.scenario_id}_{self.session_id[:8]}"
        csv_path = OUTPUT_DIR / f"{stem}.csv"
        parquet_path = OUTPUT_DIR / f"{stem}.parquet"
        csv_exporter.to_csv(csv_path)
        parquet_exporter.to_parquet(labelled, parquet_path)
        return {"csv": _display_path(csv_path), "parquet": _display_path(parquet_path)}

    def reset(self) -> list[TelemetryFrame]:
        with self.lock:
            self.engine = _new_engine(self.config, self.scenario_id)
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
