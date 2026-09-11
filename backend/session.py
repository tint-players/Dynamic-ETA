from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from uuid import uuid4

from simulator.config_loader import load_simulation_config
from simulator.engine import SimulationEngine
from simulator.models import SimulationConfig, TelemetryFrame


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@dataclass
class SimulationSession:
    session_id: str
    scenario_id: str
    scenario_name: str
    config: SimulationConfig
    engine: SimulationEngine
    playback_speed: float = 1.0
    playing: bool = False
    lock: Lock = field(default_factory=Lock)

    def snapshot(self) -> TelemetryFrame:
        with self.lock:
            return self.engine.snapshot()

    def tick(self) -> list[TelemetryFrame]:
        with self.lock:
            return self.engine.tick()

    def reset(self) -> TelemetryFrame:
        with self.lock:
            self.engine = SimulationEngine(self.config, scenario_id=self.scenario_id)
            self.playing = False
            self.playback_speed = 1.0
            return self.engine.snapshot()


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, SimulationSession] = {}

    @staticmethod
    def list_scenarios() -> list[str]:
        return sorted(path.name for path in EXAMPLES_DIR.glob("*.yaml"))

    @staticmethod
    def load_scenario(scenario_name: str) -> SimulationConfig:
        # Only allow known example basenames. Arbitrary filesystem paths are rejected.
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
            engine=SimulationEngine(config, scenario_id=scenario_id),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> SimulationSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session: {session_id}") from exc


sessions = SessionManager()
